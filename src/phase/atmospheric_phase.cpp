#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/phase.h>
#include <drjit/dynamic.h>
#include <drjit/texture.h>
#include <drjit/tensor.h>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AtmosphericPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    using Texture3 = dr::Texture<Float, 3>;
    using Distr2D = Marginal2D<Float, 1, true>;

    AtmosphericPhaseFunction(const Properties &props) : Base(props) {
        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);

        Vector3f up = props.get<Vector3f>("up", Vector3f(0.f, 0.f, 1.f));
        m_up = dr::normalize(up);

        // Read XML overrides if they exist, otherwise default to -1.0
        m_hg_weight = props.get<float>("hg_weight", -1.0f);
        m_g = props.get<float>("g", -1.0f);

        std::string filename = Thread::thread()->file_resolver()->resolve(props.get<std::string>("filename")).string();
        load_binary(filename);
    }

    void load_binary(const std::string& filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f) Throw("Could not open phase function file: %s", filename.c_str());

        char magic[8];
        f.read(magic, 8);
        if (std::strncmp(magic, "ATMPHASE", 8) != 0) Throw("Invalid signature in phase file");

        uint32_t version;
        f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

        // Demand Version 3
        if (version != 3) Throw("Unsupported phase file version: %d. Please clear your Python cache.", version);

        f.read(reinterpret_cast<char*>(&m_num_angles), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&m_num_phi_bins), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&m_num_wavelengths), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&m_lambda_min), sizeof(float));
        f.read(reinterpret_cast<char*>(&m_lambda_max), sizeof(float));

        // --- READ HYBRID PARAMS ---
        float file_hg, file_g;
        f.read(reinterpret_cast<char*>(&file_hg), sizeof(float));
        f.read(reinterpret_cast<char*>(&file_g), sizeof(float));

        // If the XML didn't specify an override, use the physically correct baked data
        if (m_hg_weight < 0.0f) m_hg_weight = file_hg;
        if (m_g < 0.0f) m_g = file_g;

        // Read the actual LUT payload
        size_t total_elements = m_num_angles * m_num_phi_bins * m_num_wavelengths;
        std::vector<float> host_data(total_elements);
        f.read(reinterpret_cast<char*>(host_data.data()), total_elements * sizeof(float));

        // --- BRAIN 1: The Raw Intensity Texture (Evaluator) ---
        // We load the pure Python data into a Texture3 for smooth, interpolation-safe lookups
        using FloatStorage = DynamicBuffer<Float>;
        FloatStorage device_data = dr::load<FloatStorage>(host_data.data(), total_elements);
        size_t shape[4] = { m_num_angles, m_num_phi_bins, m_num_wavelengths, 1 }; // Z, Y, X layout
        dr::Tensor<FloatStorage> tensor(device_data, 4, shape);
        m_volume = std::make_unique<Texture3>(tensor, true, true, dr::FilterMode::Linear, dr::WrapMode::Clamp);

        // --- BRAIN 2: The Area-Weighted Sampler ---
        // The sampler MUST know the physical bucket sizes, so we bake the Jacobian here.
        std::vector<float> distr_data(total_elements);
        float d_theta = dr::Pi<float> / std::max(float(m_num_angles - 1), 1.0f);

        for (size_t t = 0; t < m_num_angles; ++t) {
            float theta_top = std::max(float(t) - 0.5f, 0.f) * d_theta;
            float theta_bot = std::min(float(t) + 0.5f, float(m_num_angles - 1)) * d_theta;
            float solid_angle_factor = std::max(std::cos(theta_top) - std::cos(theta_bot), 1e-8f);

            for (size_t p = 0; p < m_num_phi_bins; ++p) {
                for (size_t w = 0; w < m_num_wavelengths; ++w) {
                    size_t src_idx = t * (m_num_phi_bins * m_num_wavelengths) + p * m_num_wavelengths + w;
                    size_t dst_idx = w * (m_num_angles * m_num_phi_bins) + t * m_num_phi_bins + p;
                    distr_data[dst_idx] = std::max(host_data[src_idx] * solid_angle_factor, 1e-10f);
                }
            }
        }

        std::vector<float> wvl_nodes(m_num_wavelengths);
        for (uint32_t i = 0; i < m_num_wavelengths; ++i) {
            wvl_nodes[i] = m_num_wavelengths > 1 ?
                m_lambda_min + (m_lambda_max - m_lambda_min) * (float(i) / float(m_num_wavelengths - 1)) : m_lambda_min;
        }

        ScalarVector2u distr_extents(m_num_phi_bins, m_num_angles);
        std::array<uint32_t, 1> param_res = { m_num_wavelengths };
        std::array<const float*, 1> param_values = { wvl_nodes.data() };
        m_distr = Distr2D(distr_data.data(), distr_extents, param_res, param_values);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext&,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASK_ARGUMENT(active);

        // Frame must be centered on the LIGHT ray (wo), not the view ray!
        Vector3f forward = dr::normalize(wo);
        Vector3f s_raw = dr::cross(m_up, forward);
        Float s_norm_sqr = dr::squared_norm(s_raw);
        Mask valid_s = s_norm_sqr > 1e-12f;
        Vector3f s = dr::select(valid_s, s_raw * dr::rsqrt(dr::maximum(s_norm_sqr, 1e-16f)), Frame3f(forward).s);
        Vector3f t = dr::cross(forward, s);

        // Evaluate the incoming view ray (-mi.wi) in the sun's local frame
        Vector3f local_view = Vector3f(dr::dot(-mi.wi, s), dr::dot(-mi.wi, t), dr::dot(-mi.wi, forward));
        Float cos_theta = dr::clip(local_view.z(), -1.f, 1.f);
        Float theta = dr::acos(cos_theta);
        Float phi   = dr::atan2(-local_view.y(), local_view.x());

        Float u = dr::select(phi < 0.f, phi + dr::TwoPi<Float>, phi) * dr::InvTwoPi<Float>;
        Float v = theta * dr::InvPi<Float>;

        // 1. Evaluate Analytical HG Lobe
        Float denom = 1.f + m_g * m_g - 2.f * m_g * cos_theta;
        Float hg_val = dr::InvFourPi<Float> * (1.f - m_g * m_g) / (denom * dr::safe_sqrt(dr::maximum(denom, 1e-7f)));

        // 2. Evaluate Tabulated Residual Lobe
        Spectrum lut_val;
        for (size_t i = 0; i < dr::size_v<Spectrum>; ++i) {
            Float w_i = (mi.wavelengths[i] - m_lambda_min) / (m_lambda_max - m_lambda_min);

            // DrJit requires us to pass a pointer for the output
            Float tex_val;
            m_volume->eval(dr::Array<Float, 3>(w_i, u, v), &tex_val, active);
            lut_val[i] = tex_val;
        }

        // 3. Blend exactly by the missing energy fraction
        Spectrum final_val = m_hg_weight * hg_val + (1.f - m_hg_weight) * lut_val;

        Float final_pdf = dr::mean(final_val);

        return { dr::select(dr::isnan(final_val), 0.f, final_val),
                 dr::maximum(dr::select(dr::isnan(final_pdf), 1e-9f, final_pdf), 1e-9f) };
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);
        Mask sample_hg = sample1 < m_hg_weight;

        Float s1_hg = sample1 / dr::maximum(m_hg_weight, 1e-5f);
        Float s1_lut = (sample1 - m_hg_weight) / dr::maximum(1.f - m_hg_weight, 1e-5f);

        // --- Sample HG ---
        Float sqr_term = (1.f - m_g * m_g) / (1.f - m_g + 2.f * m_g * sample2.y());
        Float cos_theta_hg = (1.f + m_g * m_g - sqr_term * sqr_term) / (2.f * m_g);
        cos_theta_hg = dr::select(dr::abs(m_g) < 1e-4f, 1.f - 2.f * sample2.y(), cos_theta_hg);
        Float sin_theta_hg = dr::safe_sqrt(1.f - cos_theta_hg * cos_theta_hg);
        Float phi_hg = dr::TwoPi<Float> * sample2.x();

        // ---  Sample LUT ---
        UInt32 num_channels = dr::size_v<Spectrum>;
        Float scaled_s1 = s1_lut * num_channels;
        UInt32 channel = dr::minimum(UInt32(scaled_s1), num_channels - 1);
        Float hero_lambda = dr::select(channel == 0, mi.wavelengths[0],
                            dr::select(channel == 1, mi.wavelengths[1],
                            dr::select(channel == 2, mi.wavelengths[2], mi.wavelengths[3])));
        auto [sample_pos, dummy_pdf] = m_distr.sample(Point2f(sample2.x(), sample2.y()), &hero_lambda, active);
        Float phi_lut = sample_pos.x() * dr::TwoPi<Float>;
        Float theta_lut = sample_pos.y() * dr::Pi<Float>;
        auto [sin_theta_lut, cos_theta_lut] = dr::sincos(theta_lut);

        // --- MERGE BRANCHES ---
        Float cos_theta = dr::select(sample_hg, cos_theta_hg, cos_theta_lut);
        Float sin_theta = dr::select(sample_hg, sin_theta_hg, sin_theta_lut);
        Float phi = dr::select(sample_hg, phi_hg, phi_lut);
        auto [sin_phi, cos_phi] = dr::sincos(phi);

        Vector3f wo_local(sin_theta * cos_phi, -sin_theta * sin_phi, cos_theta);

        // Transform to world space
        Vector3f forward = -dr::normalize(mi.wi);
        Vector3f s_raw = dr::cross(m_up, forward);
        Float s_n_sqr = dr::squared_norm(s_raw);
        Vector3f s = dr::select(s_n_sqr > 1e-12f, s_raw * dr::rsqrt(dr::maximum(s_n_sqr, 1e-16f)), Frame3f(forward).s);
        Vector3f t = dr::cross(forward, s);

        Vector3f wo = s * wo_local.x() + t * wo_local.y() + forward * wo_local.z();

        auto [weight_vals, pdf] = eval_pdf(ctx, mi, wo, active);

        return { wo, dr::select(pdf > 1e-9f, weight_vals / pdf, 0.f), pdf };
    }

    std::string to_string() const override { return "AtmosphericPhase[Hybrid]"; }
    MI_DECLARE_CLASS(AtmosphericPhaseFunction)
private:
    Vector3f m_up;
    std::unique_ptr<Texture3> m_volume;
    Distr2D m_distr;
    float m_hg_weight, m_g;
    uint32_t m_num_angles, m_num_phi_bins, m_num_wavelengths;
    float m_lambda_min, m_lambda_max;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)