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

        std::string filename = Thread::thread()->file_resolver()->resolve(props.get<std::string>("filename")).string();
        load_binary(filename);
    }

    void load_binary(const std::string& filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f) Throw("Could not open phase function file: %s", filename.c_str());

        char magic[8];
        f.read(magic, 8);
        if (std::strncmp(magic, "ATMPHASE", 8) != 0) {
            Throw("Invalid signature in phase file: %s", filename.c_str());
        }

        uint32_t version;
        f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        if (version != 2) Throw("Unsupported phase file version: %d", version);

        f.read(reinterpret_cast<char*>(&m_num_angles), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&m_num_phi_bins), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&m_num_wavelengths), sizeof(uint32_t));

        f.read(reinterpret_cast<char*>(&m_lambda_min), sizeof(float));
        f.read(reinterpret_cast<char*>(&m_lambda_max), sizeof(float));

        size_t total_elements = m_num_angles * m_num_phi_bins * m_num_wavelengths;
        std::vector<float> host_data(total_elements);
        f.read(reinterpret_cast<char*>(host_data.data()), total_elements * sizeof(float));

        std::vector<float> distr_data(total_elements);

        // --- THE ONLY CHANGE: Area Weighing ---
        float d_theta = dr::Pi<float> / float(m_num_angles - 1);

        for (size_t t = 0; t < m_num_angles; ++t) {
            // Calculate the physical area of this theta row's spherical bucket
            float theta_top = std::max(float(t) - 0.5f, 0.f) * d_theta;
            float theta_bot = std::min(float(t) + 0.5f, float(m_num_angles - 1)) * d_theta;
            float solid_angle_factor = dr::cos(theta_top) - dr::cos(theta_bot);

            for (size_t p = 0; p < m_num_phi_bins; ++p) {
                for (size_t w = 0; w < m_num_wavelengths; ++w) {
                    size_t src_idx = t * (m_num_phi_bins * m_num_wavelengths) + p * m_num_wavelengths + w;
                    size_t dst_idx = w * (m_num_angles * m_num_phi_bins) + t * m_num_phi_bins + p;

                    // The sampler now sees Energy (Intensity * Area), not just Intensity
                    distr_data[dst_idx] = host_data[src_idx] * solid_angle_factor;
                }
            }
        }

        std::vector<float> wvl_nodes(m_num_wavelengths);
        for (uint32_t i = 0; i < m_num_wavelengths; ++i) {
            wvl_nodes[i] = m_lambda_min + (m_lambda_max - m_lambda_min) * (float(i) / (m_num_wavelengths - 1));
        }

        ScalarVector2u distr_extents(m_num_phi_bins, m_num_angles);
        std::array<uint32_t, 1> param_res = { m_num_wavelengths };
        std::array<const float*, 1> param_values = { wvl_nodes.data() };

        m_distr = Distr2D(distr_data.data(), distr_extents, param_res, param_values);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &/* ctx */,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASK_ARGUMENT(active);

        Vector3f forward = -mi.wi;
        Vector3f s = dr::cross(m_up, forward);
        Float s_norm = dr::norm(s);
        Mask valid_s = s_norm > 1e-6f;
        s = dr::select(valid_s, s / dr::maximum(s_norm, 1e-8f), Frame3f(forward).s);
        Vector3f t = dr::cross(forward, s);

        Vector3f local_wo = Vector3f(
            dr::dot(wo, s),
            dr::dot(wo, t),
            dr::dot(wo, forward)
        );

        Float cos_theta = dr::clip(local_wo.z(), -1.f, 1.f);
        Float theta = dr::acos(cos_theta);
        Float phi   = dr::atan2(local_wo.y(), local_wo.x());

        Float u = dr::select(phi < 0.f, phi + dr::TwoPi<Float>, phi) * dr::InvTwoPi<Float>;
        Float v = theta * dr::InvPi<Float>;

        Spectrum val;
        Float pdf_avg = 0.f;

        for (size_t i = 0; i < dr::size_v<Spectrum>; ++i) {
            Float val_i = m_distr.eval(Point2f(u, v), &mi.wavelengths[i], active);

            Float scaled_val = val_i / (2.f * dr::square(dr::Pi<Float>));

            val[i] = scaled_val;
            pdf_avg += scaled_val;
        }

        pdf_avg /= dr::size_v<Spectrum>;

        return { val, dr::maximum(pdf_avg, 1e-9f) };
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        constexpr size_t Channels = dr::size_v<Spectrum>;
        UInt32 hero_idx = dr::minimum(UInt32(sample1 * Channels), Channels - 1);

        Float hero_lambda = mi.wavelengths[0];
        if constexpr (Channels > 1) {
            for (size_t i = 1; i < Channels; ++i) {
                hero_lambda = dr::select(hero_idx == UInt32(i), mi.wavelengths[i], hero_lambda);
            }
        }

        auto [sample_pos, dummy_pdf] = m_distr.sample(sample2, &hero_lambda, active);

        Float u = sample_pos.x();
        Float v = sample_pos.y();

        Float phi   = u * dr::TwoPi<Float>;
        Float theta = v * dr::Pi<Float>;

        auto [sin_theta, cos_theta] = dr::sincos(theta);
        auto [sin_phi, cos_phi]     = dr::sincos(phi);

        Vector3f wo_local(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        );

        Vector3f forward = -mi.wi;
        Vector3f s = dr::cross(m_up, forward);
        Float s_norm = dr::norm(s);
        Mask valid_s = s_norm > 1e-6f;
        s = dr::select(valid_s, s / dr::maximum(s_norm, 1e-8f), Frame3f(forward).s);
        Vector3f t = dr::cross(forward, s);

        Vector3f wo = s * wo_local.x() +
                      t * wo_local.y() +
                      forward * wo_local.z();

        auto [weight_vals, pdf] = eval_pdf(ctx, mi, wo, active);

        Spectrum final_weight = dr::select(pdf > 0.f, weight_vals / pdf, 0.f);

        return { wo, final_weight, pdf };
    }

    std::string to_string() const override {
        return tfm::format("AtmosphericPhase[\n"
                           "  lambda_min = %f,\n"
                           "  lambda_max = %f,\n"
                           "  angles = %i,\n"
                           "  phi_bins = %i,\n"
                           "  wavelengths = %i\n"
                           "]",
                           m_lambda_min, m_lambda_max,
                           m_num_angles, m_num_phi_bins, m_num_wavelengths);
    }

    MI_DECLARE_CLASS(AtmosphericPhaseFunction)
private:
    Vector3f m_up;
    Distr2D m_distr;

    uint32_t m_num_angles;
    uint32_t m_num_phi_bins;
    uint32_t m_num_wavelengths;
    float m_lambda_min;
    float m_lambda_max;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)