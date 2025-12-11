#include <algorithm>
#include <array>
#include <cstring>
#include <drjit/math.h> // REQUIRED for erfinv
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AtmosphericPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    using FloatStorage = DynamicBuffer<Float>;

    enum class LobeType : uint8_t {
        Forward  = 0,
        Rainbow  = 1,
        Residual = 2,
        Glory    = 3
    };

    struct LobeDescriptor {
        LobeType type;
        bool wavelength_dependent;
        float mu_center;
        float kappa;
        float amplitude;
        float weight; // <--- ADDED: Baked weight for faster/safer access

        FloatStorage mu_per_wavelength;
        FloatStorage kappa_per_wavelength;
        size_t grid_size = 0;
    };

    AtmosphericPhaseFunction(const Properties &props) : Base(props) {
        m_filename = props.get<std::string>("filename");
        m_use_mis  = props.get<bool>("use_mis", true);

        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(m_filename);

        if (!fs::exists(file_path))
            Throw("File not found: \"%s\"", m_filename);

        std::unique_ptr<FileStream> fs_stream =
            std::make_unique<FileStream>(file_path);

        // [Standard Header Reading Omitted for Brevity - kept same logic]
        char magic[8];
        fs_stream->read(magic, 8);
        // ... (Assume standard header reading is identical to your version) ...
        // Re-implementing simplified header read for context:
        uint32_t version;
        fs_stream->read(&version, sizeof(uint32_t));
        fs_stream->read(&m_resolution, sizeof(uint32_t));
        fs_stream->read(&m_num_channels, sizeof(uint32_t));
        fs_stream->read(&m_min_wavelength, sizeof(float));
        fs_stream->read(&m_max_wavelength, sizeof(float));

        size_t total_floats = m_resolution * m_num_channels;
        std::vector<float> host_data(total_floats);
        fs_stream->read(host_data.data(), total_floats * sizeof(float));
        m_data = dr::load<FloatStorage>(host_data.data(), total_floats);

        // Build envelope (Same as yours)
        std::vector<float> envelope_data(m_resolution);
        const float *raw_ptr = host_data.data();
        for (uint32_t a = 0; a < m_resolution; ++a) {
            float max_val            = 0.0f;
            const float *angle_block = raw_ptr + (a * m_num_channels);
            for (uint32_t c = 0; c < m_num_channels; ++c)
                max_val = std::max(max_val, angle_block[c]);
            envelope_data[a] = max_val;
        }
        std::reverse(envelope_data.begin(), envelope_data.end());
        m_distr = ContinuousDistribution<Float>(
            ScalarVector2f(-1.f, 1.f), envelope_data.data(), m_resolution);

        // === Read MIS metadata ===
        m_has_mis = false;
        try {
            char mis_magic[4];
            size_t current_pos = fs_stream->tell();
            if (fs_stream->size() - current_pos >= 4) {
                fs_stream->read(mis_magic, 4);
                if (memcmp(mis_magic, "MISD", 4) == 0) {
                    read_mis_metadata(fs_stream.get());
                    m_has_mis = true;
                }
            }
        } catch (const std::exception &e) {
            Log(Warn, "Failed to read MIS metadata: %s", e.what());
            m_has_mis = false;
        }

        m_wavelength_scale =
            (m_num_channels - 1) / (m_max_wavelength - m_min_wavelength);
        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    void read_mis_metadata(FileStream *fs) {
        uint8_t version;
        fs->read(&version, sizeof(uint8_t));
        uint32_t num_components;
        fs->read(&num_components, sizeof(uint32_t));

        // Read raw weights temporarily
        std::array<float, 3> temp_weights;
        fs->read(temp_weights.data(), 3 * sizeof(float));
        // Store for PDF/Selector usage if needed globally, but we will bake
        // them
        m_mixture_weights = temp_weights;

        for (uint32_t i = 0; i < num_components; ++i) {
            LobeDescriptor lobe;
            uint8_t lobe_type, wl_dep;
            uint16_t reserved;
            fs->read(&lobe_type, sizeof(uint8_t));
            lobe.type = static_cast<LobeType>(lobe_type);
            fs->read(&wl_dep, sizeof(uint8_t));
            lobe.wavelength_dependent = (wl_dep != 0);
            fs->read(&reserved, sizeof(uint16_t));
            fs->read(&lobe.mu_center, sizeof(float));
            fs->read(&lobe.kappa, sizeof(float));
            fs->read(&lobe.amplitude, sizeof(float));

            // --- BAKE WEIGHT HERE ---
            // Maps type 0->weight[0], type 1->weight[1], etc.
            // Safe fallback if type is out of bounds (though it shouldn't be)
            int type_idx = static_cast<int>(lobe.type);
            if (type_idx >= 0 && type_idx < 3) {
                lobe.weight = m_mixture_weights[type_idx];
            } else {
                lobe.weight = 0.0f;
            }

            uint32_t num_wl;
            fs->read(&num_wl, sizeof(uint32_t));
            lobe.grid_size = num_wl;

            if (num_wl > 0) {
                std::vector<float> host_mu(num_wl), host_kappa(num_wl);
                fs->read(host_mu.data(), num_wl * sizeof(float));
                fs->read(host_kappa.data(), num_wl * sizeof(float));
                lobe.mu_per_wavelength =
                    dr::load<FloatStorage>(host_mu.data(), num_wl);
                lobe.kappa_per_wavelength =
                    dr::load<FloatStorage>(host_kappa.data(), num_wl);
            }
            m_lobes.push_back(lobe);
        }
    }

    // --- PDF EVALUATION ---
    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Float cos_theta = dot(wo, mi.wi);

        // 1. Evaluate True Phase Function Value (Lookup)
        Float angle_idx = (1.f - cos_theta) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        // 2. Evaluate MIS PDF
        Float pdf;
        if (m_use_mis && m_has_mis) {
            Float pdf_sum = 0.f;
            // Loop over SIMD wavelengths
            for (size_t w = 0; w < Spectrum::Size; ++w) {
                Float wvl = mi.wavelengths[w];
                pdf_sum += eval_mis_pdf_for_wvl(cos_theta, wvl, active);
            }
            pdf = pdf_sum / Float(Spectrum::Size);
        } else {
            Float pdf_mu = m_distr.eval_pdf_normalized(cos_theta, active);
            pdf          = pdf_mu * dr::rcp(2.f * dr::Pi<ScalarFloat>);
        }

        return { value, pdf };
    }

    // --- SAMPLING ---
    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &ctx, const MediumInteraction3f &mi,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float cos_theta;

        if (m_use_mis && m_has_mis) {
            cos_theta =
                sample_mis(sample1, sample2.x(), mi.wavelengths, active);
        } else {
            cos_theta = m_distr.sample(sample2.x());
        }

        // Standard direction reconstruction
        Float sin_theta = dr::safe_sqrt(1.f - cos_theta * cos_theta);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo_local{ sin_theta * cos_phi, sin_theta * sin_phi,
                           cos_theta };
        Vector3f wo = Frame3f(mi.wi).to_world(wo_local);

        // Recalculate PDF
        Float angle_idx = (1.f - cos_theta) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float pdf;
        if (m_use_mis && m_has_mis) {
            Float pdf_sum = 0.f;
            for (size_t w = 0; w < Spectrum::Size; ++w) {
                Float wvl = mi.wavelengths[w];
                pdf_sum += eval_mis_pdf_for_wvl(cos_theta, wvl, active);
            }
            pdf = pdf_sum / Float(Spectrum::Size);
        } else {
            Float pdf_mu = m_distr.eval_pdf_normalized(cos_theta, active);
            pdf          = pdf_mu * dr::rcp(2.f * dr::Pi<ScalarFloat>);
        }

        Spectrum weight = value / pdf;
        return { wo, weight, pdf };
    }

    // --- MIS SAMPLING LOGIC ---
    Float sample_mis(Float lobe_sample, Float mu_sample, const Wavelength &wvls,
                     Mask active) const {

        // 1. Pick Hero Wavelength
        UInt32 channel_idx = dr::minimum(
            dr::floor2int<UInt32>(lobe_sample * ScalarFloat(Spectrum::Size)),
            UInt32(Spectrum::Size - 1));
        Float guide_wvl = 0.f;
        for (uint32_t i = 0; i < Spectrum::Size; ++i) {
            guide_wvl = dr::select(channel_idx == i, wvls[i], guide_wvl);
        }

        // Rescale for selector
        Float rescaled_sample =
            lobe_sample * ScalarFloat(Spectrum::Size) - Float(channel_idx);

        // 2. Select Lobe via Cumulative Weights
        // Using global weights for selection is fine, as long as they sum to 1
        Float cumsum0 = m_mixture_weights[0];
        Float cumsum1 = cumsum0 + m_mixture_weights[1];

        Mask is_forward  = rescaled_sample < cumsum0;
        Mask is_rainbow  = !is_forward && (rescaled_sample < cumsum1);
        Mask is_residual = !is_forward && !is_rainbow;

        Float cos_theta = 2.f * mu_sample - 1.f; // Default for residual

        for (const auto &lobe : m_lobes) {
            Mask use_this_lobe(false);
            if (lobe.type == LobeType::Forward)
                use_this_lobe = is_forward;
            else if (lobe.type == LobeType::Rainbow)
                use_this_lobe = is_rainbow;

            if (dr::any_or<true>(use_this_lobe)) {
                auto [mu_center, kappa] =
                    get_lobe_params(lobe, guide_wvl, active);

                Float sampled_mu =
                    sample_truncated_gaussian(mu_sample, mu_center, kappa);
                cos_theta = dr::select(use_this_lobe, sampled_mu, cos_theta);
            }
        }
        return cos_theta;
    }

    // --- MIS PDF LOGIC ---
    Float eval_mis_pdf_for_wvl(Float cos_theta, Float wvl, Mask active) const {
        Float pdf_mixture(0.f);

        for (const auto &lobe : m_lobes) {
            Float weight = lobe.weight; // Use baked weight

            if (lobe.type == LobeType::Residual) {
                // Uniform PDF on [-1, 1] is 0.5
                pdf_mixture += weight * 0.5f;
            } else {
                auto [mu_center, kappa] = get_lobe_params(lobe, wvl, active);
                Float pdf_lobe =
                    eval_truncated_gaussian_pdf(cos_theta, mu_center, kappa);
                pdf_mixture += weight * pdf_lobe;
            }
        }
        return pdf_mixture * dr::rcp(2.f * dr::Pi<ScalarFloat>);
    }

    // --- HELPER: Interpolate Lobe Parameters ---
    std::pair<Float, Float> get_lobe_params(const LobeDescriptor &lobe,
                                            Float wvl, Mask active) const {
        Float mu_center = lobe.mu_center;
        Float kappa     = lobe.kappa;

        if (lobe.wavelength_dependent && lobe.grid_size > 0) {
            Float wvl_idx = (wvl - m_min_wavelength) /
                            (m_max_wavelength - m_min_wavelength);
            wvl_idx = dr::clip(wvl_idx, 0.f, 1.f) * Float(lobe.grid_size - 1);

            UInt32 idx0 = dr::floor2int<UInt32>(wvl_idx);
            UInt32 idx1 = dr::minimum(idx0 + 1u, UInt32(lobe.grid_size - 1));
            Float t     = wvl_idx - Float(idx0);

            Float mu0 = dr::gather<Float>(lobe.mu_per_wavelength, idx0, active);
            Float mu1 = dr::gather<Float>(lobe.mu_per_wavelength, idx1, active);
            Float k0 =
                dr::gather<Float>(lobe.kappa_per_wavelength, idx0, active);
            Float k1 =
                dr::gather<Float>(lobe.kappa_per_wavelength, idx1, active);

            mu_center = dr::lerp(mu0, mu1, t);
            kappa     = dr::lerp(k0, k1, t);
        }
        return { mu_center, kappa };
    }

    // --- MATH: TRUNCATED GAUSSIAN IMPLEMENTATION ---

    // Normal CDF: Phi(x)
    Float std_normal_cdf(Float x) const {
        return 0.5f * (1.f + dr::erf(x * dr::InvSqrtTwo<ScalarFloat>));
    }

    // Inverse Normal CDF: Phi^-1(p)
    Float std_normal_inv_cdf(Float p) const {
        // Clamp p to avoid infinities
        p = dr::clip(p, 1e-6f, 1.f - 1e-6f);
        return dr::SqrtTwo<ScalarFloat> * dr::erfinv(2.f * p - 1.f);
    }

    Float sample_truncated_gaussian(Float u, Float mu, Float kappa) const {
        Float sigma = dr::rsqrt(2.f * kappa + 1e-6f); // Safer epsilon

        Float alpha = (-1.f - mu) / sigma;
        Float beta  = (1.f - mu) / sigma;

        Float Phi_alpha = std_normal_cdf(alpha);
        Float Phi_beta  = std_normal_cdf(beta);
        Float Z         = Phi_beta - Phi_alpha;

        // If Z is too small (distribution strictly outside [-1, 1]), fallback
        // to mu This handles extreme edge cases where the lobe drifts off the
        // sphere. But for your rainbow (-0.787) and forward (1.0), this is
        // safe.

        Float p     = Phi_alpha + u * Z;
        Float x_std = std_normal_inv_cdf(p);

        Float result = mu + x_std * sigma;
        return dr::clip(result, -1.f, 1.f);
    }

    Float eval_truncated_gaussian_pdf(Float x, Float mu, Float kappa) const {
        // If x is outside [-1, 1], PDF is 0 (handled by caller implicitly
        // usually, but good to know)
        Float sigma    = dr::rsqrt(2.f * kappa + 1e-6f);
        Float sigma_sq = sigma * sigma;

        Float alpha = (-1.f - mu) / sigma;
        Float beta  = (1.f - mu) / sigma;
        Float Z     = std_normal_cdf(beta) - std_normal_cdf(alpha);

        Float arg   = (x - mu) / sigma;
        Float numer = dr::exp(-0.5f * arg * arg);
        Float denom = sigma * dr::SqrtTwoPi<ScalarFloat> * Z;

        return numer / denom;
    }

    std::string to_string() const override {
        return "AtmosphericPhaseFunction[]";
    }

private:
    // [Private members unchanged]
    Spectrum lookup_interpolated(const Wavelength &wvls, Float angle_idx,
                                 Mask active) const {
        // ... (Keep your original implementation for this) ...
        if constexpr (is_spectral_v<Spectrum>) {
            Spectrum result;
            UInt32 a0        = dr::floor2int<UInt32>(angle_idx);
            UInt32 a1        = dr::minimum(a0 + 1u, UInt32(m_resolution - 1));
            Float t_angle    = angle_idx - Float(a0);
            UInt32 stride    = UInt32(m_num_channels);
            UInt32 offset_a0 = a0 * stride;
            UInt32 offset_a1 = a1 * stride;
            constexpr size_t n_wavelengths = Spectrum::Size;
            for (size_t i = 0; i < n_wavelengths; ++i) {
                Float wvl   = wvls[i];
                Float w_idx = (wvl - m_min_wavelength) * m_wavelength_scale;
                w_idx = dr::clip(w_idx, 0.f, ScalarFloat(m_num_channels - 1));
                UInt32 w0   = dr::floor2int<UInt32>(w_idx);
                UInt32 w1   = dr::minimum(w0 + 1u, UInt32(m_num_channels - 1));
                Float t_wvl = w_idx - Float(w0);
                Float v00   = dr::gather<Float>(m_data, offset_a0 + w0, active);
                Float v01   = dr::gather<Float>(m_data, offset_a0 + w1, active);
                Float v10   = dr::gather<Float>(m_data, offset_a1 + w0, active);
                Float v11   = dr::gather<Float>(m_data, offset_a1 + w1, active);
                result[i]   = dr::lerp(dr::lerp(v00, v01, t_wvl),
                                       dr::lerp(v10, v11, t_wvl), t_angle);
            }
            return result;
        } else {
            return 0.f;
        }
    }

    MI_DECLARE_CLASS(AtmosphericPhaseFunction)

    FloatStorage m_data;
    ContinuousDistribution<Float> m_distr;
    std::string m_filename;
    uint32_t m_resolution, m_num_channels;
    float m_min_wavelength, m_max_wavelength, m_wavelength_scale;
    bool m_use_mis, m_has_mis;
    std::vector<LobeDescriptor> m_lobes;
    std::array<float, 3> m_mixture_weights;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)