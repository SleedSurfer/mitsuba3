#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>
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

    AtmosphericPhaseFunction(const Properties &props) : Base(props) {
        m_filename = props.get<std::string>("filename");

        m_up = dr::normalize(props.get<Vector3f>("up", Vector3f(0.f, 1.f, 0.f)));
        m_fscatter_elimit = props.get<float>("forward_scatter_limit", 1.0f);

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(m_filename);

        if (!fs::exists(file_path))
            Throw("File not found: \"%s\"", m_filename);

        auto stream = std::make_unique<FileStream>(file_path);

        char magic[8];
        stream->read(magic, 8);
        if (memcmp(magic, "ATMPHASE", 8) != 0)
            Throw("Invalid phase LUT file: \"%s\"", m_filename);

        uint32_t version;
        stream->read(&version, sizeof(uint32_t));

        if (version < 2)
            Throw("Legacy V1 binary detected. Please regenerate using V2 pipeline.");

        stream->read(&m_theta_bins, sizeof(uint32_t));
        stream->read(&m_phi_bins, sizeof(uint32_t));
        stream->read(&m_num_channels, sizeof(uint32_t));
        stream->read(&m_min_wavelength, sizeof(float));
        stream->read(&m_max_wavelength, sizeof(float));

        size_t total_floats = (size_t)m_theta_bins * (size_t)m_phi_bins * (size_t)m_num_channels;
        std::vector<float> host_data(total_floats);
        stream->read(host_data.data(), total_floats * sizeof(float));

        m_data = dr::load<FloatStorage>(host_data.data(), total_floats);

        // Precompute memory layouts for the 2D CDFs
        std::vector<float> marg_cdf_host(m_num_channels * m_theta_bins, 0.f);
        std::vector<float> cond_cdf_host(total_floats, 0.f);
        std::vector<float> norm_factors_host(m_num_channels, 0.f);

        double d_theta = dr::Pi<double> / (m_theta_bins - 1);
        double d_phi = (m_phi_bins > 1) ? (2.0 * dr::Pi<double> / (m_phi_bins - 1)) : (2.0 * dr::Pi<double>);

        std::vector<double> phi_integrals(m_theta_bins, 0.0);

        for (uint32_t c = 0; c < m_num_channels; ++c) {

            // 1. Build Conditional CDFs (Phi | Theta, Wvl)
            for (uint32_t t = 0; t < m_theta_bins; ++t) {
                double integral_phi = 0.0;
                size_t cond_offset = c * (m_theta_bins * m_phi_bins) + t * m_phi_bins;
                cond_cdf_host[cond_offset + 0] = 0.f;

                if (m_phi_bins > 1) {
                    for (uint32_t p = 1; p < m_phi_bins; ++p) {
                        float v0 = host_data[t * (m_phi_bins * m_num_channels) + (p - 1) * m_num_channels + c];
                        float v1 = host_data[t * (m_phi_bins * m_num_channels) + p * m_num_channels + c];
                        integral_phi += 0.5 * (v0 + v1) * d_phi;
                        cond_cdf_host[cond_offset + p] = (float)integral_phi;
                    }
                    double norm_phi = integral_phi > 0.0 ? integral_phi : 1.0;
                    for (uint32_t p = 0; p < m_phi_bins; ++p) {
                        cond_cdf_host[cond_offset + p] /= (float)norm_phi;
                    }
                    cond_cdf_host[cond_offset + m_phi_bins - 1] = 1.f;
                } else {
                    double raw_val = host_data[t * m_num_channels + c];
                    integral_phi = raw_val * 2.0 * dr::Pi<double>;
                    cond_cdf_host[cond_offset + 0] = 1.f;
                }
                phi_integrals[t] = integral_phi;
            }

            // 2. Build Marginal CDF (Theta | Wvl)
            double integral_theta = 0.0;
            size_t marg_offset = c * m_theta_bins;
            marg_cdf_host[marg_offset + 0] = 0.f;

            for (uint32_t t = 1; t < m_theta_bins; ++t) {
                double mu0 = std::cos((t - 1) * d_theta);
                double mu1 = std::cos(t * d_theta);
                double d_mu = mu0 - mu1;

                integral_theta += 0.5 * (phi_integrals[t - 1] + phi_integrals[t]) * d_mu;
                marg_cdf_host[marg_offset + t] = (float)integral_theta;
            }

            double norm_theta = integral_theta > 0.0 ? integral_theta : 1.0;
            norm_factors_host[c] = (float)norm_theta;

            for (uint32_t t = 0; t < m_theta_bins; ++t) {
                marg_cdf_host[marg_offset + t] /= (float)norm_theta;
            }
            marg_cdf_host[marg_offset + m_theta_bins - 1] = 1.f;
        }

        m_marginal_cdf = dr::load<FloatStorage>(marg_cdf_host.data(), marg_cdf_host.size());
        m_conditional_cdf = dr::load<FloatStorage>(cond_cdf_host.data(), cond_cdf_host.size());
        m_pdf_norm = dr::load<FloatStorage>(norm_factors_host.data(), m_num_channels);

        dr::eval(m_data, m_marginal_cdf, m_conditional_cdf, m_pdf_norm);

        m_wavelength_scale = (m_num_channels > 1) ? (m_num_channels - 1) / (m_max_wavelength - m_min_wavelength) : 0.f;

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext & /*ctx*/,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        // 1. Calculate the spherical coordinates of the outgoing ray 'wo'
        Float mu = -dr::dot(wo, mi.wi);
        Float theta = dr::acos(dr::clip(mu, -1.f, 1.f));

        Vector3f s = dr::cross(m_up, mi.wi);
        Float s_norm = dr::norm(s);
        Mask valid_s = s_norm > 1e-6f;
        s = dr::select(valid_s, s / dr::maximum(s_norm, 1e-8f), Frame3f(mi.wi).s);
        Vector3f t = dr::cross(mi.wi, s);

        Float wo_s = dr::dot(wo, s);
        Float wo_t = dr::dot(wo, t);
        Float phi = dr::atan2(wo_t, wo_s);
        phi = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        // 2. Fetch INTERPOLATED data (Linear space, smooth gradient)
        Spectrum actual_value = lookup_interpolated(mi.wavelengths, theta, phi, active);
        actual_value = dr::maximum(actual_value, 0.f);

        // 3. APPLY THE LIMITER (This is the 'value' we give to the integrator)
        Spectrum value = dr::minimum(actual_value, m_fscatter_elimit);

        // 4. CALCULATE CONTINUOUS PDF
        // We MUST use the squashed 'value' here so that (value / pdf) stays balanced.
        Float pdf_avg = 0.f;
        if constexpr (is_spectral_v<Spectrum>) {
            for (size_t i = 0; i < Spectrum::Size; ++i) {
                Float w_idx_i = dr::clip((mi.wavelengths[i] - m_min_wavelength) * m_wavelength_scale,
                                          0.f, ScalarFloat(m_num_channels - 1));
                UInt32 w_int_i = dr::round2int<UInt32>(w_idx_i);

                // Get the normalization factor for this wavelength
                Float norm_i = dr::gather<Float>(m_pdf_norm, w_int_i, active);

                // The continuous PDF is the value divided by the total integral
                pdf_avg += value[i] / dr::maximum(norm_i, 1e-8f);
            }
            pdf_avg /= ScalarFloat(Spectrum::Size);
        }

        return { value, pdf_avg };
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext & /*ctx*/,
           const MediumInteraction3f &mi,
           Float sample1, const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // --- Driver Wavelength Selection ---
        Float driver_wvl = mi.wavelengths[0];
        if constexpr (is_spectral_v<Spectrum>) {
            UInt32 driver_idx = dr::minimum(dr::floor2int<UInt32>(sample1 * Spectrum::Size), UInt32(Spectrum::Size - 1));
            for (size_t i = 1; i < Spectrum::Size; ++i) {
                driver_wvl = dr::select(driver_idx == UInt32(i), mi.wavelengths[i], driver_wvl);
            }
        }

        Float w_idx = dr::clip((driver_wvl - m_min_wavelength) * m_wavelength_scale, 0.f, ScalarFloat(m_num_channels - 1));
        UInt32 w_int = dr::round2int<UInt32>(w_idx);

        // --- Continuous CDF Inversion (The Jitter Fix) ---
        UInt32 t_int = sample_cdf_discrete(sample2.x(), m_marginal_cdf, w_int * m_theta_bins, m_theta_bins, active);
        Float cdf_t0 = dr::gather<Float>(m_marginal_cdf, w_int * m_theta_bins + t_int, active);
        Float cdf_t1 = dr::gather<Float>(m_marginal_cdf, w_int * m_theta_bins + dr::minimum(t_int + 1u, m_theta_bins - 1u), active);
        Float t_frac = dr::select(cdf_t1 > cdf_t0, (sample2.x() - cdf_t0) / (cdf_t1 - cdf_t0), 0.f);
        Float d_theta = dr::Pi<Float> / Float(m_theta_bins - 1);
        Float theta = (Float(t_int) + t_frac) * d_theta;

        UInt32 cond_offset = w_int * (m_theta_bins * m_phi_bins) + t_int * m_phi_bins;
        UInt32 p_int = sample_cdf_discrete(sample2.y(), m_conditional_cdf, cond_offset, m_phi_bins, active);
        Float d_phi = 2.f * dr::Pi<Float> / Float(m_phi_bins > 1 ? m_phi_bins - 1 : 1);
        Float phi;
        if (m_phi_bins > 1) {
            Float cdf_p0 = dr::gather<Float>(m_conditional_cdf, cond_offset + p_int, active);
            Float cdf_p1 = dr::gather<Float>(m_conditional_cdf, cond_offset + dr::minimum(p_int + 1u, m_phi_bins - 1u), active);
            Float p_frac = dr::select(cdf_p1 > cdf_p0, (sample2.y() - cdf_p0) / (cdf_p1 - cdf_p0), 0.f);
            phi = (Float(p_int) + p_frac) * d_phi;
        } else {
            phi = sample2.y() * 2.f * dr::Pi<Float>;
        }

        // --- Vector Math ---
        Vector3f s = dr::cross(m_up, mi.wi);
        Float s_norm = dr::norm(s);
        Mask valid_s = s_norm > 1e-6f;
        s = dr::select(valid_s, s / dr::maximum(s_norm, 1e-8f), Frame3f(mi.wi).s);
        Vector3f t = dr::cross(mi.wi, s);
        auto [sin_theta, cos_theta] = dr::sincos(theta);
        auto [sin_phi, cos_phi]     = dr::sincos(phi);
        Vector3f wo_local{ sin_theta * cos_phi, sin_theta * sin_phi, -cos_theta };
        Vector3f wo = s * wo_local.x() + t * wo_local.y() + mi.wi * wo_local.z();

        // --- Evaluation (Linear Space) ---
        Spectrum actual_value = lookup_interpolated(mi.wavelengths, theta, phi, active);
        actual_value = dr::maximum(actual_value, 0.f);

        // --- Squash ---
        Spectrum value = dr::minimum(actual_value, m_fscatter_elimit);

        // --- PDF (Synced with Squash) ---
        Float pdf_avg = 0.f;
        if constexpr (is_spectral_v<Spectrum>) {
            for (size_t i = 0; i < Spectrum::Size; ++i) {
                Float w_idx_i = dr::clip((mi.wavelengths[i] - m_min_wavelength) * m_wavelength_scale,
                                          0.f, ScalarFloat(m_num_channels - 1));
                UInt32 w_int_i = dr::round2int<UInt32>(w_idx_i);
                Float norm_i = dr::gather<Float>(m_pdf_norm, w_int_i, active);
                pdf_avg += value[i] / dr::maximum(norm_i, 1e-8f);
            }
            pdf_avg /= ScalarFloat(Spectrum::Size);
        }

        // --- Final MIS-Corrected Weight ---
        Spectrum weight = dr::select(pdf_avg > 1e-8f, value / pdf_avg, 0.f);

        return { wo, weight, pdf_avg };
    }

    std::string to_string() const override {
        return tfm::format(
            "AtmosphericPhaseFunction[LUT-only, Spectral-Anisotropic]\n"
            "  filename = \"%s\",\n"
            "  resolution = %ux%u,\n"
            "  forward_scatter_limit = %f,\n"
            "  wavelength_bins = %u,\n"
            "  wavelength_range_nm = [%f, %f]\n"
            "]",
            m_filename, m_theta_bins, m_phi_bins, m_num_channels,
            m_min_wavelength, m_max_wavelength
        );
    }

private:

    Spectrum lookup_nearest(const Wavelength &wvls, Float theta, Float phi, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Spectrum result;

            // Snap directly to the nearest integer bin index
            UInt32 a_idx = dr::clip(dr::round2int<UInt32>(theta * (Float(m_theta_bins - 1) * dr::InvPi<Float>)), 0u, m_theta_bins - 1u);
            UInt32 p_idx = dr::clip(dr::round2int<UInt32>(phi * (Float(m_phi_bins > 1 ? m_phi_bins - 1 : 1) * dr::InvTwoPi<Float>)), 0u, UInt32(m_phi_bins > 1 ? m_phi_bins - 1 : 0));

            UInt32 stride_p = m_num_channels;
            UInt32 stride_a = m_phi_bins * m_num_channels;

            for (size_t i = 0; i < Spectrum::Size; ++i) {
                Float w_idx = dr::clip((wvls[i] - m_min_wavelength) * m_wavelength_scale, 0.f, ScalarFloat(m_num_channels - 1));
                UInt32 w_int = dr::clip(dr::round2int<UInt32>(w_idx), 0u, m_num_channels - 1u);

                result[i] = dr::gather<Float>(m_data, a_idx * stride_a + p_idx * stride_p + w_int, active);
            }
            return result;
        } else {
            return 0.f;
        }
    }

    MI_INLINE UInt32 sample_cdf_discrete(const Float &u, const FloatStorage& cdf_array, UInt32 base_offset, UInt32 resolution, Mask active) const {
        UInt32 lo = 0u, hi = resolution - 1u;

        for (int it = 0; it < 16; ++it) {
            UInt32 mid = (lo + hi) >> 1;
            Float c = dr::gather<Float>(cdf_array, base_offset + mid, active);
            Mask go_left = active && (u <= c);
            hi = dr::select(go_left, mid, hi);
            lo = dr::select(go_left, lo, mid + 1u);
        }

        return dr::minimum(lo, resolution - 1u);
    }

    // Trilinear Interpolation (Theta x Phi x Wvl)
    Spectrum lookup_interpolated(const Wavelength &wvls, Float theta, Float phi, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Spectrum result;

            Float a_idx = dr::clip(theta * (Float(m_theta_bins - 1) * dr::InvPi<Float>), 0.f, ScalarFloat(m_theta_bins - 1));
            Float p_idx = dr::clip(phi * (Float(m_phi_bins > 1 ? m_phi_bins - 1 : 1) * dr::InvTwoPi<Float>), 0.f, ScalarFloat(m_phi_bins > 1 ? m_phi_bins - 1 : 1));

            UInt32 a0 = dr::minimum(dr::floor2int<UInt32>(a_idx), UInt32(m_theta_bins - 2));
            UInt32 a1 = a0 + 1u;
            Float t_a = a_idx - Float(a0);

            UInt32 p0 = dr::minimum(dr::floor2int<UInt32>(p_idx), UInt32(m_phi_bins > 1 ? m_phi_bins - 2 : 0));
            UInt32 p1 = p0 + (m_phi_bins > 1 ? 1u : 0u);
            Float t_p = p_idx - Float(p0);

            UInt32 stride_p = m_num_channels;
            UInt32 stride_a = m_phi_bins * m_num_channels;

            for (size_t i = 0; i < Spectrum::Size; ++i) {
                Float w_idx = dr::clip((wvls[i] - m_min_wavelength) * m_wavelength_scale, 0.f, ScalarFloat(m_num_channels - 1));
                UInt32 w0 = dr::minimum(dr::floor2int<UInt32>(w_idx), UInt32(m_num_channels - 2));
                UInt32 w1 = w0 + 1u;
                Float t_w = w_idx - Float(w0);

                auto get_val = [&](UInt32 a, UInt32 p, UInt32 w) {
                    return dr::gather<Float>(m_data, a * stride_a + p * stride_p + w, active);
                };

                Float v000 = get_val(a0, p0, w0), v001 = get_val(a0, p0, w1);
                Float v010 = get_val(a0, p1, w0), v011 = get_val(a0, p1, w1);
                Float v100 = get_val(a1, p0, w0), v101 = get_val(a1, p0, w1);
                Float v110 = get_val(a1, p1, w0), v111 = get_val(a1, p1, w1);

                Float v00 = dr::lerp(v000, v001, t_w);
                Float v01 = dr::lerp(v010, v011, t_w);
                Float v10 = dr::lerp(v100, v101, t_w);
                Float v11 = dr::lerp(v110, v111, t_w);

                Float v0 = dr::lerp(v00, v01, t_p);
                Float v1 = dr::lerp(v10, v11, t_p);

                result[i] = dr::lerp(v0, v1, t_a);
            }
            return result;
        } else {
            Throw("AtmosphericPhase is wavelength dependent and needs spectral variant to work.");
            return 0.f;
        }
    }

    FloatStorage m_data;
    FloatStorage m_marginal_cdf;
    FloatStorage m_conditional_cdf;
    FloatStorage m_pdf_norm;

    std::string m_filename;
    Vector3f m_up;
    uint32_t m_theta_bins = 0;
    uint32_t m_phi_bins = 0;
    uint32_t m_num_channels = 0;
    float m_min_wavelength = 0.f;
    float m_max_wavelength = 0.f;
    float m_wavelength_scale = 0.f;
    float m_fscatter_elimit = 1.0f;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)