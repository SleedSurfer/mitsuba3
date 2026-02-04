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
        stream->read(&m_resolution, sizeof(uint32_t));
        stream->read(&m_num_channels, sizeof(uint32_t));
        stream->read(&m_min_wavelength, sizeof(float));
        stream->read(&m_max_wavelength, sizeof(float));

        //(Angle x Wavelength interleaved)
        size_t total_floats = (size_t) m_resolution * (size_t) m_num_channels;
        std::vector<float> host_data(total_floats);
        stream->read(host_data.data(), total_floats * sizeof(float));

        m_data = dr::load<FloatStorage>(host_data.data(), total_floats);

        std::vector<float> cdf_host(total_floats, 0.f);
        std::vector<float> norm_factors_host(m_num_channels, 0.f);

        float d_theta = float(dr::Pi<double>) / float(m_resolution - 1);

        for (uint32_t c = 0; c < m_num_channels; ++c) {
            double integral = 0.0;
            size_t cdf_offset = (size_t)c * (size_t)m_resolution;

            cdf_host[cdf_offset + 0] = 0.f;

            for (uint32_t i = 1; i < m_resolution; ++i) {
                float theta0 = (i - 1) * d_theta;
                float theta1 = i * d_theta;

                double mu0 = std::cos(theta0);
                double mu1 = std::cos(theta1);
                double d_mu = mu0 - mu1;

                // Lookup raw values (Interleaved layout: [Angle][Channel])
                float p0 = host_data[(i - 1) * m_num_channels + c];
                float p1 = host_data[i * m_num_channels + c];

                // Trapezoidal Integration
                integral += 0.5f * (p0 + p1) * d_mu;

                cdf_host[cdf_offset + i] = (float) integral;
            }

            float norm = (float) integral;
            if (norm <= 0.f) norm = 1.f;

            norm_factors_host[c] = norm;

            for (uint32_t i = 0; i < m_resolution; ++i)
                cdf_host[cdf_offset + i] /= norm;

            cdf_host[cdf_offset + m_resolution - 1] = 1.f;
        }

        m_cdf = dr::load<FloatStorage>(cdf_host.data(), total_floats);
        m_pdf_norm = dr::load<FloatStorage>(norm_factors_host.data(), m_num_channels);

        if (m_num_channels == 1) {
            m_wavelength_scale = 0.f;
        } else {
            m_wavelength_scale = (m_num_channels - 1) / (m_max_wavelength - m_min_wavelength);
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext & /*ctx*/,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Float mu = -dot(wo, mi.wi);
        Float mu_clamped = dr::clip(mu, -1.f, 1.f);
        Float theta = dr::acos(mu_clamped);

        Float angle_idx = theta * (Float(m_resolution - 1) * dr::InvPi<Float>);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));

        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float w_idx = (mi.wavelengths[0] - m_min_wavelength) * m_wavelength_scale;
        w_idx = dr::clip(w_idx, 0.f, ScalarFloat(m_num_channels - 1));
        UInt32 w_int = dr::round2int<UInt32>(w_idx);

        Float norm = dr::gather<Float>(m_pdf_norm, w_int, active);

        Float pdf = 0.f;
        if constexpr (is_spectral_v<Spectrum>) {
             // For spectral modes (Mono channel)
             pdf = value[0] / norm;
        } else {
             // For RGB modes (scalar_rgb), 'value' is a Color<float, 3>
             pdf = dr::mean(value) / norm;
        }
        pdf *= dr::InvTwoPi<Float>;
        return { value, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext & /*ctx*/,
           const MediumInteraction3f &mi,
           Float /*sample1*/, const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float w_idx = (mi.wavelengths[0] - m_min_wavelength) * m_wavelength_scale;
        w_idx = dr::clip(w_idx, 0.f, ScalarFloat(m_num_channels - 1));
        UInt32 w_int = dr::round2int<UInt32>(w_idx);

        Float theta = sample_theta_spectral(sample2.x(), w_int, active);

        auto [sin_theta, cos_theta] = dr::sincos(theta);
        Float mu = cos_theta;

        auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo_local{ sin_theta * cos_phi, sin_theta * sin_phi, -mu };
        Vector3f wo = Frame3f(mi.wi).to_world(wo_local);

        Float angle_idx = theta * (Float(m_resolution - 1) * dr::InvPi<Float>);
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float norm = dr::gather<Float>(m_pdf_norm, w_int, active);
        Float pdf = 0.f;

        if constexpr (is_spectral_v<Spectrum>) {
             pdf = value[0] / norm;
        } else {
             pdf = dr::mean(value) / norm;
        }
        pdf *= dr::InvTwoPi<Float>;

        return { wo, Spectrum(1.f), pdf };
    }

    std::string to_string() const override {
        return tfm::format(
            "AtmosphericPhaseFunction[LUT-only, Spectral-CDF]\n"
            "  filename = \"%s\",\n"
            "  resolution = %u,\n"
            "  wavelength_bins = %u,\n"
            "  wavelength_range_nm = [%f, %f]\n"
            "]",
            m_filename, m_resolution, m_num_channels,
            m_min_wavelength, m_max_wavelength
        );
    }

private:
    Spectrum lookup_interpolated(const Wavelength &wvls, Float angle_idx, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Spectrum result;

            UInt32 a0        = dr::floor2int<UInt32>(angle_idx + 1e-6f); //TODO better look at this later

            a0 = dr::minimum(a0, UInt32(m_resolution - 2));
            UInt32 a1        = a0 + 1u;

            Float t_angle    = angle_idx - Float(a0);

            UInt32 stride    = UInt32(m_num_channels);
            UInt32 offset_a0 = a0 * stride;
            UInt32 offset_a1 = a1 * stride;

            constexpr size_t n_wavelengths = Spectrum::Size;
            for (size_t i = 0; i < n_wavelengths; ++i) {
                Float wvl   = wvls[i];
                Float w_idx = (wvl - m_min_wavelength) * m_wavelength_scale;
                w_idx = dr::clip(w_idx, 0.f, ScalarFloat(m_num_channels - 1));

                UInt32 w0   = dr::floor2int<UInt32>(w_idx + 1e-6f);
                w0          = dr::minimum(w0, UInt32(m_num_channels - 2));
                UInt32 w1   = w0 + 1u;
                Float t_wvl = w_idx - Float(w0);

                Float v00 = dr::gather<Float>(m_data, offset_a0 + w0, active);
                Float v01 = dr::gather<Float>(m_data, offset_a0 + w1, active);
                Float v10 = dr::gather<Float>(m_data, offset_a1 + w0, active);
                Float v11 = dr::gather<Float>(m_data, offset_a1 + w1, active);

                result[i] = dr::lerp(dr::lerp(v00, v01, t_wvl), dr::lerp(v10, v11, t_wvl), t_angle);
            }
            return result;
        } else {
            return 0.f;
        }
    }


    MI_INLINE Float sample_theta_spectral(const Float &u, UInt32 w_int, Mask active) const {
        UInt32 base_offset = w_int * UInt32(m_resolution);

        UInt32 lo = 0u;
        UInt32 hi = UInt32(m_resolution - 1);

        for (int it = 0; it < 13; ++it) {
            UInt32 mid = (lo + hi) >> 1;
            Float c = dr::gather<Float>(m_cdf, base_offset + mid, active);
            Mask go_left = active && (u <= c);
            hi = dr::select(go_left, mid, hi);
            lo = dr::select(go_left, lo, mid + 1u);
        }

        UInt32 idx = dr::minimum(lo, UInt32(m_resolution - 1));
        UInt32 idx0 = dr::select(idx > 0u, idx - 1u, 0u);

        Float c0 = dr::gather<Float>(m_cdf, base_offset + idx0, active);
        Float c1 = dr::gather<Float>(m_cdf, base_offset + idx, active);

        Float denom = dr::maximum(c1 - c0, 1e-12f);
        Float s = dr::clip((u - c0) / denom, 0.f, 1.f);
        Float angle_idx_f = Float(idx0) + s;

        return angle_idx_f * (dr::Pi<Float> / Float(m_resolution - 1));
    }

    FloatStorage m_data;      // Raw Data: [Angle x Wavelength] (Interleaved)
    FloatStorage m_cdf;       // CDF:      [Wavelength x Angle] (Block/SoA)
    FloatStorage m_pdf_norm;  // Integral of p(theta) per wavelength

    std::string m_filename;
    uint32_t m_resolution = 0;
    uint32_t m_num_channels = 0;
    float m_min_wavelength = 0.f;
    float m_max_wavelength = 0.f;
    float m_wavelength_scale = 0.f;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)