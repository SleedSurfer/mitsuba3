#include <algorithm>
#include <cstring>
#include <limits>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

/**
 * AtmosphericPhaseFunction (stripped LUT-only, JIT-safe, linear λ interpolation)
 *
 * This removes *all* MIS metadata / lobe logic and samples *directly from the LUT*.
 *
 * LUT contains phase p_ω(μ, λ) normalized such that:
 *    ∫_{S^2} p_ω dω = ∫_{-1}^1 p_ω(μ) 2π dμ = 1    (per wavelength)
 *
 * Sampling scheme implemented here:
 *  - Pick a "hero lane" j uniformly among Spectrum::Size lanes using sample1
 *  - Use that lane's wavelength λ_j to define a 1D distribution over μ:
 *      p_μ(μ | λ_j) = 2π * p_ω(μ | λ_j)
 *    To get linear interpolation in wavelength, we build *two* discrete CDFs from the LUT bins
 *    neighboring λ_j and mix them by t in [0,1].
 *  - Sample μ by inverting the mixed CDF (binary search using dr::gather)
 *  - Sample φ uniformly
 *
 * PDF returned:
 *  - We return the *marginal* density of wo under the above sampling procedure:
 *      p_ω(wo) = (1/N) Σ_{lane i} p_ω(wo | λ_i)
 *    where p_ω(wo | λ_i) = p_μ(μ | λ_i) / (2π) and p_μ is the same mixed-λ distribution.
 *
 * Notes:
 *  - This implementation is JIT-safe: no selecting C++ objects based on vector indices.
 *  - Distributions are represented as buffers and accessed with dr::gather.
 *  - Everything is discrete over the LUT μ grid (angles). We do linear interpolation
 *    between the two neighboring μ bins during inversion + pdf evaluation.
 */
template <typename Float, typename Spectrum>
class AtmosphericPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    using FloatStorage = DynamicBuffer<Float>;

    AtmosphericPhaseFunction(const Properties &props) : Base(props) {
        m_filename = props.get<std::string>("filename");

        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(m_filename);

        if (!fs::exists(file_path))
            Throw("File not found: \"%s\"", m_filename);

        auto stream = std::make_unique<FileStream>(file_path);

        char magic[8];
        stream->read(magic, 8);
        if (memcmp(magic, "ATMPHASE", 8) != 0)
            Throw("Invalid phase LUT file (bad magic), expected 'ATMPHASE': \"%s\"", m_filename);

        uint32_t version;
        stream->read(&version, sizeof(uint32_t));
        stream->read(&m_resolution, sizeof(uint32_t));   // num_angles
        stream->read(&m_num_channels, sizeof(uint32_t)); // num_wavelength bins
        stream->read(&m_min_wavelength, sizeof(float));
        stream->read(&m_max_wavelength, sizeof(float));

        if (m_resolution < 2 || m_num_channels < 1)
            Throw("Invalid LUT dimensions in \"%s\" (angles=%u, wavelengths=%u).",
                  m_filename, m_resolution, m_num_channels);

        size_t total_floats = (size_t) m_resolution * (size_t) m_num_channels;
        std::vector<float> host_data(total_floats);
        stream->read(host_data.data(), total_floats * sizeof(float));

        m_data = dr::load<FloatStorage>(host_data.data(), total_floats);

        std::vector<float> pdf_mu_host(total_floats);
        std::vector<float> cdf_mu_host(total_floats);

        const float inv_two_pi = 1.f / (2.f * float(dr::Pi<double>)); // constant, used only as comment-guide
        (void) inv_two_pi;

        float d_mu = 2.f / float(m_resolution - 1);

        for (uint32_t c = 0; c < m_num_channels; ++c) {
            float sum = 0.f;
            for (uint32_t r = 0; r < m_resolution; ++r) {
                uint32_t a = (m_resolution - 1) - r;
                float v_omega = host_data[(size_t) a * (size_t) m_num_channels + c];
                v_omega = std::max(v_omega, 0.f);

                // Convert to unnormalized density in mu; factor 2π cancels after normalization,
                // but including it makes the intent explicit.
                float v_mu = v_omega * (2.f * float(dr::Pi<double>));

                pdf_mu_host[(size_t) c * (size_t) m_resolution + r] = v_mu;
                sum += v_mu;
            }

            // Normalize discretely in mu domain: sum(pdf_mu)*d_mu = 1  => pdf_mu /= (sum*d_mu)
            float norm = sum * d_mu;
            if (!(norm > 0.f))
                norm = 1.f; // avoid division by zero; will become near-zero distribution

            float cdf = 0.f;
            for (uint32_t r = 0; r < m_resolution; ++r) {
                float &p = pdf_mu_host[(size_t) c * (size_t) m_resolution + r];
                p /= norm;
                // CDF over continuous mu uses integral: accumulate p * d_mu
                cdf += p * d_mu;
                cdf_mu_host[(size_t) c * (size_t) m_resolution + r] = cdf;
            }

            // Force last entry exactly to 1 (helps numerical safety in inversion)
            cdf_mu_host[(size_t) c * (size_t) m_resolution + (m_resolution - 1)] = 1.f;
        }

        m_pdf_mu_bins = dr::load<FloatStorage>(pdf_mu_host.data(), total_floats);
        m_cdf_mu_bins = dr::load<FloatStorage>(cdf_mu_host.data(), total_floats);

        m_inv_d_mu = 1.f / d_mu;

        if (m_num_channels == 1) {
            m_wavelength_scale = 0.f;
        } else {
            m_wavelength_scale =
                (m_num_channels - 1) / (m_max_wavelength - m_min_wavelength);
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext & /*ctx*/,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Float mu = dot(wo, mi.wi);

        // Truth
        Float angle_idx = (1.f - mu) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        // Sampling pdf (marginal over hero lane selection)
        Float pdf_sum = 0.f;
        for (size_t i = 0; i < Spectrum::Size; ++i) {
            Float wvl = mi.wavelengths[i];
            pdf_sum += pdf_omega_for_wavelength(mu, wvl, active);
        }

        Float pdf = pdf_sum / Float(Spectrum::Size);
        return { value, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext & /*ctx*/,
           const MediumInteraction3f &mi,
           Float sample1, const Point2f &sample2,
           Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // 1) Pick hero lane uniformly
        UInt32 lane = dr::minimum(
            dr::floor2int<UInt32>(sample1 * ScalarFloat(Spectrum::Size)),
            UInt32(Spectrum::Size - 1)
        );

        Float hero_wvl = 0.f;
        for (uint32_t i = 0; i < Spectrum::Size; ++i)
            hero_wvl = dr::select(lane == i, mi.wavelengths[i], hero_wvl);

        // 2) Sample μ from mixed-λ CDF
        Float mu = sample_mu_for_wavelength(hero_wvl, sample2.x(), active);

        // 3) Sample φ uniformly
        Float sin_theta = dr::safe_sqrt(1.f - mu * mu);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());

        Vector3f wo_local{ sin_theta * cos_phi,
                           sin_theta * sin_phi,
                           mu };

        Vector3f wo = Frame3f(mi.wi).to_world(wo_local);

        // 4) Truth + marginal pdf
        Float angle_idx = (1.f - mu) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float pdf_sum = 0.f;
        for (size_t i = 0; i < Spectrum::Size; ++i) {
            Float wvl = mi.wavelengths[i];
            pdf_sum += pdf_omega_for_wavelength(mu, wvl, active);
        }

        Float pdf = pdf_sum / Float(Spectrum::Size);
        Spectrum weight = dr::select(pdf > 0.f, value / pdf, 0.f);

        return { wo, weight, pdf };
    }

    std::string to_string() const override {
        return tfm::format(
            "AtmosphericPhaseFunction[LUT-only, JIT-safe]\n"
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
    // ---- LUT evaluation (unchanged from your code) ----
    Spectrum lookup_interpolated(const Wavelength &wvls, Float angle_idx,
                                 Mask active) const {
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

                Float v00 = dr::gather<Float>(m_data, offset_a0 + w0, active);
                Float v01 = dr::gather<Float>(m_data, offset_a0 + w1, active);
                Float v10 = dr::gather<Float>(m_data, offset_a1 + w0, active);
                Float v11 = dr::gather<Float>(m_data, offset_a1 + w1, active);

                result[i] = dr::lerp(dr::lerp(v00, v01, t_wvl),
                                     dr::lerp(v10, v11, t_wvl), t_angle);
            }
            return result;
        } else {
            return 0.f;
        }
    }

    // ---- Wavelength -> (bin0, bin1, t) for linear interpolation ----
    MI_INLINE std::tuple<UInt32, UInt32, Float> wl_to_bins(const Float &wvl) const {
        if (m_num_channels == 1)
            return { UInt32(0), UInt32(0), 0.f };

        Float x = (wvl - m_min_wavelength) * m_wavelength_scale;
        x = dr::clip(x, 0.f, ScalarFloat(m_num_channels - 1));

        UInt32 i0 = dr::floor2int<UInt32>(x);
        UInt32 i1 = dr::minimum(i0 + 1u, UInt32(m_num_channels - 1));
        Float t   = x - Float(i0);
        return { i0, i1, t };
    }

    // ---- Evaluate pdf_omega(wo | wavelength) using mixed-λ discrete pdf_mu ----
    MI_INLINE Float pdf_omega_for_wavelength(const Float &mu, const Float &wvl, Mask active) const {
        // Convert mu in [-1,1] to discrete mu-index in [0, A-1] where index 0 => mu=-1.
        // mu_idx_f = (mu + 1)/2 * (A-1)
        Float mu_idx_f = (mu + 1.f) * 0.5f * Float(m_resolution - 1);
        mu_idx_f = dr::clip(mu_idx_f, 0.f, ScalarFloat(m_resolution - 1));

        UInt32 m0 = dr::floor2int<UInt32>(mu_idx_f);
        UInt32 m1 = dr::minimum(m0 + 1u, UInt32(m_resolution - 1));
        Float tm  = mu_idx_f - Float(m0);

        auto [b0, b1, tw] = wl_to_bins(wvl);

        UInt32 base0 = b0 * UInt32(m_resolution);
        UInt32 base1 = b1 * UInt32(m_resolution);

        Float p00 = dr::gather<Float>(m_pdf_mu_bins, base0 + m0, active);
        Float p01 = dr::gather<Float>(m_pdf_mu_bins, base0 + m1, active);
        Float p10 = dr::gather<Float>(m_pdf_mu_bins, base1 + m0, active);
        Float p11 = dr::gather<Float>(m_pdf_mu_bins, base1 + m1, active);

        Float p0 = dr::lerp(p00, p01, tm);
        Float p1 = dr::lerp(p10, p11, tm);
        Float p_mu = dr::lerp(p0, p1, tw);

        // Convert back to solid-angle density: p_omega = p_mu / (2π)
        return p_mu * dr::rcp(2.f * dr::Pi<ScalarFloat>);
    }

    // ---- Sample mu from mixed-λ CDF by inversion (binary search) ----
    MI_INLINE Float sample_mu_for_wavelength(const Float &wvl, const Float &u, Mask active) const {
        auto [b0, b1, tw] = wl_to_bins(wvl);

        UInt32 base0 = b0 * UInt32(m_resolution);
        UInt32 base1 = b1 * UInt32(m_resolution);

        // Binary search for smallest idx such that CDF(idx) >= u.
        UInt32 lo = 0u;
        UInt32 hi = UInt32(m_resolution - 1);

        // resolution is 1024..4096, so 12 iterations is enough for <=4096
        // (works for any power-of-two-ish but also fine generally).
        for (int it = 0; it < 12; ++it) {
            UInt32 mid = (lo + hi) >> 1;

            Float c0 = dr::gather<Float>(m_cdf_mu_bins, base0 + mid, active);
            Float c1 = dr::gather<Float>(m_cdf_mu_bins, base1 + mid, active);
            Float c  = dr::lerp(c0, c1, tw);

            Mask go_left = active && (u <= c);
            hi = dr::select(go_left, mid, hi);
            lo = dr::select(go_left, lo, mid + 1u);
        }

        UInt32 idx = dr::minimum(lo, UInt32(m_resolution - 1));

        // Interpolate within the bin using local linear CDF segment.
        UInt32 idx0 = dr::select(idx > 0u, idx - 1u, 0u);
        UInt32 idx1 = idx;

        Float c0a = dr::gather<Float>(m_cdf_mu_bins, base0 + idx0, active);
        Float c0b = dr::gather<Float>(m_cdf_mu_bins, base0 + idx1, active);
        Float c1a = dr::gather<Float>(m_cdf_mu_bins, base1 + idx0, active);
        Float c1b = dr::gather<Float>(m_cdf_mu_bins, base1 + idx1, active);

        Float Ca = dr::lerp(c0a, c1a, tw);
        Float Cb = dr::lerp(c0b, c1b, tw);

        // Avoid division by zero if segment is flat
        Float denom = dr::maximum(Cb - Ca, 1e-12f);
        Float s = (u - Ca) / denom;
        s = dr::clip(s, 0.f, 1.f);

        // Convert index-space to mu in [-1,1]
        // mu_idx_f = idx0 + s
        Float mu_idx_f = Float(idx0) + s;

        // mu = 2*(mu_idx_f/(A-1)) - 1
        Float mu = dr::fmadd(mu_idx_f, 2.f / Float(m_resolution - 1), -1.f);
        return dr::clip(mu, -1.f, 1.f);
    }

    MI_DECLARE_CLASS(AtmosphericPhaseFunction)

private:
    // LUT table for truth evaluation (angle-major storage)
    FloatStorage m_data;

    // Discrete distributions in mu-domain (mu_idx 0 => mu=-1, last => mu=+1)
    FloatStorage m_pdf_mu_bins; // length = channels * resolution, normalized so ∑ p_mu*d_mu = 1
    FloatStorage m_cdf_mu_bins; // length = channels * resolution, last is 1

    std::string m_filename;
    uint32_t m_resolution = 0;
    uint32_t m_num_channels = 0;
    float m_min_wavelength = 0.f;
    float m_max_wavelength = 0.f;
    float m_wavelength_scale = 0.f;
    float m_inv_d_mu = 0.f;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)