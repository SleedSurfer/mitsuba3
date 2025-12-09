#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/core/distr_1d.h>
#include <algorithm> // For std::reverse
#include <vector>
#include <cstring>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AtmosphericPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    AtmosphericPhaseFunction(const Properties &props) : Base(props) {
        m_filename = props.get<std::string>("filename");

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(m_filename);

        if (!fs::exists(file_path))
            Throw("File not found: \"%s\"", m_filename);

        std::unique_ptr<FileStream> fs_stream = std::make_unique<FileStream>(file_path);

        char magic[8];
        fs_stream->read(magic, 8);
        if (memcmp(magic, "ATMPHASE", 8) != 0) Throw("Invalid magic header");

        uint32_t version; fs_stream->read(&version, sizeof(uint32_t));
        fs_stream->read(&m_resolution, sizeof(uint32_t));
        fs_stream->read(&m_num_channels, sizeof(uint32_t));
        fs_stream->read(&m_min_wavelength, sizeof(float));
        fs_stream->read(&m_max_wavelength, sizeof(float));

        // Layout: Interleaved [Angle0_Wvl0, Angle0_Wvl1... AngleN_WvlM]
        size_t total_floats = m_resolution * m_num_channels;
        std::vector<float> host_data(total_floats);
        fs_stream->read(host_data.data(), total_floats * sizeof(float));

        m_data = dr::load<FloatStorage>(host_data.data(), total_floats);

        std::vector<float> envelope_data(m_resolution);
        const float* raw_ptr = host_data.data();

        for (uint32_t a = 0; a < m_resolution; ++a) {
            float max_val = 0.0f;

            // Pointer to the start of this angle's wavelength block (LUT order)
            const float* angle_block = raw_ptr + (a * m_num_channels);

            for (uint32_t c = 0; c < m_num_channels; ++c) {
                float val = angle_block[c];
                if (val > max_val)
                    max_val = val;
            }

            // At this point:
            //   a = 0        -> mu = +1  (forward)
            //   a = res - 1  -> mu = -1 (backward)
            envelope_data[a] = max_val;
        }

        // Flip to match ContinuousDistribution's domain [-1, 1]:
        //   pdf[0]   = value at mu = -1 (backward)
        //   pdf[last]= value at mu = +1 (forward)
        std::reverse(envelope_data.begin(), envelope_data.end());

        // Create the angular sampling distribution over cos(theta) in [-1, 1]
        m_distr = ContinuousDistribution<Float>(
            ScalarVector2f(-1.f, 1.f),
            envelope_data.data(),
            m_resolution
        );

        m_wavelength_scale = (m_num_channels - 1) / (m_max_wavelength - m_min_wavelength);
        m_flags = +PhaseFunctionFlags::Anisotropic;
        m_components.push_back(m_flags);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);


        Float cos_theta = dot(wo, mi.wi);

        // Map cosθ ∈ [-1,1] onto LUT index [0, m_resolution-1]
        Float angle_idx = (1.f - cos_theta) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));

        // Spectral phase value p(μ, λ)
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        // 1D pdf over μ (normalized)
        Float pdf_mu = m_distr.eval_pdf_normalized(cos_theta, active);

        // Convert to solid-angle pdf: uniform in φ
        Float pdf = pdf_mu * dr::rcp(2.f * dr::Pi<ScalarFloat>);

        return { value, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // 1. Sample cos_theta from angular envelope distribution over [-1,1]
        Float cos_theta = m_distr.sample(sample2.x());

        // 2. Compute local/world direction (unchanged)
        Float sin_theta = dr::safe_sqrt(1.f - cos_theta * cos_theta);
        auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());

        Vector3f wo_local { sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };
        Vector3f wo = Frame3f(mi.wi).to_world(wo_local);

        // 3. Map cosθ to LUT index according to generator convention
        Float angle_idx = (1.f - cos_theta) * 0.5f * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));

        // 4. Lookup spectral phase value p(μ, λ)
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        // 5. 1D pdf over μ from ContinuousDistribution (normalized)
        Float pdf_mu = m_distr.eval_pdf_normalized(cos_theta, active);

        // Convert to solid-angle pdf: p_sample(ω) = pdf_mu / (2π)
        Float pdf = pdf_mu * dr::rcp(2.f * dr::Pi<ScalarFloat>);

        Spectrum weight = value / pdf;
        
        return { wo, weight, pdf };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AtmosphericPhaseFunction[" << std::endl
            << "  filename = \"" << m_filename << "\"," << std::endl
            << "  resolution = " << m_resolution << std::endl
            << "]";
        return oss.str();
    }

private:
    Spectrum lookup_interpolated(const Wavelength &wvls, Float angle_idx, Mask active) const {
        if constexpr (is_spectral_v<Spectrum>) {
            Spectrum result;

            UInt32 a0 = dr::floor2int<UInt32>(angle_idx);
            UInt32 a1 = dr::minimum(a0 + 1u, UInt32(m_resolution - 1));
            Float t_angle = angle_idx - Float(a0);

            UInt32 stride = UInt32(m_num_channels);
            UInt32 offset_a0 = a0 * stride;
            UInt32 offset_a1 = a1 * stride;

            constexpr size_t n_wavelengths = Spectrum::Size;

            #pragma unroll
            for (size_t i = 0; i < n_wavelengths; ++i) {
                Float wvl = wvls[i];
                Float w_idx = (wvl - m_min_wavelength) * m_wavelength_scale;
                w_idx = dr::clip(w_idx, 0.f, ScalarFloat(m_num_channels - 1));

                UInt32 w0 = dr::floor2int<UInt32>(w_idx);
                UInt32 w1 = dr::minimum(w0 + 1u, UInt32(m_num_channels - 1));
                Float t_wvl = w_idx - Float(w0);

                UInt32 i00 = offset_a0 + w0;
                UInt32 i01 = offset_a0 + w1;
                UInt32 i10 = offset_a1 + w0;
                UInt32 i11 = offset_a1 + w1;

                Float v00 = dr::gather<Float>(m_data, i00, active);
                Float v01 = dr::gather<Float>(m_data, i01, active);
                Float v10 = dr::gather<Float>(m_data, i10, active);
                Float v11 = dr::gather<Float>(m_data, i11, active);

                Float val_a0 = dr::lerp(v00, v01, t_wvl);
                Float val_a1 = dr::lerp(v10, v11, t_wvl);

                result[i] = dr::lerp(val_a0, val_a1, t_angle);
            }
            return result;
        } else {
            return 0.f;
        }
    }

    MI_DECLARE_CLASS(AtmosphericPhaseFunction)

    using FloatStorage = DynamicBuffer<Float>;
    FloatStorage m_data;
    ContinuousDistribution<Float> m_distr;

    std::string m_filename;
    uint32_t m_resolution, m_num_channels;
    float m_min_wavelength, m_max_wavelength, m_wavelength_scale;
};

MI_EXPORT_PLUGIN(AtmosphericPhaseFunction)
NAMESPACE_END(mitsuba)