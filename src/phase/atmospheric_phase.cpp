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

            // Pointer to the start of this angle's wavelength block
            const float* angle_block = raw_ptr + (a * m_num_channels);

            for (uint32_t c = 0; c < m_num_channels; ++c) {
                float val = angle_block[c];
                if (val > max_val) {
                    max_val = val;
                }
            }
            envelope_data[a] = max_val;
        }

        // 3. Orientation Fix (Keep consistent with your previous logic)
        // If your file is 0->180, but Mitsuba is Forward->Back, we reverse.
        std::reverse(envelope_data.begin(), envelope_data.end());

        // 4. Create the Distribution
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

        Float u = (1.f - cos_theta) * 0.5f;

        Float angle_idx = u * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));

        // 3. Spectral Value
        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float pdf = m_distr.eval_pdf(cos_theta);

        return { value, pdf };
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float sample1,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // 1. Sample cos_theta from Green Distribution
        Float cos_theta = m_distr.sample(sample2.x());

        // 2. Compute Local Direction
        Float sin_theta = dr::safe_sqrt(1.f - cos_theta * cos_theta);
        auto [sin_phi, cos_phi] = dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());

        Vector3f wo_local { sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };

        // 3. FRAME CORRECTION (CRITICAL)
        // We must transform wo_local to world space relative to the INCOMING RAY (mi.wi).
        // This aligns the "Forward" peak (Z+) with the light direction.
        // Frame3f(vector) constructs a frame where 'vector' is the Z axis.
        Vector3f wo = Frame3f(mi.wi).to_world(wo_local);

        Float u = (1.f - cos_theta) * 0.5f;
        Float angle_idx = u * Float(m_resolution - 1);
        angle_idx = dr::clip(angle_idx, 0.f, ScalarFloat(m_resolution - 1));

        Spectrum value = lookup_interpolated(mi.wavelengths, angle_idx, active);

        Float pdf = m_distr.eval_pdf(cos_theta);

        Spectrum weight = value / pdf;
        dr::masked(weight, pdf == 0.f) = 0.f;

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