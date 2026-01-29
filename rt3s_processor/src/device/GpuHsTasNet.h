#ifndef GPUA_GPUHSTASNET_H_INCLUDED
#define GPUA_GPUHSTASNET_H_INCLUDED

#include <gpu_primitives/Conv1D.h>
#include <gpu_primitives/GpuRingbuffer.h>
#include <gpu_primitives/Linear.h>
#include <gpu_primitives/LSTM.h>
#include <gpu_primitives/RMSNorm.h>
#include <gpu_primitives/Spectrogram.h>

#if !defined(GPU_AUDIO_DEVICE)
#include <istream>
#endif

namespace gpua {

template <typename TYPE>
struct GpuHsTasNet {
    MultiChannelRingBuffer<2u, 1024u, float> m_input_history;
    MultiChannelRingBuffer<8u, 1024u, float> m_spec_output_history;
    MultiChannelRingBuffer<8u, 1024u, float> m_waveform_output_history;

    Spectrogram m_stft;

    Linear<2052u, 500u, TYPE, true> m_spec_encode;
    RMSNorm<500u, TYPE, false> m_spec_norm;
    Linear<500u, 8208u, TYPE, true> m_spec_mask_linear;
    Conv1D<2u, 3000u, 1024u, 512u, TYPE, true> m_conv_encode;
    Linear<1500u, 500u, TYPE, true> m_basis_to_embed;
    RMSNorm<500u, TYPE, false> m_waveform_norm;
    Linear<500u, 6000u, TYPE, true> m_waveform_mask_linear;
    ConvTranspose1D<1500u, 2u, 1024u, 512u, TYPE, false> m_conv_decode;

    LSTM<500u, 500u, 2u, TYPE> m_pre_spec_branch;
    LSTM<500u, 500u, 2u, TYPE> m_post_spec_branch;
    LSTM<1000u, 1000u, 2u, TYPE> m_fusion_branch;
    LSTM<500u, 500u, 2u, TYPE> m_pre_waveform_branch;
    LSTM<500u, 500u, 2u, TYPE> m_post_waveform_branch;

#if !defined(GPU_AUDIO_DEVICE)
    GpuHsTasNet(std::istream& weights_in, uint64_t lut, uint64_t inv_lut) {
        Init();
        SetWeights(weights_in);
        SetSpecLUTs(lut, inv_lut);
    }

    void Init() {
        m_input_history.Init();
        m_spec_output_history.Init();
        m_waveform_output_history.Init();
    }
    void SetWeights(std::istream& win) {
        m_spec_encode.SetWeights(win);
        m_spec_norm.SetWeights(win);
        m_spec_mask_linear.SetWeights(win);
        m_conv_encode.SetWeights(win);
        m_basis_to_embed.SetWeights(win);
        m_waveform_norm.SetWeights(win);
        m_waveform_mask_linear.SetWeights(win);
        m_conv_decode.SetWeights(win);

        auto SetLSTMLayerWeights = [&win](auto& lstm_layer) {
            lstm_layer.SetWeights(win);
        };

        m_pre_spec_branch.visitEachLSTMLayer(SetLSTMLayerWeights);
        m_post_spec_branch.visitEachLSTMLayer(SetLSTMLayerWeights);
        m_fusion_branch.visitEachLSTMLayer(SetLSTMLayerWeights);
        m_pre_waveform_branch.visitEachLSTMLayer(SetLSTMLayerWeights);
        m_post_waveform_branch.visitEachLSTMLayer(SetLSTMLayerWeights);
    }
    void SetSpecLUTs(uint64_t lut, uint64_t inv_lut) {
        m_stft.m_twiddle_LUT = lut;
        m_stft.m_twiddle_inverse_LUT = inv_lut;
    }
#endif
};

} // namespace gpua

#endif /* GPUA_GPUHSTASNET_H_INCLUDED */
