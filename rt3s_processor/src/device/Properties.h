/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef GPU_AUDIO_RT3S_PROPERTIES_INCLUDED
#define GPU_AUDIO_RT3S_PROPERTIES_INCLUDED

#include "ScrambledNames.h"

#include <platform/Abstraction.h>

namespace gpua {
// parameter struct passed to each task independent of network type
struct ProcessorParameter {
    uint64_t d_hstasnet {};
    uint32_t buffer_capacity {};
    uint32_t ringbuffer_cursor {};
};

// common params
struct RMSNormParams {
    uint64_t p_data_in {};
    uint64_t p_data_norm_out {};
};

template <uint32_t BLOCKS, uint32_t BLOCK_SIZE, uint32_t GROUP_SIZE>
struct LstmCommonParams {
    static __program_scope constexpr uint32_t Blocks = BLOCKS;
    static __program_scope constexpr uint32_t BlockSize = BLOCK_SIZE;
    static __program_scope constexpr uint32_t GroupSize = GROUP_SIZE;

    uint64_t p_in {};
    uint64_t p_out {};
    uint32_t running_counter {};
};
// 0. enqueue input
using EnqueueInputParams = void;

// 1.1 pre fusion spec
struct SpecParams {
    uint64_t p_spec_reordered_out {};
    uint32_t signal_length {};
    uint32_t hop_size {};
};

struct SpecEncParams {
    uint64_t p_spec_reordered_in {};
    uint64_t p_spec_encoded_out {};
    uint32_t nblocks {};
};

// 1.2 pre fusion waveform
struct ConvEncodeParams {
    uint64_t p_conv_enc_out {};
    uint32_t nblocks {};
};

struct ConvActivationParams {
    uint64_t p_conv_enc_in {};
    uint64_t p_audio_basis_out {};
    uint32_t nblocks {};
};

struct BasisEmbedParams {
    uint64_t p_audio_basis_in {};
    uint64_t p_waveform_encoded_out {};
    uint32_t nblocks {};
};

// 2.1 pre-fusion lstm
struct PreSpecWaveformLstmParams {
    using LstmParams = LstmCommonParams<64, 128, 32>;
    static __program_scope constexpr uint32_t Blocks = 2u * LstmParams::Blocks;
    static __program_scope constexpr uint32_t BlockSize = LstmParams::BlockSize;
    LstmParams lstm_params[2];
};

// 2.2 fusion
struct FusionParams {
    using LstmParams = LstmCommonParams<128, 128, 32>;
    LstmParams lstm_params;
    uint64_t p_spec_waveform_encoded_in {};
};

// 2.3 post-fusion lstms
struct PostSpecWaveformLstmParams {
    using LstmParams = LstmCommonParams<64, 128, 32>;
    static __program_scope constexpr uint32_t Blocks = 2u * LstmParams::Blocks;
    static __program_scope constexpr uint32_t BlockSize = LstmParams::BlockSize;
    LstmParams lstm_params[2];
};

// 2.4 post-post-fusion rms normalization
struct SpecWaveformNormParams {
    RMSNormParams norm_params[2];
};

// 3.1 post spec
struct SpecMaskParams {
    uint64_t p_spec_post_rnn_norm_in {};
    uint64_t p_spec_mask_linear_out {};
    uint32_t nblocks {};
};

struct SpecMaskApplyParams {
    uint64_t p_spec_mask_linear_in {};
    uint64_t p_spec_reordered_in {};
    uint64_t p_spec_per_source_out {};
    uint64_t p_phase_per_source_out {};
    uint32_t nblocks {};
};

struct InvSpecParams {
    uint64_t p_spec_per_source_in {};
    uint64_t p_phase_per_source_in {};
    uint32_t signal_length {};
    uint32_t hop_size {};
    uint32_t nblocks {};
};

// 3.2 post waveform
struct WaveformMaskParams {
    uint64_t p_waveform_post_rnn_norm_in {};
    uint64_t p_waveform_mask_linear_out {};
    uint32_t nblocks {};
};

struct WaveformMaskApplyParams {
    uint64_t p_waveform_mask_linear_in {};
    uint64_t p_audio_basis_in {};
    uint64_t p_basis_per_source_out {};
    uint32_t nblocks {};
};

struct ConvDecodeParams {
    uint64_t p_basis_per_source_in {};
    uint32_t nblocks {};
};

// 4. combine branches
struct CombineBranchesParams {
    uint32_t buffer_length {};
};

} // namespace gpua

#endif /* GPU_AUDIO_RT3S_PROPERTIES_INCLUDED */
