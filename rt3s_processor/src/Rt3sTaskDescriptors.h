/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef GPU_AUDIO_RT3S_TASK_DESCRIPTORS_INCLUDED
#define GPU_AUDIO_RT3S_TASK_DESCRIPTORS_INCLUDED

#include "Properties.h"
#include "GpuHsTasNet.h"

#include <processor_api/GpuTaskData.h>

using GPUA::processor::v2::GpuTaskData;
using GPUA::processor::v2::ProcessingFlag;

namespace {
template <typename T>
static inline T divup(T a, T b) {
    return (a + b - 1) / b;
}
} // namespace
namespace RT3S {
// enqueu input
void enqueue_input_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 0u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 18u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = 0u;
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// spec pre fusion
void spec_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 1u;
    gpu_task.thread_count = 64u;
    gpu_task.block_count = 2u;
    gpu_task.shared_mem_size = (8 * 64 + 64) * sizeof(float);
    gpu_task.task_param_size = sizeof(gpua::SpecParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void spec_encode_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 2u;
    // 1 x 2052 x 500; 128 warps; 4 cols of len 2052 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 16u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::SpecEncParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// waveform pre fusion
void conv_encode_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 3u;
    // 3000 x 2048 x 1; 256 warps; 12 rows of len 2048 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 32u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::ConvEncodeParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void conv_activation_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 4u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = divup(1500u, gpu_task.thread_count);
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::ConvActivationParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void basis_embed_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 5u;
    // 500 x 1500 x 1; 128 warps; 4 rows of len 1500 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 16u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::BasisEmbedParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// pre-fusion lstm
void pre_spec_waveform_lstm_task_0(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 6u;
    gpu_task.thread_count = gpua::PreSpecWaveformLstmParams::BlockSize;
    gpu_task.block_count = gpua::PreSpecWaveformLstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_pre_spec_branch.template SmemSize<gpua::PreSpecWaveformLstmParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::PreSpecWaveformLstmParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void pre_spec_waveform_lstm_task_1(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 7u;
    gpu_task.thread_count = gpua::PreSpecWaveformLstmParams::BlockSize;
    gpu_task.block_count = gpua::PreSpecWaveformLstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_pre_spec_branch.template SmemSize<gpua::PreSpecWaveformLstmParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::PreSpecWaveformLstmParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// fusion
void fusion_task_0(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 8u;
    gpu_task.thread_count = gpua::FusionParams::LstmParams::BlockSize;
    gpu_task.block_count = gpua::FusionParams::LstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_fusion_branch.template SmemSize<gpua::FusionParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::FusionParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void fusion_task_1(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 9u;
    gpu_task.thread_count = gpua::FusionParams::LstmParams::BlockSize;
    gpu_task.block_count = gpua::FusionParams::LstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_fusion_branch.template SmemSize<gpua::FusionParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::FusionParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// post-fusion lstm
void post_spec_waveform_lstm_task_0(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 10u;
    gpu_task.thread_count = gpua::PostSpecWaveformLstmParams::BlockSize;
    gpu_task.block_count = gpua::PostSpecWaveformLstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_post_spec_branch.template SmemSize<gpua::PostSpecWaveformLstmParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::PostSpecWaveformLstmParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void post_spec_waveform_lstm_task_1(GpuTaskData& gpu_task, gpua::GpuHsTasNet<float>* h_hstasnet) {
    gpu_task.entry_idx = 11u;
    gpu_task.thread_count = gpua::PostSpecWaveformLstmParams::BlockSize;
    gpu_task.block_count = gpua::PostSpecWaveformLstmParams::Blocks;
    gpu_task.shared_mem_size = h_hstasnet->m_post_spec_branch.template SmemSize<gpua::PostSpecWaveformLstmParams::LstmParams::Blocks>();
    gpu_task.task_param_size = sizeof(gpua::PostSpecWaveformLstmParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// post-post-fustion lstm rms norm.
void spec_waveform_rms_norm_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 12u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 2u;
    gpu_task.shared_mem_size = gpu_task.thread_count / 32u * sizeof(float);
    gpu_task.task_param_size = sizeof(gpua::SpecWaveformNormParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// spec post fusion
void spec_mask_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 13u;
    // 1 x 500 x 8208; 256 warps; 33 cols of len 500 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 32u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::SpecMaskParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void spec_mask_apply_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 14u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = divup(2u * 4u * 2u * 513u, gpu_task.thread_count);
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::SpecMaskApplyParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void inv_spec_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 15u;
    gpu_task.thread_count = 64u;
    gpu_task.block_count = 2u;
    gpu_task.shared_mem_size = (8u * 64u + 64u) * sizeof(float);
    gpu_task.task_param_size = sizeof(gpua::InvSpecParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

// waveform post fusion
void waveform_mask_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 16u;
    // 1 x 500 x 6000; 256 warps; 24 rows of len 500 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 32u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::WaveformMaskParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void waveform_mask_apply_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 17u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = divup(6000u, gpu_task.thread_count);
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::WaveformMaskApplyParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void conv_decode_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 18u;
    // 4 x 1500 x 2048; 512 warps; 4 cols of len 1500 per warp
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 64u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::ConvDecodeParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

void combine_branches_task(GpuTaskData& gpu_task) {
    gpu_task.entry_idx = 19u;
    gpu_task.thread_count = 256u;
    gpu_task.block_count = 8u;
    gpu_task.shared_mem_size = 0u;
    gpu_task.task_param_size = sizeof(gpua::CombineBranchesParams);
    gpu_task.processing_flags = ProcessingFlag::eProcessingFlagDefault;
}

} // namespace RT3S

#endif /* GPU_AUDIO_RT3S_TASK_DESCRIPTORS_INCLUDED */
