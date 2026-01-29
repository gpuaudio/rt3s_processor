/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#include "Rt3sProcessor.h"
#include "Rt3sTaskDescriptors.h"

#include <processor_api/GpuTaskData.h>
#include <processor_api/PortChangedFlags.h>
#include <processor_api/ProcessorSpecification.h>
#include <processor_api/MemoryManager.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <float.h>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <map>
#include <stdio.h>

using namespace GPUA::processor::v2;

static constexpr uint32_t c_signal_length {1024};
static constexpr uint32_t c_hop_size {c_signal_length / 2};
static constexpr uint32_t c_nbins {c_signal_length / 2 + 1};
static constexpr uint32_t c_nchannels {2u};
static constexpr uint32_t c_buffer_capacity {c_hop_size};
static constexpr uint32_t c_nsources {4u};
static constexpr uint32_t c_nspec_features {500u};
static constexpr uint32_t c_nwaveform_features {500u};

template <uint32_t size, uint32_t alignment = 64u>
struct aligned_size {
    static_assert(alignment % sizeof(float) == 0, "aligned_size: alignment muust be a multiple of sizeof(float)");
    static constexpr uint32_t value {(size + alignment - 1u) / alignment * alignment};
};

static constexpr char const* c_buffer_names[15] = {
    "spec_reordered",
    "spec_waveform_encoded",
    "conv_encoded",
    "audio_basis",
    "spec_waveform_pre_rnn",
    "spec_waveform_with_residual",
    "spec_post_rnn",
    "spec_post_rnn_norm",
    "spec_mask_linear",
    "spec_real_4x2ch",
    "spec_imag_4x2ch",
    "waveform_post_rnn",
    "waveform_post_rnn_norm",
    "waveform_mask_linear",
    "audio_basis_per_source"};

static constexpr uint32_t c_buffer_sizes_bytes[15] = {
    /* d_spec_reordered */
    aligned_size<2u * c_nchannels * c_nbins * sizeof(float)>::value,
    /* d_spec_waveform_encoded */
    aligned_size<(c_nspec_features + c_nwaveform_features * sizeof(float))>::value,
    /* d_conv_encoded */
    aligned_size<3000u * sizeof(float)>::value,
    /* d_audio_basis */
    aligned_size<1500u * sizeof(float)>::value,
    /* d_spec_waveform_pre_rnn */
    aligned_size<(c_nspec_features + c_nwaveform_features) * sizeof(float)>::value,
    /* d_spec_waveform_with_residual */
    aligned_size<(c_nspec_features + c_nwaveform_features) * sizeof(float)>::value,
    /* d_spec_post_rnn */
    aligned_size<c_nspec_features * sizeof(float)>::value,
    /* d_spec_post_rnn_norm */
    aligned_size<c_nspec_features * sizeof(float)>::value,
    /* d_spec_mask_linear */
    aligned_size<c_nsources * c_nchannels * 2u /* complex */ * c_nbins * sizeof(float)>::value,
    /* d_spec_real_4x2ch */
    aligned_size<c_nsources * c_nchannels * c_nbins * sizeof(float)>::value,
    /* d_spec_imag_4x2ch */
    aligned_size<c_nsources * c_nchannels * c_nbins * sizeof(float)>::value,
    /* d_waveform_post_rnn */
    aligned_size<c_nwaveform_features * sizeof(float)>::value,
    /* d_waveform_post_rnn_norm */
    aligned_size<c_nwaveform_features * sizeof(float)>::value,
    /* d_waveform_mask_linear */
    aligned_size<6000u * sizeof(float)>::value,
    /* d_audio_basis_per_source */
    aligned_size<c_nsources * 1500u * sizeof(float)>::value};

static constexpr uint32_t c_buffer_offsets_bytes[15] = {
    /* d_spec_reordered */
    0,
    c_buffer_sizes_bytes[0],
    /* d_spec_waveform_encoded */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1],
    /* d_conv_encoded */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2],
    /* d_audio_basis */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3],
    /* d_spec_waveform_pre_rnn */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4],
    /* d_spec_waveform_with_residual */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5],
    /* d_spec_post_rnn */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6],
    /* d_spec_post_rnn_norm */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7],
    /* d_spec_mask_linear */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8],
    /* d_spec_real_4x2ch */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9],
    /* d_spec_imag_4x2ch */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9] + c_buffer_sizes_bytes[10],
    /* d_waveform_post_rnn_norm */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9] + c_buffer_sizes_bytes[10] + c_buffer_sizes_bytes[11],
    /* d_waveform_mask_linear */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9] + c_buffer_sizes_bytes[10] + c_buffer_sizes_bytes[11] + c_buffer_sizes_bytes[12],
    /* d_audio_basis_per_source */
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9] + c_buffer_sizes_bytes[10] + c_buffer_sizes_bytes[11] + c_buffer_sizes_bytes[12] + c_buffer_sizes_bytes[13],
};

static constexpr uint32_t c_buffer_total_bytes {
    c_buffer_sizes_bytes[0] + c_buffer_sizes_bytes[1] + c_buffer_sizes_bytes[2] + c_buffer_sizes_bytes[3] + c_buffer_sizes_bytes[4] + c_buffer_sizes_bytes[5] + c_buffer_sizes_bytes[6] + c_buffer_sizes_bytes[7] + c_buffer_sizes_bytes[8] + c_buffer_sizes_bytes[9] + c_buffer_sizes_bytes[10] + c_buffer_sizes_bytes[11] + c_buffer_sizes_bytes[12] + c_buffer_sizes_bytes[13] + c_buffer_sizes_bytes[14]};

static constexpr bool g_debug {false};
static constexpr bool g_generate_debug_data {false};
static constexpr char const* g_debug_file_name {"test_debug.out"};

void print_debug_data(uint32_t call, float const* debug_buffer) {
    std::ofstream out;
    if (call == 0u) {
        out.open(g_debug_file_name, std::ios::binary | std::ios::out);
    }
    else {
        out.open(g_debug_file_name, std::ios::binary | std::ios::app);
    }
    // write call index
    out.write(reinterpret_cast<char const*>(&call), sizeof(call));
    out.write(reinterpret_cast<char const*>(debug_buffer), c_buffer_total_bytes);
    out.close();
}

void compare_debug_data(uint32_t call, float const* debug_buffer, float tolerance = 1e-4f) {
    std::ifstream in(g_debug_file_name, std::ios::binary | std::ios::in);
    std::vector<float> reference(c_buffer_total_bytes / sizeof(float));

    uint32_t current_call {};
    std::vector<float> current_debug_buffer(c_buffer_total_bytes / sizeof(float));
    do {
        in.read(reinterpret_cast<char*>(&current_call), sizeof(current_call));
        in.read(reinterpret_cast<char*>(current_debug_buffer.data()), c_buffer_total_bytes);
    } while (current_call != call);
    in.close();

    for (uint32_t buffer_id {}; buffer_id < 15; ++buffer_id) {
        auto reference_out = current_debug_buffer.data() + c_buffer_offsets_bytes[buffer_id] / sizeof(float);
        auto candidate_out = debug_buffer + c_buffer_offsets_bytes[buffer_id] / sizeof(float);
        for (uint32_t i {0u}; i < c_buffer_sizes_bytes[buffer_id] / sizeof(float); ++i) {
            if (std::fabsf(reference_out[i] - candidate_out[i]) > tolerance) {
                printf("CALL %d: Mismatch in buffer %s [%u] == %f != %f\n", call, c_buffer_names[buffer_id], i, candidate_out[i], reference_out[i]);
                break;
            }
            // else {
            //     printf("Match in task %u @ %u: %u == %u\n", task_id, i, candidate_out[i], reference_out[i]);
            // }
        }
    }
}

Module&
    Rt3sProcessor::GetModule() const noexcept {
    return m_module;
}

ErrorCode Rt3sProcessor::SetData(void* data, uint32_t data_size) noexcept {
    return ErrorCode::eSuccess;
}

ErrorCode Rt3sProcessor::GetData(void* data, uint32_t& data_size) const noexcept {
    return ErrorCode::eFail;
}

uint32_t Rt3sProcessor::GetInputPortCount() const noexcept {
    return 1u;
}

ErrorCode Rt3sProcessor::GetInputPort(uint32_t index, InputPort*& port) noexcept {
    if (index == 0) {
        port = m_input_port.get();
        return ErrorCode::eSuccess;
    }
    port = nullptr;
    return ErrorCode::eOutOfRange;
}

::ErrorCode Rt3sProcessor::OnBlueprintRebuild(const ::ProcessorBlueprint*& blueprint) noexcept {
    m_changed = m_input_port->m_changed = false;
    blueprint = &m_proc_data;
    return ErrorCode::eSuccess;
}

::ErrorCode Rt3sProcessor::PrepareForProcess(const ::LaunchData& data, uint32_t expected_chunks) noexcept {
    // not required right now; could potentially change configuration or similar in SetData
    // SetData(data.app_data, data.app_data_size);

    // if something changed that requires a blueprint re-build
    if (m_changed || m_input_port->m_changed) {
        return ErrorCode::eBlueprintUpdateNeeded;
    }
    return ErrorCode::eNoChangesNeeded;
}

::ErrorCode Rt3sProcessor::PrepareChunk(void* proc_data, void** task_data, uint32_t chunk_id) noexcept {
    ////////////////////////////////
    // DEBUGGING
    uint32_t ready_buffer = m_running_counter & 0x1u;

    if constexpr (g_debug) {
        if (m_running_counter >= 2 && m_running_counter < 130) {
            m_memory_manager.MemCpyGpuToCpu(h_debug_buffer.data(), *d_device_buffer[ready_buffer], 0u, c_buffer_total_bytes);

            if (g_generate_debug_data) {
                print_debug_data(m_running_counter - 2u, h_debug_buffer.data());
            }
            else {
                compare_debug_data(m_running_counter - 2u, h_debug_buffer.data());
            }
        }
    }

    SetTaskBuffers(d_device_buffer[ready_buffer]->GetGpuPointer());

    ////////////////////////////////
    // prepare the next processing call

    ////////////////////////////////
    // set processor parameters
    gpua::ProcessorParameter* proc_params = reinterpret_cast<gpua::ProcessorParameter*>(proc_data);
    proc_params->d_hstasnet = m_d_hstasnet->GetGpuPointer();
    proc_params->buffer_capacity = c_buffer_capacity;
    proc_params->ringbuffer_cursor = m_ringbuffer_cursor;

    ////////////////////////////////
    // set task parameters
    // enqueu input
    auto enqueu_input = reinterpret_cast<void*>(task_data[0]);

    // spec pre fusion
    *reinterpret_cast<gpua::SpecParams*>(task_data[1]) = m_spec_params;
    *reinterpret_cast<gpua::SpecEncParams*>(task_data[2]) = m_spec_enc_params;

    // waveform pre fusion
    *reinterpret_cast<gpua::ConvEncodeParams*>(task_data[3]) = m_conv_encode_params;
    *reinterpret_cast<gpua::ConvActivationParams*>(task_data[4]) = m_conv_activation_params;
    *reinterpret_cast<gpua::BasisEmbedParams*>(task_data[5]) = m_basis_embed_params;

    // pre-fusion lstm
    *reinterpret_cast<gpua::PreSpecWaveformLstmParams*>(task_data[6]) = m_pre_spec_waveform_lstm_params;
    *reinterpret_cast<gpua::PreSpecWaveformLstmParams*>(task_data[7]) = m_pre_spec_waveform_lstm_params;

    // fusion
    *reinterpret_cast<gpua::FusionParams*>(task_data[8]) = m_fusion_params;
    *reinterpret_cast<gpua::FusionParams*>(task_data[9]) = m_fusion_params;

    // post-fusion lstm
    *reinterpret_cast<gpua::PostSpecWaveformLstmParams*>(task_data[10]) = m_post_spec_waveform_lstm_params;
    *reinterpret_cast<gpua::PostSpecWaveformLstmParams*>(task_data[11]) = m_post_spec_waveform_lstm_params;

    // post-post-fusion lstm rms norm.
    *reinterpret_cast<gpua::SpecWaveformNormParams*>(task_data[12]) = m_spec_waveform_norm_params;

    // spec post fusion
    *reinterpret_cast<gpua::SpecMaskParams*>(task_data[13]) = m_spec_mask_params;
    *reinterpret_cast<gpua::SpecMaskApplyParams*>(task_data[14]) = m_spec_mask_apply_params;
    *reinterpret_cast<gpua::InvSpecParams*>(task_data[15]) = m_inv_spec_params;

    // waveform post fusion
    *reinterpret_cast<gpua::WaveformMaskParams*>(task_data[16]) = m_waveform_mask_params;
    *reinterpret_cast<gpua::WaveformMaskApplyParams*>(task_data[17]) = m_waveform_mask_apply_params;
    *reinterpret_cast<gpua::ConvDecodeParams*>(task_data[18]) = m_conv_decode_params;

    // combine branch outputs
    *reinterpret_cast<gpua::CombineBranchesParams*>(task_data[19]) = m_combine_branches_params;

    ////////////////////////////////
    // update counters
    AdvanceCounters();

    return ErrorCode::eSuccess;
}

void Rt3sProcessor::OnProcessingEnd(bool after_fat_transfer) noexcept {
}

::ProcessorProfiler* Rt3sProcessor::GetProcessorProfiler() noexcept {
    return nullptr;
}

void Rt3sProcessor::AdvanceCounters() {
    m_ringbuffer_cursor += c_buffer_capacity;
    ++m_running_counter;

    m_pre_spec_waveform_lstm_params.lstm_params[0].running_counter = m_running_counter;
    m_pre_spec_waveform_lstm_params.lstm_params[1].running_counter = m_running_counter;

    m_fusion_params.lstm_params.running_counter = m_running_counter;

    m_post_spec_waveform_lstm_params.lstm_params[0].running_counter = m_running_counter;
    m_post_spec_waveform_lstm_params.lstm_params[1].running_counter = m_running_counter;
}

void Rt3sProcessor::SetTaskBuffers(uint64_t d_device_buffers) {
    // spec mask apply <- spectrogram task -> spec encode task
    uint64_t d_spec_reordered = d_device_buffers + c_buffer_offsets_bytes[0];
    // spec encode task -> pre spec lstm
    uint64_t d_spec_waveform_encoded = d_device_buffers + c_buffer_offsets_bytes[1];
    uint64_t d_spec_encoded = d_spec_waveform_encoded;

    // conv encode -> conv activation
    uint64_t d_conv_encoded = d_device_buffers + c_buffer_offsets_bytes[2];
    // waveform mask apply <- conv activation -> waveform encode task
    uint64_t d_audio_basis = d_device_buffers + c_buffer_offsets_bytes[3];
    // waveform encode task -> pre waveform lstm
    uint64_t d_waveform_encoded = d_spec_encoded + c_nspec_features * sizeof(float);

    // pre spec lstm -> fusion lstm <- pre waveform lstm
    uint64_t d_spec_waveform_pre_rnn = d_device_buffers + c_buffer_offsets_bytes[4];
    uint64_t d_spec_pre_rnn = d_spec_waveform_pre_rnn;
    uint64_t d_waveform_pre_rnn = d_spec_pre_rnn + c_nspec_features * sizeof(float);

    // post spec lstm <- fusion -> post waveform lstm
    uint64_t d_spec_waveform_with_residual = d_device_buffers + c_buffer_offsets_bytes[5];
    uint64_t d_spec_with_residual = d_spec_waveform_with_residual;
    uint64_t d_waveform_with_residual = d_spec_with_residual + c_nspec_features * sizeof(float);

    // post spec lstm -> spec norm
    uint64_t d_spec_post_rnn = d_device_buffers + c_buffer_offsets_bytes[6];
    // spec norm -> spec mask
    uint64_t d_spec_post_rnn_norm = d_device_buffers + c_buffer_offsets_bytes[7];
    // spec mask -> spec mask reorder
    uint64_t d_spec_mask_linear = d_device_buffers + c_buffer_offsets_bytes[8];

    // spec mask reorder -> spec inverse task
    uint64_t d_spec_real_4x2ch = d_device_buffers + c_buffer_offsets_bytes[9];
    uint64_t d_spec_imag_4x2ch = d_device_buffers + c_buffer_offsets_bytes[10];

    // post waveform lstm -> waveform norm
    uint64_t d_waveform_post_rnn = d_device_buffers + c_buffer_offsets_bytes[11];
    // waveform norm -> waveform mask
    uint64_t d_waveform_post_rnn_norm = d_device_buffers + c_buffer_offsets_bytes[12];
    // waveform mask -> waveform mask reorder
    uint64_t d_waveform_mask_linear = d_device_buffers + c_buffer_offsets_bytes[13];
    // waveform mask reorder -> conv decode
    uint64_t d_audio_basis_per_source = d_device_buffers + c_buffer_offsets_bytes[14];

    ////////////////////////////////
    // initialize gpu task parameters
    // pre fusion spec
    m_spec_params = {
        .p_spec_reordered_out = d_spec_reordered,
        .signal_length = c_signal_length,
        .hop_size = c_hop_size};
    m_spec_enc_params = {
        .p_spec_reordered_in = d_spec_reordered,
        .p_spec_encoded_out = d_spec_encoded,
        .nblocks = m_gpu_task[2].block_count};

    // pre fusion waveform
    m_conv_encode_params = {
        .p_conv_enc_out = d_conv_encoded,
        .nblocks = m_gpu_task[3].block_count};
    m_conv_activation_params = {
        .p_conv_enc_in = d_conv_encoded,
        .p_audio_basis_out = d_audio_basis,
        .nblocks = m_gpu_task[4].block_count};
    m_basis_embed_params = {
        .p_audio_basis_in = d_audio_basis,
        .p_waveform_encoded_out = d_waveform_encoded,
        .nblocks = m_gpu_task[5].block_count};

    // pre-fusion lstm
    m_pre_spec_waveform_lstm_params = {
        .lstm_params = {
            {.p_in = d_spec_encoded,
                .p_out = d_spec_pre_rnn,
                .running_counter = m_running_counter},
            {.p_in = d_waveform_encoded,
                .p_out = d_waveform_pre_rnn,
                .running_counter = m_running_counter}}};

    // fusion
    m_fusion_params = {
        .lstm_params = {
            .p_in = d_spec_waveform_pre_rnn,
            .p_out = d_spec_waveform_with_residual,
            .running_counter = m_running_counter},
        .p_spec_waveform_encoded_in = d_spec_waveform_encoded};

    // post-fusion lstm
    m_post_spec_waveform_lstm_params = {
        .lstm_params = {
            {.p_in = d_spec_with_residual,
                .p_out = d_spec_post_rnn,
                .running_counter = m_running_counter},
            {.p_in = d_waveform_with_residual,
                .p_out = d_waveform_post_rnn,
                .running_counter = m_running_counter}}};

    // post-post-fusion rms
    m_spec_waveform_norm_params = {
        .norm_params = {
            {.p_data_in = d_spec_post_rnn,
                .p_data_norm_out = d_spec_post_rnn_norm},
            {.p_data_in = d_waveform_post_rnn,
                .p_data_norm_out = d_waveform_post_rnn_norm}}};

    // post fusion spec
    m_spec_mask_params = {
        .p_spec_post_rnn_norm_in = d_spec_post_rnn_norm,
        .p_spec_mask_linear_out = d_spec_mask_linear,
        .nblocks = m_gpu_task[13].block_count};
    m_spec_mask_apply_params = {
        .p_spec_mask_linear_in = d_spec_mask_linear,
        .p_spec_reordered_in = d_spec_reordered,
        .p_spec_per_source_out = d_spec_real_4x2ch,
        .p_phase_per_source_out = d_spec_imag_4x2ch,
        .nblocks = m_gpu_task[14].block_count};
    m_inv_spec_params = {
        .p_spec_per_source_in = d_spec_real_4x2ch,
        .p_phase_per_source_in = d_spec_imag_4x2ch,
        .signal_length = c_signal_length,
        .hop_size = c_hop_size,
        .nblocks = m_gpu_task[15].block_count};

    // post fusion waveform
    m_waveform_mask_params = {
        .p_waveform_post_rnn_norm_in = d_waveform_post_rnn_norm,
        .p_waveform_mask_linear_out = d_waveform_mask_linear,
        .nblocks = m_gpu_task[16].block_count};
    m_waveform_mask_apply_params = {
        .p_waveform_mask_linear_in = d_waveform_mask_linear,
        .p_audio_basis_in = d_audio_basis,
        .p_basis_per_source_out = d_audio_basis_per_source,
        .nblocks = m_gpu_task[17].block_count};
    m_conv_decode_params = {
        .p_basis_per_source_in = d_audio_basis_per_source,
        .nblocks = m_gpu_task[18].block_count};

    // combine branches
    m_combine_branches_params = {
        .buffer_length = c_buffer_capacity // Only full buffers supported
    };
}

void Rt3sProcessor::SetUpCommon(Rt3sConfig::Specification const* spec) {
    // specify what type of output port the processor has and create it
    PortInfo output_port_info {};
    output_port_info.type = PortType::eRegularPort;
    output_port_info.data_type = PortDataType::eSample32;
    output_port_info.capacity_in_bytes = c_buffer_capacity * sizeof(float);
    output_port_info.size_in_bytes = 0;
    output_port_info.channel_count = c_nsources * c_nchannels;
    output_port_info.grain = c_buffer_capacity * sizeof(float);
    m_output_port = m_port_factory.CreateDataPort(0, output_port_info);

    // create the processor's input port
    m_input_port = std::make_unique<Rt3sInputPort>(m_output_port.get());

    // copy-once data: lookup tables for specification
    m_d_twiddle_LUT = m_memory_manager.AllocateGpuMemory(MaxTwiddleLUT * 2 * sizeof(float));
    m_d_twiddle_inverse_LUT = m_memory_manager.AllocateGpuMemory(MaxTwiddleLUT * 2 * sizeof(float));

    // copy-once data: device HS-TasNet (network parameters and state)
    m_d_hstasnet = m_memory_manager.AllocateGpuMemory(sizeof(gpua::GpuHsTasNet<float>));

    ////////////////////////////////
    // initialize the host network
    ParamStreamBuf weights_in_buf(spec->params, spec->params_bytes);
    std::istream weights_in(&weights_in_buf);
    weights_in.unsetf(std::ios::skipws);
    auto hstasnet = std::make_unique<gpua::GpuHsTasNet<float>>(weights_in, m_d_twiddle_LUT->GetGpuPointer(), m_d_twiddle_inverse_LUT->GetGpuPointer());

    // transfer copy-once data to the gpu
    m_memory_manager.MemCpyCpuToGpu(*m_d_twiddle_LUT.get(), 0u, &twiddle_LUT[0], MaxTwiddleLUT * 2 * sizeof(float));
    m_memory_manager.MemCpyCpuToGpu(*m_d_twiddle_inverse_LUT.get(), 0u, &twiddle_inverse_LUT[0], MaxTwiddleLUT * 2 * sizeof(float));
    m_memory_manager.MemCpyCpuToGpu(*m_d_hstasnet.get(), 0u, hstasnet.get(), sizeof(gpua::GpuHsTasNet<float>));

    ////////////////////////////////
    // configure the gpu tasks
    // enqueue
    RT3S::enqueue_input_task(m_gpu_task[0]);

    // spec pre fusion
    RT3S::spec_task(m_gpu_task[1]);
    RT3S::spec_encode_task(m_gpu_task[2]);

    // waveform pre fusion
    RT3S::conv_encode_task(m_gpu_task[3]);
    RT3S::conv_activation_task(m_gpu_task[4]);
    RT3S::basis_embed_task(m_gpu_task[5]);

    // pre-fusion lstm
    RT3S::pre_spec_waveform_lstm_task_0(m_gpu_task[6], hstasnet.get());
    RT3S::pre_spec_waveform_lstm_task_1(m_gpu_task[7], hstasnet.get());

    // fusion
    RT3S::fusion_task_0(m_gpu_task[8], hstasnet.get());
    RT3S::fusion_task_1(m_gpu_task[9], hstasnet.get());

    // post-fusion lstm
    RT3S::post_spec_waveform_lstm_task_0(m_gpu_task[10], hstasnet.get());
    RT3S::post_spec_waveform_lstm_task_1(m_gpu_task[11], hstasnet.get());

    // post-post-fusion lstm rms norm.
    RT3S::spec_waveform_rms_norm_task(m_gpu_task[12]);

    // spec post fusion
    RT3S::spec_mask_task(m_gpu_task[13]);
    RT3S::spec_mask_apply_task(m_gpu_task[14]);
    RT3S::inv_spec_task(m_gpu_task[15]);

    // waveform post fusion
    RT3S::waveform_mask_task(m_gpu_task[16]);
    RT3S::waveform_mask_apply_task(m_gpu_task[17]);
    RT3S::conv_decode_task(m_gpu_task[18]);

    // combine branches
    RT3S::combine_branches_task(m_gpu_task[19]);

    ////////////////////////////////
    // allocate device buffers
    // DEBUG BUFFER
    d_device_buffer0 = m_memory_manager.AllocateGpuMemory(c_buffer_total_bytes);
    d_device_buffer[0] = d_device_buffer0.get();
    d_device_buffer1 = m_memory_manager.AllocateGpuMemory(c_buffer_total_bytes);
    d_device_buffer[1] = d_device_buffer1.get();
    h_debug_buffer.clear();
    h_debug_buffer.resize(c_buffer_total_bytes / sizeof(float), 0.0f);
}

Rt3sProcessor::Rt3sProcessor(::ProcessorSpecification& specification, ::Module& module) :
    m_module {module},
    m_proc_data {.num_calls = 1u, .proc_param_size = sizeof(gpua::ProcessorParameter), .processing_end_callback_requirements = ::ProcessorEndCallback::eNoCallback, .task_count = 20u, .task_information = &m_gpu_task[0]},
    m_port_factory {specification.port_factory},
    m_memory_manager {specification.memory_manager} {
    Rt3sConfig::Specification const* spec = reinterpret_cast<Rt3sConfig::Specification const*>(specification.user_data);
    if (specification.data_size != sizeof(Rt3sConfig::Specification) || spec->ThisMagic != spec->Magic) {
        throw std::runtime_error("Error in Rt3sProcessor::Rt3sProcessor: invalid specification provided");
    }
    Rt3sProcessor::SetUpCommon(spec);
}
