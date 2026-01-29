/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#ifndef GPU_AUDIO_RT3S_PROCESSOR_CUH_INCLUDED
#define GPU_AUDIO_RT3S_PROCESSOR_CUH_INCLUDED

#ifdef GPU_AUDIO_RTC_ONLINE
#ifndef GPU_AUDIO_RTC_ONLINE_STDINT
#define GPU_AUDIO_RTC_ONLINE_STDINT
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
#endif // ifndef GPU_AUDIO_RTC_ONLINE_STDINT
#endif // ifdef GPU_AUDIO_RTC_ONLINE

#include "Properties.h"
#include "GpuHsTasNet.h"

#include <gpu_primitives/GpuMatrix.h>
#include <gpu_primitives/RMSNorm.h>
#include <gpu_primitives/Spectrogram.h>

#include <platform/Abstraction.h>
#include <scheduler/device/defaultcontext.cuh>

namespace gpua {
class Rt3sDevice {
    struct AddOperatorArray {
        struct AddOperator {
            __device_addr float* m_out;
            __device_addr const float* m_add;

            __device_fct __forceinline_fct float operator=(float value) {
                return *m_out = *m_add + value;
            }
        };

        __device_addr float* m_out;
        __device_addr const float* m_add;

        __device_fct __forceinline_fct AddOperator operator[](uint32_t off) {
            return AddOperator {m_out + off, m_add + off};
        }
    };
    template <uint32_t Layer, typename LSTMClass, typename TaskParamType, typename Context>
    __device_fct __forceinline_fct void general_lstm_task(Context context, __device_addr LSTMClass& lstm, __device_addr ProcessorParameter* processor_param, __device_addr TaskParamType* task_param) __device_addr {
        // auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        auto vec_in = reinterpret_cast<__device_addr typename LSTMClass::InVecType const*>(task_param->p_in);
        auto vec_out = reinterpret_cast<__device_addr typename LSTMClass::OutVecType*>(task_param->p_out);
        __threadgroup_addr float* smem = reinterpret_cast<__threadgroup_addr float*>(context.smem());

        if constexpr (LSTMClass::NumLayers == Layer + 1) {
            AddOperatorArray add_op_array {&(*vec_out)[0], &(*vec_in)[0]};
            lstm.template Process<Layer, TaskParamType::Blocks, TaskParamType::BlockSize, TaskParamType::GroupSize>(context, smem, add_op_array, &(*vec_in)[0], task_param->running_counter);
        }
        else {
            lstm.template Process<Layer, TaskParamType::Blocks, TaskParamType::BlockSize, TaskParamType::GroupSize>(context, smem, &(*vec_out)[0], &(*vec_in)[0], task_param->running_counter);
        }
    }

public:
    // mandatory explicit defined constructor
    __device_fct Rt3sDevice() __device_addr {
    }

    // mandatory explicitly defined destructor
    __device_fct ~Rt3sDevice() __device_addr {
    }

    // mandatory init function; can be used to initialize processor data members
    template <class Context>
    __device_fct void init(Context context, unsigned int maxBufferLength) __device_addr {
    }

    // Every task of the processor must match the following interface:
    // ```
    // template <class Context>
    // __device_fct void process(Context context, __device_addr gain::ProcessorParameter* processor_param, __device_addr gain::TaskParameter* task_param,
    //                           __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr;
    // ```
    //
    // - `Context` provides all the call specific information, like `threadId`, `blockId`, `callId`, shared memory, synchronization ...
    //    - Depending on the scheduling environment and the task type, we specialize the `Context` as needed this is why it is a template parameter
    //
    // - `ProcessorParameter* processor_param` is a parameter that can be set per call of the processor
    //
    // - `TaskParameter* task_param` is a parameter specific to the task and is set for every task individually
    //    - All tasks get the same `ProcessorParameter` and individual `TaskParameters`
    //    - The host interface has to provide the sizes required for the parameters (see GainProcessor::m_proc_data and GainProcessor::m_gpu_task)
    //
    // - `float** input` points to the input. input[p][s] is sample s of port p.
    //    Layout: all samples of the first channel, all samples of the second channel, ...
    // - `float** output` points to allocated device memory for the output. output[p][s] is sample s of port p.
    //    Layout: all samples of the first channel, all samples of the second channel, ...
    //
    // ================================
    // Basic functionalities of `Context` are:
    //    - `call()` to get the call id : [0, GainProcessor::m_proc_data::num_calls - 1]
    //    - `blockId()` to get the blockId : [0, GainProcessor::m_gpu_task::block_count - 1]
    //    - `threadId()` to get the threadId : [0, GainProcessor::m_gpu_task::thread_count - 1]
    //    - `blockDim()` to get the blockSize : GainProcessor::m_gpu_task::thread_count
    //    - `smem()` to get the registered shared memory : GainProcessor::m_gpu_task::shared_mem_size bytes
    //    - `synchronize()` to synchronize all threads in the block
    //
    // Note:
    //  - exclusively use `context.synchronize()` for synchronization of a block. Platform specific sync operations might hang
    //  - use `context.blockId()`, `context.threadId()`, `context.blockDim()` instead of platform specific alternatives, which might be wrong
    //  - you can use multiple blocks,e.g., one or multiple per channel - Make sure in the host processor that enough blocks are requested (GainProcessor::m_gpu_task::block_count)
    //  - you can use multiple calls to split the processing of a longer buffer into smaller grain-sized portions. You can also do that with a loop inside the task.
    //    However, when you use multiple calls execution can parallelize better if there are multiple processors in the chain, as we can already execute the next processor
    //    when parts of its input, i.e., the current processors output, are available. It will guarantee that, within a processor, a grain-sized portion of the input
    //    will only be processed when the previous portion has been processed.

    ////////////////////////////////////////////////////////////////
    // 0. enqueue input (task 0)
    template <typename Context>
    __device_fct __forceinline_fct void enqueue_input_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr void* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        // one block per channel; offset to their input/output samples
        uint32_t ch = context.blockId();
        uint32_t block_offset = ch * processor_param->buffer_capacity;

        if (ch < 2u) {
            // stereo audio input; directly used in conv_encode_taks and spec_task
            __device_addr auto& ringbuffer = hstasnet->m_input_history;

            ringbuffer.StoreChannelData(context, ch, processor_param->ringbuffer_cursor, processor_param->buffer_capacity, input[0] + block_offset);
        }
        else if (ch < 10u) {
            // spectrogram branch output history; required for overlap in inv_spec_task
            __device_addr auto& ringbuffer = hstasnet->m_spec_output_history;

            // set [cursor, currsor + 512] to 0
            ringbuffer.StoreChannelConstant(context, ch - 2u, processor_param->ringbuffer_cursor, processor_param->buffer_capacity, 0.0f);
        }
        else if (ch < 18u) {
            // waveform branch output history; required for overlap in conv_decode_task
            __device_addr auto& ringbuffer = hstasnet->m_waveform_output_history;

            // set [cursor, currsor + 512] to 0
            ringbuffer.StoreChannelConstant(context, ch - 10u, processor_param->ringbuffer_cursor, processor_param->buffer_capacity, 0.0f);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 1.1 spectral pre fusion (task 1)
    template <typename Context>
    __device_fct __forceinline_fct void spec_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr SpecParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // Encoder: Spectrogram
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        __device_addr auto& ringbuffer = hstasnet->m_input_history;

        // one block per channel; offset to their input/output samples
        uint32_t ch = context.blockId();
        auto input_data = ringbuffer.GetLinearChannelAccess(ch, processor_param->ringbuffer_cursor - processor_param->buffer_capacity);

        // compute the spectrogram of 1024 smaples; 512 previous samples + 512 current samples
        hstasnet->m_stft.spec_512_8_8_8_phase_hann(context,
            static_cast<__threadgroup_addr float*>(context.smem()),
            input_data /* audio samples */,
            reinterpret_cast<__device_addr float*>(task_param->p_spec_reordered_out),
            task_param->signal_length, task_param->hop_size);
    }

    template <typename Context>
    __device_fct __forceinline_fct void spec_encode_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr SpecEncParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        //  linear layer 1x2052x500 -> 1x500
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        // reordered spectrogram; input to linear layer
        auto vec_in = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_encode)::InVecType const*>(task_param->p_spec_reordered_in);
        // encoded spectreogram; output of linear layer
        auto vec_out = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_encode)::OutVecType*>(task_param->p_spec_encoded_out);
        hstasnet->m_spec_encode.Process(context, vec_out, vec_in, task_param->nblocks);
    }

    ////////////////////////////////////////////////////////////////
    // 1.2 waveform pre fusion (task 3)
    template <typename Context>
    __device_fct __forceinline_fct void conv_encode_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr ConvEncodeParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        __device_addr auto& ringbuffer = hstasnet->m_input_history;

        // get access to 1024 samples of each channel: 2048x1 samples; input to convolution
        auto input_data = ringbuffer.template GetFlattenedAccess<1024u>(0u, processor_param->ringbuffer_cursor - processor_param->buffer_capacity);
        // convolution output: 1x3000
        auto vec_out = reinterpret_cast<__device_addr decltype(hstasnet->m_conv_encode)::OutVecType*>(task_param->p_conv_enc_out);
        // compute the convolution: 3000x2048 * 2048x1 + 3000x1
        hstasnet->m_conv_encode.Process(context, *vec_out, input_data, task_param->nblocks);
    }

    // task 4
    template <typename Context>
    __device_fct __forceinline_fct void conv_activation_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr ConvActivationParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        uint32_t tid = context.threadId() + context.blockId() * context.blockDim();
        uint32_t grid_dim = context.blockDim() * task_param->nblocks;
        // convolution output
        auto vec_in = reinterpret_cast<__device_addr float const*>(task_param->p_conv_enc_in);
        auto vec_out = reinterpret_cast<__device_addr float*>(task_param->p_audio_basis_out);
        // apply activation: relu[0..1500) * sigmout[1500..3000)
        for (uint32_t oid {tid}; oid < 1500u; oid += grid_dim) {
            vec_out[oid] = ActivationFuncRelu::Eval(vec_in[oid]) * ActivationFuncSigmoid::Eval(vec_in[oid + 1500]);
        }
    }

    // task 5
    template <typename Context>
    __device_fct __forceinline_fct void basis_embed_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr BasisEmbedParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        // compute linear layer: 500x1500 * 1500x1 -> 500x1
        auto vec_in = reinterpret_cast<__device_addr decltype(hstasnet->m_basis_to_embed)::InVecType const*>(task_param->p_audio_basis_in);
        auto vec_out = reinterpret_cast<__device_addr decltype(hstasnet->m_basis_to_embed)::OutVecType*>(task_param->p_waveform_encoded_out);
        hstasnet->m_basis_to_embed.Process(context, vec_out, vec_in, task_param->nblocks);
    }

    ////////////////////////////////////////////////////////////////
    // 2.1 pre-fusion lstms (task 6)
    template <typename Context>
    __device_fct __forceinline_fct void pre_spec_waveform_lstm_task_0(Context context, __device_addr ProcessorParameter* processor_param, __device_addr PreSpecWaveformLstmParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 0 500-> 500
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        constexpr uint32_t blocks_per_branch = PreSpecWaveformLstmParams::LstmParams::Blocks;
        Wip::DefaultContext subcontext(context.threadId(), 32u, context.blockDim(), context.blockId() % blocks_per_branch, blocks_per_branch, reinterpret_cast<__threadgroup_addr char*>(context.smem()));
        if (context.blockId() < blocks_per_branch) {
            // spectral banch lstm layer 0
            general_lstm_task<0>(subcontext, hstasnet->m_pre_spec_branch, processor_param, &task_param->lstm_params[0]);
        }
        else {
            // waveform branch lstm layer 0
            general_lstm_task<0>(subcontext, hstasnet->m_pre_waveform_branch, processor_param, &task_param->lstm_params[1]);
        }
    }
    // task 7
    template <typename Context>
    __device_fct __forceinline_fct void pre_spec_waveform_lstm_task_1(Context context, __device_addr ProcessorParameter* processor_param, __device_addr PreSpecWaveformLstmParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 1 500-> 500
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        constexpr uint32_t blocks_per_branch = PreSpecWaveformLstmParams::LstmParams::Blocks;
        Wip::DefaultContext subcontext(context.threadId(), 32u, context.blockDim(), context.blockId() % blocks_per_branch, blocks_per_branch, reinterpret_cast<__threadgroup_addr char*>(context.smem()));
        if (context.blockId() < blocks_per_branch) {
            // spectral banch lstm layer 1
            general_lstm_task<1>(subcontext, hstasnet->m_pre_spec_branch, processor_param, &task_param->lstm_params[0]);
        }
        else {
            // waveform branch lstm layer 1
            general_lstm_task<1>(subcontext, hstasnet->m_pre_waveform_branch, processor_param, &task_param->lstm_params[1]);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 2.2 fusion lstm (task 8)
    template <typename Context>
    __device_fct __forceinline_fct void fusion_task_0(Context context, __device_addr ProcessorParameter* processor_param, __device_addr FusionParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 0 1000 -> 1000
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        // fusion: combined lstm; input from spectral pre-fusion lstm [0..500) and waveform pre-fusion lstm [500...1000)
        general_lstm_task<0>(context, hstasnet->m_fusion_branch, processor_param, &task_param->lstm_params);
    }
    // task 9
    template <typename Context>
    __device_fct __forceinline_fct void fusion_task_1(Context context, __device_addr ProcessorParameter* processor_param, __device_addr FusionParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 1 1000 -> 1000
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        general_lstm_task<1>(context, hstasnet->m_fusion_branch, processor_param, &task_param->lstm_params);

        // add p_spec_waveform_encoded to fusion lstm output
        constexpr uint32_t HiddenSize = decltype(hstasnet->m_fusion_branch)::HiddenSize;
        constexpr uint32_t Blocks = decltype(task_param->lstm_params)::Blocks;
        constexpr uint32_t BlockSize = decltype(task_param->lstm_params)::BlockSize;
        constexpr uint32_t OutputsPerBlock = ((HiddenSize + Blocks - 1) / Blocks);

        // avoid race-condition: each thread only uses data he computed himself
        uint32_t start_row = context.blockId() * OutputsPerBlock;
        for (uint32_t out_index = context.threadId(); out_index < OutputsPerBlock && start_row + out_index < HiddenSize; out_index += BlockSize) {
            uint32_t out_row = start_row + out_index;
            reinterpret_cast<__device_addr float*>(task_param->lstm_params.p_out)[out_row] += reinterpret_cast<__device_addr float const*>(task_param->p_spec_waveform_encoded_in)[out_row];
        }
    }

    ////////////////////////////////////////////////////////////////
    // 2.3 post-fusion lstms (task 10)
    template <typename Context>
    __device_fct __forceinline_fct void post_spec_waveform_lstm_task_0(Context context, __device_addr ProcessorParameter* processor_param, __device_addr PostSpecWaveformLstmParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 0 500-> 500
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        constexpr uint32_t blocks_per_branch = PostSpecWaveformLstmParams::LstmParams::Blocks;
        Wip::DefaultContext subcontext(context.threadId(), 32u, context.blockDim(), context.blockId() % blocks_per_branch, blocks_per_branch, reinterpret_cast<__threadgroup_addr char*>(context.smem()));
        if (context.blockId() < blocks_per_branch) {
            // spectral banch lstm layer 0; input [0..500) of fusion task output
            general_lstm_task<0>(subcontext, hstasnet->m_post_spec_branch, processor_param, &task_param->lstm_params[0]);
        }
        else {
            // waveform branch lstm layer 0; input [500..1000) of fusion task output
            general_lstm_task<0>(subcontext, hstasnet->m_post_waveform_branch, processor_param, &task_param->lstm_params[1]);
        }
    }
    template <typename Context>
    __device_fct __forceinline_fct void post_spec_waveform_lstm_task_1(Context context, __device_addr ProcessorParameter* processor_param, __device_addr PostSpecWaveformLstmParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // LSTM layer 1 500-> 500
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        constexpr uint32_t blocks_per_branch = PostSpecWaveformLstmParams::LstmParams::Blocks;
        Wip::DefaultContext subcontext(context.threadId(), 32u, context.blockDim(), context.blockId() % blocks_per_branch, blocks_per_branch, reinterpret_cast<__threadgroup_addr char*>(context.smem()));
        if (context.blockId() < blocks_per_branch) {
            // spectral banch lstm layer 1
            general_lstm_task<1>(subcontext, hstasnet->m_post_spec_branch, processor_param, &task_param->lstm_params[0]);
        }
        else {
            // waveform branch lstm layer 1
            general_lstm_task<1>(subcontext, hstasnet->m_post_waveform_branch, processor_param, &task_param->lstm_params[1]);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 2.4 post-post-fusion lstms rms normalization (task 12)
    template <typename Context>
    __device_fct __forceinline_fct void spec_waveform_rms_norm_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr SpecWaveformNormParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        constexpr uint32_t BLOCK_DIM {256u};

        if (context.blockId() == 0) {
            // normalize output of spectral post fusion lstm
            auto vec_in_s = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_norm)::InVecType const*>(task_param->norm_params[0].p_data_in);
            auto vec_out_s = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_norm)::OutVecType*>(task_param->norm_params[0].p_data_norm_out);
            using SmemTypeSpec = decltype(hstasnet->m_spec_norm)::Smem<BLOCK_DIM>;
            hstasnet->m_spec_norm.Process<BLOCK_DIM>(context, reinterpret_cast<__threadgroup_addr SmemTypeSpec*>(context.smem()), *vec_out_s, *vec_in_s);
        }
        else if (context.blockId() == 1) {
            // normalize output of waveform post fusion lstm
            auto vec_in_w = reinterpret_cast<__device_addr decltype(hstasnet->m_waveform_norm)::InVecType const*>(task_param->norm_params[1].p_data_in);
            auto vec_out_w = reinterpret_cast<__device_addr decltype(hstasnet->m_waveform_norm)::OutVecType*>(task_param->norm_params[1].p_data_norm_out);
            using SmemTypeWaveform = decltype(hstasnet->m_waveform_norm)::Smem<BLOCK_DIM>;
            hstasnet->m_waveform_norm.Process<BLOCK_DIM>(context, reinterpret_cast<__threadgroup_addr SmemTypeWaveform*>(context.smem()), *vec_out_w, *vec_in_w);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 3.1 spectral post fusion (task 13)
    template <typename Context>
    __device_fct __forceinline_fct void spec_mask_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr SpecMaskParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        // normalized lstm output 1x500
        auto vec_in = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_mask_linear)::InVecType const*>(task_param->p_spec_post_rnn_norm_in);
        // linear spectral mask 1x8208
        auto vec_out = reinterpret_cast<__device_addr decltype(hstasnet->m_spec_mask_linear)::OutVecType*>(task_param->p_spec_mask_linear_out);
        // linear layer 1x500 * 500x8208
        hstasnet->m_spec_mask_linear.Process(context, vec_out, vec_in, task_param->nblocks);
    }

    // task 14
    template <typename Context>
    __device_fct __forceinline_fct void spec_mask_apply_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr SpecMaskApplyParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        uint32_t tid = context.threadId() + context.blockId() * context.blockDim();
        uint32_t grid_dim = context.blockDim() * task_param->nblocks;

        constexpr uint32_t nchannels {2u};
        constexpr uint32_t nbins {513u};
        constexpr uint32_t ncomplex {2u};
        constexpr uint32_t nsources {4u};
        constexpr uint32_t nlids {nchannels * nbins * ncomplex * nsources};

        // linear spectral branch mask
        auto mask_linear = reinterpret_cast<__device_addr float const*>(task_param->p_spec_mask_linear_in);
        // the spectrogram from the input
        auto spec_phase_1x2 = reinterpret_cast<__device_addr float const*>(task_param->p_spec_reordered_in);
        // per source spectrograms
        auto spec_4x2 = reinterpret_cast<__device_addr float*>(task_param->p_spec_per_source_out);
        auto phase_4x2 = reinterpret_cast<__device_addr float*>(task_param->p_phase_per_source_out);

        for (uint32_t i {tid}; i < nlids; i += grid_dim) {
            uint32_t ti {i};

            uint32_t const source = ti % nsources;
            ti /= nsources;

            uint32_t const complex = ti % ncomplex;
            ti /= ncomplex;

            uint32_t const bin = ti % nbins;
            ti /= nbins;

            uint32_t const channel = ti % nchannels;

            uint32_t const within_source_off = channel * nbins + bin;
            auto src_ptr = spec_phase_1x2 + channel * 2 * nbins + 2 * bin + complex;
            auto dst_ptr = (complex ? phase_4x2 : spec_4x2) + source * nchannels * nbins + within_source_off;

            // applyy the mask to the input spectrogram to compute the source spectrogram
            *dst_ptr = *(mask_linear + i) * *src_ptr;
        }
    }

    // task 15
    template <typename Context>
    __device_fct __forceinline_fct void inv_spec_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr InvSpecParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // Decoder: i-Spectrogram
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        __device_addr auto& ringbuffer = hstasnet->m_spec_output_history;

        // 8 channels = 4 sources with 2 channels each
        for (uint32_t ch {context.blockId()}; ch < 8u; ch += task_param->nblocks) {
            // get linear access to [cursor - 512]
            auto output_data = ringbuffer.GetLinearChannelAccess(ch, processor_param->ringbuffer_cursor - processor_param->buffer_capacity);

            constexpr uint32_t nbins {513u};
            uint32_t block_offset = ch * nbins;
            // compute the inverse spectrogram of each source to get reconstructed audio samples
            hstasnet->m_stft.inv_spec_512_8_8_8_hann(context,
                static_cast<__threadgroup_addr float*>(context.smem()),
                reinterpret_cast<__device_addr float const*>(task_param->p_spec_per_source_in) + block_offset /* spectrogram */,
                reinterpret_cast<__device_addr float const*>(task_param->p_phase_per_source_in) + block_offset /* phase */,
                output_data /* recon. audio*/,
                task_param->signal_length, task_param->hop_size,
                processor_param->buffer_capacity, processor_param->buffer_capacity);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 3.2 waveform post fusion (task 16)
    template <typename Context>
    __device_fct __forceinline_fct void waveform_mask_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr WaveformMaskParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);

        // normalized output of post-fusion lstm 1x500
        auto vec_in = reinterpret_cast<__device_addr decltype(hstasnet->m_waveform_mask_linear)::InVecType const*>(task_param->p_waveform_post_rnn_norm_in);
        // waveform branch linear mask 1x6000
        auto vec_out = reinterpret_cast<__device_addr decltype(hstasnet->m_waveform_mask_linear)::OutVecType*>(task_param->p_waveform_mask_linear_out);
        // linear layer 1x500 * 500x6000
        hstasnet->m_waveform_mask_linear.Process(context, vec_out, vec_in, task_param->nblocks);
    }

    // task 17
    template <typename Context>
    __device_fct __forceinline_fct void waveform_mask_apply_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr WaveformMaskApplyParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        // 4x1500 per source mask
        auto mask_per_source = reinterpret_cast<__device_addr float const*>(task_param->p_waveform_mask_linear_in);
        // 1x1500 audio basis of the input
        auto basis = reinterpret_cast<__device_addr float const*>(task_param->p_audio_basis_in);
        // 4x1500 audio basis per source
        auto basis_per_source = reinterpret_cast<__device_addr float*>(task_param->p_basis_per_source_out);

        uint32_t tid = context.threadId() + context.blockId() * context.blockDim();
        uint32_t grid_dim = context.blockDim() * task_param->nblocks;
        // apply the mask to the audio basis of the input to get a per source audio basis
        for (uint32_t oid {tid}; oid < 6000u; oid += grid_dim) {
            basis_per_source[oid] = mask_per_source[oid] * basis[oid % 1500];
        }
    }

    // task 18
    template <typename Context>
    __device_fct __forceinline_fct void conv_decode_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr ConvDecodeParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        __device_addr auto& ringbuffer = hstasnet->m_waveform_output_history;

        // transposed convolution of the per source audio basis to get a reconstructed audio signal
        for (uint32_t source {0u}; source < 4u; ++source) {
            // 1x1500 audio basis of the source
            auto vec_in = reinterpret_cast<__device_addr decltype(hstasnet->m_conv_decode)::InVecType const*>(task_param->p_basis_per_source_in + source * 1500u * sizeof(float));
            // stereo audio output 2x1024. Only the first 512 samples are final, as they overlap with the 512 samples of each channel from the prev. call
            auto output_data = ringbuffer.template GetFlattenedAccess<1024u>(source * 2u, processor_param->ringbuffer_cursor - processor_param->buffer_capacity);

            hstasnet->m_conv_decode.template Process<AccumulateOp>(context, output_data, *vec_in, task_param->nblocks);
        }
    }

    ////////////////////////////////////////////////////////////////
    // 4. combine branches (task 19)
    template <typename Context>
    __device_fct __forceinline_fct void combine_branches_task(Context context, __device_addr ProcessorParameter* processor_param, __device_addr CombineBranchesParams* task_param, __device_addr float* __device_addr* input, __device_addr float* __device_addr* output) __device_addr {
        auto hstasnet = reinterpret_cast<__device_addr gpua::GpuHsTasNet<float>*>(processor_param->d_hstasnet);
        uint32_t ch = context.blockId();

        // get access to [cursor - 512, cursor] in spec and waveform output
        uint32_t ringbuffer_offset = processor_param->ringbuffer_cursor - processor_param->buffer_capacity;
        auto spec_recon_in = hstasnet->m_spec_output_history.GetLinearChannelAccess(ch, ringbuffer_offset);
        auto waveform_recon_in = hstasnet->m_waveform_output_history.GetLinearChannelAccess(ch, ringbuffer_offset);

        // add [cursor - 512, cursor] from both branches and write the sum into the output to get the final output
        __device_addr float* recon_out = output[0] + ch * processor_param->buffer_capacity;
        for (uint32_t sid {context.threadId()}; sid < task_param->buffer_length; sid += context.blockDim()) {
            recon_out[sid] = spec_recon_in[sid] + waveform_recon_in[sid];
        }
    }
};

} // namespace gpua

#endif /* GPU_AUDIO_RT3S_PROCESSOR_CUH_INCLUDED */
