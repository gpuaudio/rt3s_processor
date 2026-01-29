/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#include "Rt3sProcessor.cuh"
#include <scheduler/device/processor.cuh>

// - `DeclareProcessorStep` must come before `DeclareProcessor`
// - `DeclareProcessorStep` parameters are:
//    - full processor name (with namespace and template parameters)
//    - an increasing integer to mark the tasks 0, 1, 2... (this integer is used to extract the actual task)
//    - the name of the method (currently no additional template parameters are supported)
//    - the processor parameter type (must match the function definition and has to be the same for all tasks)
//    - the task parameter type (must match the function definition and can be different for each task)
// - `DeclareProcessor` parameters are:
//    - full processor name (with namespace and template parameters)
//    - the number of tasks (must match the increasing integer from DeclareProcessorStep)

// 0. enqueu input
DeclareProcessorStep(gpua::Rt3sDevice, 0, enqueue_input_task, float, gpua::ProcessorParameter, void);

// 1.1 spec pre fusion
DeclareProcessorStep(gpua::Rt3sDevice, 1, spec_task, float, gpua::ProcessorParameter, gpua::SpecParams);
DeclareProcessorStep(gpua::Rt3sDevice, 2, spec_encode_task, float, gpua::ProcessorParameter, gpua::SpecEncParams);

// 1.2 waveform pre fusion
DeclareProcessorStep(gpua::Rt3sDevice, 3, conv_encode_task, float, gpua::ProcessorParameter, gpua::ConvEncodeParams);
DeclareProcessorStep(gpua::Rt3sDevice, 4, conv_activation_task, float, gpua::ProcessorParameter, gpua::ConvActivationParams);
DeclareProcessorStep(gpua::Rt3sDevice, 5, basis_embed_task, float, gpua::ProcessorParameter, gpua::BasisEmbedParams);

// 2.1 pre-fusion lstms
DeclareProcessorStep(gpua::Rt3sDevice, 6, pre_spec_waveform_lstm_task_0, float, gpua::ProcessorParameter, gpua::PreSpecWaveformLstmParams);
DeclareProcessorStep(gpua::Rt3sDevice, 7, pre_spec_waveform_lstm_task_1, float, gpua::ProcessorParameter, gpua::PreSpecWaveformLstmParams);

// 2.2 fusion
DeclareProcessorStep(gpua::Rt3sDevice, 8, fusion_task_0, float, gpua::ProcessorParameter, gpua::FusionParams);
DeclareProcessorStep(gpua::Rt3sDevice, 9, fusion_task_1, float, gpua::ProcessorParameter, gpua::FusionParams);

// 2.3 post-fusion lstms
DeclareProcessorStep(gpua::Rt3sDevice, 10, post_spec_waveform_lstm_task_0, float, gpua::ProcessorParameter, gpua::PostSpecWaveformLstmParams);
DeclareProcessorStep(gpua::Rt3sDevice, 11, post_spec_waveform_lstm_task_1, float, gpua::ProcessorParameter, gpua::PostSpecWaveformLstmParams);

// 2.4 post-post-fusion lstm rms norm
DeclareProcessorStep(gpua::Rt3sDevice, 12, spec_waveform_rms_norm_task, float, gpua::ProcessorParameter, gpua::SpecWaveformNormParams);

// 3.1 spec post fusion
DeclareProcessorStep(gpua::Rt3sDevice, 13, spec_mask_task, float, gpua::ProcessorParameter, gpua::SpecMaskParams);
DeclareProcessorStep(gpua::Rt3sDevice, 14, spec_mask_apply_task, float, gpua::ProcessorParameter, gpua::SpecMaskApplyParams);
DeclareProcessorStep(gpua::Rt3sDevice, 15, inv_spec_task, float, gpua::ProcessorParameter, gpua::InvSpecParams);

// 3.2 waveform post fusion
DeclareProcessorStep(gpua::Rt3sDevice, 16, waveform_mask_task, float, gpua::ProcessorParameter, gpua::WaveformMaskParams);
DeclareProcessorStep(gpua::Rt3sDevice, 17, waveform_mask_apply_task, float, gpua::ProcessorParameter, gpua::WaveformMaskApplyParams);
DeclareProcessorStep(gpua::Rt3sDevice, 18, conv_decode_task, float, gpua::ProcessorParameter, gpua::ConvDecodeParams);

// 4. combine branch outputs
DeclareProcessorStep(gpua::Rt3sDevice, 19, combine_branches_task, float, gpua::ProcessorParameter, gpua::CombineBranchesParams);

DeclareProcessor(gpua::Rt3sDevice, 20);
