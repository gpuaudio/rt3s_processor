/*
 * Copyright (c) 2022 Braingines SA - All Rights Reserved
 * Unauthorized copying of this file is strictly prohibited
 * Proprietary and confidential
 */

#include "Rt3sModule.h"
#include "Rt3sProcessor.h"

Rt3sModule::Rt3sModule(const GPUA::processor::v2::ModuleSpecification& specification) :
    GPUA::processor::v2::ModuleBase(specification) {}

GPUA::processor::v2::ErrorCode Rt3sModule::CreateProcessor(GPUA::processor::v2::ProcessorSpecification& specification, GPUA::processor::v2::Processor*& processor) noexcept {
    try {
        processor = new Rt3sProcessor(specification, *this);
        return GPUA::processor::v2::ErrorCode::eSuccess;
    }
    catch (...) {
    }
    processor = nullptr;
    return GPUA::processor::v2::ErrorCode::eFail;
}

GPUA::processor::v2::ErrorCode Rt3sModule::DeleteProcessor(GPUA::processor::v2::Processor* processor) noexcept {
    if (processor) {
        delete processor;
        return GPUA::processor::v2::ErrorCode::eSuccess;
    }
    return GPUA::processor::v2::ErrorCode::eFail;
}
