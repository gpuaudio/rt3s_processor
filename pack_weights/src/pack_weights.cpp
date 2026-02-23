#include <cassert>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "common/utils.h"

namespace {

enum class DataType : uint8_t {
    INVALID,
    VECTOR,
    MATRIX,
    TENSOR
};

// rows (in channels, features), cols (out channels, features), pages (kernel size)
using Dim = std::vector<int>;

using Metadata = std::tuple<std::string, DataType, Dim>;

auto getWeightFiles(const std::string& data_path) -> std::unordered_map<std::string_view, Metadata> {
    using namespace std::string_view_literals;
    return {
        {"spec_encode_bias"sv, Metadata {data_path + "/" + "data_0_spec_encode_bias.npy", DataType::VECTOR, {500}}},
        {"spec_encode_weight"sv, Metadata {data_path + "/" + "data_1_spec_encode_weight.npy", DataType::MATRIX, {500, 2052}}},
        {"audio_conv_encode_bias"sv, Metadata {data_path + "/" + "data_2_audio_conv_encode_bias.npy", DataType::VECTOR, {3000}}},
        {"audio_conv_encode_weight"sv, Metadata {data_path + "/" + "data_3_audio_conv_encode_weight.npy", DataType::TENSOR, {3000, 2, 1024}}},
        {"basis_to_embed_bias"sv, Metadata {data_path + "/" + "data_4_basis_to_embed_bias.npy", DataType::VECTOR, {500}}},
        {"basis_to_embed_weight"sv, Metadata {data_path + "/" + "data_5_basis_to_embed_weight.npy", DataType::MATRIX, {500, 1500}}},
        {"pre_spec_branch_weight_ih_l0"sv, Metadata {data_path + "/" + "data_6_pre_spec_branch_weight_ih_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_spec_branch_weight_hh_l0"sv, Metadata {data_path + "/" + "data_7_pre_spec_branch_weight_hh_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_spec_branch_bias_ih_l0"sv, Metadata {data_path + "/" + "data_8_pre_spec_branch_bias_ih_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_spec_branch_bias_hh_l0"sv, Metadata {data_path + "/" + "data_9_pre_spec_branch_bias_hh_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_spec_branch_weight_ih_l1"sv, Metadata {data_path + "/" + "data_10_pre_spec_branch_weight_ih_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_spec_branch_weight_hh_l1"sv, Metadata {data_path + "/" + "data_11_pre_spec_branch_weight_hh_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_spec_branch_bias_ih_l1"sv, Metadata {data_path + "/" + "data_12_pre_spec_branch_bias_ih_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_spec_branch_bias_hh_l1"sv, Metadata {data_path + "/" + "data_13_pre_spec_branch_bias_hh_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_waveform_branch_weight_ih_l0"sv, Metadata {data_path + "/" + "data_14_pre_waveform_branch_weight_ih_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_waveform_branch_weight_hh_l0"sv, Metadata {data_path + "/" + "data_15_pre_waveform_branch_weight_hh_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_waveform_branch_bias_ih_l0"sv, Metadata {data_path + "/" + "data_16_pre_waveform_branch_bias_ih_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_waveform_branch_bias_hh_l0"sv, Metadata {data_path + "/" + "data_17_pre_waveform_branch_bias_hh_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_waveform_branch_weight_ih_l1"sv, Metadata {data_path + "/" + "data_18_pre_waveform_branch_weight_ih_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_waveform_branch_weight_hh_l1"sv, Metadata {data_path + "/" + "data_19_pre_waveform_branch_weight_hh_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"pre_waveform_branch_bias_ih_l1"sv, Metadata {data_path + "/" + "data_20_pre_waveform_branch_bias_ih_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"pre_waveform_branch_bias_hh_l1"sv, Metadata {data_path + "/" + "data_21_pre_waveform_branch_bias_hh_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"fusion_branch_weight_ih_l0"sv, Metadata {data_path + "/" + "data_22_fusion_branch_weight_ih_l0.npy", DataType::MATRIX, {4 * 1000, 1000}}},
        {"fusion_branch_weight_hh_l0"sv, Metadata {data_path + "/" + "data_23_fusion_branch_weight_hh_l0.npy", DataType::MATRIX, {4 * 1000, 1000}}},
        {"fusion_branch_bias_ih_l0"sv, Metadata {data_path + "/" + "data_24_fusion_branch_bias_ih_l0.npy", DataType::VECTOR, {4 * 1000}}},
        {"fusion_branch_bias_hh_l0"sv, Metadata {data_path + "/" + "data_25_fusion_branch_bias_hh_l0.npy", DataType::VECTOR, {4 * 1000}}},
        {"fusion_branch_weight_ih_l1"sv, Metadata {data_path + "/" + "data_26_fusion_branch_weight_ih_l1.npy", DataType::MATRIX, {4 * 1000, 1000}}},
        {"fusion_branch_weight_hh_l1"sv, Metadata {data_path + "/" + "data_27_fusion_branch_weight_hh_l1.npy", DataType::MATRIX, {4 * 1000, 1000}}},
        {"fusion_branch_bias_ih_l1"sv, Metadata {data_path + "/" + "data_28_fusion_branch_bias_ih_l1.npy", DataType::VECTOR, {4 * 1000}}},
        {"fusion_branch_bias_hh_l1"sv, Metadata {data_path + "/" + "data_29_fusion_branch_bias_hh_l1.npy", DataType::VECTOR, {4 * 1000}}},
        {"post_spec_branch_weight_ih_l0"sv, Metadata {data_path + "/" + "data_30_post_spec_branch_weight_ih_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_spec_branch_weight_hh_l0"sv, Metadata {data_path + "/" + "data_31_post_spec_branch_weight_hh_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_spec_branch_bias_ih_l0"sv, Metadata {data_path + "/" + "data_32_post_spec_branch_bias_ih_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"post_spec_branch_bias_hh_l0"sv, Metadata {data_path + "/" + "data_33_post_spec_branch_bias_hh_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"post_spec_branch_weight_ih_l1"sv, Metadata {data_path + "/" + "data_34_post_spec_branch_weight_ih_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_spec_branch_weight_hh_l1"sv, Metadata {data_path + "/" + "data_35_post_spec_branch_weight_hh_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_spec_branch_bias_ih_l1"sv, Metadata {data_path + "/" + "data_36_post_spec_branch_bias_ih_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"post_spec_branch_bias_hh_l1"sv, Metadata {data_path + "/" + "data_37_post_spec_branch_bias_hh_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"post_waveform_branch_weight_ih_l0"sv, Metadata {data_path + "/" + "data_38_post_waveform_branch_weight_ih_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_waveform_branch_weight_hh_l0"sv, Metadata {data_path + "/" + "data_39_post_waveform_branch_weight_hh_l0.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_waveform_branch_bias_ih_l0"sv, Metadata {data_path + "/" + "data_40_post_waveform_branch_bias_ih_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"post_waveform_branch_bias_hh_l0"sv, Metadata {data_path + "/" + "data_41_post_waveform_branch_bias_hh_l0.npy", DataType::VECTOR, {4 * 500}}},
        {"post_waveform_branch_weight_ih_l1"sv, Metadata {data_path + "/" + "data_42_post_waveform_branch_weight_ih_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_waveform_branch_weight_hh_l1"sv, Metadata {data_path + "/" + "data_43_post_waveform_branch_weight_hh_l1.npy", DataType::MATRIX, {4 * 500, 500}}},
        {"post_waveform_branch_bias_ih_l1"sv, Metadata {data_path + "/" + "data_44_post_waveform_branch_bias_ih_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"post_waveform_branch_bias_hh_l1"sv, Metadata {data_path + "/" + "data_45_post_waveform_branch_bias_hh_l1.npy", DataType::VECTOR, {4 * 500}}},
        {"to_spec_masks_RMSNorm_weight"sv, Metadata {data_path + "/" + "data_46_to_spec_masks_RMSNorm_weight.npy", DataType::VECTOR, {500}}},
        {"to_spec_masks_linear_weight"sv, Metadata {data_path + "/" + "data_47_to_spec_masks_linear_weight.npy", DataType::MATRIX, {8208, 500}}},
        {"to_spec_masks_linear_bias"sv, Metadata {data_path + "/" + "data_48_to_spec_masks_linear_bias.npy", DataType::VECTOR, {8208}}},
        {"to_waveform_masks_RMSNorm_weight"sv, Metadata {data_path + "/" + "data_49_to_waveform_masks_RMSNorm_weight.npy", DataType::VECTOR, {500}}},
        {"to_waveform_masks_linear_weight"sv, Metadata {data_path + "/" + "data_50_to_waveform_masks_linear_weight.npy", DataType::MATRIX, {6000, 500}}},
        {"to_waveform_masks_linear_bias"sv, Metadata {data_path + "/" + "data_51_to_waveform_masks_linear_bias.npy", DataType::VECTOR, {6000}}},
        {"conv_decode_weight"sv, Metadata {data_path + "/" + "data_52_conv_decode_weight.npy", DataType::VECTOR, {/*todo*/}}},
        {"conv_decode_windowed_filters"sv, Metadata {data_path + "/" + "data_53_conv_decode_windowed_filters.npy", DataType::TENSOR, {1500, 2, 1024}}}};
}

auto getDataList() -> std::vector<std::string_view> {
    using namespace std::string_view_literals;
    return {
        "spec_encode_weight"sv,
        "spec_encode_bias"sv,
        "to_spec_masks_RMSNorm_weight"sv,
        "to_spec_masks_linear_weight"sv,
        "to_spec_masks_linear_bias"sv,
        "audio_conv_encode_weight"sv,
        "audio_conv_encode_bias"sv,
        "basis_to_embed_weight"sv,
        "basis_to_embed_bias"sv,
        "to_waveform_masks_RMSNorm_weight"sv,
        "to_waveform_masks_linear_weight"sv,
        "to_waveform_masks_linear_bias"sv,
        "conv_decode_windowed_filters"sv,
        "pre_spec_branch_weight_ih_l0"sv,
        "pre_spec_branch_weight_hh_l0"sv,
        "pre_spec_branch_bias_ih_l0"sv,
        "pre_spec_branch_bias_hh_l0"sv,
        "pre_spec_branch_weight_ih_l1"sv,
        "pre_spec_branch_weight_hh_l1"sv,
        "pre_spec_branch_bias_ih_l1"sv,
        "pre_spec_branch_bias_hh_l1"sv,
        "post_spec_branch_weight_ih_l0"sv,
        "post_spec_branch_weight_hh_l0"sv,
        "post_spec_branch_bias_ih_l0"sv,
        "post_spec_branch_bias_hh_l0"sv,
        "post_spec_branch_weight_ih_l1"sv,
        "post_spec_branch_weight_hh_l1"sv,
        "post_spec_branch_bias_ih_l1"sv,
        "post_spec_branch_bias_hh_l1"sv,
        "fusion_branch_weight_ih_l0"sv,
        "fusion_branch_weight_hh_l0"sv,
        "fusion_branch_bias_ih_l0"sv,
        "fusion_branch_bias_hh_l0"sv,
        "fusion_branch_weight_ih_l1"sv,
        "fusion_branch_weight_hh_l1"sv,
        "fusion_branch_bias_ih_l1"sv,
        "fusion_branch_bias_hh_l1"sv,
        "pre_waveform_branch_weight_ih_l0"sv,
        "pre_waveform_branch_weight_hh_l0"sv,
        "pre_waveform_branch_bias_ih_l0"sv,
        "pre_waveform_branch_bias_hh_l0"sv,
        "pre_waveform_branch_weight_ih_l1"sv,
        "pre_waveform_branch_weight_hh_l1"sv,
        "pre_waveform_branch_bias_ih_l1"sv,
        "pre_waveform_branch_bias_hh_l1"sv,
        "post_waveform_branch_weight_ih_l0"sv,
        "post_waveform_branch_weight_hh_l0"sv,
        "post_waveform_branch_bias_ih_l0"sv,
        "post_waveform_branch_bias_hh_l0"sv,
        "post_waveform_branch_weight_ih_l1"sv,
        "post_waveform_branch_weight_hh_l1"sv,
        "post_waveform_branch_bias_ih_l1"sv,
        "post_waveform_branch_bias_hh_l1"sv};
}

template <typename T>
void storeData(std::ostream& output, const T& data, const Dim& dim) {
    if (dim.size() == 1) {
        assert(dim[0] == data.size());
    }
    if (dim.size() == 2) {
        assert(dim[0] == data.rows());
        assert(dim[1] == data.cols());
    }
    auto data_f = data.template cast<float>().eval();
    size_t nbytes = data_f.size() * sizeof(float);
    output.write(reinterpret_cast<char const*>(&nbytes), sizeof(nbytes));
    output.write(reinterpret_cast<char const*>(data_f.data()), nbytes);
}

template <>
void storeData<Eigen::Tensor<double, 3>>(std::ostream& output, const Eigen::Tensor<double, 3>& data, const Dim& dim) {
    assert(dim.size() == 3);
    assert(dim[0] == data.dimension(0));
    assert(dim[1] == data.dimension(1));
    assert(dim[2] == data.dimension(2));
    const auto in_channels = data.dimension(0);
    const auto out_channels = data.dimension(1);
    const auto kernel_size = data.dimension(2);
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(in_channels, out_channels * kernel_size);
    for (int ic {}; ic < in_channels; ++ic) {
        for (int oc {}; oc < out_channels; ++oc) {
            for (int k {}; k < kernel_size; ++k) {
                weights(ic, oc * kernel_size + k) = data(ic, oc, k);
            }
        }
    }

    auto data_f = weights.cast<float>().eval();
    size_t nbytes = data_f.size() * sizeof(float);
    output.write(reinterpret_cast<char const*>(&nbytes), sizeof(nbytes));
    output.write(reinterpret_cast<char const*>(data_f.data()), nbytes);
}

} // namespace

int main(int argc, char const* argv[]) {
    std::string const data_path = argc > 1 ? argv[1] : "./data_npy";
    std::string const output_path = argc > 2 ? argv[2] : "./params.bw";

    std::ofstream wout(output_path, std::ios::binary | std::ios::out);
    if (!wout) {
        std::cerr << "Could not open file for writing: " << output_path << std::endl;
        return 1;
    }

    const auto npy_paths = getWeightFiles(data_path);
    const auto data_list = getDataList();
    for (const auto& list_item : data_list) {
        const auto& [path, type, dim] = npy_paths.at(list_item);
        const auto filepath = std::filesystem::path {path};
        if (!std::filesystem::exists(filepath)) {
            std::cerr << "File not found: " << filepath << std::endl;
            continue;
        }
        switch (type) {
        case DataType::VECTOR: {
            assert(dim.size() == 1);
            const auto& data = load_npy_vector(path);
            storeData(wout, data, dim);
            break;
        }
        case DataType::MATRIX: {
            assert(dim.size() == 2);
            const auto& data = load_npy_matrix(path);
            storeData(wout, data, dim);
            break;
        }
        case DataType::TENSOR: {
            assert(dim.size() == 3);
            const auto& data = load_npy_tensor<double, 3>(path);
            storeData(wout, data, dim);
            break;
        }
        case DataType::INVALID:
            std::cerr << "Invalid item: " << list_item << std::endl;
            break;
        }
    }
    return 0;
}
