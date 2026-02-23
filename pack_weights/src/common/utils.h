#pragma once

#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <complex>
#include <tuple>
#include <variant>
#include <cmath>
#include <stdexcept>

#include "npy.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

// Utility: load npy file into Eigen vector
inline Eigen::VectorXd load_npy_vector(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    auto npy = npy::read_npy<float>(in);
    std::vector<int> shape;
    for (auto d : npy.shape)
        if (d != 1) shape.push_back(d);
    if (shape.size() != 1)
        throw std::runtime_error("Only 1D npy arrays are supported for load_npy_vector");
    Eigen::VectorXf vec = Eigen::Map<Eigen::VectorXf>(npy.data.data(), shape[0]);
    return vec.cast<double>();
}

// Utility: load npy file into Eigen matrix
inline Eigen::MatrixXd load_npy_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    auto npy = npy::read_npy<float>(in);
    std::vector<int> shape;
    for (auto d : npy.shape)
        if (d != 1) shape.push_back(d);
    if (shape.size() == 1) {
        Eigen::MatrixXf mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>(npy.data.data(), shape[0], 1);
        return mat.cast<double>();
    }
    else if (shape.size() == 2 && !npy.fortran_order) {
        Eigen::MatrixXf mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(npy.data.data(), shape[1], shape[0]);
        return mat.transpose().cast<double>(); // transpose to column-major
    }
    else {
        Eigen::MatrixXf mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(npy.data.data(), shape[0], shape[1]);
        return mat.cast<double>();
    }
    throw std::runtime_error("Only 1D or 2D npy arrays are supported");
}

// Utility: load npy file into Eigen tensor of arbitrary dimension
template <typename Scalar, int Rank>
Eigen::Tensor<Scalar, Rank> load_npy_tensor(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    using NpyScalar = std::conditional_t<std::is_same_v<Scalar, std::complex<float>> || std::is_same_v<Scalar, std::complex<double>>, std::complex<float>, float>;
    auto npy = npy::read_npy<NpyScalar>(in);

    // Remove singleton dimensions
    std::vector<size_t> shape;
    for (auto d : npy.shape)
        if (d != 1) shape.push_back(d);
    if (shape.size() != Rank) throw std::runtime_error("Rank mismatch");

    // Eigen expects column-major, so reverse shape for mapping
    Eigen::array<Eigen::Index, Rank> dims;
    if (!npy.fortran_order)
        std::reverse(shape.begin(), shape.end());
    for (int i = 0; i < Rank; ++i)
        dims[i] = shape[i];

    // Map as row-major, then permute to column-major
    Eigen::Tensor<NpyScalar, Rank> tensor_rowmajor(dims);
    std::copy_n(npy.data.data(), npy.data.size(), tensor_rowmajor.data());

    Eigen::Tensor<Scalar, Rank> tensor;

    if (!npy.fortran_order) {
        // Permute dimensions to reverse order
        Eigen::array<int, Rank> shuffle;
        for (int i = 0; i < Rank; ++i)
            shuffle[i] = Rank - 1 - i;

        tensor = tensor_rowmajor.shuffle(shuffle).template cast<Scalar>();
    }
    else {
        tensor = tensor_rowmajor.template cast<Scalar>();
    }
    return tensor;
}
