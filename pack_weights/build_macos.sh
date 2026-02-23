#!/usr/bin/env bash
set -e

mkdir -p build
pushd build
    cmake -G Xcode ..
    cmake --build . --config RelWithDebInfo
popd
