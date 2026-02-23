@echo off

set _BUILD_DIR=build

set _BUILD_TYPE=RelWithDebInfo
::set _BUILD_TYPE=Release
::set _BUILD_TYPE=Debug

cmake -B %_BUILD_DIR% -DCMAKE_BUILD_TYPE=%_BUILD_TYPE%

cmake --build %_BUILD_DIR% --config %_BUILD_TYPE%
