#!/usr/bin/env bash
# where-teacher-acts: build the on-device latency benchmark (qj_bench) for arm64 Android with the NDK, against the
# headers and library of the ONNX Runtime Android package (the .aar from Maven Central, unzipped).
#
#   bash android/build.sh <ndk_dir> <unzipped_ort_aar_dir> <out_dir>
#   bash android/build.sh $ANDROID_HOME/ndk/28.2.13676358 android/vendor/ort-1.26.0 android/build
#
# The library carries its own C++ runtime and needs only the system libraries, so the binary links the C++
# runtime statically and the two files (qj_bench, libonnxruntime.so) are all the device needs. Segments
# are aligned to 16 KB, as the library's are, so both run on 4 KB and 16 KB page devices.
set -euo pipefail
NDK=${1:?NDK directory}
ORT=${2:?unzipped ONNX Runtime Android package}
OUT=${3:?output directory}
API=${API:-24}
HERE="$(cd "$(dirname "$0")" && pwd)"
CXX="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android${API}-clang++"
[ -x "$CXX" ] || { echo "no compiler at $CXX" >&2; exit 1; }
[ -f "$ORT/headers/onnxruntime_cxx_api.h" ] || { echo "no ONNX Runtime headers under $ORT/headers" >&2; exit 1; }
[ -f "$ORT/jni/arm64-v8a/libonnxruntime.so" ] || { echo "no arm64 library under $ORT/jni" >&2; exit 1; }
mkdir -p "$OUT"
"$CXX" -std=c++17 -O2 -Wall -Wextra -I "$ORT/headers" "$HERE/qj_bench.cpp" \
    -L "$ORT/jni/arm64-v8a" -lonnxruntime -static-libstdc++ -Wl,-z,max-page-size=16384 \
    -o "$OUT/qj_bench"
cp "$ORT/jni/arm64-v8a/libonnxruntime.so" "$OUT/libonnxruntime.so"
echo "built $OUT/qj_bench (API $API) with $(basename "$ORT")"
