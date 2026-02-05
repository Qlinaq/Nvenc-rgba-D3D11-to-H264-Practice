// NvencExport/RgbaToNv12.h

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    // GPU ÄÚ´æ°æ±¾
    cudaError_t ConvertRgbaToNv12_GPU(
        const uint8_t* d_rgba,
        uint8_t* d_nv12,
        int width,
        int height,
        cudaStream_t stream = 0);

    // CPU¡úGPU °æ±¾
    cudaError_t ConvertRgbaToNv12_HostToDevice(
        const uint8_t* h_rgba,
        uint8_t* d_nv12,
        uint8_t* d_rgba_temp,
        int width,
        int height,
        cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif#pragma once
