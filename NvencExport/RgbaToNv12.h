// RgbaToNv12.h
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

    void LaunchRGBAtoNV12Direct(
        unsigned long long d_rgba,
        unsigned long long dst_y_ptr,
        unsigned long long dst_uv_ptr,
        int width,
        int height,
        int dst_pitch,
        void* cuda_context
    );

    void CleanupRGBAtoNV12();

#ifdef __cplusplus
}
#endif