// NvencExport/RgbaToNv12.cu

#include <cuda_runtime.h>
#include <stdint.h>

// CUDA 内核：RGBA 转 NV12
__global__ void RgbaToNv12Kernel(
    const uint8_t* __restrict__ rgba,
    uint8_t* __restrict__ yPlane,
    uint8_t* __restrict__ uvPlane,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 读取 RGBA
    int rgbaIdx = (y * width + x) * 4;
    int r = rgba[rgbaIdx + 0];
    int g = rgba[rgbaIdx + 1];
    int b = rgba[rgbaIdx + 2];

    // 计算 Y（每个像素都有）
    int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    Y = min(max(Y, 0), 255);
    yPlane[y * width + x] = (uint8_t)Y;

    // 计算 UV（每 2x2 像素块共享一个 UV）
    if ((x % 2 == 0) && (y % 2 == 0)) {
        int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        int V = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        U = min(max(U, 0), 255);
        V = min(max(V, 0), 255);

        int uvIdx = (y / 2) * width + (x / 2) * 2;
        uvPlane[uvIdx + 0] = (uint8_t)U;
        uvPlane[uvIdx + 1] = (uint8_t)V;
    }
}

// 优化版本：使用共享内存，2x2 像素块平均计算 UV
__global__ void RgbaToNv12KernelOptimized(
    const uint8_t* __restrict__ rgba,
    uint8_t* __restrict__ yPlane,
    uint8_t* __restrict__ uvPlane,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 读取 RGBA
    int rgbaIdx = (y * width + x) * 4;
    int r = rgba[rgbaIdx + 0];
    int g = rgba[rgbaIdx + 1];
    int b = rgba[rgbaIdx + 2];

    // 计算并写入 Y
    int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    yPlane[y * width + x] = (uint8_t)min(max(Y, 0), 255);

    // 只有 2x2 块的左上角像素计算 UV（使用 4 个像素的平均值）
    if ((x % 2 == 0) && (y % 2 == 0) && (x + 1 < width) && (y + 1 < height)) {
        int sumR = 0, sumG = 0, sumB = 0;

        // 采样 2x2 块的 4 个像素
#pragma unroll
        for (int dy = 0; dy < 2; dy++) {
#pragma unroll
            for (int dx = 0; dx < 2; dx++) {
                int idx = ((y + dy) * width + (x + dx)) * 4;
                sumR += rgba[idx + 0];
                sumG += rgba[idx + 1];
                sumB += rgba[idx + 2];
            }
        }

        // 平均值
        int avgR = sumR / 4;
        int avgG = sumG / 4;
        int avgB = sumB / 4;

        int U = ((-38 * avgR - 74 * avgG + 112 * avgB + 128) >> 8) + 128;
        int V = ((112 * avgR - 94 * avgG - 18 * avgB + 128) >> 8) + 128;

        int uvIdx = (y / 2) * width + (x / 2) * 2;
        uvPlane[uvIdx + 0] = (uint8_t)min(max(U, 0), 255);
        uvPlane[uvIdx + 1] = (uint8_t)min(max(V, 0), 255);
    }
}

// C++ 接口
extern "C" {

    // GPU 内存版本：RGBA 和 NV12 都在 GPU 上
    cudaError_t ConvertRgbaToNv12_GPU(
        const uint8_t* d_rgba,    // GPU 上的 RGBA 数据
        uint8_t* d_nv12,          // GPU 上的 NV12 输出
        int width,
        int height,
        cudaStream_t stream = 0)
    {
        uint8_t* yPlane = d_nv12;
        uint8_t* uvPlane = d_nv12 + width * height;

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        RgbaToNv12KernelOptimized << <grid, block, 0, stream >> > (
            d_rgba, yPlane, uvPlane, width, height
            );

        return cudaGetLastError();
    }

    // CPU→GPU 版本：RGBA 在 CPU，转换后 NV12 在 GPU
    cudaError_t ConvertRgbaToNv12_HostToDevice(
        const uint8_t* h_rgba,    // CPU 上的 RGBA 数据
        uint8_t* d_nv12,          // GPU 上的 NV12 输出
        uint8_t* d_rgba_temp,     // GPU 临时缓冲区（需预先分配）
        int width,
        int height,
        cudaStream_t stream = 0)
    {
        size_t rgbaSize = width * height * 4;

        // 上传 RGBA 到 GPU
        cudaError_t err = cudaMemcpyAsync(d_rgba_temp, h_rgba, rgbaSize,
            cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return err;

        // GPU 上转换
        return ConvertRgbaToNv12_GPU(d_rgba_temp, d_nv12, width, height, stream);
    }

}  // extern "C"