#include "pch.h"
#include "NvencExport.h"
#include "RgbaToNv12.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <fstream>
#include <vector>
#include <mutex>
#include <string>

#include "D:/Video_Codec_SDK_13.0.37/Samples/NvCodec/NvEncoder/NvEncoderCuda.h"
#include "D:/Video_Codec_SDK_13.0.37/Samples/NvCodec/NvEncoder/NvEncoderD3D11.h"

class BitstreamWriter {
public:
    BitstreamWriter(const char* path) {
        file_.open(path, std::ios::binary);
    }
    ~BitstreamWriter() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    bool IsOpen() const { return file_.is_open(); }

    void Write(const std::vector<uint8_t>& data) {
        if (file_.is_open() && !data.empty()) {
            file_.write(reinterpret_cast<const char*>(data.data()), data.size());
        }
    }

private:
    std::ofstream file_;
};

// ============================================
// RGBA 数组编码（GPU Kernel 版本）
// ============================================
extern "C" NVENC_API int EncodeRGBAToH264(
    const char* rgba_frames[],
    int arr_size,
    const char* out_file_path)
{
    if (!rgba_frames || arr_size <= 0 || arr_size > 100 || !out_file_path) {
        return -1;
    }

    const int width = 800;
    const int height = 600;
    const int fps = 30;

    NvEncoderCuda* encoder = nullptr;
    CUdeviceptr d_rgba = 0;
    CUcontext cuContext = nullptr;
    CUdevice cuDevice = 0;
    bool usesPrimaryCtx = false;

    try {
        // ========== Driver API 初始化 ==========
        CUresult cuResult = cuInit(0);
        if (cuResult != CUDA_SUCCESS) {
            printf("cuInit failed: %d\n", cuResult);
            return -2;
        }

        int deviceCount = 0;
        cuResult = cuDeviceGetCount(&deviceCount);
        printf("CUDA Device count: %d\n", deviceCount);

        if (deviceCount == 0) {
            printf("No CUDA devices found\n");
            return -3;
        }

        cuResult = cuDeviceGet(&cuDevice, 0);
        if (cuResult != CUDA_SUCCESS) {
            printf("cuDeviceGet failed: %d\n", cuResult);
            return -3;
        }

        char deviceName[256];
        cuDeviceGetName(deviceName, 256, cuDevice);
        printf("Using CUDA Device: %s\n", deviceName);

        // 使用 Primary Context
        cuResult = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
        if (cuResult != CUDA_SUCCESS) {
            printf("cuDevicePrimaryCtxRetain failed: %d\n", cuResult);
            return -4;
        }
        usesPrimaryCtx = true;

        cuResult = cuCtxSetCurrent(cuContext);
        if (cuResult != CUDA_SUCCESS) {
            printf("cuCtxSetCurrent failed: %d\n", cuResult);
            cuDevicePrimaryCtxRelease(cuDevice);
            return -4;
        }
        printf("CUDA Context: %p\n", cuContext);

        // 分配 GPU 内存
        size_t rgbaSize = width * height * 4;
        cuResult = cuMemAlloc(&d_rgba, rgbaSize);
        if (cuResult != CUDA_SUCCESS) {
            printf("cuMemAlloc failed: %d\n", cuResult);
            cuDevicePrimaryCtxRelease(cuDevice);
            return -6;
        }

        // 创建编码器
        encoder = new NvEncoderCuda(cuContext, width, height, NV_ENC_BUFFER_FORMAT_NV12);

        NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
        initializeParams.encodeConfig = &encodeConfig;

        encoder->CreateDefaultEncoderParams(
            &initializeParams,
            NV_ENC_CODEC_H264_GUID,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO_HIGH_QUALITY
        );

        initializeParams.frameRateNum = fps;
        initializeParams.frameRateDen = 1;

        encodeConfig.rcParams.averageBitRate = 2000000;
        encodeConfig.rcParams.maxBitRate = 4000000;
        encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        encodeConfig.encodeCodecConfig.h264Config.idrPeriod = encodeConfig.gopLength;

        encoder->CreateEncoder(&initializeParams);

        BitstreamWriter writer(out_file_path);
        if (!writer.IsOpen()) {
            encoder->DestroyEncoder();
            delete encoder;
            cuMemFree(d_rgba);
            cuDevicePrimaryCtxRelease(cuDevice);
            return -5;
        }

        std::vector<NvEncOutputFrame> vPacket;

        for (int i = 0; i < arr_size; i++) {
            // 上传 RGBA 到 GPU
            cuMemcpyHtoD(d_rgba, rgba_frames[i], rgbaSize);

            const NvEncInputFrame* inputFrame = encoder->GetNextInputFrame();

            uint8_t* dst_y = reinterpret_cast<uint8_t*>(inputFrame->inputPtr);
            uint8_t* dst_uv = dst_y + inputFrame->chromaOffsets[0];

            // 调用 GPU Kernel（PTX 版本）
            LaunchRGBAtoNV12Direct(
                d_rgba,
                (unsigned long long)dst_y,
                (unsigned long long)dst_uv,
                width,
                height,
                (int)inputFrame->pitch,
                cuContext
            );

            encoder->EncodeFrame(vPacket);

            for (const auto& packet : vPacket) {
                writer.Write(packet.frame);
            }
            vPacket.clear();
        }

        encoder->EndEncode(vPacket);
        for (const auto& packet : vPacket) {
            writer.Write(packet.frame);
        }

        CleanupRGBAtoNV12();
        encoder->DestroyEncoder();
        delete encoder;
        cuMemFree(d_rgba);
        cuDevicePrimaryCtxRelease(cuDevice);

        return 0;
    }
    catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
        CleanupRGBAtoNV12();
        if (encoder) {
            encoder->DestroyEncoder();
            delete encoder;
        }
        if (d_rgba) cuMemFree(d_rgba);
        if (usesPrimaryCtx) cuDevicePrimaryCtxRelease(cuDevice);
        return -100;
    }
}

// ============================================
// D3D11 部分（保持不变）
// ============================================

static NvEncoderD3D11* g_d3d11Encoder = nullptr;
static BitstreamWriter* g_d3d11Writer = nullptr;
static ID3D11Device* g_d3d11Device = nullptr;
static ID3D11DeviceContext* g_d3d11Context = nullptr;
static std::mutex g_d3d11Mutex;
static int g_width = 0;
static int g_height = 0;

extern "C" NVENC_API int EncodeD3D11Texture(
    ID3D11Texture2D * texture,
    const char* out_file_path,
    bool flag)
{
    std::lock_guard<std::mutex> lock(g_d3d11Mutex);

    try {
        if (flag && g_d3d11Encoder == nullptr) {
            if (!texture || !out_file_path) {
                return -1;
            }

            texture->GetDevice(&g_d3d11Device);
            if (!g_d3d11Device) {
                return -2;
            }

            g_d3d11Device->GetImmediateContext(&g_d3d11Context);
            if (!g_d3d11Context) {
                g_d3d11Device->Release();
                g_d3d11Device = nullptr;
                return -3;
            }

            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            g_width = desc.Width;
            g_height = desc.Height;

            g_d3d11Encoder = new NvEncoderD3D11(
                g_d3d11Device,
                g_width,
                g_height,
                NV_ENC_BUFFER_FORMAT_NV12
            );

            NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
            NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
            initializeParams.encodeConfig = &encodeConfig;

            g_d3d11Encoder->CreateDefaultEncoderParams(
                &initializeParams,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO_HIGH_QUALITY
            );

            initializeParams.frameRateNum = 30;
            initializeParams.frameRateDen = 1;

            encodeConfig.rcParams.averageBitRate = 2000000;
            encodeConfig.rcParams.maxBitRate = 4000000;
            encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
            encodeConfig.encodeCodecConfig.h264Config.idrPeriod = encodeConfig.gopLength;

            g_d3d11Encoder->CreateEncoder(&initializeParams);

            g_d3d11Writer = new BitstreamWriter(out_file_path);
            if (!g_d3d11Writer->IsOpen()) {
                g_d3d11Encoder->DestroyEncoder();
                delete g_d3d11Encoder;
                g_d3d11Encoder = nullptr;
                delete g_d3d11Writer;
                g_d3d11Writer = nullptr;
                g_d3d11Context->Release();
                g_d3d11Context = nullptr;
                g_d3d11Device->Release();
                g_d3d11Device = nullptr;
                return -4;
            }
        }

        if (flag && g_d3d11Encoder && texture) {
            const NvEncInputFrame* inputFrame = g_d3d11Encoder->GetNextInputFrame();

            ID3D11Texture2D* encoderInputTexture =
                reinterpret_cast<ID3D11Texture2D*>(inputFrame->inputPtr);

            g_d3d11Context->CopyResource(encoderInputTexture, texture);

            std::vector<NvEncOutputFrame> vPacket;
            g_d3d11Encoder->EncodeFrame(vPacket);

            for (const auto& packet : vPacket) {
                g_d3d11Writer->Write(packet.frame);
            }
        }

        if (!flag && g_d3d11Encoder) {
            std::vector<NvEncOutputFrame> vPacket;
            g_d3d11Encoder->EndEncode(vPacket);

            for (const auto& packet : vPacket) {
                g_d3d11Writer->Write(packet.frame);
            }

            g_d3d11Encoder->DestroyEncoder();
            delete g_d3d11Encoder;
            g_d3d11Encoder = nullptr;

            delete g_d3d11Writer;
            g_d3d11Writer = nullptr;

            if (g_d3d11Context) {
                g_d3d11Context->Release();
                g_d3d11Context = nullptr;
            }

            if (g_d3d11Device) {
                g_d3d11Device->Release();
                g_d3d11Device = nullptr;
            }

            g_width = 0;
            g_height = 0;
        }

        return 0;
    }
    catch (const std::exception& e) {
        if (g_d3d11Encoder) {
            g_d3d11Encoder->DestroyEncoder();
            delete g_d3d11Encoder;
            g_d3d11Encoder = nullptr;
        }

        if (g_d3d11Writer) {
            delete g_d3d11Writer;
            g_d3d11Writer = nullptr;
        }

        if (g_d3d11Context) {
            g_d3d11Context->Release();
            g_d3d11Context = nullptr;
        }

        if (g_d3d11Device) {
            g_d3d11Device->Release();
            g_d3d11Device = nullptr;
        }

        g_width = 0;
        g_height = 0;

        return -100;
    }
}