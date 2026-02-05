#include "pch.h"
#include "NvencExport.h"

#include <cuda.h>
#include <d3d11.h>
#include <fstream>
#include <vector>
#include <mutex>
#include <string>
#include <algorithm>

// 用绝对路径测试
#include "D:/Video_Codec_SDK_13.0.37/Samples/NvCodec/NvEncoder/NvEncoderCuda.h"
#include "D:/Video_Codec_SDK_13.0.37/Samples/NvCodec/NvEncoder/NvEncoderD3D11.h"
// ============================================
// 辅助类：管理输出文件
// ============================================
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
// 辅助函数：RGBA 转 NV12
// ============================================
inline uint8_t ClampByte(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<uint8_t>(value);
}

void ConvertRGBAtoNV12_CPU(const uint8_t* rgba, uint8_t* nv12, int width, int height) {
    uint8_t* yPlane = nv12;
    uint8_t* uvPlane = nv12 + width * height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgbaIdx = (y * width + x) * 4;
            uint8_t r = rgba[rgbaIdx + 0];
            uint8_t g = rgba[rgbaIdx + 1];
            uint8_t b = rgba[rgbaIdx + 2];

            int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            yPlane[y * width + x] = ClampByte(Y);

            if (y % 2 == 0 && x % 2 == 0) {
                int U = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int V = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

                int uvIdx = (y / 2) * width + x;
                uvPlane[uvIdx + 0] = ClampByte(U);
                uvPlane[uvIdx + 1] = ClampByte(V);
            }
        }
    }
}

// ============================================
// 任务一：RGBA 数组编码
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

    CUcontext cuContext = nullptr;
    CUdevice cuDevice = 0;
    NvEncoderCuda* encoder = nullptr;
    bool primaryCtxRetained = false;

    try {
        CUresult cuResult = cuInit(0);
        if (cuResult != CUDA_SUCCESS) {
            return -2;
        }

        cuResult = cuDeviceGet(&cuDevice, 0);
        if (cuResult != CUDA_SUCCESS) {
            return -3;
        }

        cuResult = cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
        if (cuResult != CUDA_SUCCESS) {
            return -4;
        }
        primaryCtxRetained = true;

        cuResult = cuCtxSetCurrent(cuContext);
        if (cuResult != CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(cuDevice);
            return -4;
        }

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
            cuDevicePrimaryCtxRelease(cuDevice);
            return -5;
        }

        size_t nv12Size = width * height * 3 / 2;
        std::vector<uint8_t> nv12Buffer(nv12Size);

        std::vector<NvEncOutputFrame> vPacket;

        for (int i = 0; i < arr_size; i++) {
            ConvertRGBAtoNV12_CPU(
                reinterpret_cast<const uint8_t*>(rgba_frames[i]),
                nv12Buffer.data(),
                width, height
            );

            const NvEncInputFrame* inputFrame = encoder->GetNextInputFrame();

            NvEncoderCuda::CopyToDeviceFrame(
                cuContext,
                nv12Buffer.data(),
                0,
                (CUdeviceptr)inputFrame->inputPtr,
                inputFrame->pitch,
                width,
                height,
                CU_MEMORYTYPE_HOST,
                inputFrame->bufferFormat,
                inputFrame->chromaOffsets,
                inputFrame->numChromaPlanes
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

        encoder->DestroyEncoder();
        delete encoder;
        cuDevicePrimaryCtxRelease(cuDevice);

        return 0;
    }
    catch (const std::exception& e) {
        if (encoder) {
            encoder->DestroyEncoder();
            delete encoder;
        }
        if (primaryCtxRetained) {
            cuDevicePrimaryCtxRelease(cuDevice);
        }
        return -100;
    }
}

// ============================================
// 任务二：D3D11 纹理流式编码
// ============================================

// 全局状态（用于跨调用保持编码器状态）
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
        // ========================================
        // flag=true 且编码器未初始化：开始新的编码会话
        // ========================================
        if (flag && g_d3d11Encoder == nullptr) {
            if (!texture || !out_file_path) {
                return -1;
            }

            // 从纹理获取设备
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

            // 获取纹理尺寸
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            g_width = desc.Width;
            g_height = desc.Height;

            // 创建编码器
            g_d3d11Encoder = new NvEncoderD3D11(
                g_d3d11Device,
                g_width,
                g_height,
                NV_ENC_BUFFER_FORMAT_NV12
            );

            // 配置编码参数
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

            // 打开输出文件
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

        // ========================================
        // flag=true 且有纹理：编码一帧
        // ========================================
        if (flag && g_d3d11Encoder && texture) {
            // 获取编码器的输入帧
            const NvEncInputFrame* inputFrame = g_d3d11Encoder->GetNextInputFrame();

            // 获取输入纹理（编码器内部缓冲区）
            ID3D11Texture2D* encoderInputTexture =
                reinterpret_cast<ID3D11Texture2D*>(inputFrame->inputPtr);

            // 将输入纹理复制到编码器缓冲区
            g_d3d11Context->CopyResource(encoderInputTexture, texture);

            // 编码
            std::vector<NvEncOutputFrame> vPacket;
            g_d3d11Encoder->EncodeFrame(vPacket);

            // 写入输出
            for (const auto& packet : vPacket) {
                g_d3d11Writer->Write(packet.frame);
            }
        }

        // ========================================
        // flag=false：结束编码，清理资源
        // ========================================
        if (!flag && g_d3d11Encoder) {
            // 刷新编码器
            std::vector<NvEncOutputFrame> vPacket;
            g_d3d11Encoder->EndEncode(vPacket);

            // 写入剩余数据
            for (const auto& packet : vPacket) {
                g_d3d11Writer->Write(packet.frame);
            }

            // 清理资源
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
        // 异常处理：清理所有资源
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