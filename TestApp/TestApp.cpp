#include <iostream>
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>
#include <cmath>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "NvencExport.lib")

using Microsoft::WRL::ComPtr;

// 任务一接口（批量编码）
extern "C" __declspec(dllimport) int EncodeRGBAToH264(
    const char* rgba_frames[],
    int arr_size,
    const char* out_file_path
);

// 任务二接口（流式编码）
extern "C" __declspec(dllimport) int EncodeD3D11Texture(
    ID3D11Texture2D * texture,
    const char* out_file_path,
    bool flag
);

// ==================== D3D11 辅助函数 ====================

bool CreateD3D11Device(ComPtr<ID3D11Device>& device, ComPtr<ID3D11DeviceContext>& context)
{
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        featureLevels,
        1,
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context
    );

    if (FAILED(hr)) {
        std::cerr << "D3D11CreateDevice failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    return true;
}

ComPtr<ID3D11Texture2D> CreateNV12Texture(ID3D11Device* device, int width, int height)
{
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &texture);

    if (FAILED(hr)) {
        std::cerr << "CreateTexture2D (NV12) failed: 0x" << std::hex << hr << std::dec << std::endl;
        return nullptr;
    }

    return texture;
}

ComPtr<ID3D11Texture2D> CreateStagingTexture(ID3D11Device* device, int width, int height)
{
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    desc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &texture);

    if (FAILED(hr)) {
        std::cerr << "CreateTexture2D (Staging) failed: 0x" << std::hex << hr << std::dec << std::endl;
        return nullptr;
    }

    return texture;
}

void GenerateNV12Frame(
    std::vector<uint8_t>& yPlane,
    std::vector<uint8_t>& uvPlane,
    int width,
    int height,
    int frameIndex)
{
    int offset = frameIndex * 3;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = (x + y + offset) % 256;
            yPlane[y * width + x] = static_cast<uint8_t>(value);
        }
    }

    int uvWidth = width;
    int uvHeight = height / 2;
    for (int y = 0; y < uvHeight; y++) {
        for (int x = 0; x < uvWidth; x += 2) {
            int idx = y * uvWidth + x;
            uvPlane[idx] = static_cast<uint8_t>((128 + frameIndex * 2) % 256);
            uvPlane[idx + 1] = static_cast<uint8_t>((128 + x / 2) % 256);
        }
    }
}

bool WriteNV12ToStagingTexture(
    ID3D11DeviceContext* context,
    ID3D11Texture2D* stagingTexture,
    const std::vector<uint8_t>& yPlane,
    const std::vector<uint8_t>& uvPlane,
    int width,
    int height)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context->Map(stagingTexture, 0, D3D11_MAP_WRITE, 0, &mapped);

    if (FAILED(hr)) {
        std::cerr << "Map staging texture failed: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    uint8_t* dst = static_cast<uint8_t*>(mapped.pData);
    int dstPitch = mapped.RowPitch;

    for (int y = 0; y < height; y++) {
        memcpy(dst + y * dstPitch, yPlane.data() + y * width, width);
    }

    uint8_t* uvDst = dst + height * dstPitch;
    int uvHeight = height / 2;
    for (int y = 0; y < uvHeight; y++) {
        memcpy(uvDst + y * dstPitch, uvPlane.data() + y * width, width);
    }

    context->Unmap(stagingTexture, 0);
    return true;
}

// ==================== 任务一：RGBA 批量编码测试 ====================

int TestTask1_RGBA()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Task 1: RGBA Array Batch Encoding ===" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int width = 800;
    const int height = 600;
    const int frameCount = 60;  // 最多100帧
    const char* outputPath = "output_rgba.h264";

    std::cout << "Output file: " << outputPath << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frame count: " << frameCount << std::endl;
    std::cout << std::endl;

    std::vector<std::vector<uint8_t>> frameBuffers(frameCount);
    std::vector<const char*> framePtrs(frameCount);

    std::cout << "Generating " << frameCount << " RGBA frames..." << std::endl;

    for (int i = 0; i < frameCount; i++) {
        frameBuffers[i].resize(width * height * 4);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 4;
                frameBuffers[i][idx + 0] = static_cast<uint8_t>((x + i * 3) % 256);      // R
                frameBuffers[i][idx + 1] = static_cast<uint8_t>((y + i * 2) % 256);      // G
                frameBuffers[i][idx + 2] = static_cast<uint8_t>((x + y + i * 4) % 256);  // B
                frameBuffers[i][idx + 3] = 255;                                           // A
            }
        }

        framePtrs[i] = reinterpret_cast<const char*>(frameBuffers[i].data());

        if ((i + 1) % 20 == 0) {
            std::cout << "Generated frame " << (i + 1) << "/" << frameCount << std::endl;
        }
    }

    std::cout << "All frames generated. Starting encoding..." << std::endl;

    int result = EncodeRGBAToH264(framePtrs.data(), frameCount, outputPath);

    if (result == 0) {
        std::cout << std::endl;
        std::cout << "=== Task 1 completed successfully! ===" << std::endl;
        std::cout << "Output file: " << outputPath << std::endl;
        std::cout << "Verify: ffplay " << outputPath << std::endl;
    }
    else {
        std::cerr << "=== Task 1 failed with error: " << result << " ===" << std::endl;
        std::cerr << "Error codes: -1=invalid params, -2=cuInit, -3=cuDeviceGet, -4=cuContext, -5=file open, -100=exception" << std::endl;
    }

    return result;
}

// ==================== 任务二：D3D11 纹理编码测试 ====================

int TestTask2_D3D11()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Task 2: D3D11 NV12 Texture Encoding ===" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int width = 800;
    const int height = 600;
    const int frameCount = 60;
    const char* outputPath = "output_d3d11.h264";

    std::cout << "Creating D3D11 device..." << std::endl;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;

    if (!CreateD3D11Device(device, context)) {
        std::cerr << "Failed to create D3D11 device" << std::endl;
        return -1;
    }
    std::cout << "D3D11 device created successfully" << std::endl;

    std::cout << "Creating textures..." << std::endl;
    ComPtr<ID3D11Texture2D> nv12Texture = CreateNV12Texture(device.Get(), width, height);
    ComPtr<ID3D11Texture2D> stagingTexture = CreateStagingTexture(device.Get(), width, height);

    if (!nv12Texture || !stagingTexture) {
        std::cerr << "Failed to create textures" << std::endl;
        return -1;
    }
    std::cout << "Textures created successfully" << std::endl;

    std::vector<uint8_t> yPlane(width * height);
    std::vector<uint8_t> uvPlane(width * height / 2);

    std::cout << std::endl;
    std::cout << "Output file: " << outputPath << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frame count: " << frameCount << std::endl;
    std::cout << std::endl;

    int result = 0;

    for (int i = 0; i < frameCount; i++) {
        GenerateNV12Frame(yPlane, uvPlane, width, height, i);

        if (!WriteNV12ToStagingTexture(context.Get(), stagingTexture.Get(), yPlane, uvPlane, width, height)) {
            std::cerr << "Failed to write frame " << i << std::endl;
            result = -1;
            break;
        }

        context->CopyResource(nv12Texture.Get(), stagingTexture.Get());

        if (i == 0) {
            result = EncodeD3D11Texture(nv12Texture.Get(), outputPath, true);
        }
        else {
            result = EncodeD3D11Texture(nv12Texture.Get(), nullptr, true);
        }

        if (result != 0) {
            std::cerr << "EncodeD3D11Texture failed at frame " << i << ", error: " << result << std::endl;
            break;
        }

        if ((i + 1) % 10 == 0 || i == 0) {
            std::cout << "Encoded frame " << (i + 1) << "/" << frameCount << std::endl;
        }
    }

    if (result == 0) {
        std::cout << "Finishing encoding..." << std::endl;
        result = EncodeD3D11Texture(nullptr, nullptr, false);

        if (result != 0) {
            std::cerr << "EndEncode failed, error: " << result << std::endl;
        }
    }
    else {
        EncodeD3D11Texture(nullptr, nullptr, false);
    }

    if (result == 0) {
        std::cout << std::endl;
        std::cout << "=== Task 2 completed successfully! ===" << std::endl;
        std::cout << "Output file: " << outputPath << std::endl;
        std::cout << "Verify: ffplay " << outputPath << std::endl;
    }
    else {
        std::cerr << "=== Task 2 failed with error: " << result << " ===" << std::endl;
    }

    return result;
}


int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "  NVENC Encoding Test Program" << std::endl;
    std::cout << "========================================" << std::endl;

    int testMode = 0; // 0 = both, 1 = task1 only, 2 = task2 only

    if (argc > 1) {
        testMode = atoi(argv[1]);
    }

    std::cout << "\nUsage: TestApp.exe [mode]" << std::endl;
    std::cout << "  0 = Run both tasks (default)" << std::endl;
    std::cout << "  1 = Run Task 1 only (RGBA batch)" << std::endl;
    std::cout << "  2 = Run Task 2 only (D3D11 stream)" << std::endl;
    std::cout << "\nCurrent mode: " << testMode << std::endl;

    int result1 = 0, result2 = 0;

    if (testMode == 0 || testMode == 1) {
        result1 = TestTask1_RGBA();
    }

    if (testMode == 0 || testMode == 2) {
        result2 = TestTask2_D3D11();
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    if (testMode == 0 || testMode == 1) {
        std::cout << "Task 1 (RGBA batch):  " << (result1 == 0 ? "PASSED" : "FAILED") << std::endl;
    }
    if (testMode == 0 || testMode == 2) {
        std::cout << "Task 2 (D3D11 stream): " << (result2 == 0 ? "PASSED" : "FAILED") << std::endl;
    }

    std::cout << std::endl;
    if (result1 == 0 && (testMode == 0 || testMode == 1)) {
        std::cout << "Task 1 output: output_rgba.h264" << std::endl;
    }
    if (result2 == 0 && (testMode == 0 || testMode == 2)) {
        std::cout << "Task 2 output: output_d3d11.h264" << std::endl;
    }

    return (result1 != 0 || result2 != 0) ? -1 : 0;
}