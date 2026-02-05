#include <iostream>
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>
#include <cmath>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "NvencExport.lib")

using Microsoft::WRL::ComPtr;

// 直接声明导入函数
extern "C" __declspec(dllimport) int EncodeD3D11Texture(
    ID3D11Texture2D * texture,
    const char* out_file_path,
    bool flag);

// 创建 D3D11 设备和上下文
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
        std::cerr << "D3D11CreateDevice failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

// 创建 NV12 纹理
ComPtr<ID3D11Texture2D> CreateNV12Texture(
    ID3D11Device* device,
    int width,
    int height)
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
        std::cerr << "CreateTexture2D (NV12) failed: 0x" << std::hex << hr << std::endl;
        return nullptr;
    }

    return texture;
}

// 创建 Staging 纹理（用于 CPU 写入）
ComPtr<ID3D11Texture2D> CreateStagingTexture(
    ID3D11Device* device,
    int width,
    int height)
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
        std::cerr << "CreateTexture2D (Staging) failed: 0x" << std::hex << hr << std::endl;
        return nullptr;
    }

    return texture;
}

// 生成测试 NV12 数据（渐变动画）
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

// 将 NV12 数据写入 Staging 纹理
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
        std::cerr << "Map staging texture failed: 0x" << std::hex << hr << std::endl;
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

int main()
{
    std::cout << "=== Task 2 Test: D3D11 NV12 Texture Encoding ===" << std::endl;

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

    std::cout << "Starting encoding..." << std::endl;
    std::cout << "Output file: " << outputPath << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frame count: " << frameCount << std::endl;

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
        std::cout << "=== Encoding completed successfully! ===" << std::endl;
        std::cout << "Output file: " << outputPath << std::endl;
        std::cout << std::endl;
        std::cout << "To verify, run:" << std::endl;
        std::cout << "  ffplay " << outputPath << std::endl;
    }
    else {
        std::cerr << std::endl;
        std::cerr << "=== Encoding failed with error: " << result << " ===" << std::endl;
    }

    return result;
}