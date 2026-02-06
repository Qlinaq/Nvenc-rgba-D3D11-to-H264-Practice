# NVENC D3D11 RGBA to H264 Practice

## 项目说明

实现 x64 DLL 动态库接口，输入特定数据，使用 NVIDIA NVENC 硬件编码器，把数据转为 H.264 视频流，保存到文件。

---

## 功能接口

### Task 1: RGBA 数组编码

- **输入**：`char* rgba[800 * 600 * 4]`, `int arr_size`（不大于100）, `const char* out_file_path`
- **处理**：RGBA → NV12 转换 + NVENC 编码
- **输出**：H.264 流，帧率 30fps，数组大小影响时长

```cpp
int EncodeRGBAToH264(
    const char* rgba_frames[],  // RGBA 帧数组
    int arr_size,               // 帧数量
    const char* out_file_path   // 输出文件路径
);

// 使用示例
std::vector<char*> frames(60);
// 填充 RGBA 数据 
EncodeRGBAToH264(frames.data(), 60, "output.h264");
```

### Task 2: D3D11 纹理流式编码

- **输入**：`ID3D11Texture2D* texture`（800×600，DXGI_FORMAT_NV12）, `const char* out_file_path`, `bool flag`
- **处理**：GPU 直接编码（零拷贝）
- **输出**：H.264 流

```cpp
int EncodeD3D11Texture(
    ID3D11Texture2D* texture,   // NV12 格式纹理
    const char* out_file_path,  // 输出文件路径
    bool flag                   // true=编码帧, false=结束编码
);

// 使用示例
EncodeD3D11Texture(texture, "output.h264", true);   // 编码帧
EncodeD3D11Texture(texture, "output.h264", true);   // 编码帧
EncodeD3D11Texture(nullptr, nullptr, false);        // 结束编码
```

---

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | CPU 进行 RGBA→NV12 转换 |
| `cuda` | GPU (CUDA) 进行 RGBA→NV12 转换 |

---
## CUDA branch
### PTX 兼容性问题
CUDA 分支使用 Parallel Thread Execution 内联代码实现 RGBA→NV12 转换。

不同 GPU 架构需要对应的 PTX 版本不同

如果遇到 `cuModuleLoadData failed: 218` 错误，说明 PTX 版本与 GPU 不兼容。

解决方案：修改 `RgbaToNv12.cu` 中的 PTX 头部：

### PTX
 RTX 40 系列 (当前 RTX 4060)
- .version 8.0
- .target sm_89

RTX 30 系列
- .version 7.0
- .target sm_86

RTX 20 系列 
- .version 6.4
- .target sm_75
---

## 依赖

- Visual Studio 2019
- NVIDIA Video Codec SDK 13.0
- CUDA Toolkit
- Windows SDK (D3D11)
- NVIDIA 4060 GPU (支持 NVENC)

---

## 文件结构

```
NvencExport/
├── NvencExport/          # DLL 项目
│   ├── NvencExport.cpp   # 主要实现
│   ├── NvencExport.h     # 导出头文件
│   ├── RgbaToNv12.cu     # CUDA 转换核心 (cuda分支)
│   └── RgbaToNv12.h      # CUDA 头文件 (cuda分支)
├── TestApp/              # 测试程序
│   └── TestApp.cpp
├── x64/Debug/            # 编译输出
└── README.md
```

---


## 验证输出

```bash
ffplay output_rgba.h264
ffplay output_d3d11.h264
```



