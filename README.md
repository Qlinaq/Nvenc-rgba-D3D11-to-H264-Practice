# NVENC D3D11 RGBA to H264 Practice

## 项目说明
实现以下x64 dll动态库接口，输入特定数据，使用NVIDIA NVENC，把数据转为H264视频流，保存到文件：

### Task 1: RGBA 数组编码
- 输入：:char* rgba[800 * 600 * 4], int arr_size（不大于100）, const char* out_file_path
- 处理：CPU 转换为 NV12 + CUDA 上传
- 输出：H.264 流 帧率30fps，数组大小影响时长
- ```cpp
int EncodeRGBAToH264(
    const char* rgba_frames[],  // RGBA 帧数组
    int arr_size,               // 帧数量
    const char* out_file_path   // 输出文件路径
);

// Task 1
std::vector<char*> frames(60);
// 填充 RGBA 数据 
EncodeRGBAToH264(frames.data(), 60, "output.h264");

### Task 2: D3D11 ID3D11Texture2D
- 输入：ID3D11Texture2D* texture（800 * 600，DXGI_FORMAT_NV12）, const char* out_file_path, bool flag
- 处理：GPU 直接编码（零拷贝）
- 输出：H.264 流
-```cpp
int EncodeD3D11Texture(
    ID3D11Texture2D* texture,   // NV12 格式纹理
    const char* out_file_path,  // 输出文件路径
    bool flag                   // true=编码帧, false=结束编码
);

// Task 2
EncodeD3D11Texture(texture, "output.h264", true);   // 编码帧
EncodeD3D11Texture(texture, "output.h264", true);   // 编码帧
EncodeD3D11Texture(nullptr, nullptr, false);        // 结束

## 依赖
- Visual Studio 2019
- NVIDIA Video Codec SDK 13.0
- CUDA Toolkit
- Windows SDK (D3D11)
- NVIDIA GPU (支持 NVENC)

##Branch
| `main` | CPU 进行 RGBA→NV12 转换 |
| `cuda` | GPU (CUDA) 进行 RGBA→NV12 转换 |

## 文件结构
- NvencExport/ - DLL 项目
- TestApp/ - 测试程序
- x64/Debug/ - 编译输出





