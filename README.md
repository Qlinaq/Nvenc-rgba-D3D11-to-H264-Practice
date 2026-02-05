# NVENC D3D11 RGBA to H264 Practice

## 项目说明
实现以下x64 dll动态库接口，输入特定数据，使用NVIDIA NVENC，把数据转为H264视频流，保存到文件：

### Task 1: RGBA 数组编码
- 输入：:char* rgba[800 * 600 * 4], int arr_size（不大于100）, const char* out_file_path
- 处理：CPU 转换为 NV12 + CUDA 上传
- 输出：H.264 流 帧率30fps，数组大小影响时长

### Task 2: D3D11 ID3D11Texture2D
- 输入：ID3D11Texture2D* texture（800 * 600，DXGI_FORMAT_NV12）, const char* out_file_path, bool flag
- 处理：GPU 直接编码（零拷贝）
- 输出：H.264 流

## 编译环境
- Visual Studio 2019
- NVIDIA Video Codec SDK 13.0
- CUDA Toolkit
- Windows 11

## 文件结构
- NvencExport/ - DLL 项目
- TestApp/ - 测试程序
- x64/Debug/ - 编译输出


