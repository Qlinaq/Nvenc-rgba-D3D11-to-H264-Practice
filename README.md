# NVENC D3D11 RGBA to H264 Practice

## 项目说明
使用 NVIDIA Video Codec SDK 实现两个编码任务：

### Task 1: RGBA 数组编码
- 输入：RGBA 像素数组
- 处理：CPU 转换为 NV12 + CUDA 上传
- 输出：H.264 流

### Task 2: D3D11 NV12 纹理编码
- 输入：D3D11 NV12 纹理
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

## 使用方法
1. 打开 NvencExport.sln
2. 编译 Release x64
3. 运行 TestApp.exe
