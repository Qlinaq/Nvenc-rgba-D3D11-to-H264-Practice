#pragma once

#ifdef NVENCEXPORT_EXPORTS
#define NVENC_API __declspec(dllexport)
#else
#define NVENC_API __declspec(dllimport)
#endif

#include <d3d11.h>

extern "C" {
    // 接口1: 将RGBA帧数组编码为H264视频文件
    // 参数：
    //   rgba_frames: RGBA数据指针数组，每个元素指向 800*600*4 字节的数据
    //   arr_size: 数组大小（帧数），1-100
    //   out_file_path: 输出文件路径
    // 返回值：0=成功，负数=错误码
    NVENC_API int EncodeRGBAToH264(
        const char* rgba_frames[],
        int arr_size,
        const char* out_file_path
    );

    // 接口2: D3D11纹理实时编码
    // 参数：
    //   texture: ID3D11Texture2D指针，800x600，NV12格式
    //   out_file_path: 输出文件路径
    //   flag: true=开始/继续编码，false=结束编码并保存
    // 返回值：0=成功，负数=错误码
    NVENC_API int EncodeD3D11Texture(
        ID3D11Texture2D* texture,
        const char* out_file_path,
        bool flag
    );
}