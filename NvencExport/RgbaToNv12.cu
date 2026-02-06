// RgbaToNv12.cu
#include <cuda.h>
#include <cstdint>
#include <cstdio>

// PTX 代码 - 兼容 sm_89 (RTX 40 系列)
static const char* g_ptxSource = R"(
.version 8.0
.target sm_89
.address_size 64

.visible .entry RGBAtoNV12Kernel(
    .param .u64 param_rgba,
    .param .u64 param_dst_y,
    .param .u64 param_dst_uv,
    .param .u32 param_width,
    .param .u32 param_height,
    .param .u32 param_pitch
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<40>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [param_rgba];
    ld.param.u64    %rd2, [param_dst_y];
    ld.param.u64    %rd3, [param_dst_uv];
    ld.param.u32    %r1, [param_width];
    ld.param.u32    %r2, [param_height];
    ld.param.u32    %r3, [param_pitch];

    mov.u32         %r4, %ctaid.x;
    mov.u32         %r5, %ntid.x;
    mov.u32         %r6, %tid.x;
    mad.lo.s32      %r7, %r4, %r5, %r6;

    mov.u32         %r8, %ctaid.y;
    mov.u32         %r9, %ntid.y;
    mov.u32         %r10, %tid.y;
    mad.lo.s32      %r11, %r8, %r9, %r10;

    setp.ge.s32     %p1, %r7, %r1;
    setp.ge.s32     %p2, %r11, %r2;
    or.pred         %p3, %p1, %p2;
    @%p3 bra        DONE;

    mad.lo.s32      %r12, %r11, %r1, %r7;
    shl.b32         %r13, %r12, 2;
    cvt.u64.u32     %rd4, %r13;
    add.u64         %rd5, %rd1, %rd4;

    ld.global.u8    %r14, [%rd5];
    ld.global.u8    %r15, [%rd5+1];
    ld.global.u8    %r16, [%rd5+2];

    mul.lo.s32      %r17, %r14, 66;
    mul.lo.s32      %r18, %r15, 129;
    mul.lo.s32      %r19, %r16, 25;
    add.s32         %r20, %r17, %r18;
    add.s32         %r20, %r20, %r19;
    add.s32         %r20, %r20, 128;
    shr.s32         %r20, %r20, 8;
    add.s32         %r20, %r20, 16;
    max.s32         %r20, %r20, 0;
    min.s32         %r20, %r20, 255;

    mad.lo.s32      %r21, %r11, %r3, %r7;
    cvt.u64.u32     %rd6, %r21;
    add.u64         %rd7, %rd2, %rd6;
    st.global.u8    [%rd7], %r20;

    and.b32         %r22, %r7, 1;
    and.b32         %r23, %r11, 1;
    or.b32          %r24, %r22, %r23;
    setp.ne.s32     %p1, %r24, 0;
    @%p1 bra        DONE;

    mul.lo.s32      %r25, %r14, -38;
    mul.lo.s32      %r26, %r15, -74;
    mul.lo.s32      %r27, %r16, 112;
    add.s32         %r28, %r25, %r26;
    add.s32         %r28, %r28, %r27;
    add.s32         %r28, %r28, 128;
    shr.s32         %r28, %r28, 8;
    add.s32         %r28, %r28, 128;
    max.s32         %r28, %r28, 0;
    min.s32         %r28, %r28, 255;

    mul.lo.s32      %r29, %r14, 112;
    mul.lo.s32      %r30, %r15, -94;
    mul.lo.s32      %r31, %r16, -18;
    add.s32         %r32, %r29, %r30;
    add.s32         %r32, %r32, %r31;
    add.s32         %r32, %r32, 128;
    shr.s32         %r32, %r32, 8;
    add.s32         %r32, %r32, 128;
    max.s32         %r32, %r32, 0;
    min.s32         %r32, %r32, 255;

    shr.s32         %r33, %r11, 1;
    mad.lo.s32      %r34, %r33, %r3, %r7;
    cvt.u64.u32     %rd8, %r34;
    add.u64         %rd9, %rd3, %rd8;

    st.global.u8    [%rd9], %r28;
    st.global.u8    [%rd9+1], %r32;

DONE:
    ret;
}
)";

static CUmodule g_module = nullptr;
static CUfunction g_kernel = nullptr;
static bool g_initialized = false;

static bool InitKernel(CUcontext ctx)
{
    if (g_initialized && g_kernel != nullptr) {
        return true;
    }

    cuCtxSetCurrent(ctx);

    // 使用 JIT 编译选项
    CUjit_option options[] = {
        CU_JIT_TARGET_FROM_CUCONTEXT,
        CU_JIT_OPTIMIZATION_LEVEL
    };
    void* optionValues[] = {
        nullptr,
        (void*)4
    };

    CUresult result = cuModuleLoadDataEx(
        &g_module,
        g_ptxSource,
        2,
        options,
        optionValues
    );

    if (result != CUDA_SUCCESS) {
        printf("  [ERROR] cuModuleLoadDataEx failed: %d\n", result);
        printf("  [INFO] Trying cuModuleLoadData...\n");

        // 尝试不带选项
        result = cuModuleLoadData(&g_module, g_ptxSource);
        if (result != CUDA_SUCCESS) {
            printf("  [ERROR] cuModuleLoadData also failed: %d\n", result);
            return false;
        }
    }

    result = cuModuleGetFunction(&g_kernel, g_module, "RGBAtoNV12Kernel");
    if (result != CUDA_SUCCESS) {
        printf("  [ERROR] cuModuleGetFunction failed: %d\n", result);
        cuModuleUnload(g_module);
        g_module = nullptr;
        return false;
    }

    g_initialized = true;
    printf("  [INFO] GPU Kernel initialized successfully (PTX)\n");
    return true;
}

extern "C" void LaunchRGBAtoNV12Direct(
    unsigned long long d_rgba,
    unsigned long long dst_y_ptr,
    unsigned long long dst_uv_ptr,
    int width,
    int height,
    int dst_pitch,
    void* cuda_context)
{
    CUcontext ctx = (CUcontext)cuda_context;
    cuCtxSetCurrent(ctx);

    if (!InitKernel(ctx)) {
        printf("  [ERROR] Failed to initialize kernel\n");
        return;
    }

    void* args[] = {
        &d_rgba,
        &dst_y_ptr,
        &dst_uv_ptr,
        &width,
        &height,
        &dst_pitch
    };

    unsigned int blockX = 16;
    unsigned int blockY = 16;
    unsigned int gridX = (width + blockX - 1) / blockX;
    unsigned int gridY = (height + blockY - 1) / blockY;

    CUresult result = cuLaunchKernel(
        g_kernel,
        gridX, gridY, 1,
        blockX, blockY, 1,
        0,
        nullptr,
        args,
        nullptr
    );

    if (result != CUDA_SUCCESS) {
        printf("  [ERROR] cuLaunchKernel failed: %d\n", result);
        return;
    }

    cuCtxSynchronize();
}

extern "C" void CleanupRGBAtoNV12()
{
    if (g_module) {
        cuModuleUnload(g_module);
        g_module = nullptr;
        g_kernel = nullptr;
        g_initialized = false;
    }
}