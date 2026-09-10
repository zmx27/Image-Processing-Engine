#pragma once

// RAII wrappers for the driver resources: CUcontext, CUmodule, CUdeviceptr, pinned
// host memory and CUstream. No raw cuCtxDestroy / cuModuleUnload / cuMemFree /
// cuStreamDestroy appears in logic code (CLAUDE.md conventions).

#include <cuda.h>

#include <cstddef>
#include <string>
#include <utility>

#include "backend/cuda/cuda_check.h"

namespace imgjit::cuda {

// CLAUDE.md invariant 1: exactly one thread constructs this, once, at startup, and
// never releases or migrates it. Non-movable so it cannot drift off that thread.
class CudaContext {
 public:
  explicit CudaContext(int device_ordinal = 0) {
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device_, device_ordinal));
    CU_CHECK(cuCtxCreate(&context_, 0, device_));
  }

  ~CudaContext() { CU_CHECK_NOTHROW(cuCtxDestroy(context_)); }

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  CUcontext get() const { return context_; }
  CUdevice device() const { return device_; }

  // The `--gpu-architecture` NVRTC should target for this device, e.g. "compute_75".
  std::string compute_arch() const {
    int major = 0;
    int minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
    return "compute_" + std::to_string(major) + std::to_string(minor);
  }

 private:
  CUdevice device_{};
  CUcontext context_{};
};

// Owns a module loaded from a PTX image, whether that PTX came from nvcc ahead of
// time (Phase 1a) or from NVRTC at runtime (Phase 1b) — cuModuleLoadData cannot
// tell the difference, which is exactly why the two checkpoints share this type.
class CudaModule {
 public:
  // `ptx_image` must be null-terminated PTX text.
  explicit CudaModule(const void* ptx_image) { CU_CHECK(cuModuleLoadData(&module_, ptx_image)); }

  ~CudaModule() { reset(); }

  CudaModule(CudaModule&& other) noexcept : module_(std::exchange(other.module_, nullptr)) {}

  CudaModule& operator=(CudaModule&& other) noexcept {
    if (this != &other) {
      reset();
      module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
  }

  CudaModule(const CudaModule&) = delete;
  CudaModule& operator=(const CudaModule&) = delete;

  CUfunction get_function(const char* name) const {
    CUfunction function{};
    CU_CHECK(cuModuleGetFunction(&function, module_, name));
    return function;
  }

 private:
  void reset() noexcept {
    if (module_ != nullptr) {
      CU_CHECK_NOTHROW(cuModuleUnload(module_));
      module_ = nullptr;
    }
  }

  CUmodule module_{};
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : size_(bytes) { CU_CHECK(cuMemAlloc(&pointer_, bytes)); }

  ~DeviceBuffer() { reset(); }

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, 0)), size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = std::exchange(other.pointer_, 0);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  CUdeviceptr get() const { return pointer_; }
  std::size_t size() const { return size_; }

 private:
  void reset() noexcept {
    if (pointer_ != 0) {
      CU_CHECK_NOTHROW(cuMemFree(pointer_));
      pointer_ = 0;
      size_ = 0;
    }
  }

  CUdeviceptr pointer_{};
  std::size_t size_{};
};

// Page-locked host memory, allocated once at worker startup to back the frame slots
// (CLAUDE.md invariant 2 — the allocation implicitly synchronizes, so it must never
// happen per frame). Hands out std::byte* so IBackend::allocate_slots can return it
// without leaking a driver type past this directory.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(std::size_t bytes) : size_(bytes) {
    CU_CHECK(cuMemAllocHost(&pointer_, bytes));
  }

  ~PinnedHostBuffer() { reset(); }

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = std::exchange(other.pointer_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  std::byte* data() const { return static_cast<std::byte*>(pointer_); }
  std::size_t size() const { return size_; }

 private:
  void reset() noexcept {
    if (pointer_ != nullptr) {
      CU_CHECK_NOTHROW(cuMemFreeHost(pointer_));
      pointer_ = nullptr;
      size_ = 0;
    }
  }

  void* pointer_{nullptr};
  std::size_t size_{};
};

// Phase 5 runs every copy and launch on ONE of these, deliberately (docs/PLAN.md
// Phase 5): a single stream isolates "is the plumbing correct" from "does async
// overlap work", so a Phase 5 failure is never ambiguous between the two. Phase 6
// holds K of them and adds the events that gate buffer return; this type does not
// change, only how many exist.
//
// CU_STREAM_NON_BLOCKING rather than the default: a stream created with the default
// flag implicitly synchronizes with the NULL stream, which would silently serialize
// the K streams Phase 6 creates against any library that touches the legacy default.
// Nothing here uses the NULL stream, so the two are equivalent today — this is the
// habit that stays correct when there are four of them.
class CudaStream {
 public:
  CudaStream() { CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING)); }

  ~CudaStream() { reset(); }

  CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      reset();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CUstream get() const { return stream_; }

 private:
  void reset() noexcept {
    if (stream_ != nullptr) {
      CU_CHECK_NOTHROW(cuStreamDestroy(stream_));
      stream_ = nullptr;
    }
  }

  CUstream stream_{};
};

}  // namespace imgjit::cuda
