#pragma once

// RAII wrappers for the driver resources: CUcontext, CUmodule, CUdeviceptr, pinned
// host memory, CUstream and CUevent. No raw cuCtxDestroy / cuModuleUnload / cuMemFree /
// cuStreamDestroy / cuEventDestroy appears in logic code (CLAUDE.md conventions).

#include <cuda.h>

#include <unistd.h>

#include <cstddef>
#include <new>
#include <string>
#include <utility>

#include "backend/cuda/cuda_check.h"

namespace imgjit::cuda {

// CLAUDE.md invariant 1: exactly one thread ever constructs this, and only that thread
// destroys it. Non-movable so it cannot drift off that thread. Phase 6 is what made
// "once, at startup" too strong a claim to keep: an illegal access poisons a context
// permanently, so recovery is destroying this and building another one — still on the
// same thread, and still with nothing else in the process holding a driver handle.
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

// Page-locked host memory the DRIVER owns, allocated once at worker startup (CLAUDE.md
// invariant 2 — the allocation implicitly synchronizes, so it must never happen per
// frame). Phase 6 uses it for the per-stream device-to-host staging buffers, which live
// and die entirely inside the backend.
//
// It is deliberately NOT what backs the frame slots any more; see RegisteredHostBuffer
// below for why context recreation made that unsafe.
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

// Host memory THIS PROCESS owns, page-locked in place with cuMemHostRegister rather
// than allocated by the driver. It backs the frame slots, and the distinction is a
// correctness one that Phase 6 forced.
//
// cuCtxDestroy frees every cuMemAllocHost allocation made in that context. The frame
// slots cannot take part in that: the server's SlotPool holds their base pointer for
// the life of the process, and connection threads are recv()ing into individual slots
// at the instant a context recreation happens — so driver-owned slot memory would leave
// every reader thread writing into a freed buffer, which is the silent-corruption class
// this project keeps out of the design on purpose.
//
// Owning the pages ourselves makes recreation a matter of dropping the registration and
// taking it again: the ADDRESS NEVER MOVES, so nothing outside this directory has to
// learn that anything happened. The window between detach() and attach() is not a
// hazard either — the memory stays perfectly valid to write, it merely stops being
// page-locked, so a recv() landing in it during recovery is a correct copy that misses
// out on DMA it was not about to use anyway.
//
// Page-aligned because cuMemHostRegister pins whole pages: handing it an interior
// pointer is not portably accepted, and the rounding is two lines.
class RegisteredHostBuffer {
 public:
  explicit RegisteredHostBuffer(std::size_t bytes)
      : alignment_(page_size()),
        size_(round_up(bytes, alignment_)),
        memory_(static_cast<std::byte*>(
            ::operator new(size_, std::align_val_t(alignment_)))) {
    attach();
  }

  ~RegisteredHostBuffer() {
    detach();
    ::operator delete(memory_, std::align_val_t(alignment_));
  }

  // Non-movable as well as non-copyable: an owner outside this directory holds data()
  // indefinitely, so an address that can be moved out from under it is exactly the bug
  // this type exists to make impossible.
  RegisteredHostBuffer(const RegisteredHostBuffer&) = delete;
  RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;

  // Page-locks the buffer into the CURRENT context. Called at startup and again after
  // every context recreation.
  void attach() {
    if (!registered_) {
      CU_CHECK(cuMemHostRegister(memory_, size_, 0));
      registered_ = true;
    }
  }

  // Nothrow: this runs while tearing a poisoned context down, where the unregister is
  // very likely to report the sticky error that started the recovery. Losing the
  // registration is fine — cuCtxDestroy drops it regardless.
  void detach() noexcept {
    if (registered_) {
      CU_CHECK_NOTHROW(cuMemHostUnregister(memory_));
      registered_ = false;
    }
  }

  std::byte* data() const { return memory_; }
  std::size_t size() const { return size_; }

 private:
  static std::size_t page_size() {
    const long reported = ::sysconf(_SC_PAGESIZE);
    return reported > 0 ? static_cast<std::size_t>(reported) : std::size_t{4096};
  }

  static std::size_t round_up(std::size_t value, std::size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
  }

  std::size_t alignment_;
  std::size_t size_;
  std::byte* memory_;
  bool registered_{false};
};

// Phase 5 ran every copy and launch on ONE of these, deliberately (docs/PLAN.md
// Phase 5): a single stream isolates "is the plumbing correct" from "does async
// overlap work", so a Phase 5 failure was never ambiguous between the two. Phase 6
// holds K of them and adds the events that gate buffer return; this type did not
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

// The instrument behind CLAUDE.md invariant 3, and the most important type Phase 6
// added. Recorded on a stream after that frame's last transfer, it is the ONLY thing
// permitted to say a frame is finished: not the return of cuLaunchKernel, not the
// return of the async copy, both of which come back while the work is still queued.
// Anything the frame touched — its two device buffers, its staging buffer, and the
// pinned input slot the server releases on the completion — stays off-limits until
// cuEventQuery on this reports success.
//
// CU_EVENT_DISABLE_TIMING by default, because the event that gates buffer return is
// never read for a time, and disabling timing makes both the record and the query
// cheaper — which matters on a path polled once per worker iteration. Phase 7's
// kernel-time pair is the one place that reads an elapsed time off an event, and it
// asks for CU_EVENT_DEFAULT explicitly.
class CudaEvent {
 public:
  explicit CudaEvent(unsigned int flags = CU_EVENT_DISABLE_TIMING) {
    CU_CHECK(cuEventCreate(&event_, flags));
  }

  ~CudaEvent() { reset(); }

  CudaEvent(CudaEvent&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      reset();
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CUevent get() const { return event_; }

 private:
  void reset() noexcept {
    if (event_ != nullptr) {
      CU_CHECK_NOTHROW(cuEventDestroy(event_));
      event_ = nullptr;
    }
  }

  CUevent event_{};
};

}  // namespace imgjit::cuda
