/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <simd/simd.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <optional>
#include <exception>
#include <stdexcept>
#include <string>

#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000) || \
    (defined(__TV_OS_VERSION_MAX_ALLOWED) && __TV_OS_VERSION_MAX_ALLOWED >= 180000) || \
    (defined(__WATCH_OS_VERSION_MAX_ALLOWED) && __WATCH_OS_VERSION_MAX_ALLOWED >= 110000)
#define ET_METAL_SDK_HAS_MTL_MATH_COMPILE_OPTIONS 1
#else
#define ET_METAL_SDK_HAS_MTL_MATH_COMPILE_OPTIONS 0
#endif

#if !ET_METAL_SDK_HAS_MTL_MATH_COMPILE_OPTIONS
// When building with an older SDK, declare newer Metal compile option symbols so we can still
// use them behind runtime availability checks.
typedef NS_ENUM(NSInteger, MTLMathMode) {
    MTLMathModeSafe = 0,
    MTLMathModeRelaxed = 1,
    MTLMathModeFast = 2,
};

typedef NS_ENUM(NSInteger, MTLMathFloatingPointFunctions) {
    MTLMathFloatingPointFunctionsFast = 0,
    MTLMathFloatingPointFunctionsPrecise = 1,
};

@interface MTLCompileOptions ()
@property(readwrite, nonatomic) MTLMathMode mathMode;
@property(readwrite, nonatomic) MTLMathFloatingPointFunctions mathFloatingPointFunctions;
@end
#endif

namespace executorch {
namespace backends {
namespace metal {

// =======================
// Exception-Safe Dispatch Function (similar to PyTorch MPS)
// =======================

void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)()) {
    __block std::optional<std::exception_ptr> block_exception;
    dispatch_sync(queue, ^() {
        try {
            block();
        } catch (...) {
            block_exception = std::current_exception();
        }
    });
    if (block_exception) {
        std::rethrow_exception(*block_exception);
    }
}

// =======================
// Global Variables and Storage
// ================


// Global Metal buffer mapping - accessible for MPS shim
std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

namespace {
// A view's address mapped to the address of the buffer it lives in, counted
// because several tensors can be views of the same address. For a view of CPU
// memory, `base` is the start of a CPU region in cpu_regions.
struct MetalView {
    void* base;
    int32_t count;
    bool cpu = false;
};
std::unordered_map<void*, MetalView> ptr_to_view;

// Real sizes and strides of the strided views, by tensor.
struct StridedView {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
};
std::unordered_map<const void*, StridedView> strided_views;
// CPU memory that views of it are bound into, the way views of a Metal
// allocation are bound into its buffer: one no-copy buffer over the whole
// region, which all its views and the region's own address share, so Metal
// orders their uses. Separate buffers over overlapping memory would not be. The
// buffer lives as long as the memory, not its views: views come and go all the
// time, and a new buffer for the same memory would not be ordered with work
// still queued on the old one.
struct CpuRegion {
    id<MTLBuffer> buffer;
    // Memory the runtime allocated: the region lives until the memory is freed
    // (metal_release_cpu_region). Other memory can be freed and its address
    // reused behind the runtime's back, so its region only lives while views
    // of it do, and `views` counts their handles.
    bool owned;
    int32_t views;
};
std::unordered_map<void*, CpuRegion> cpu_regions;
} // namespace

bool metal_resolve_buffer(void* ptr, id<MTLBuffer>* buffer, size_t* offset) {
    void* base = ptr;
    auto view = ptr_to_view.find(ptr);
    if (view != ptr_to_view.end()) {
        base = view->second.base;
        if (view->second.cpu) {
            auto region = cpu_regions.find(base);
            if (region == cpu_regions.end()) {
                return false;
            }
            *buffer = region->second.buffer;
            *offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(base);
            return true;
        }
    }

    auto it = ptr_to_mtl_buffer.find(base);
    if (it == ptr_to_mtl_buffer.end()) {
        // The start of a CPU region is bound into the region's buffer too.
        auto region = cpu_regions.find(ptr);
        if (region == cpu_regions.end()) {
            return false;
        }
        *buffer = region->second.buffer;
        *offset = 0;
        return true;
    }
    *buffer = it->second;
    *offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(base);
    return true;
}

void metal_record_strided_view(
    const void* tensor,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides) {
    strided_views[tensor] = {std::move(sizes), std::move(strides)};
}

void metal_share_strided_view(const void* from, const void* to) {
    auto it = strided_views.find(from);
    if (it != strided_views.end()) {
        StridedView view = it->second;
        strided_views[to] = std::move(view);
    }
}

void metal_forget_strided_view(const void* tensor) {
    strided_views.erase(tensor);
}

bool metal_is_strided_view(const void* tensor) {
    return strided_views.find(tensor) != strided_views.end();
}

id<MTLBuffer> metal_packed_copy_of_strided_view(
    const executorch::runtime::etensor::Tensor& tensor) {
    if (strided_views.find(&tensor) == strided_views.end()) {
        return nil;
    }
    id<MTLComputeCommandEncoder> encoder = getCurrentMetalStream()->commandEncoder();
    if (!encoder) {
        ET_LOG(Error, "metal_packed_copy_of_strided_view: no command encoder");
        return nil;
    }
    return ETMetalKernelFunction::encodePackedCopyOfStridedView(encoder, tensor);
}

bool metal_copy_strided_view(
    const executorch::runtime::etensor::Tensor& tensor,
    void* dst) {
    @autoreleasepool {
        id<MTLBuffer> packed = metal_packed_copy_of_strided_view(tensor);
        if (!packed) {
            return false;
        }
        getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
        std::memcpy(dst, [packed contents], tensor.nbytes());
        return true;
    }
}

// Metal buffer pool with best-fit matching and LRU eviction.
// On free, buffers are recycled into a sorted pool. On alloc, the smallest
// buffer >= requested size is returned (if within the headroom bound). When the
// pool exceeds its max size, least-recently-used buffers are released.
//
// The headroom limits internal fragmentation: a cached buffer is only
// reused if its size is at most min(2x requested, requested + kMaxHeadroom).
static const size_t kMaxHeadroom = 32768; // 32KB

struct PoolEntry {
    id<MTLBuffer> buffer;
    size_t size;
};

class MetalBufferPool {
public:
    explicit MetalBufferPool(size_t max_bytes = 256 * 1024 * 1024)
        : max_bytes_(max_bytes), cached_bytes_(0) {}

    id<MTLBuffer> reuse(size_t size) {
        auto it = size_map_.lower_bound(size);
        // Use saturating arithmetic to avoid size_t overflow.
        size_t double_size = (size > SIZE_MAX / 2) ? SIZE_MAX : 2 * size;
        size_t size_plus_headroom = (size > SIZE_MAX - kMaxHeadroom) ? SIZE_MAX : size + kMaxHeadroom;
        size_t max_acceptable = std::min(double_size, size_plus_headroom);
        if (it == size_map_.end() || it->first > max_acceptable) {
            return nil;
        }

        auto lru_it = it->second;
        id<MTLBuffer> buffer = lru_it->buffer;
        cached_bytes_ -= lru_it->size;
        lru_list_.erase(lru_it);
        size_map_.erase(it);
        return buffer;
    }

    void recycle(id<MTLBuffer> buffer) {
        size_t size = [buffer length];

        // Don't pool buffers larger than half the max pool size — a single
        // large buffer would evict all the small frequently-reused buffers.
        if (size > max_bytes_ / 2) {
            [buffer release];
            return;
        }

        lru_list_.push_front({buffer, size});
        size_map_.insert({size, lru_list_.begin()});
        cached_bytes_ += size;

        while (cached_bytes_ > max_bytes_ && !lru_list_.empty()) {
            evict_oldest();
        }
    }

    void clear() {
        for (auto& entry : lru_list_) {
            [entry.buffer release];
        }
        lru_list_.clear();
        size_map_.clear();
        cached_bytes_ = 0;
    }

private:
    void evict_oldest() {
        auto tail = std::prev(lru_list_.end());
        cached_bytes_ -= tail->size;

        // Find and remove the matching size_map_ entry for this LRU tail.
        // O(k) where k = number of cached buffers of the same size (typically 1-2).
        auto range = size_map_.equal_range(tail->size);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == tail) {
                size_map_.erase(it);
                break;
            }
        }

        [tail->buffer release];
        lru_list_.erase(tail);
    }

    size_t max_bytes_;
    size_t cached_bytes_;
    std::list<PoolEntry> lru_list_;       // newest at front, oldest at back
    std::multimap<size_t, std::list<PoolEntry>::iterator> size_map_;
};

static constexpr size_t kMaxPoolSizeMB = 16384; // 16 GB upper bound

static MetalBufferPool& get_metal_buffer_pool() {
    static auto* pool = [] {
        size_t max_bytes = 256 * 1024 * 1024; // 256 MB default
        const char* env = std::getenv("ET_METAL_BUFFER_POOL_SIZE_MB");
        if (env) {
            char* end = nullptr;
            long mb = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && mb > 0 &&
                static_cast<unsigned long>(mb) <= kMaxPoolSizeMB) {
                max_bytes = static_cast<size_t>(mb) * 1024 * 1024;
            }
        }
        return new MetalBufferPool(max_bytes);
    }();
    return *pool;
}

// Global storage to keep shared_ptr alive while raw pointers are used
static std::unordered_map<ETMetalKernelFunction*, std::shared_ptr<ETMetalKernelFunction>> function_storage;
static std::unordered_map<ETMetalShaderLibrary*, std::unique_ptr<ETMetalShaderLibrary>> library_storage;

// Static singleton instance for default stream
ETMetalStream* ETMetalStream::defaultStream_ = nullptr;

// Thread-local current stream
static thread_local ETMetalStream* currentStream_ = nullptr;

// =======================
// Metal Helper Functions (C Interface)
// =======================

extern "C" {

void* metal_allocate_buffer(long bytes) {
    if (bytes <= 0) {
        ET_LOG(Error, "Invalid Metal buffer allocation size: %ld", bytes);
        return nullptr;
    }
    size_t size = static_cast<size_t>(bytes);

    // Check the buffer pool first (best-fit with bounded headroom)
    auto& pool = get_metal_buffer_pool();
    id<MTLBuffer> buffer = pool.reuse(size);
    if (buffer) {
        void* ptr = [buffer contents];
        ptr_to_mtl_buffer[ptr] = buffer;
        ET_LOG(Debug, "Reused %zu byte Metal buffer from pool (requested %ld)", [buffer length], bytes);
        return ptr;
    }

    // Pool miss — allocate a new buffer
    ETMetalStream* stream = getCurrentMetalStream();
    id<MTLDevice> device = stream->device();
    if (!device) {
        ET_LOG(Error, "Failed to get Metal device from stream");
        return nullptr;
    }

    @autoreleasepool {
        buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer) {
            ET_LOG(Error, "Failed to allocate %zu bytes on Metal device", size);
            return nullptr;
        }

        void* ptr = [buffer contents];
        ptr_to_mtl_buffer[ptr] = buffer;

        ET_LOG(Debug, "Allocated %zu bytes on Metal device", size);
        return ptr;
    }
}

void metal_deallocate_buffer(void* ptr) {
    if (!ptr) {
        return;
    }
    auto it = ptr_to_mtl_buffer.find(ptr);
    if (it != ptr_to_mtl_buffer.end()) {
        id<MTLBuffer> buffer = it->second;
        ET_LOG(Debug, "Recycling %zu byte Metal buffer to pool (ptr %p)", (size_t)[buffer length], ptr);
        ptr_to_mtl_buffer.erase(it);
        get_metal_buffer_pool().recycle(buffer);
    } else {
        ET_LOG(Error, "Failed to find Metal buffer for pointer %p", ptr);
    }
}

void metal_cleanup_resources() {
    for (auto& pair : ptr_to_mtl_buffer) {
        [pair.second release];
    }
    ptr_to_mtl_buffer.clear();
    ptr_to_view.clear();
    strided_views.clear();
    for (auto& pair : cpu_regions) {
        [pair.second.buffer release];
    }
    cpu_regions.clear();
    get_metal_buffer_pool().clear();
}

bool metal_buffer_nocopy(void* ptr, size_t nbytes, bool map_ptr_to_buffer) {
    id<MTLDevice> device = get_metal_device();
    id<MTLBuffer> subBuffer = [device newBufferWithBytesNoCopy:ptr
                                                        length:nbytes
                                                        options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared
                                                    deallocator:nil];
    if (!subBuffer) {
        ET_LOG(Error, "metal_buffer_nocopy: Failed to create no-copy buffer (ptr=%p, nbytes=%zu)", ptr, nbytes);
        return false;
    }

    if (map_ptr_to_buffer) {
        ptr_to_mtl_buffer[ptr] = subBuffer;  // Map contents to buffer
    }

    return true;
}

bool metal_register_view(void* view_ptr, void* base_ptr) {
    // A view of a view lives in the same buffer as its parent.
    auto parent = ptr_to_view.find(base_ptr);
    void* base = parent != ptr_to_view.end() ? parent->second.base : base_ptr;
    if (ptr_to_mtl_buffer.find(base) == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "metal_register_view: %p is not inside a Metal buffer", base_ptr);
        return false;
    }

    auto it = ptr_to_view.find(view_ptr);
    if (it == ptr_to_view.end()) {
        ptr_to_view[view_ptr] = {base, 1};
    } else {
        it->second.base = base;
        it->second.count++;
    }
    return true;
}

void metal_retain_view(void* view_ptr) {
    auto it = ptr_to_view.find(view_ptr);
    if (it != ptr_to_view.end()) {
        it->second.count++;
        if (it->second.cpu) {
            auto region = cpu_regions.find(it->second.base);
            if (region != cpu_regions.end()) {
                region->second.views++;
            }
        }
    }
}

bool metal_register_cpu_view(
    void* view_ptr,
    size_t view_nbytes,
    void* region,
    size_t region_nbytes,
    bool owned) {
    auto view = ptr_to_view.find(view_ptr);
    if (view != ptr_to_view.end() && (!view->second.cpu || view->second.base != region)) {
        ET_LOG(Error, "metal_register_cpu_view: %p is already a view of other memory", view_ptr);
        return false;
    }

    const size_t needed = std::max(
        region_nbytes,
        static_cast<size_t>(static_cast<uint8_t*>(view_ptr) - static_cast<uint8_t*>(region)) + view_nbytes);
    auto it = cpu_regions.find(region);
    if (it == cpu_regions.end() || [it->second.buffer length] < needed) {
        // Default CPU caching: the CPU keeps reading and writing this memory.
        id<MTLBuffer> buffer = [get_metal_device() newBufferWithBytesNoCopy:region
                                                                     length:needed
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
        if (!buffer) {
            ET_LOG(Error, "metal_register_cpu_view: failed to wrap %zu bytes at %p", needed, region);
            return false;
        }
        if (it == cpu_regions.end()) {
            it = cpu_regions.emplace(region, CpuRegion{buffer, owned, 0}).first;
        } else {
            // Only for memory whose extent was not known up front. Work already
            // queued uses the old buffer, which Metal does not relate to the new
            // one over the same memory, so it has to be done first.
            getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
            [it->second.buffer release];
            it->second.buffer = buffer;
        }
    }
    it->second.views++;

    if (view == ptr_to_view.end()) {
        ptr_to_view[view_ptr] = {region, 1, true};
    } else {
        view->second.count++;
    }
    return true;
}

bool metal_cpu_view_region(void* ptr, void** region) {
    auto it = ptr_to_view.find(ptr);
    if (it == ptr_to_view.end() || !it->second.cpu) {
        return false;
    }
    *region = it->second.base;
    return true;
}

bool metal_is_cpu_view(void* ptr) {
    void* region = nullptr;
    return metal_cpu_view_region(ptr, &region);
}

bool metal_is_cpu_memory(void* ptr) {
    return metal_is_cpu_view(ptr) || cpu_regions.find(ptr) != cpu_regions.end();
}

bool metal_release_cpu_region(void* region) {
    auto it = cpu_regions.find(region);
    if (it == cpu_regions.end()) {
        return false;
    }
    // Queued work may still use the memory through this buffer, which does
    // not own the memory: it has to be done before the memory goes.
    getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    [it->second.buffer release];
    cpu_regions.erase(it);
    return true;
}

bool metal_unregister_view(void* view_ptr) {
    auto it = ptr_to_view.find(view_ptr);
    if (it == ptr_to_view.end()) {
        return false;
    }
    if (it->second.cpu) {
        auto region = cpu_regions.find(it->second.base);
        if (region != cpu_regions.end() && --region->second.views <= 0 &&
            !region->second.owned) {
            metal_release_cpu_region(it->second.base);
        }
    }
    if (--it->second.count > 0) {
        return false;
    }
    ptr_to_view.erase(it);
    return true;
}

bool metal_is_device_pointer(void* ptr) {
    id<MTLBuffer> buffer = nil;
    size_t offset = 0;
    return metal_resolve_buffer(ptr, &buffer, &offset);
}

int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device) {
    if (!src || !dst || nbytes == 0) {
        ET_LOG(Error, "Metal copy: Invalid parameters");
        return -1;
    }

    @autoreleasepool {
        // Case 1: Device-to-device copy - use GPU blit encoder (most efficient).
        // CPU memory with a Metal buffer (metal_register_cpu_view) counts as a
        // device pointer, but CPU code reads and writes it directly after the
        // copy returns, so a copy involving it is made on the CPU, after a wait.
        if (src_is_device && dst_is_device &&
            !metal_is_cpu_memory(const_cast<void*>(src)) &&
            !metal_is_cpu_memory(dst)) {
            id<MTLBuffer> srcBuffer = nil;
            id<MTLBuffer> dstBuffer = nil;
            size_t srcOffset = 0;
            size_t dstOffset = 0;

            if (metal_resolve_buffer(const_cast<void*>(src), &srcBuffer, &srcOffset) &&
                metal_resolve_buffer(dst, &dstBuffer, &dstOffset)) {

                // Use Metal's blit encoder for GPU-accelerated copy
                ETMetalStream* stream = getCurrentMetalStream();
                stream->copy(srcBuffer, dstBuffer, nbytes, srcOffset, dstOffset, SyncType::NONE);

                ET_LOG(Debug, "Metal device-to-device copy (GPU blit): %zu bytes", nbytes);
                return 0;
            }

            ET_LOG(Error, "Metal copy: Device pointers not found in buffer map");
            return -1;
        }

        // Case 2: Host-to-device or device-to-host - use memcpy with shared memory
        // Since Metal uses shared storage mode, CPU and GPU access the same memory.
        // What the GPU still has to do with it must be done before the CPU
        // touches it: writes to a device source, and reads or writes of a device
        // destination.
        if (src_is_device || dst_is_device) {
            ETMetalStream* stream = getCurrentMetalStream();
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        std::memcpy(dst, src, nbytes);

        ET_LOG(Debug, "Metal memory copy (memcpy): %zu bytes, src_device=%d, dst_device=%d",
               nbytes, src_is_device, dst_is_device);
    }

    return 0;
}

id<MTLDevice> get_metal_device() {
    // Use stream-based device access
    ETMetalStream* stream = getCurrentMetalStream();
    return stream->device();
}

id<MTLCommandQueue> get_metal_command_queue() {
    // Use stream-based queue access
    ETMetalStream* stream = getCurrentMetalStream();
    return stream->commandQueue();
}

} // extern "C"

// =======================
// ETMetalShaderLibrary Implementation
// =======================

ETMetalShaderLibrary::ETMetalShaderLibrary(const std::string& source) : shaderSource_(source) {
    compileLibrary();
}

ETMetalShaderLibrary::~ETMetalShaderLibrary() {
    @autoreleasepool {
        if (library_) {
            [library_ release];
            library_ = nil;
        }

        for (auto& pair : pipelineStates_) {
            [pair.second.first release];
            [pair.second.second release];
        }
        pipelineStates_.clear();
    }
}

void ETMetalShaderLibrary::compileLibrary() {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
            return;
        }

        NSString* sourceString = [NSString stringWithUTF8String:shaderSource_.c_str()];
        NSError* error = nil;

        MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }

        library_ = [device newLibraryWithSource:sourceString options:options error:&error];
        if (!library_ || error) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to compile shader library: %s",
                   error ? [[error localizedDescription] UTF8String] : "unknown error");
            return;
        }

        [library_ retain];
        ET_LOG(Debug, "ETMetalShaderLibrary: Successfully compiled shader library");
    }
}

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> ETMetalShaderLibrary::getLibraryPipelineState(const std::string& functionName) {
    auto it = pipelineStates_.find(functionName);
    if (it != pipelineStates_.end()) {
        return it->second;
    }

    @autoreleasepool {
        if (!library_) {
            ET_LOG(Error, "ETMetalShaderLibrary: Library not compiled");
            return {nil, nil};
        }

        id<MTLDevice> device = get_metal_device();
        if (!device) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
            return {nil, nil};
        }

        NSString* funcName = [NSString stringWithUTF8String:functionName.c_str()];
        id<MTLFunction> function = [library_ newFunctionWithName:funcName];
        if (!function) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get function '%s'", functionName.c_str());
            return {nil, nil};
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState || error) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to create pipeline state for '%s': %s",
                   functionName.c_str(), error ? [[error localizedDescription] UTF8String] : "unknown error");
            [function release];
            return {nil, nil};
        }

        [pipelineState retain];
        [function retain];
        pipelineStates_[functionName] = {pipelineState, function};

        ET_LOG(Debug, "ETMetalShaderLibrary: Created pipeline state for function '%s'", functionName.c_str());
        return {pipelineState, function};
    }
}

std::shared_ptr<ETMetalKernelFunction> ETMetalShaderLibrary::getKernelFunction(const std::string& name) {
    auto pipelineStatePair = getLibraryPipelineState(name);
    if (!pipelineStatePair.first || !pipelineStatePair.second) {
        ET_LOG(Error, "ETMetalShaderLibrary::getKernelFunction: Failed to get pipeline state for '%s'", name.c_str());
        return nullptr;
    }

    return std::make_shared<ETMetalKernelFunction>(pipelineStatePair.first, pipelineStatePair.second);
}

// =======================
// ETMetalKernelFunction Implementation
// =======================

ETMetalKernelFunction::ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func)
    : cps_(cps), func_(func), encoder_(nil) {
    if (cps_) [cps_ retain];
    if (func_) [func_ retain];
}

ETMetalKernelFunction::~ETMetalKernelFunction() {
    @autoreleasepool {
        // Don't release encoder_ here - the stream owns it
        // Only clean up our own references
        if (cps_) {
            [cps_ release];
            cps_ = nil;
        }
        if (func_) {
            [func_ release];
            func_ = nil;
        }

        encoder_ = nil; // Clear reference without releasing
        clearBindings();
    }
}

void ETMetalKernelFunction::bindBuffer(unsigned idx, id<MTLBuffer> buffer, size_t offset) {
    [encoder_ setBuffer:buffer offset:offset atIndex:idx];
    if (bindings_.size() <= idx) {
        bindings_.resize(idx + 1);
    }
    Binding& binding = bindings_[idx];
    [buffer retain];
    [binding.buffer release];
    binding.buffer = buffer;
    binding.offset = offset;
    binding.bytes.clear();
    binding.set = true;
}

void ETMetalKernelFunction::bindBytes(unsigned idx, const void* data, size_t size) {
    [encoder_ setBytes:data length:size atIndex:idx];
    if (bindings_.size() <= idx) {
        bindings_.resize(idx + 1);
    }
    Binding& binding = bindings_[idx];
    [binding.buffer release];
    binding.buffer = nil;
    binding.offset = 0;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    binding.bytes.assign(bytes, bytes + size);
    binding.set = true;
}

void ETMetalKernelFunction::restoreBindings() {
    [encoder_ setComputePipelineState:cps_];
    for (size_t idx = 0; idx < bindings_.size(); idx++) {
        const Binding& binding = bindings_[idx];
        if (!binding.set) {
            continue;
        }
        if (binding.buffer) {
            [encoder_ setBuffer:binding.buffer offset:binding.offset atIndex:idx];
        } else {
            [encoder_ setBytes:binding.bytes.data() length:binding.bytes.size() atIndex:idx];
        }
    }
}

void ETMetalKernelFunction::clearBindings() {
    for (Binding& binding : bindings_) {
        [binding.buffer release];
    }
    bindings_.clear();
}

void ETMetalKernelFunction::startEncoding() {
    @autoreleasepool {
        clearBindings();
        // Don't retain/release the encoder - just get reference from stream
        ETMetalStream* stream = getCurrentMetalStream();
        encoder_ = stream->commandEncoder(); // Use stream's managed encoder
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction: Failed to get encoder from stream");
            return;
        }

        // Don't retain - stream owns the encoder
        [encoder_ setComputePipelineState:cps_];

        ET_LOG(Debug, "ETMetalKernelFunction: Started encoding with stream-managed encoder");
    }
}

namespace {

// Copies the elements of a strided view, in row-major order, into a packed
// buffer. Any slots would do: a kernel function whose arguments the gather
// overwrites binds them again afterwards (restoreBindings).
constexpr uint32_t kGatherMaxDims = 16;
constexpr unsigned kGatherSrcIndex = 28;
constexpr unsigned kGatherDstIndex = 29;
constexpr unsigned kGatherParamsIndex = 30;

struct GatherParams {
    uint32_t ndim;
    uint32_t sizes[kGatherMaxDims];
    uint32_t strides[kGatherMaxDims];
};

const char* kGatherShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct GatherParams {
    uint ndim;
    uint sizes[16];
    uint strides[16];
};

template <typename T>
kernel void gather_strided(
    device const T* src [[buffer(28)]],
    device T* dst [[buffer(29)]],
    constant GatherParams& p [[buffer(30)]],
    uint gid [[thread_position_in_grid]]) {
    uint remaining = gid;
    uint offset = 0;
    for (uint i = 0; i < p.ndim; ++i) {
        const uint d = p.ndim - 1 - i;
        offset += (remaining % p.sizes[d]) * p.strides[d];
        remaining /= p.sizes[d];
    }
    dst[gid] = src[offset];
}

template [[host_name("gather_strided_1")]] kernel void gather_strided<uchar>(device const uchar*, device uchar*, constant GatherParams&, uint);
template [[host_name("gather_strided_2")]] kernel void gather_strided<ushort>(device const ushort*, device ushort*, constant GatherParams&, uint);
template [[host_name("gather_strided_4")]] kernel void gather_strided<uint>(device const uint*, device uint*, constant GatherParams&, uint);
template [[host_name("gather_strided_8")]] kernel void gather_strided<ulong>(device const ulong*, device ulong*, constant GatherParams&, uint);
)";

std::shared_ptr<ETMetalKernelFunction> gather_kernel(size_t element_size) {
    static ETMetalShaderLibrary library(kGatherShaderSource);
    return library.getKernelFunction("gather_strided_" + std::to_string(element_size));
}

} // namespace

MTLBuffer_t ETMetalKernelFunction::encodePackedCopyOfStridedView(
    MTLComputeCommandEncoder_t encoder,
    const executorch::runtime::etensor::Tensor& tensor) {
    auto it = strided_views.find(&tensor);
    if (it == strided_views.end()) {
        return nil;
    }
    const StridedView& view = it->second;
    const size_t element_size = tensor.element_size();
    const size_t numel = tensor.numel();
    if (view.sizes.size() > kGatherMaxDims || numel > UINT32_MAX ||
        (element_size != 1 && element_size != 2 && element_size != 4 && element_size != 8)) {
        ET_LOG(Error, "encodePackedCopyOfStridedView: unsupported view (ndim %zu, numel %zu, element size %zu)",
               view.sizes.size(), numel, element_size);
        return nil;
    }

    id<MTLBuffer> src = nil;
    size_t src_offset = 0;
    if (!metal_resolve_buffer(tensor.mutable_data_ptr(), &src, &src_offset)) {
        ET_LOG(Error, "encodePackedCopyOfStridedView: the view is not in a Metal buffer");
        return nil;
    }
    id<MTLBuffer> dst = [get_metal_device() newBufferWithLength:std::max<size_t>(numel * element_size, 1)
                                                        options:MTLResourceStorageModeShared];
    if (!dst) {
        ET_LOG(Error, "encodePackedCopyOfStridedView: failed to allocate %zu bytes", numel * element_size);
        return nil;
    }
    [dst autorelease];

    GatherParams params = {};
    params.ndim = static_cast<uint32_t>(view.sizes.size());
    uint64_t extent = 0;
    for (size_t d = 0; d < view.sizes.size(); d++) {
        params.sizes[d] = static_cast<uint32_t>(view.sizes[d]);
        params.strides[d] = static_cast<uint32_t>(view.strides[d]);
        if (view.sizes[d] > 0) {
            extent += static_cast<uint64_t>(view.sizes[d] - 1) * static_cast<uint64_t>(view.strides[d]);
        }
    }
    if (extent > UINT32_MAX) {
        ET_LOG(Error, "encodePackedCopyOfStridedView: view spans %llu elements, more than the gather can index",
               (unsigned long long)extent);
        return nil;
    }

    auto gather = gather_kernel(element_size);
    if (!gather) {
        return nil;
    }
    id<MTLComputePipelineState> cps = gather->cps_;
    [encoder setComputePipelineState:cps];
    [encoder setBuffer:src offset:src_offset atIndex:kGatherSrcIndex];
    [encoder setBuffer:dst offset:0 atIndex:kGatherDstIndex];
    [encoder setBytes:&params length:sizeof(params) atIndex:kGatherParamsIndex];
    const uint64_t group = std::min<uint64_t>([cps maxTotalThreadsPerThreadgroup], std::max<size_t>(numel, 1));
    // Straight onto the encoder: the stream's dispatch accounting may flush,
    // which would end the encoder under the op that is still binding to it.
    [encoder dispatchThreads:MTLSizeMake(std::max<size_t>(numel, 1), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    return dst;
}

void ETMetalKernelFunction::setArg(
    unsigned idx,
    const executorch::runtime::etensor::Tensor& tensor,
    ArgAccess access) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    if (access != ArgAccess::kStridedInPlace && metal_is_strided_view(&tensor)) {
        // The tensor's own strides are those of a packed tensor, so binding the
        // memory as it is would have the kernel read or write the wrong
        // elements.
        if (access == ArgAccess::kWrite) {
            ET_LOG(Error, "ETMetalKernelFunction::setArg: argument %u is written through a view that is not densely packed, which is unsupported", idx);
            // The kernel will not be dispatched: let go of what it bound.
            clearBindings();
            throw std::runtime_error("kernel writes through a non-packed view");
        }
        id<MTLBuffer> packed = encodePackedCopyOfStridedView(encoder_, tensor);
        if (!packed) {
            ET_LOG(Error, "ETMetalKernelFunction::setArg: failed to pack the view for argument %u", idx);
            clearBindings();
            throw std::runtime_error("failed to pack a non-packed view");
        }
        restoreBindings();
        bindBuffer(idx, packed, 0);
        return;
    }

    void* data_ptr = tensor.mutable_data_ptr();
    size_t totalSize = tensor.numel() * tensor.element_size();

    id<MTLBuffer> mtlBuffer = nil;
    size_t bufferOffset = 0;
    if (metal_resolve_buffer(data_ptr, &mtlBuffer, &bufferOffset)) {
        // Use existing Metal buffer; a view binds its parent at an offset
        bindBuffer(idx, mtlBuffer, bufferOffset);
        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set Metal buffer at index %u (size: %zu)", idx, totalSize);
    } else {
        // Handle CPU tensor data
        if (totalSize <= 4096) {
            // Use setBytes for small data (more efficient)
            bindBytes(idx, data_ptr, totalSize);
            ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set CPU tensor via setBytes at index %u (size: %zu)", idx, totalSize);
        } else {
            // Create temporary buffer for large data (should be rare)
            @autoreleasepool {
                id<MTLDevice> device = get_metal_device();
                if (device) {
                    id<MTLBuffer> tempBuffer = [device newBufferWithBytes:data_ptr
                                                                   length:totalSize
                                                                  options:MTLResourceStorageModeShared];
                    if (tempBuffer) {
                        // The binding keeps it until dispatch, and the command
                        // buffer after that.
                        bindBuffer(idx, tempBuffer, 0);
                        [tempBuffer release];
                        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set large CPU tensor via temporary buffer at index %u (size: %zu)", idx, totalSize);
                    } else {
                        ET_LOG(Error, "ETMetalKernelFunction::setArg: Failed to create temporary buffer for index %u", idx);
                    }
                } else {
                    ET_LOG(Error, "ETMetalKernelFunction::setArg: No Metal device available for index %u", idx);
                }
            }
        }
    }
}

void ETMetalKernelFunction::setArg(unsigned idx, int64_t val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    bindBytes(idx, &val, sizeof(int64_t));
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set int64_t value %lld at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, uint32_t val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    bindBytes(idx, &val, sizeof(uint32_t));
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set uint32_t value %u at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, float val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    bindBytes(idx, &val, sizeof(float));
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set float value %f at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, bool val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    bindBytes(idx, &val, sizeof(bool));
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set bool value %s at index %u", val ? "true" : "false", idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, const void* data, size_t size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    bindBytes(idx, data, size);
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set bytes at index %u (size: %zu)", idx, size);
}

void ETMetalKernelFunction::setArgUint3(unsigned idx, uint32_t x, uint32_t y, uint32_t z) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArgUint3: No active encoder");
        return;
    }

    // Use SIMD library's uint3 type which matches Metal shader's uint3 layout
    simd_uint3 val = {x, y, z};
    bindBytes(idx, &val, sizeof(simd_uint3));
    ET_LOG(Debug, "ETMetalKernelFunction::setArgUint3: Set uint3{%u, %u, %u} at index %u", x, y, z, idx);
}

void ETMetalKernelFunction::dispatchSingle(uint64_t length) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingle: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    getCurrentMetalStream()->notifyDispatch();
    clearBindings();
    encoder_ = nil; // May be invalidated by flush; re-obtain via startEncoding()
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingle: Dispatched with length %llu, group size %llu", length, actualGroupSize);

}

void ETMetalKernelFunction::dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingleWithGroupSize: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = group_size > 0 ? std::min(group_size, maxThreadsPerGroup) : std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    getCurrentMetalStream()->notifyDispatch();
    clearBindings();
    encoder_ = nil; // May be invalidated by flush; re-obtain via startEncoding()
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingleWithGroupSize: Dispatched with length %llu, group size %llu", length, actualGroupSize);

}

void ETMetalKernelFunction::dispatchArray(const uint64_t* length, size_t length_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length[0]);
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    getCurrentMetalStream()->notifyDispatch();
    clearBindings();
    encoder_ = nil; // May be invalidated by flush; re-obtain via startEncoding()
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArray: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

}

void ETMetalKernelFunction::dispatchArrayWithGroupSize(const uint64_t* length, size_t length_size,
                                                      const uint64_t* group_size, size_t group_size_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = maxThreadsPerGroup;
        if (group_size && group_size_size > 0) {
            actualGroupSize = std::min(maxThreadsPerGroup, group_size[0]);
        }
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        if (group_size && group_size_size >= 2) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        if (group_size && group_size_size >= 3) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
            groupZ = std::min(static_cast<uint64_t>(group_size[2]), length_size > 2 ? length[2] : 1);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    getCurrentMetalStream()->notifyDispatch();
    clearBindings();
    encoder_ = nil; // May be invalidated by flush; re-obtain via startEncoding()
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

}

void ETMetalKernelFunction::dispatchThreadgroups(uint64_t gridX, uint64_t gridY, uint64_t gridZ,
                                                  uint64_t threadsX, uint64_t threadsY, uint64_t threadsZ) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: No active encoder");
        return;
    }

    if (!cps_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: No compute pipeline state");
        return;
    }

    // Calculate total threads per threadgroup
    uint64_t totalThreads = threadsX * threadsY * threadsZ;

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    // Validate total thread count
    if (totalThreads > maxThreadsPerGroup) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: Requested %llu total threads per threadgroup exceeds device maximum of %llu",
               (unsigned long long)totalThreads, (unsigned long long)maxThreadsPerGroup);
        return;
    }

    MTLSize threadgroupsPerGrid = MTLSizeMake(gridX, gridY, gridZ);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsX, threadsY, threadsZ);

    [encoder_ dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    getCurrentMetalStream()->notifyDispatch();
    clearBindings();
    encoder_ = nil; // May be invalidated by flush; re-obtain via startEncoding()

    ET_LOG(Debug, "ETMetalKernelFunction::dispatchThreadgroups: Dispatched grid [%llu, %llu, %llu] with threadgroup [%llu, %llu, %llu]",
           (unsigned long long)gridX, (unsigned long long)gridY, (unsigned long long)gridZ,
           (unsigned long long)threadsX, (unsigned long long)threadsY, (unsigned long long)threadsZ);
}

void ETMetalKernelFunction::runCommandBlock(std::function<void(void)> f) {
    // Use dispatch_sync with the stream's serial queue for thread safety and synchronization
    // This matches PyTorch's approach: dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ...)
    ETMetalStream* stream = getCurrentMetalStream();
    dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
            f();
        }
    });

    ET_LOG(Debug, "ETMetalKernelFunction::runCommandBlock: Executed command block with dispatch_sync");
}

// =======================
// ETMetalStream Implementation
// =======================

// Returns the default commitAndContinue flush interval for the given device.
// The GPU architecture name (e.g. "applegpu_g14g") ends with a chip class:
//   'p' = iPhone, 'g' = base/Pro, 's' = Max, 'd' = Ultra
// See also: https://github.com/ml-explore/mlx (Metal backend flush-interval
// selection uses the same heuristic).
static int getDefaultFlushInterval(MTLDevice_t device, const char** outArch) {
    char suffix = 'g';
    NSString* arch = nil;

    if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
        id architecture = [device architecture];
        if (architecture != nil) {
            arch = [architecture name];
            const char* str = arch ? [arch UTF8String] : nullptr;
            if (str) {
                size_t len = strlen(str);
                if (len > 0) {
                    suffix = str[len - 1];
                }
            }
        }
    }

    if (!arch) {
        arch = @"unknown";
    }
    if (outArch) {
        *outArch = [arch UTF8String];
    }

    switch (suffix) {
        case 'p': return 20;  // iPhone
        case 'g': return 40;  // base/Pro
        case 's': return 50;  // Max
        case 'd': return 50;  // Ultra
        default:  return 40;
    }
}

ETMetalStream::ETMetalStream()
    : device_(nil), commandQueue_(nil), commandBuffer_(nil), prevCommandBuffer_(nil),
      commandEncoder_(nil), serialQueue_(nullptr), enableCommitAndContinue_(true),
      flushInterval_(0), dispatchCount_(0) {
    @autoreleasepool {
        // Create device and command queue
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            ET_LOG(Error, "ETMetalStream: Failed to create Metal device");
            return;
        }
        [device_ retain];

        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            ET_LOG(Error, "ETMetalStream: Failed to create Metal command queue");
            return;
        }
        [commandQueue_ retain];

        // Create serial queue for thread safety
        serialQueue_ = dispatch_queue_create("metal gpu stream", nullptr);

        // Dispatch pipelining: periodically call [commandBuffer commitAndContinue]
        // every flushInterval_ dispatches so the driver can prepare the next batch
        // while the GPU executes the current one. Enabled by default for all Apple
        // Silicon. Set ET_METAL_FLUSH_INTERVAL=0 to disable.
        const char* archStr = nullptr;
        flushInterval_ = getDefaultFlushInterval(device_, &archStr);
        ET_LOG(Info, "ETMetalStream: arch='%s', flush interval=%d",
               archStr ? archStr : "unknown", flushInterval_);
    }
}

ETMetalStream::~ETMetalStream() {
    @autoreleasepool {
        // Synchronize before cleanup
        synchronize(SyncType::COMMIT_AND_WAIT);

        // Clean up command encoder
        if (commandEncoder_) {
            [commandEncoder_ release];
            commandEncoder_ = nil;
        }

        // Clean up command buffers
        if (commandBuffer_) {
            [commandBuffer_ release];
            commandBuffer_ = nil;
        }
        if (prevCommandBuffer_) {
            [prevCommandBuffer_ release];
            prevCommandBuffer_ = nil;
        }

        // Clean up command queue and device
        if (commandQueue_) {
            [commandQueue_ release];
            commandQueue_ = nil;
        }
        if (device_) {
            [device_ release];
            device_ = nil;
        }

        // Clean up serial queue
        if (serialQueue_) {
            dispatch_release(serialQueue_);
            serialQueue_ = nullptr;
        }

        ET_LOG(Debug, "ETMetalStream: Destroyed stream");
    }
}

ETMetalStream* ETMetalStream::getDefaultStream() {
    if (!defaultStream_) {
        defaultStream_ = new ETMetalStream();
    }
    return defaultStream_;
}

// Lazy command buffer creation (use MPSCommandBuffer like PyTorch)
MPSCommandBuffer* ETMetalStream::commandBuffer() {
    if (!commandBuffer_) {
        if (!commandQueue_) {
            ET_LOG(Error, "ETMetalStream::commandBuffer: No command queue available");
            return nil;
        }

        commandBuffer_ = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue_];
        if (!commandBuffer_) {
            ET_LOG(Error, "ETMetalStream::commandBuffer: Failed to create command buffer");
            return nil;
        }
        [commandBuffer_ retain];

        ET_LOG(Debug, "ETMetalStream::commandBuffer: Created lazy command buffer %p", commandBuffer_);
    }

    return commandBuffer_;
}

// Lazy command encoder creation
id<MTLComputeCommandEncoder> ETMetalStream::commandEncoder() {
    if (!commandEncoder_) {
        MPSCommandBuffer* cmdBuffer = commandBuffer();
        if (!cmdBuffer) {
            ET_LOG(Error, "ETMetalStream::commandEncoder: Failed to get command buffer");
            return nil;
        }

        commandEncoder_ = [cmdBuffer computeCommandEncoder];
        if (!commandEncoder_) {
            ET_LOG(Error, "ETMetalStream::commandEncoder: Failed to create command encoder");
            return nil;
        }
        [commandEncoder_ retain];

        ET_LOG(Debug, "ETMetalStream::commandEncoder: Created lazy command encoder %p", commandEncoder_);
    }

    return commandEncoder_;
}

// Synchronization with SyncType - matches PyTorch's approach (no dispatch_sync here)
void ETMetalStream::synchronize(SyncType syncType) {
    endKernelCoalescing();

    switch (syncType) {
        case SyncType::NONE:
            // Do nothing - no commit
            break;
        case SyncType::COMMIT:
            commit();
            break;
        case SyncType::COMMIT_AND_WAIT:
            commitAndWait();
            break;
        case SyncType::COMMIT_AND_CONTINUE:
            if (enableCommitAndContinue_) {
                commitAndContinue();
            } else {
                ET_LOG(Error, "ETMetalStream::synchronize: CommitAndContinue requested but disabled");
                commit();
            }
            break;
        case SyncType::COMMIT_ADAPTIVE:
            // Simple adaptive policy - could be enhanced with memory pressure detection
            // TODO: Could add memory pressure detection like PyTorch does
            commit();
            break;
    }

    ET_LOG(Debug, "ETMetalStream::synchronize: Completed with SyncType %d", static_cast<int>(syncType));
}

// Encoder coalescing management
void ETMetalStream::endKernelCoalescing() {
    if (commandEncoder_) {
        [commandEncoder_ endEncoding];
        [commandEncoder_ release];
        commandEncoder_ = nil;
        ET_LOG(Debug, "ETMetalStream::endKernelCoalescing: Ended encoder coalescing");
    }
}

// Dispatch pipelining
void ETMetalStream::setFlushInterval(int interval) {
    flushInterval_ = interval;
    dispatchCount_ = 0;
    ET_LOG(Info, "ETMetalStream: flush interval set to %d dispatches", interval);
}

void ETMetalStream::notifyDispatch() {
    if (!enableCommitAndContinue_ || flushInterval_ <= 0) {
        return;
    }
    dispatchCount_++;
    if (dispatchCount_ >= flushInterval_) {
        dispatchCount_ = 0;
        // Submit current work to GPU via commitAndContinue. The command buffer
        // stays alive for further encoding, enabling pipelined execution.
        endKernelCoalescing();
        if (commandBuffer_) {
            [commandBuffer_ commitAndContinue];
        }
    }
}

// Commit methods
void ETMetalStream::commit() {
    if (!commandBuffer_) {
        ET_LOG(Error, "ETMetalStream::commit: No command buffer to commit");
        return;
    }

    [commandBuffer_ commit];
    ET_LOG(Debug, "ETMetalStream::commit: Committed buffer %p", commandBuffer_);

    [commandBuffer_ release];
    commandBuffer_ = nil;
    dispatchCount_ = 0;
}

void ETMetalStream::commitAndWait() {
    // Handle previous command buffer first
    if (prevCommandBuffer_) {
        [prevCommandBuffer_ waitUntilCompleted];
        [prevCommandBuffer_ release];
        prevCommandBuffer_ = nil;
    }

    // Commit the final batch and wait for all work (including prior
    // commitAndContinue batches) to complete on the GPU.
    if (commandBuffer_) {
        [commandBuffer_ commit];
        [commandBuffer_ waitUntilCompleted];
        [commandBuffer_ release];
        commandBuffer_ = nil;
    }

    dispatchCount_ = 0;
    ET_LOG(Debug, "ETMetalStream::commitAndWait: Committed and waited for completion");
}

void ETMetalStream::commitAndContinue() {
    if (!commandBuffer_) {
        ET_LOG(Error, "ETMetalStream::commitAndContinue: No command buffer to commit");
        return;
    }

    // Commit buffer and allow immediate reuse for better performance
    [commandBuffer_ commit];
    ET_LOG(Debug, "ETMetalStream::commitAndContinue: Committed buffer %p with continue", commandBuffer_);

    // The buffer handles synchronization internally for commit-and-continue
}

void ETMetalStream::flush() {
    if (commandBuffer_) {
        [commandBuffer_ commit];

        if (!enableCommitAndContinue_) {
            // Keep the command buffer for later waiting if commit-and-continue is disabled
            prevCommandBuffer_ = commandBuffer_;
        } else {
            [commandBuffer_ release];
        }
        commandBuffer_ = nil;
        dispatchCount_ = 0;

        ET_LOG(Debug, "ETMetalStream::flush: Flushed command buffer");
    }
}

// Memory operations
void ETMetalStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType) {
    if (length == 0) {
        return;
    }

    dispatch_sync(serialQueue_, ^{
        @autoreleasepool {
            endKernelCoalescing();
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

            [blitEncoder fillBuffer:buffer range:NSMakeRange(offset, length) value:value];
            [blitEncoder endEncoding];
            synchronize(syncType);

            ET_LOG(Debug, "ETMetalStream::fill: Filled buffer with value %u, length %zu, offset %zu", value, length, offset);
        }
    });
}

void ETMetalStream::copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
                        size_t srcOffset, size_t dstOffset, SyncType syncType) {

    if (length == 0) {
        return;
    }

    // Check that offsets are within buffer bounds before copying
    if (!srcBuffer || !dstBuffer) {
        ET_LOG(Error, "ETMetalStream::copy: Source or destination buffer is nil");
        return;
    }
    NSUInteger srcBufferLength = [srcBuffer length];
    NSUInteger dstBufferLength = [dstBuffer length];
    if (srcOffset + length > srcBufferLength) {
        ET_LOG(Error, "ETMetalStream::copy: Source offset (%zu) + length (%zu) exceeds source buffer size (%zu)", srcOffset, length, srcBufferLength);
        return;
    }
    if (dstOffset + length > dstBufferLength) {
        ET_LOG(Error, "ETMetalStream::copy: Destination offset (%zu) + length (%zu) exceeds destination buffer size (%zu)", dstOffset, length, dstBufferLength);
        return;
    }

    dispatch_sync(serialQueue_, ^{
        @autoreleasepool {
            endKernelCoalescing();
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

            // Handle large copies in chunks
            constexpr size_t max_copy_size = 0x80000000; // 2GB
            size_t bytes_copied = 0;
            size_t bytes_remaining = length;

            while (bytes_remaining > 0) {
                NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remaining);
                [blitEncoder copyFromBuffer:srcBuffer
                               sourceOffset:(NSUInteger)srcOffset + bytes_copied
                                   toBuffer:dstBuffer
                          destinationOffset:(NSUInteger)dstOffset + bytes_copied
                                       size:bytes_to_copy];
                bytes_copied += bytes_to_copy;
                bytes_remaining -= bytes_to_copy;
            }

            [blitEncoder endEncoding];
            synchronize(syncType);

            ET_LOG(Debug, "ETMetalStream::copy: Copied %zu bytes from offset %zu to offset %zu", length, srcOffset, dstOffset);
        }
    });
}


void ETMetalStream::synchronize() {
    synchronize(SyncType::COMMIT_AND_WAIT);
}

bool ETMetalStream::isEmpty() const {
    return !commandBuffer_ && !commandEncoder_;
}

void ETMetalStream::executeMPSGraph(
    MPSGraph* mpsGraph,
    NSDictionary* feeds,
    NSDictionary* results,
    SyncType syncType,
    bool settle_aliases) {
    // Use dispatch_sync_with_rethrow exactly like PyTorch does for MPSGraph execution
    dispatch_sync_with_rethrow(serialQueue_, ^() {
        @autoreleasepool {
            // An alias (see get_mtl_buffer) is a separate MTLBuffer over memory
            // another buffer covers, and Metal orders nothing between the two.
            // Settle that memory on both sides of this graph, all within this
            // block so that no other work on the stream can come in between.
            if (settle_aliases) {
                synchronize(SyncType::COMMIT_AND_WAIT);
            }
            endKernelCoalescing();

            [mpsGraph encodeToCommandBuffer:commandBuffer()
                                      feeds:feeds
                           targetOperations:nil
                          resultsDictionary:results
                        executionDescriptor:nil];

            if (settle_aliases) {
                synchronize(SyncType::COMMIT_AND_WAIT);
            }
        }
    });
}

// =======================
// Global Storage Management Functions
// =======================

void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr) {
    function_storage[raw_function] = function_shared_ptr;
}

void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library) {
    library_storage[raw_library] = std::move(library);
}

bool removeFunctionHandle(ETMetalKernelFunction* raw_function) {
    auto it = function_storage.find(raw_function);
    if (it != function_storage.end()) {
        function_storage.erase(it);
        return true;
    }
    return false;
}

bool removeLibraryHandle(ETMetalShaderLibrary* raw_library) {
    auto it = library_storage.find(raw_library);
    if (it != library_storage.end()) {
        library_storage.erase(it);
        return true;
    }
    return false;
}

// =======================
// Global Stream Access Functions
// =======================

ETMetalStream* getCurrentMetalStream() {
    if (!currentStream_) {
        currentStream_ = ETMetalStream::getDefaultStream();
    }
    return currentStream_;
}

void setCurrentMetalStream(ETMetalStream* stream) {
    currentStream_ = stream;
}

// =======================
// Metal Stream Synchronization Functions
// =======================

void synchronize_metal_stream() {
    @autoreleasepool {
        // Use the ETMetalStream for proper synchronization
        ETMetalStream* stream = getCurrentMetalStream();
        stream->synchronize(SyncType::COMMIT_AND_WAIT);

        ET_LOG(Debug, "synchronize_metal_stream: Stream synchronized with COMMIT_AND_WAIT");
    }
}

void synchronize_metal_stream_with_type(int sync_type) {
    @autoreleasepool {
        ETMetalStream* stream = getCurrentMetalStream();
        SyncType syncTypeEnum = static_cast<SyncType>(sync_type);
        stream->synchronize(syncTypeEnum);

        ET_LOG(Debug, "synchronize_metal_stream_with_type: Stream synchronized with SyncType %d", sync_type);
    }
}

} // namespace metal
} // namespace backends
} // namespace executorch
