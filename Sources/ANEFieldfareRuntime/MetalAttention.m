#import "MetalAttention.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t queryHeads, keyValueHeads, headDimension, rotaryAngles;
    uint32_t positionCount, firstPosition, capacity, slidingWindow, fullAttention;
    float ropeTheta, epsilon, attentionScale;
} MetalAttentionParams;

@interface ANEFieldfareMetalAttentionImpl : NSObject
@property(nonatomic) id<MTLDevice> device;
@property(nonatomic) id<MTLCommandQueue> queue;
@property(nonatomic) id<MTLComputePipelineState> prepare;
@property(nonatomic) id<MTLComputePipelineState> prepareDirect;
@property(nonatomic) id<MTLComputePipelineState> attend;
@property(nonatomic) id<MTLComputePipelineState> packDirect;
@property(nonatomic) id<MTLBuffer> queryInput;
@property(nonatomic) id<MTLBuffer> keyInput;
@property(nonatomic) id<MTLBuffer> valueInput;
@property(nonatomic) id<MTLBuffer> queryPrepared;
@property(nonatomic) id<MTLBuffer> combined;
@property(nonatomic) id<MTLBuffer> queryNorm;
@property(nonatomic) id<MTLBuffer> keyNorm;
@property(nonatomic) NSString *lastError;
@property(nonatomic) ANEFieldfareMetalAttentionInfo info;
@end
@implementation ANEFieldfareMetalAttentionImpl
@end

static double milliseconds(uint64_t ticks) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1.0e6;
}

static void setError(ANEFieldfareMetalAttentionImpl *impl, NSString *message) {
    if (!impl.lastError.length) impl.lastError = message;
}

static NSUInteger roundedPageBytes(NSUInteger bytes) {
    NSUInteger page = (NSUInteger)getpagesize();
    return (bytes + page - 1) / page * page;
}

static const char *copyError(NSString *message) {
    return strdup((message.length ? message : @"unknown Metal attention error").UTF8String);
}

ANEFieldfareMetalAttention *ANEFieldfareMetalAttentionCreate(
    const char *metalSourcePath, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!metalSourcePath) return NULL;
        ANEFieldfareMetalAttentionImpl *impl = [ANEFieldfareMetalAttentionImpl new];
        impl.device = MTLCreateSystemDefaultDevice();
        impl.queue = [impl.device newCommandQueue];
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:
            [NSString stringWithUTF8String:metalSourcePath]
            encoding:NSUTF8StringEncoding error:&error];
        id<MTLLibrary> library = source ? [impl.device newLibraryWithSource:source
            options:nil error:&error] : nil;
        id<MTLFunction> prepare = [library newFunctionWithName:@"attention_prepare_f32"];
        id<MTLFunction> prepareDirect = [library newFunctionWithName:@"attention_prepare_ane_tile"];
        id<MTLFunction> attend = [library newFunctionWithName:@"attention_causal_f32"];
        id<MTLFunction> packDirect = [library newFunctionWithName:@"attention_pack_ane_tile"];
        impl.prepare = prepare ? [impl.device newComputePipelineStateWithFunction:prepare error:&error] : nil;
        impl.prepareDirect = prepareDirect ? [impl.device newComputePipelineStateWithFunction:prepareDirect error:&error] : nil;
        impl.attend = attend ? [impl.device newComputePipelineStateWithFunction:attend error:&error] : nil;
        impl.packDirect = packDirect ? [impl.device newComputePipelineStateWithFunction:packDirect error:&error] : nil;
        const NSUInteger q = 64ull * 16 * 512 * sizeof(float);
        const NSUInteger kv = 64ull * 8 * 256 * sizeof(float);
        impl.queryInput = [impl.device newBufferWithLength:q options:MTLResourceStorageModeShared];
        impl.keyInput = [impl.device newBufferWithLength:kv options:MTLResourceStorageModeShared];
        impl.valueInput = [impl.device newBufferWithLength:kv options:MTLResourceStorageModeShared];
        impl.queryPrepared = [impl.device newBufferWithLength:q options:MTLResourceStorageModeShared];
        impl.combined = [impl.device newBufferWithLength:q options:MTLResourceStorageModeShared];
        impl.queryNorm = [impl.device newBufferWithLength:512 * sizeof(float)
            options:MTLResourceStorageModeShared];
        impl.keyNorm = [impl.device newBufferWithLength:512 * sizeof(float)
            options:MTLResourceStorageModeShared];
        if (!impl.device || !impl.queue || !library || !impl.prepare || !impl.prepareDirect ||
            !impl.attend || !impl.packDirect ||
            !impl.queryInput || !impl.keyInput || !impl.valueInput || !impl.queryPrepared ||
            !impl.combined || !impl.queryNorm || !impl.keyNorm) {
            setError(impl, error.localizedDescription ?: @"Metal attention construction failed");
            if (errorMessage) *errorMessage = strdup(impl.lastError.UTF8String);
            return NULL;
        }
        return (__bridge_retained ANEFieldfareMetalAttention *)impl;
    }
}

bool ANEFieldfareMetalAttentionCall(
    ANEFieldfareMetalAttention *raw,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *query,
    const float *key,
    const float *value,
    float *combined) {
    @autoreleasepool {
        ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
        if (!impl || !weights || !cache || !query || !key || !value || !combined ||
            !positionCount || positionCount > 64 || firstPosition > UINT32_MAX ||
            (firstPosition + positionCount > cache->capacity &&
             cache->descriptor.kind == ANEFieldfareAttentionFull)) return false;
        const ANEFieldfareAttentionDescriptor *d = &weights->descriptor;
        uint32_t queryWidth = d->queryHeads * d->headDimension;
        uint32_t keyWidth = d->keyValueHeads * d->headDimension;
        NSUInteger queryBytes = (NSUInteger)positionCount * queryWidth * sizeof(float);
        NSUInteger keyBytes = (NSUInteger)positionCount * keyWidth * sizeof(float);
        NSUInteger cacheBytes = (NSUInteger)cache->capacity * keyWidth * sizeof(float);
        if (((uintptr_t)cache->keys % (uintptr_t)getpagesize()) ||
            ((uintptr_t)cache->values % (uintptr_t)getpagesize())) {
            setError(impl, @"KV cache is not page aligned");
            return false;
        }
        id<MTLBuffer> keys = [impl.device newBufferWithBytesNoCopy:cache->keys
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> values = [impl.device newBufferWithBytesNoCopy:cache->values
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        if (!keys || !values) {
            setError(impl, @"Metal rejected page-aligned KV cache");
            return false;
        }
        memcpy(impl.queryInput.contents, query, queryBytes);
        memcpy(impl.keyInput.contents, key, keyBytes);
        memcpy(impl.valueInput.contents, value, keyBytes);
        memcpy(impl.queryNorm.contents, weights->queryNormValues,
               d->headDimension * sizeof(float));
        memcpy(impl.keyNorm.contents, weights->keyNormValues,
               d->headDimension * sizeof(float));
        for (uint32_t row = 0; row < positionCount; ++row) {
            uint64_t absolute = firstPosition + row;
            uint64_t slot = d->kind == ANEFieldfareAttentionFull
                ? absolute : absolute % cache->capacity;
            cache->positions[slot] = absolute;
        }
        MetalAttentionParams params = {
            d->queryHeads, d->keyValueHeads, d->headDimension, d->rotaryAngles,
            positionCount, (uint32_t)firstPosition, (uint32_t)cache->capacity,
            d->slidingWindow, d->kind == ANEFieldfareAttentionFull,
            d->ropeTheta, d->rmsEpsilon, d->attentionScale,
        };
        uint64_t started = mach_absolute_time();
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl.prepare];
        [encoder setBuffer:impl.queryInput offset:0 atIndex:0];
        [encoder setBuffer:impl.keyInput offset:0 atIndex:1];
        [encoder setBuffer:impl.valueInput offset:0 atIndex:2];
        [encoder setBuffer:impl.queryNorm offset:0 atIndex:3];
        [encoder setBuffer:impl.keyNorm offset:0 atIndex:4];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:5];
        [encoder setBuffer:keys offset:0 atIndex:6];
        [encoder setBuffer:values offset:0 atIndex:7];
        [encoder setBytes:&params length:sizeof(params) atIndex:8];
        NSUInteger prepareThreads = (NSUInteger)positionCount *
            (d->queryHeads + d->keyValueHeads);
        [encoder dispatchThreads:MTLSizeMake(prepareThreads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                impl.prepare.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder setComputePipelineState:impl.attend];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:0];
        [encoder setBuffer:keys offset:0 atIndex:1];
        [encoder setBuffer:values offset:0 atIndex:2];
        [encoder setBuffer:impl.combined offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        uint64_t maximumCausal = firstPosition + positionCount;
        if (d->kind == ANEFieldfareAttentionSliding && maximumCausal > d->slidingWindow)
            maximumCausal = d->slidingWindow;
        [encoder setThreadgroupMemoryLength:(NSUInteger)maximumCausal * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(positionCount, d->queryHeads, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            setError(impl, command.error.localizedDescription ?: @"Metal attention failed");
            return false;
        }
        memcpy(combined, impl.combined.contents, queryBytes);
        ANEFieldfareMetalAttentionInfo info = impl.info;
        info.dispatchCount += 1;
        info.positionCount += positionCount;
        info.hostWaitCount += 1;
        info.wallMilliseconds += milliseconds(mach_absolute_time() - started);
        info.gpuMilliseconds += (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        impl.info = info;
        return true;
    }
}

bool ANEFieldfareMetalAttentionCallProjected(
    ANEFieldfareMetalAttention *raw,
    void *affineContext,
    ANEFieldfareExternalAffineEncode encodeAffine,
    ANEFieldfareExternalAffineComplete completeAffine,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    const uint8_t *tensorIndexBase,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *query,
    const float *key,
    const float *value,
    float *projected) {
    @autoreleasepool {
        ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
        if (!impl || !affineContext || !encodeAffine || !completeAffine || !weights ||
            !cache || !tensorIndexBase || !query ||
            !key || !value || !projected || !positionCount || positionCount != 1 ||
            firstPosition > UINT32_MAX) return false;
        const ANEFieldfareAttentionDescriptor *d = &weights->descriptor;
        uint32_t queryWidth = d->queryHeads * d->headDimension;
        uint32_t keyWidth = d->keyValueHeads * d->headDimension;
        NSUInteger queryBytes = (NSUInteger)queryWidth * sizeof(float);
        NSUInteger keyBytes = (NSUInteger)keyWidth * sizeof(float);
        NSUInteger cacheBytes = (NSUInteger)cache->capacity * keyWidth * sizeof(float);
        if (((uintptr_t)cache->keys % (uintptr_t)getpagesize()) ||
            ((uintptr_t)cache->values % (uintptr_t)getpagesize())) return false;
        id<MTLBuffer> keys = [impl.device newBufferWithBytesNoCopy:cache->keys
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> values = [impl.device newBufferWithBytesNoCopy:cache->values
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        if (!keys || !values) return false;
        memcpy(impl.queryInput.contents, query, queryBytes);
        memcpy(impl.keyInput.contents, key, keyBytes);
        memcpy(impl.valueInput.contents, value, keyBytes);
        memcpy(impl.queryNorm.contents, weights->queryNormValues,
               d->headDimension * sizeof(float));
        memcpy(impl.keyNorm.contents, weights->keyNormValues,
               d->headDimension * sizeof(float));
        uint64_t slot = d->kind == ANEFieldfareAttentionFull
            ? firstPosition : firstPosition % cache->capacity;
        cache->positions[slot] = firstPosition;
        MetalAttentionParams params = {
            d->queryHeads, d->keyValueHeads, d->headDimension, d->rotaryAngles,
            1, (uint32_t)firstPosition, (uint32_t)cache->capacity,
            d->slidingWindow, d->kind == ANEFieldfareAttentionFull,
            d->ropeTheta, d->rmsEpsilon, d->attentionScale,
        };
        uint64_t started = mach_absolute_time();
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl.prepare];
        [encoder setBuffer:impl.queryInput offset:0 atIndex:0];
        [encoder setBuffer:impl.keyInput offset:0 atIndex:1];
        [encoder setBuffer:impl.valueInput offset:0 atIndex:2];
        [encoder setBuffer:impl.queryNorm offset:0 atIndex:3];
        [encoder setBuffer:impl.keyNorm offset:0 atIndex:4];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:5];
        [encoder setBuffer:keys offset:0 atIndex:6];
        [encoder setBuffer:values offset:0 atIndex:7];
        [encoder setBytes:&params length:sizeof(params) atIndex:8];
        [encoder dispatchThreads:MTLSizeMake(d->queryHeads + d->keyValueHeads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                impl.prepare.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder setComputePipelineState:impl.attend];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:0];
        [encoder setBuffer:keys offset:0 atIndex:1];
        [encoder setBuffer:values offset:0 atIndex:2];
        [encoder setBuffer:impl.combined offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        uint64_t maximumCausal = firstPosition + 1;
        if (d->kind == ANEFieldfareAttentionSliding && maximumCausal > d->slidingWindow)
            maximumCausal = d->slidingWindow;
        [encoder setThreadgroupMemoryLength:(NSUInteger)maximumCausal * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1, d->queryHeads, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        CFTypeRef outputBuffer = NULL;
        if (!encodeAffine(
                affineContext, tensorIndexBase, &weights->outputProjection,
                (__bridge CFTypeRef)command, (__bridge CFTypeRef)impl.combined,
                &outputBuffer)) return false;
        [command commit];
        [command waitUntilCompleted];
        bool valid = completeAffine(
            affineContext, (__bridge CFTypeRef)command, outputBuffer, started, projected);
        if (!valid) return false;
        ANEFieldfareMetalAttentionInfo info = impl.info;
        ++info.dispatchCount;
        ++info.positionCount;
        ++info.hostWaitCount;
        info.wallMilliseconds += milliseconds(mach_absolute_time() - started);
        info.gpuMilliseconds += (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        impl.info = info;
        return true;
    }
}

bool ANEFieldfareMetalAttentionEncodeDirect(
    ANEFieldfareMetalAttention *raw,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    IOSurfaceRef qkvOutput,
    uint32_t queryChannels,
    uint32_t keyChannels,
    bool valueStacked,
    IOSurfaceRef outputProjectionInput,
    CFTypeRef metalSharedEvent,
    uint64_t waitValue,
    uint64_t signalValue,
    CFTypeRef *commandBuffer,
    const char **errorMessage) {
    @autoreleasepool {
        if (commandBuffer) *commandBuffer = NULL;
        if (errorMessage) *errorMessage = NULL;
        ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
        if (!impl || !weights || !cache || !qkvOutput || !outputProjectionInput ||
            !metalSharedEvent || !waitValue || signalValue <= waitValue || !commandBuffer ||
            !positionCount || positionCount > 64 || firstPosition > UINT32_MAX) {
            if (errorMessage) *errorMessage = copyError(@"invalid direct attention arguments");
            return false;
        }
        const ANEFieldfareAttentionDescriptor *d = &weights->descriptor;
        uint32_t expectedQuery = d->queryHeads * d->headDimension;
        uint32_t expectedKey = d->keyValueHeads * d->headDimension;
        if (queryChannels != expectedQuery || keyChannels != expectedKey) {
            if (errorMessage) *errorMessage = copyError(@"direct attention channel mismatch");
            return false;
        }
        NSUInteger page = (NSUInteger)getpagesize();
        void *qkvBase = IOSurfaceGetBaseAddress(qkvOutput);
        void *projectionBase = IOSurfaceGetBaseAddress(outputProjectionInput);
        NSUInteger qkvBytes = IOSurfaceGetAllocSize(qkvOutput);
        NSUInteger projectionBytes = IOSurfaceGetAllocSize(outputProjectionInput);
        NSUInteger cacheBytes = (NSUInteger)cache->capacity * expectedKey * sizeof(float);
        if (!qkvBase || !projectionBase || ((uintptr_t)qkvBase % page) ||
            ((uintptr_t)projectionBase % page) || ((uintptr_t)cache->keys % page) ||
            ((uintptr_t)cache->values % page) || qkvBytes % page || projectionBytes % page) {
            if (errorMessage) *errorMessage = copyError(@"direct attention surface/cache alignment failed");
            return false;
        }
        id<MTLBuffer> qkv = [impl.device newBufferWithBytesNoCopy:qkvBase length:qkvBytes
            options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> projection = [impl.device newBufferWithBytesNoCopy:projectionBase
            length:projectionBytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> keys = [impl.device newBufferWithBytesNoCopy:cache->keys
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> values = [impl.device newBufferWithBytesNoCopy:cache->values
            length:roundedPageBytes(cacheBytes) options:MTLResourceStorageModeShared deallocator:nil];
        if (!qkv || !projection || !keys || !values ||
            qkv.contents != qkvBase || projection.contents != projectionBase) {
            if (errorMessage) *errorMessage = copyError(@"direct attention Metal buffer guard failed");
            return false;
        }
        memcpy(impl.queryNorm.contents, weights->queryNormValues,
               d->headDimension * sizeof(float));
        memcpy(impl.keyNorm.contents, weights->keyNormValues,
               d->headDimension * sizeof(float));
        for (uint32_t row = 0; row < positionCount; ++row) {
            uint64_t absolute = firstPosition + row;
            uint64_t slot = d->kind == ANEFieldfareAttentionFull
                ? absolute : absolute % cache->capacity;
            cache->positions[slot] = absolute;
        }
        MetalAttentionParams params = {
            d->queryHeads, d->keyValueHeads, d->headDimension, d->rotaryAngles,
            positionCount, (uint32_t)firstPosition, (uint32_t)cache->capacity,
            d->slidingWindow, d->kind == ANEFieldfareAttentionFull,
            d->ropeTheta, d->rmsEpsilon, d->attentionScale,
        };
        uint32_t stacked = valueStacked ? 1 : 0;
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)metalSharedEvent;
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        [command encodeWaitForEvent:event value:waitValue];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl.prepareDirect];
        [encoder setBuffer:qkv offset:0 atIndex:0];
        [encoder setBuffer:impl.queryNorm offset:0 atIndex:1];
        [encoder setBuffer:impl.keyNorm offset:0 atIndex:2];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:3];
        [encoder setBuffer:keys offset:0 atIndex:4];
        [encoder setBuffer:values offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(params) atIndex:6];
        [encoder setBytes:&queryChannels length:sizeof(queryChannels) atIndex:7];
        [encoder setBytes:&keyChannels length:sizeof(keyChannels) atIndex:8];
        [encoder setBytes:&stacked length:sizeof(stacked) atIndex:9];
        NSUInteger prepareThreads = (NSUInteger)positionCount *
            (d->queryHeads + d->keyValueHeads);
        [encoder dispatchThreads:MTLSizeMake(prepareThreads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                impl.prepareDirect.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder setComputePipelineState:impl.attend];
        [encoder setBuffer:impl.queryPrepared offset:0 atIndex:0];
        [encoder setBuffer:keys offset:0 atIndex:1];
        [encoder setBuffer:values offset:0 atIndex:2];
        [encoder setBuffer:impl.combined offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        uint64_t maximumCausal = firstPosition + positionCount;
        if (d->kind == ANEFieldfareAttentionSliding && maximumCausal > d->slidingWindow)
            maximumCausal = d->slidingWindow;
        [encoder setThreadgroupMemoryLength:(NSUInteger)maximumCausal * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(positionCount, d->queryHeads, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:impl.packDirect];
        [encoder setBuffer:impl.combined offset:0 atIndex:0];
        [encoder setBuffer:projection offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)expectedQuery * 64, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                impl.packDirect.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding];
        [command encodeSignalEvent:event value:signalValue];
        [command commit];
        ANEFieldfareMetalAttentionInfo info = impl.info;
        info.positionCount += positionCount;
        info.directEventPositionCount += positionCount;
        impl.info = info;
        *commandBuffer = CFBridgingRetain(command);
        return true;
    }
}

bool ANEFieldfareMetalAttentionCompleteDirect(
    ANEFieldfareMetalAttention *raw,
    CFTypeRef commandBuffer,
    uint64_t startedTicks,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
        if (!impl || !commandBuffer) return false;
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)commandBuffer;
        [command waitUntilCompleted];
        bool valid = command.status == MTLCommandBufferStatusCompleted;
        if (valid) {
            ANEFieldfareMetalAttentionInfo info = impl.info;
            info.dispatchCount += 1;
            info.directEventDispatchCount += 1;
            info.directEventHostWaitCount += 1;
            info.directEventWallMilliseconds += milliseconds(mach_absolute_time() - startedTicks);
            info.gpuMilliseconds += (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            impl.info = info;
        } else if (errorMessage) {
            *errorMessage = copyError(command.error.localizedDescription ?: @"direct Metal attention failed");
        }
        CFRelease(commandBuffer);
        return valid;
    }
}

ANEFieldfareMetalAttentionInfo ANEFieldfareMetalAttentionGetInfo(
    const ANEFieldfareMetalAttention *raw) {
    ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
    return impl ? impl.info : (ANEFieldfareMetalAttentionInfo){0};
}

const char *ANEFieldfareMetalAttentionLastError(const ANEFieldfareMetalAttention *raw) {
    ANEFieldfareMetalAttentionImpl *impl = (__bridge ANEFieldfareMetalAttentionImpl *)raw;
    return impl.lastError.UTF8String ?: "none";
}

void ANEFieldfareMetalAttentionDestroy(ANEFieldfareMetalAttention *raw) {
    if (raw) CFRelease(raw);
}
