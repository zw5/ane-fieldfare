#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <math.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

enum { Hidden = 2816, Outputs = 128, Positions = 64, Repetitions = 8 };

static double milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)ticks * info.numer / info.denom / 1e6;
}

static bool initializeWeights(__fp16 *carrier,
                              const ANEFieldfareTensor *weights,
                              uint32_t stride) {
    memset(carrier, 0, (size_t)Hidden * stride * sizeof(*carrier));
    for (uint32_t column = 0; column < Hidden; ++column)
        for (uint32_t row = 0; row < Outputs; ++row) {
            float value = 0;
            if (!ANEFieldfareAffineValue(weights, 8, 64, row, column, &value)) return false;
            carrier[(uint64_t)column * stride + Positions + row] = (__fp16)value;
        }
    return true;
}

static void cpuFillActivation(__fp16 *carrier, uint32_t stride, float scale) {
    for (uint32_t column = 0; column < Hidden; ++column) {
        int32_t centered = (int32_t)(column % 257) - 128;
        carrier[(uint64_t)column * stride] = (__fp16)((float)centered * scale);
    }
}

static void poisonOutput(IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    float *output = IOSurfaceGetBaseAddress(surface);
    for (uint32_t index = 0; index < Outputs * Positions; ++index) output[index] = NAN;
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSDictionary *validateOutput(IOSurfaceRef inputSurface,
                                    IOSurfaceRef outputSurface,
                                    uint32_t stride) {
    IOSurfaceLock(inputSurface, kIOSurfaceLockReadOnly, NULL);
    const __fp16 *carrier = IOSurfaceGetBaseAddress(inputSurface);
    float expected[Outputs] = {0};
    for (uint32_t row = 0; row < Outputs; ++row) {
        double sum = 0;
        for (uint32_t column = 0; column < Hidden; ++column)
            sum += (float)carrier[(uint64_t)column * stride] *
                   (float)carrier[(uint64_t)column * stride + Positions + row];
        expected[row] = (float)sum;
    }
    IOSurfaceUnlock(inputSurface, kIOSurfaceLockReadOnly, NULL);

    IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    const float *output = IOSurfaceGetBaseAddress(outputSurface);
    double maxError = 0, squared = 0;
    uint32_t failures = 0, poisonRemaining = 0, inactiveFailures = 0;
    for (uint32_t row = 0; row < Outputs; ++row) {
        float actual = output[(uint64_t)row * Positions];
        if (isnan(actual)) ++poisonRemaining;
        if (!isfinite(actual)) { ++failures; continue; }
        double error = fabs((double)actual - expected[row]);
        if (error > maxError) maxError = error;
        squared += error * error;
        if (error > 0.03 + 0.02 * fabs(expected[row])) ++failures;
        for (uint32_t position = 1; position < Positions; ++position) {
            float inactive = output[(uint64_t)row * Positions + position];
            if (isnan(inactive)) ++poisonRemaining;
            if (!isfinite(inactive) || fabsf(inactive) > 0.001f) ++inactiveFailures;
        }
    }
    IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    return @{
        @"valid": @(failures == 0 && inactiveFailures == 0 && poisonRemaining == 0),
        @"max_absolute_error": @(maxError), @"rmse": @(sqrt(squared / Outputs)),
        @"active_tolerance_failures": @(failures),
        @"inactive_lane_failures": @(inactiveFailures),
        @"poison_values_remaining": @(poisonRemaining)
    };
}

typedef NS_ENUM(uint32_t, ProducerKind) {
    ProducerCPU,
    ProducerMetalBlit,
    ProducerMetalDirect,
};

static NSDictionary *runArm(NSString *name,
                            ProducerKind kind,
                            ANEFieldfareANEProgram *program,
                            IOSurfaceRef inputSurface,
                            IOSurfaceRef outputSurface,
                            id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> aneInputBuffer,
                            id<MTLBuffer> separateBuffer,
                            const __fp16 *weightTemplate,
                            uint64_t inputBytes,
                            uint32_t stride,
                            bool *valid) {
    NSMutableArray *runs = [NSMutableArray array];
    for (uint32_t repetition = 0; repetition < Repetitions; ++repetition) {
        float scale = 0.0009765625f * (1.0f + 0.0625f * repetition);
        IOSurfaceLock(inputSurface, 0, NULL);
        __fp16 *carrier = IOSurfaceGetBaseAddress(inputSurface);
        memcpy(carrier, weightTemplate, inputBytes);
        IOSurfaceUnlock(inputSurface, 0, NULL);
        if (kind == ProducerMetalBlit) memcpy(separateBuffer.contents, weightTemplate, inputBytes);
        poisonOutput(outputSurface);

        uint64_t fullStarted = mach_absolute_time();
        double producerWall = 0, producerGPU = 0;
        bool producerValid = true;
        if (kind == ProducerCPU) {
            IOSurfaceLock(inputSurface, 0, NULL);
            cpuFillActivation(IOSurfaceGetBaseAddress(inputSurface), stride, scale);
            IOSurfaceUnlock(inputSurface, 0, NULL);
            producerWall = milliseconds(mach_absolute_time() - fullStarted);
        } else {
            uint64_t producerStarted = mach_absolute_time();
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:kind == ProducerMetalDirect ? aneInputBuffer : separateBuffer
                        offset:0 atIndex:0];
            [encoder setBytes:&stride length:sizeof(stride) atIndex:1];
            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(Hidden, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                                                     pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
            if (kind == ProducerMetalBlit) {
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                [blit copyFromBuffer:separateBuffer sourceOffset:0
                            toBuffer:aneInputBuffer destinationOffset:0 size:(NSUInteger)inputBytes];
                [blit endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            producerWall = milliseconds(mach_absolute_time() - producerStarted);
            producerGPU = command.GPUEndTime >= command.GPUStartTime
                ? (command.GPUEndTime - command.GPUStartTime) * 1000.0 : 0;
            producerValid = command.status == MTLCommandBufferStatusCompleted;
        }

        const char *aneError = NULL;
        ANEFieldfareANEEvaluationTiming aneTiming = {0};
        bool aneValid = producerValid && ANEFieldfareANEProgramEvaluate(
            program, 0, 5ULL * NSEC_PER_SEC, &aneTiming, &aneError);
        double fullWall = milliseconds(mach_absolute_time() - fullStarted);
        NSDictionary *oracle = aneValid ? validateOutput(inputSurface, outputSurface, stride) : @{};
        bool runValid = producerValid && aneValid && [oracle[@"valid"] boolValue];
        *valid = *valid && runValid;
        [runs addObject:@{
            @"repetition": @(repetition), @"scale": @(scale),
            @"warm_timing_sample": @(repetition > 0),
            @"producer_completed": @(producerValid), @"producer_wall_ms": @(producerWall),
            @"producer_gpu_ms": @(producerGPU),
            @"ane_enqueue_ms": @(aneTiming.enqueueMilliseconds),
            @"ane_completion_ms": @(aneTiming.completionMilliseconds),
            @"producer_start_to_ane_completion_ms": @(fullWall),
            @"oracle": oracle,
            @"error": aneError ? [NSString stringWithUTF8String:aneError] : @"none"
        }];
        free((void *)aneError);
    }
    NSArray *full = [runs valueForKey:@"producer_start_to_ane_completion_ms"];
    double sum = 0, warmFull = 0, warmProducer = 0, warmGPU = 0;
    for (NSNumber *value in full) sum += value.doubleValue;
    for (uint32_t repetition = 1; repetition < Repetitions; ++repetition) {
        NSDictionary *run = runs[repetition];
        warmFull += [run[@"producer_start_to_ane_completion_ms"] doubleValue];
        warmProducer += [run[@"producer_wall_ms"] doubleValue];
        warmGPU += [run[@"producer_gpu_ms"] doubleValue];
    }
    return @{
        @"name": name, @"repetitions": @(Repetitions), @"runs": runs,
        @"bytes_written_by_producer": kind == ProducerMetalBlit
            ? @(inputBytes) : @(Hidden * sizeof(__fp16)),
        @"explicit_transfer_bytes": kind == ProducerMetalBlit ? @(inputBytes) : @0,
        @"mean_producer_start_to_ane_completion_ms": @(sum / Repetitions),
        @"warm_repetitions": @(Repetitions - 1),
        @"warm_mean_producer_wall_ms": @(warmProducer / (Repetitions - 1)),
        @"warm_mean_producer_gpu_ms": @(warmGPU / (Repetitions - 1)),
        @"warm_mean_producer_start_to_ane_completion_ms": @(warmFull / (Repetitions - 1)),
        @"all_runs_valid": @(*valid)
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        const uint32_t stride = Positions + Outputs;
        const uint64_t inputBytes = (uint64_t)Hidden * stride * sizeof(__fp16);
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *path = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor router = {0};
        const char *creationError = NULL;
        bool valid = ANEFieldfareTensorIndexOpen(path.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(
                &index, "language_model.model.layers.0.router.proj.weight", &router);
        ANEFieldfareANEDynamicProjection *projection = valid
            ? ANEFieldfareANEDynamicProjectionCreate(Hidden, Outputs, 1, &creationError) : NULL;
        valid = valid && projection;
        ANEFieldfareANEProgram *program = ANEFieldfareANEDynamicProjectionProgram(projection);
        IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
        IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);

        __fp16 *weightTemplate = calloc(inputBytes / sizeof(__fp16), sizeof(__fp16));
        valid = valid && weightTemplate && initializeWeights(weightTemplate, &router, stride);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSString *source =
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "kernel void fill_activation(device half *carrier [[buffer(0)]], "
             "constant uint &stride [[buffer(1)]], constant float &scale [[buffer(2)]], "
             "uint gid [[thread_position_in_grid]]) { if (gid >= 2816) return; "
             "int centered = int(gid % 257) - 128; carrier[ulong(gid) * stride] = half(float(centered) * scale); }";
        NSError *metalError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&metalError];
        id<MTLFunction> function = [library newFunctionWithName:@"fill_activation"];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&metalError] : nil;

        IOSurfaceLock(inputSurface, 0, NULL);
        void *aneInputAddress = IOSurfaceGetBaseAddress(inputSurface);
        id<MTLBuffer> aneInputBuffer = [device newBufferWithBytesNoCopy:aneInputAddress
            length:(NSUInteger)inputBytes options:MTLResourceStorageModeShared deallocator:nil];
        IOSurfaceUnlock(inputSurface, 0, NULL);
        id<MTLBuffer> separateBuffer = [device newBufferWithLength:(NSUInteger)inputBytes
                                                           options:MTLResourceStorageModeShared];
        bool directSupported = aneInputBuffer != nil;
        valid = valid && device && queue && pipeline && separateBuffer && directSupported;

        bool cpuValid = valid, blitValid = valid, directValid = valid;
        NSDictionary *cpu = valid ? runArm(
            @"cpu_fill_to_ane", ProducerCPU, program, inputSurface, outputSurface,
            queue, pipeline, aneInputBuffer, separateBuffer, weightTemplate,
            inputBytes, stride, &cpuValid) : @{};
        NSDictionary *blit = valid ? runArm(
            @"metal_separate_buffer_full_blit_to_ane", ProducerMetalBlit, program,
            inputSurface, outputSurface, queue, pipeline, aneInputBuffer, separateBuffer,
            weightTemplate, inputBytes, stride, &blitValid) : @{};
        NSDictionary *direct = valid ? runArm(
            @"metal_direct_write_ane_iosurface_backing", ProducerMetalDirect, program,
            inputSurface, outputSurface, queue, pipeline, aneInputBuffer, separateBuffer,
            weightTemplate, inputBytes, stride, &directValid) : @{};
        valid = valid && cpuValid && blitValid && directValid;

        ANEFieldfareANEProgramInfo info = ANEFieldfareANEDynamicProjectionGetInfo(projection);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.gpu-ane-transfer.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"shape": @{ @"hidden": @(Hidden), @"outputs": @(Outputs),
                          @"positions": @(Positions), @"carrier_bytes": @(inputBytes),
                          @"activation_bytes": @(Hidden * sizeof(__fp16)) },
            @"lifecycle": @{ @"ane_compile_count": @1, @"ane_load_count": @1,
                              @"ane_compile_ms": @(info.compileMilliseconds),
                              @"ane_load_ms": @(info.loadMilliseconds),
                              @"metal_library_compile_count": @1,
                              @"metal_pipeline_load_count": @1 },
            @"direct_iosurface_backed_metal_buffer_supported": @(directSupported),
            @"synchronization": @{
                @"metal": @"commit plus waitUntilCompleted before ANE begin",
                @"ane": @"completion-handler semaphore with five-second timeout",
                @"full_timing": @"producer start through ANE completion and output visibility"
            },
            @"arms": @[cpu, blit, direct],
            @"metal_error": metalError.localizedDescription ?: @"none",
            @"error": creationError ? [NSString stringWithUTF8String:creationError] : @"none",
            @"claim_boundary": @"One loaded authentic layer-0 router projection consumed nontrivial FP16 activations produced by CPU, by Metal in a separate shared buffer followed by an explicit full-carrier blit, and by Metal writing directly into a shared MTLBuffer over the mapped ANE input IOSurface backing. Every repetition waited for Metal completion before ANE submission, timed producer start through ANE completion, poisoned the full output, and passed an independent CPU oracle. Direct shared-storage visibility is validated for this host and lifecycle; this does not establish overlap, shared-event signaling, zero-copy across arbitrary IOSurface layouts, full-layer execution, or generation."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        free((void *)creationError); free(weightTemplate);
        ANEFieldfareANEDynamicProjectionDestroy(projection);
        ANEFieldfareTensorIndexClose(&index);
        return valid ? 0 : 1;
    }
}
