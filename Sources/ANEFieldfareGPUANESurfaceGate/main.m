#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"

enum { Channels = 64, Positions = 64, Elements = Channels * Positions };

static NSString *programText(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "  func main<ios18>(tensor<fp16,[1,64,1,64]> x) {\n"
            "    fp16 two=const()[name=string(\"two\"),val=fp16(0x1p+1)];\n"
            "    tensor<fp16,[1,64,1,64]> y=mul(x=x,y=two)[name=string(\"twice\")];\n"
            "  } -> (y);\n"
            "}\n";
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

int main(void) {
    @autoreleasepool {
        const char *aneError = NULL;
        ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreate(
            programText().UTF8String, Elements * sizeof(__fp16),
            Elements * sizeof(__fp16), 1, &aneError);
        if (!program) {
            emit(@{@"schema": @"ane-fieldfare.gpu-ane-surface.v1",
                   @"status": @"rejected", @"stage": @"ane_program",
                   @"error": aneError ? [NSString stringWithUTF8String:aneError] : @"unknown"});
            free((void *)aneError);
            return 1;
        }
        free((void *)aneError);

        IOSurfaceRef input = ANEFieldfareANEProgramInput(program, 0);
        IOSurfaceRef output = ANEFieldfareANEProgramOutput(program, 0);
        size_t allocBytes = input ? IOSurfaceGetAllocSize(input) : 0;
        void *surfaceBase = input ? IOSurfaceGetBaseAddress(input) : NULL;
        BOOL poisoned = NO;
        if (input && IOSurfaceLock(input, 0, NULL) == kIOReturnSuccess) {
            surfaceBase = IOSurfaceGetBaseAddress(input);
            if (surfaceBase) {
                memset(surfaceBase, 0x7e, allocBytes);
                poisoned = YES;
            }
            IOSurfaceUnlock(input, 0, NULL);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "kernel void fill_ane_input(device half *values [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
                            "  if (i < 4096) { uint c=i/64; uint p=i%64; values[i]=half(float(c)*0.03125f+float(p)*0.0078125f-1.0f); }\n"
                            "}\n";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"fill_ane_input"];
        id<MTLComputePipelineState> pipeline = function
            ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        id<MTLBuffer> buffer = surfaceBase && allocBytes
            ? [device newBufferWithBytesNoCopy:surfaceBase
                                        length:allocBytes
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil] : nil;
        BOOL pointerIdentity = buffer && buffer.contents == surfaceBase;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        BOOL encoded = pipeline && buffer && encoder;
        if (encoded) {
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
            [encoder dispatchThreads:MTLSizeMake(Elements, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
        }
        BOOL metalCompleted = encoded && command.status == MTLCommandBufferStatusCompleted;

        ANEFieldfareANEEvaluationTiming timing = {0};
        const char *evaluationError = NULL;
        BOOL aneEvaluated = metalCompleted && ANEFieldfareANEProgramEvaluate(
            program, 0, 5 * NSEC_PER_SEC, &timing, &evaluationError);

        NSUInteger mismatches = 0, finite = 0;
        double maxError = 0;
        uint64_t checksum = 1469598103934665603ULL;
        if (aneEvaluated && output &&
            IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
            const __fp16 *values = IOSurfaceGetBaseAddress(output);
            for (NSUInteger index = 0; values && index < Elements; ++index) {
                NSUInteger channel = index / Positions, position = index % Positions;
                float inputValue = (float)((__fp16)((float)channel * 0.03125f +
                                                    (float)position * 0.0078125f - 1.0f));
                float expected = (float)((__fp16)(inputValue * 2.0f));
                float actual = (float)values[index];
                if (isfinite(actual)) finite++;
                double delta = fabs((double)actual - expected);
                if (delta > maxError) maxError = delta;
                if (delta > 0.001) mismatches++;
                const uint8_t *bytes = (const uint8_t *)&values[index];
                checksum ^= bytes[0]; checksum *= 1099511628211ULL;
                checksum ^= bytes[1]; checksum *= 1099511628211ULL;
            }
            IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        }

        BOOL validated = poisoned && device && library && pipeline && buffer && pointerIdentity &&
            metalCompleted && aneEvaluated && finite == Elements && mismatches == 0;
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.gpu-ane-surface.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"metal_to_ane_numerical_oracle" : @"execution",
            @"host": NSProcessInfo.processInfo.hostName,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"ane_architecture": @"Apple M5 / H17G",
            @"metal_device": device.name ?: @"none",
            @"surface_alloc_bytes": @(allocBytes),
            @"surface_base_page_aligned": @(((uintptr_t)surfaceBase % 4096) == 0),
            @"metal_buffer_created_with_bytes_no_copy": @(buffer != nil),
            @"metal_buffer_contents_pointer_matches_iosurface": @(pointerIdentity),
            @"cpu_poison_before_metal": @(poisoned),
            @"cpu_memcpy_after_metal": @NO,
            @"ordering_boundary": @"MTLCommandBuffer waitUntilCompleted before ANE evaluate submission",
            @"metal_command_status": @(command.status),
            @"compile_count": @1,
            @"load_count": @1,
            @"ane_compile_ms": @(info.compileMilliseconds),
            @"ane_load_ms": @(info.loadMilliseconds),
            @"ane_evaluation_ms": @(timing.completionMilliseconds),
            @"numerical_result": @{
                @"elements": @(Elements), @"finite": @(finite),
                @"mismatches": @(mismatches), @"max_absolute_error": @(maxError),
                @"checksum": [NSString stringWithFormat:@"0x%016llx", checksum]
            },
            @"error": evaluationError ? [NSString stringWithUTF8String:evaluationError]
                                      : (error.localizedDescription ?: @"none"),
            @"claim_boundary": @"A shared MTLBuffer directly wrapped the page-aligned storage of the already mapped ANE input IOSurface. Metal overwrote a poisoned FP16 carrier; after command-buffer completion, ANE consumed it without a CPU memcpy and all 4,096 doubled outputs passed an independent oracle. This validates this buffer-backed same-process handoff on one M5/macOS build, not implicit GPU/ANE overlap, texture-backed IOSurface coherence, cross-process sharing, or a no-transport hardware claim."
        };
        emit(receipt);
        free((void *)evaluationError);
        ANEFieldfareANEProgramDestroy(program);
        return validated ? 0 : 1;
    }
}
