#import "MetalExpertDispatch.h"
#import "Decode.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fcntl.h>
#include <errno.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    ExpertStride = 3358720,
    GateWeightOffset = 0,
    GateScaleOffset = 991232,
    GateBiasOffset = 1053184,
    UpWeightOffset = 1115136,
    UpScaleOffset = 2106368,
    UpBiasOffset = 2168320,
    DownWeightOffset = 2230272,
    DownScaleOffset = 3221504,
    DownBiasOffset = 3283456,
    // Larger caches improve route hits but contend catastrophically with ANE
    // constants and GPU resources in unified memory.  A measured 64-slot arm
    // reached 7.10 GB process footprint and regressed 81-token prefill from
    // 4.53--5.66 s to 25.68 s.  Twenty is the retained production bound.
    MaxExactCacheSlots = 20,
};

typedef struct {
    uint32_t gateWeight;
    uint32_t gateScale;
    uint32_t gateBias;
    uint32_t upWeight;
    uint32_t upScale;
    uint32_t upBias;
    uint32_t downWeight;
    uint32_t downScale;
    uint32_t downBias;
} ExpertOffsets;

@interface ANEFieldfareMetalExpertDispatcherImpl : NSObject
@property(nonatomic) int *descriptors;
@property(nonatomic) void **mappings;
@property(nonatomic) size_t *mappingLengths;
@property(nonatomic) uint32_t layerCount;
@property(nonatomic) int modelDescriptor;
@property(nonatomic) void *modelMapping;
@property(nonatomic) size_t modelMappingLength;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *expertBuffers;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *layerResources;
@property(nonatomic, strong) id<MTLComputePipelineState> phase1;
@property(nonatomic, strong) id<MTLComputePipelineState> phase1Subset;
@property(nonatomic, strong) id<MTLComputePipelineState> phase2;
@property(nonatomic, strong) id<MTLComputePipelineState> groupedPhase1;
@property(nonatomic, strong) id<MTLComputePipelineState> groupedPhase2;
@property(nonatomic, strong) id<MTLComputePipelineState> groupedReduce;
@property(nonatomic, strong) id<MTLArgumentEncoder> argumentEncoder;
@property(nonatomic, strong) id<MTLBuffer> argumentBuffer;
@property(nonatomic, strong) id<MTLBuffer> inputBuffer;
@property(nonatomic, strong) id<MTLBuffer> actsBuffer;
@property(nonatomic, strong) id<MTLBuffer> activeSlotsBuffer;
@property(nonatomic, strong) id<MTLBuffer> weightsBuffer;
@property(nonatomic, strong) id<MTLBuffer> residualBuffer;
@property(nonatomic, strong) id<MTLBuffer> outputBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedInputBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedWeightsBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedPairsBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedActsBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedPartialsBuffer;
@property(nonatomic, strong) id<MTLBuffer> groupedOutputBuffer;
@property(nonatomic, strong) id<MTLBuffer> expertStagingBuffer;
@property(nonatomic, strong) id<MTLBuffer> expertIDsBuffer;
@property(nonatomic, strong) NSMutableArray *exactCacheBuffers;
@property(nonatomic) uint32_t *exactCacheTags;
@property(nonatomic) uint64_t *exactCacheFrequency;
@property(nonatomic) uint64_t *exactCacheLastUse;
@property(nonatomic) uint64_t exactCacheClock;
@property(nonatomic) uint32_t exactCacheSlots;
@property(nonatomic, strong) id<MTLBuffer> modelBuffer;
@property(nonatomic, strong) id<MTLComputePipelineState> affine;
@property(nonatomic, strong) id<MTLComputePipelineState> affineScalarGroups;
@property(nonatomic, strong) id<MTLComputePipelineState> affineBatched;
@property(nonatomic, strong) id<MTLComputePipelineState> swiglu;
@property(nonatomic, strong) id<MTLComputePipelineState> greedyHeadRows;
@property(nonatomic, strong) id<MTLComputePipelineState> greedyHeadReduce;
@property(nonatomic, strong) id<MTLComputePipelineState> greedyHeadRowsPositions;
@property(nonatomic, strong) id<MTLComputePipelineState> greedyHeadReducePositions;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionPrepare;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionCausal;
@property(nonatomic, strong) id<MTLBuffer> denseA;
@property(nonatomic, strong) id<MTLBuffer> denseB;
@property(nonatomic, strong) id<MTLBuffer> denseInput;
@property(nonatomic, strong) id<MTLBuffer> denseOutput;
@property(nonatomic, strong) id<MTLBuffer> greedyHeadSummaries;
@property(nonatomic, strong) id<MTLBuffer> greedyHeadToken;
@property(nonatomic, strong) id<MTLBuffer> greedyHeadValue;
@property(nonatomic, strong) id<MTLBuffer> attentionQuery;
@property(nonatomic, strong) id<MTLBuffer> attentionCombined;
@property(nonatomic, strong) id<MTLBuffer> attentionQueryNorm;
@property(nonatomic, strong) id<MTLBuffer> attentionKeyNorm;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic) ANEFieldfareMetalExpertDispatchInfo info;
@end

typedef struct {
    uint32_t queryHeads, keyValueHeads, headDimension, rotaryAngles;
    uint32_t positionCount, firstPosition, capacity, slidingWindow, fullAttention;
    float ropeTheta, epsilon, attentionScale;
} FusedAttentionParams;

@implementation ANEFieldfareMetalExpertDispatcherImpl
- (instancetype)init {
    self = [super init];
    if (self) _modelDescriptor = -1;
    return self;
}
- (void)dealloc {
    _expertBuffers = nil;
    _layerResources = nil;
    for (uint32_t layer = 0; layer < _layerCount; ++layer) {
        if (_mappings && _mappings[layer] && _mappings[layer] != MAP_FAILED)
            munmap(_mappings[layer], _mappingLengths[layer]);
        if (_descriptors && _descriptors[layer] >= 0) close(_descriptors[layer]);
    }
    free(_mappingLengths);
    free(_mappings);
    free(_descriptors);
    free(_exactCacheTags);
    free(_exactCacheFrequency);
    free(_exactCacheLastUse);
    _modelBuffer = nil;
    if (_modelMapping && _modelMapping != MAP_FAILED) munmap(_modelMapping, _modelMappingLength);
    if (_modelDescriptor >= 0) close(_modelDescriptor);
}
@end

static double elapsedMilliseconds(uint64_t ticks) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1.0e6;
}

static void setError(ANEFieldfareMetalExpertDispatcherImpl *impl, NSString *message) {
    impl.lastError = message ?: @"unknown Metal expert error";
}

static id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device, NSUInteger bytes) {
    return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> makeAlignedExactCacheBuffer(id<MTLDevice> device) {
    void *bytes = NULL;
    if (posix_memalign(&bytes, 2 * 1024 * 1024, ExpertStride) != 0 || !bytes) return nil;
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:bytes
                                                    length:ExpertStride
                                                   options:MTLResourceStorageModeShared
                                               deallocator:^(void *pointer, NSUInteger length) {
        (void)length;
        free(pointer);
    }];
    if (!buffer) free(bytes);
    return buffer;
}

static bool exactCacheEnabled(void) {
    const char *staging = getenv("ANE_FIELDFARE_METAL_STAGING");
    const char *cache = getenv("ANE_FIELDFARE_EXACT_EXPERT_CACHE");
    return !staging && (!cache || strcmp(cache, "0") != 0);
}

static uint32_t requestedExactCacheSlots(void) {
    const char *value = getenv("ANE_FIELDFARE_EXACT_EXPERT_CACHE_SLOTS");
    if (!value || !*value) return MaxExactCacheSlots;
    char *end = NULL;
    unsigned long slots = strtoul(value, &end, 10);
    // A reduced dispatch pins eight distinct selected experts until both Metal
    // phases finish, so fewer than top-k slots cannot satisfy the contract.
    return end && !*end && slots >= ANEFieldfareGemma4TopK &&
        slots <= MaxExactCacheSlots ? (uint32_t)slots : 0;
}

static bool mapLayer(ANEFieldfareMetalExpertDispatcherImpl *impl,
                     uint32_t layer,
                     const char *path) {
    struct stat status = {0};
    const uint64_t expected = (uint64_t)ExpertStride * ANEFieldfareGemma4ExpertCount;
    impl.descriptors[layer] = open(path, O_RDONLY | O_CLOEXEC);
    if (impl.descriptors[layer] < 0 || fstat(impl.descriptors[layer], &status) != 0 ||
        (uint64_t)status.st_size != expected) {
        setError(impl, [NSString stringWithFormat:@"invalid packed expert layer: %s", path]);
        return false;
    }
    impl.mappingLengths[layer] = (size_t)status.st_size;
    impl.mappings[layer] = mmap(NULL, impl.mappingLengths[layer], PROT_READ, MAP_SHARED,
                                impl.descriptors[layer], 0);
    if (impl.mappings[layer] == MAP_FAILED) {
        setError(impl, [NSString stringWithFormat:@"mmap failed for compact layer: %s", path]);
        return false;
    }
    // A Metal buffer covering the complete 430 MB layer makes the resource
    // residency manager fault the complete layer even when the argument
    // encoder points at eight slices. ExpertStride is page aligned on this
    // model, so expose 128 exact expert resources instead. useResource then
    // declares only the selected 26.9 MB and preserves compact demand paging.
    for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert) {
        void *bytes = (uint8_t *)impl.mappings[layer] + (uint64_t)expert * ExpertStride;
        id<MTLBuffer> buffer = [impl.device newBufferWithBytesNoCopy:bytes
                                                              length:ExpertStride
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
        if (!buffer) {
            setError(impl, [NSString stringWithFormat:
                @"Metal rejected compact expert mapping: %s expert %u", path, expert]);
            return false;
        }
        [impl.expertBuffers addObject:buffer];
    }
    id<MTLBuffer> layerResource = [impl.device newBufferWithBytesNoCopy:impl.mappings[layer]
                                                               length:impl.mappingLengths[layer]
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    if (!layerResource) {
        setError(impl, [NSString stringWithFormat:@"Metal rejected layer resource: %s", path]);
        return false;
    }
    [impl.layerResources addObject:layerResource];
    return true;
}

static bool mapModelWeights(ANEFieldfareMetalExpertDispatcherImpl *impl,
                            const char *modelDirectory) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/model_weights.bin", modelDirectory);
    struct stat status = {0};
    if (length <= 0 || (size_t)length >= sizeof(path) ||
        (impl.modelDescriptor = open(path, O_RDONLY | O_CLOEXEC)) < 0 ||
        fstat(impl.modelDescriptor, &status) != 0 || status.st_size <= 0) {
        setError(impl, @"model_weights.bin mapping failed");
        return false;
    }
    impl.modelMappingLength = (size_t)status.st_size;
    impl.modelMapping = mmap(NULL, impl.modelMappingLength, PROT_READ, MAP_SHARED,
                             impl.modelDescriptor, 0);
    if (impl.modelMapping == MAP_FAILED) {
        setError(impl, @"model_weights.bin mmap failed");
        return false;
    }
    impl.modelBuffer = [impl.device newBufferWithBytesNoCopy:impl.modelMapping
                                                       length:impl.modelMappingLength
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!impl.modelBuffer) {
        setError(impl, @"Metal rejected model_weights.bin mapping");
        return false;
    }
    return true;
}

static ANEFieldfareMetalExpertDispatcher *createDispatcher(
    const char *modelDirectory,
    const char *singleLayerPath,
    const char *metalSourcePath,
    const char **error) {
    if (error) *error = NULL;
    if ((!modelDirectory && !singleLayerPath) || !metalSourcePath) {
        if (error) *error = "model/layer path and Metal source path are required";
        return NULL;
    }
    ANEFieldfareMetalExpertDispatcherImpl *impl = [ANEFieldfareMetalExpertDispatcherImpl new];
    impl.exactCacheSlots = requestedExactCacheSlots();
    if (!impl.exactCacheSlots) {
        setError(impl, @"ANE_FIELDFARE_EXACT_EXPERT_CACHE_SLOTS must be between 8 and 20");
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }
    impl.layerCount = modelDirectory ? 30 : 1;
    impl.descriptors = malloc(impl.layerCount * sizeof(*impl.descriptors));
    impl.mappings = calloc(impl.layerCount, sizeof(*impl.mappings));
    impl.mappingLengths = calloc(impl.layerCount, sizeof(*impl.mappingLengths));
    impl.expertBuffers = [NSMutableArray arrayWithCapacity:
        (NSUInteger)impl.layerCount * ANEFieldfareGemma4ExpertCount];
    impl.layerResources = [NSMutableArray arrayWithCapacity:impl.layerCount];
    if (!impl.descriptors || !impl.mappings || !impl.mappingLengths || !impl.expertBuffers ||
        !impl.layerResources) {
        setError(impl, @"compact model mapping allocation failed");
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }
    for (uint32_t layer = 0; layer < impl.layerCount; ++layer) impl.descriptors[layer] = -1;

    impl.device = MTLCreateSystemDefaultDevice();
    impl.queue = [impl.device newCommandQueue];
    if (!impl.device || !impl.queue) {
        setError(impl, @"Metal device or command queue unavailable");
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }
    for (uint32_t layer = 0; layer < impl.layerCount; ++layer) {
        char path[4096];
        const char *selected = singleLayerPath;
        if (modelDirectory) {
            int length = snprintf(path, sizeof(path), "%s/packed_experts/layer_%02u.bin",
                                  modelDirectory, layer);
            if (length <= 0 || (size_t)length >= sizeof(path)) {
                setError(impl, @"compact expert layer path is too long");
                if (error) *error = strdup(impl.lastError.UTF8String);
                return NULL;
            }
            selected = path;
        }
        if (!mapLayer(impl, layer, selected)) {
            if (error) *error = strdup(impl.lastError.UTF8String);
            return NULL;
        }
    }
    if (modelDirectory && !mapModelWeights(impl, modelDirectory)) {
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }

    NSError *nsError = nil;
    NSString *source = [NSString stringWithContentsOfFile:
        [NSString stringWithUTF8String:metalSourcePath]
                                               encoding:NSUTF8StringEncoding
                                                  error:&nsError];
    id<MTLLibrary> library = source ? [impl.device newLibraryWithSource:source
                                                                options:nil
                                                                  error:&nsError] : nil;
    NSString *metalDirectory = [[NSString stringWithUTF8String:metalSourcePath]
        stringByDeletingLastPathComponent];
    NSString *attentionPath = [metalDirectory stringByAppendingPathComponent:@"attention.metal"];
    NSString *attentionSource = [NSString stringWithContentsOfFile:attentionPath
        encoding:NSUTF8StringEncoding error:&nsError];
    id<MTLLibrary> attentionLibrary = attentionSource
        ? [impl.device newLibraryWithSource:attentionSource options:nil error:&nsError] : nil;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    uint32_t constantD = ANEFieldfareGemma4HiddenSize;
    uint32_t constantF = ANEFieldfareGemma4ExpertIntermediateSize;
    uint32_t constantTopK = ANEFieldfareGemma4TopK;
    bool useConstants = true;
    [constants setConstantValue:&constantD type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&constantF type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&constantTopK type:MTLDataTypeUInt atIndex:2];
    [constants setConstantValue:&useConstants type:MTLDataTypeBool atIndex:3];
    id<MTLFunction> phase1Function = [library
        newFunctionWithName:@"moe_phase1_gate_up_act_u16load"
             constantValues:constants error:&nsError];
    id<MTLFunction> phase1SubsetFunction = [library
        newFunctionWithName:@"moe_phase1_gate_up_act_subset_u16load"
             constantValues:constants error:&nsError];
    id<MTLFunction> phase2Function = [library
        newFunctionWithName:@"moe_phase2_down_reduce_k8"
             constantValues:constants error:&nsError];
    id<MTLFunction> groupedPhase1Function =
        [library newFunctionWithName:@"moe_grouped_prefill_phase1"];
    id<MTLFunction> groupedPhase2Function =
        [library newFunctionWithName:@"moe_grouped_prefill_phase2"];
    id<MTLFunction> groupedReduceFunction =
        [library newFunctionWithName:@"moe_grouped_prefill_reduce"];
    id<MTLFunction> affineFunction = [library newFunctionWithName:@"affine_int4_gemv_f32"];
    id<MTLFunction> affineScalarGroupsFunction =
        [library newFunctionWithName:@"affine_int4_gemv_f32_scalar_groups"];
    id<MTLFunction> affineBatchedFunction =
        [library newFunctionWithName:@"affine_int4_gemm_f32"];
    id<MTLFunction> swigluFunction = [library newFunctionWithName:@"swiglu_f32"];
    id<MTLFunction> greedyHeadRowsFunction =
        [library newFunctionWithName:@"greedy_head_int4_rows_f32"];
    id<MTLFunction> greedyHeadReduceFunction =
        [library newFunctionWithName:@"greedy_head_reduce"];
    id<MTLFunction> greedyHeadRowsPositionsFunction =
        [library newFunctionWithName:@"greedy_head_int4_rows_positions_f32"];
    id<MTLFunction> greedyHeadReducePositionsFunction =
        [library newFunctionWithName:@"greedy_head_reduce_positions"];
    id<MTLFunction> attentionPrepareFunction =
        [attentionLibrary newFunctionWithName:@"attention_prepare_f32"];
    id<MTLFunction> attentionCausalFunction =
        [attentionLibrary newFunctionWithName:@"attention_causal_f32"];
    impl.phase1 = phase1Function ? [impl.device newComputePipelineStateWithFunction:phase1Function
                                                                             error:&nsError] : nil;
    impl.phase1Subset = phase1SubsetFunction
        ? [impl.device newComputePipelineStateWithFunction:phase1SubsetFunction error:&nsError]
        : nil;
    impl.phase2 = phase2Function ? [impl.device newComputePipelineStateWithFunction:phase2Function
                                                                             error:&nsError] : nil;
    impl.groupedPhase1 = groupedPhase1Function
        ? [impl.device newComputePipelineStateWithFunction:groupedPhase1Function error:&nsError]
        : nil;
    impl.groupedPhase2 = groupedPhase2Function
        ? [impl.device newComputePipelineStateWithFunction:groupedPhase2Function error:&nsError]
        : nil;
    impl.groupedReduce = groupedReduceFunction
        ? [impl.device newComputePipelineStateWithFunction:groupedReduceFunction error:&nsError]
        : nil;
    impl.affine = affineFunction ? [impl.device newComputePipelineStateWithFunction:affineFunction
                                                                               error:&nsError] : nil;
    impl.affineScalarGroups = affineScalarGroupsFunction
        ? [impl.device newComputePipelineStateWithFunction:affineScalarGroupsFunction error:&nsError]
        : nil;
    impl.affineBatched = affineBatchedFunction
        ? [impl.device newComputePipelineStateWithFunction:affineBatchedFunction error:&nsError]
        : nil;
    impl.swiglu = swigluFunction ? [impl.device newComputePipelineStateWithFunction:swigluFunction
                                                                               error:&nsError] : nil;
    impl.greedyHeadRows = greedyHeadRowsFunction
        ? [impl.device newComputePipelineStateWithFunction:greedyHeadRowsFunction error:&nsError]
        : nil;
    impl.greedyHeadReduce = greedyHeadReduceFunction
        ? [impl.device newComputePipelineStateWithFunction:greedyHeadReduceFunction error:&nsError]
        : nil;
    impl.greedyHeadRowsPositions = greedyHeadRowsPositionsFunction
        ? [impl.device newComputePipelineStateWithFunction:greedyHeadRowsPositionsFunction error:&nsError]
        : nil;
    impl.greedyHeadReducePositions = greedyHeadReducePositionsFunction
        ? [impl.device newComputePipelineStateWithFunction:greedyHeadReducePositionsFunction error:&nsError]
        : nil;
    impl.attentionPrepare = attentionPrepareFunction
        ? [impl.device newComputePipelineStateWithFunction:attentionPrepareFunction error:&nsError]
        : nil;
    impl.attentionCausal = attentionCausalFunction
        ? [impl.device newComputePipelineStateWithFunction:attentionCausalFunction error:&nsError]
        : nil;
    if (!library || !impl.phase1 || !impl.phase1Subset || !impl.phase2 || !impl.groupedPhase1 ||
        !impl.groupedPhase2 || !impl.groupedReduce || !impl.affine || !impl.affineScalarGroups ||
        !impl.affineBatched || !impl.swiglu || !impl.greedyHeadRows ||
        !impl.greedyHeadReduce || !impl.greedyHeadRowsPositions ||
        !impl.greedyHeadReducePositions || !impl.attentionPrepare ||
        !impl.attentionCausal) {
        setError(impl, [NSString stringWithFormat:@"Metal kernel compilation failed: %@",
                                                   nsError.localizedDescription ?: @"unknown"]);
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }

    impl.argumentEncoder = [phase1Function newArgumentEncoderWithBufferIndex:0];
    impl.argumentBuffer = makeSharedBuffer(impl.device, impl.argumentEncoder.encodedLength);
    impl.inputBuffer = makeSharedBuffer(impl.device, 8192 * sizeof(__fp16));
    impl.actsBuffer = makeSharedBuffer(impl.device,
        ANEFieldfareGemma4TopK * ANEFieldfareGemma4ExpertIntermediateSize * sizeof(__fp16));
    impl.activeSlotsBuffer = makeSharedBuffer(
        impl.device, ANEFieldfareGemma4TopK * sizeof(uint32_t));
    impl.weightsBuffer = makeSharedBuffer(impl.device, ANEFieldfareGemma4TopK * sizeof(__fp16));
    impl.residualBuffer = makeSharedBuffer(impl.device, ANEFieldfareGemma4HiddenSize * sizeof(__fp16));
    impl.outputBuffer = makeSharedBuffer(impl.device, ANEFieldfareGemma4HiddenSize * sizeof(__fp16));
    const NSUInteger groupedRoutes = 64 * ANEFieldfareGemma4TopK;
    impl.groupedInputBuffer = makeSharedBuffer(
        impl.device, 64 * ANEFieldfareGemma4HiddenSize * sizeof(__fp16));
    impl.groupedWeightsBuffer = makeSharedBuffer(impl.device, groupedRoutes * sizeof(__fp16));
    impl.groupedPairsBuffer = makeSharedBuffer(impl.device, groupedRoutes * 4 * sizeof(uint32_t));
    impl.groupedActsBuffer = makeSharedBuffer(
        impl.device, groupedRoutes * ANEFieldfareGemma4ExpertIntermediateSize * sizeof(__fp16));
    impl.groupedPartialsBuffer = makeSharedBuffer(
        impl.device, groupedRoutes * ANEFieldfareGemma4HiddenSize * sizeof(__fp16));
    impl.groupedOutputBuffer = makeSharedBuffer(
        impl.device, 64 * ANEFieldfareGemma4HiddenSize * sizeof(__fp16));
    impl.expertStagingBuffer = makeSharedBuffer(
        impl.device, (NSUInteger)ExpertStride * ANEFieldfareGemma4TopK);
    impl.expertIDsBuffer = makeSharedBuffer(
        impl.device, ANEFieldfareGemma4TopK * sizeof(uint32_t));
    impl.exactCacheBuffers = [NSMutableArray arrayWithCapacity:
        (NSUInteger)impl.layerCount * impl.exactCacheSlots];
    impl.exactCacheTags = malloc((NSUInteger)impl.layerCount * impl.exactCacheSlots *
                                 sizeof(*impl.exactCacheTags));
    impl.exactCacheFrequency = calloc(
        (NSUInteger)impl.layerCount * ANEFieldfareGemma4ExpertCount,
        sizeof(*impl.exactCacheFrequency));
    impl.exactCacheLastUse = calloc(
        (NSUInteger)impl.layerCount * impl.exactCacheSlots,
        sizeof(*impl.exactCacheLastUse));
    for (uint32_t index = 0; index < impl.layerCount * impl.exactCacheSlots; ++index) {
        [impl.exactCacheBuffers addObject:NSNull.null];
        impl.exactCacheTags[index] = UINT32_MAX;
    }
    impl.denseA = makeSharedBuffer(impl.device, 64 * 8192 * sizeof(float));
    impl.denseB = makeSharedBuffer(impl.device, 64 * 8192 * sizeof(float));
    impl.denseInput = makeSharedBuffer(impl.device, 64 * 8192 * sizeof(float));
    impl.denseOutput = makeSharedBuffer(impl.device, 64 * 8192 * sizeof(float));
    impl.greedyHeadSummaries = makeSharedBuffer(
        impl.device, 64 * ((262144 + 7) / 8) * 2 * sizeof(float));
    impl.greedyHeadToken = makeSharedBuffer(impl.device, 64 * sizeof(uint32_t));
    impl.greedyHeadValue = makeSharedBuffer(impl.device, 64 * sizeof(float));
    impl.attentionQuery = makeSharedBuffer(impl.device, 8192 * sizeof(float));
    impl.attentionCombined = makeSharedBuffer(impl.device, 8192 * sizeof(float));
    impl.attentionQueryNorm = makeSharedBuffer(impl.device, 512 * sizeof(float));
    impl.attentionKeyNorm = makeSharedBuffer(impl.device, 512 * sizeof(float));
    if (!impl.argumentEncoder || !impl.argumentBuffer || !impl.inputBuffer ||
        !impl.actsBuffer || !impl.activeSlotsBuffer || !impl.weightsBuffer ||
        !impl.residualBuffer || !impl.outputBuffer ||
        !impl.groupedInputBuffer || !impl.groupedWeightsBuffer || !impl.groupedPairsBuffer ||
        !impl.groupedActsBuffer || !impl.groupedPartialsBuffer || !impl.groupedOutputBuffer ||
        !impl.expertStagingBuffer || !impl.expertIDsBuffer || !impl.exactCacheBuffers ||
        !impl.exactCacheTags || !impl.exactCacheFrequency || !impl.exactCacheLastUse ||
        !impl.denseA || !impl.denseB || !impl.denseInput || !impl.denseOutput ||
        !impl.greedyHeadSummaries || !impl.greedyHeadToken || !impl.greedyHeadValue ||
        !impl.attentionQuery || !impl.attentionCombined ||
        !impl.attentionQueryNorm || !impl.attentionKeyNorm) {
        setError(impl, @"Metal activation buffer allocation failed");
        if (error) *error = strdup(impl.lastError.UTF8String);
        return NULL;
    }
    memset(impl.residualBuffer.contents, 0, impl.residualBuffer.length);
    const uint64_t layerBytes = (uint64_t)ExpertStride * ANEFieldfareGemma4ExpertCount;
    impl.info = (ANEFieldfareMetalExpertDispatchInfo){
        .compactLayerBytes = layerBytes,
        .compactModelBytes = layerBytes * impl.layerCount,
        .mappedExpertBytesPerDispatch = (uint64_t)ExpertStride * ANEFieldfareGemma4TopK,
        .compactStagingBytes = (uint64_t)ExpertStride * ANEFieldfareGemma4TopK,
        .exactCacheCapacityBytes =
            (uint64_t)ExpertStride * impl.exactCacheSlots * impl.layerCount,
        .exactCacheSlotsPerLayer = impl.exactCacheSlots,
        .mappedLayerCount = impl.layerCount,
        .libraryCompileCount = 1,
        .pipelineLoadCount = 16,
        .splitQueueEnabled = getenv("ANE_FIELDFARE_METAL_SPLIT_QUEUE") != NULL,
    };
    return (__bridge_retained ANEFieldfareMetalExpertDispatcher *)impl;
}

ANEFieldfareMetalExpertDispatcher *ANEFieldfareMetalExpertDispatcherCreate(
    const char *packedLayerPath,
    const char *metalSourcePath,
    const char **error) {
    @autoreleasepool { return createDispatcher(NULL, packedLayerPath, metalSourcePath, error); }
}

ANEFieldfareMetalExpertDispatcher *ANEFieldfareMetalExpertDispatcherCreateModel(
    const char *modelDirectory,
    const char *metalSourcePath,
    const char **error) {
    @autoreleasepool { return createDispatcher(modelDirectory, NULL, metalSourcePath, error); }
}

typedef struct {
    __unsafe_unretained ANEFieldfareMetalExpertDispatcherImpl *impl;
    uint32_t layer;
    uint32_t count;
    uint32_t slots[MaxExactCacheSlots];
    uint32_t experts[MaxExactCacheSlots];
    _Atomic bool valid;
} ExactCacheFillWork;

static void fillExactCacheMiss(void *raw, size_t value) {
    ExactCacheFillWork *work = raw;
    if (value >= work->count || !atomic_load_explicit(&work->valid, memory_order_relaxed)) return;
    uint32_t slot = work->slots[value], expert = work->experts[value];
    NSUInteger index = (NSUInteger)work->layer * work->impl.exactCacheSlots + slot;
    id buffer = work->impl.exactCacheBuffers[index];
    uint8_t *destination = [buffer contents];
    size_t remaining = ExpertStride;
    off_t offset = (off_t)((uint64_t)expert * ExpertStride);
    while (remaining) {
        ssize_t got = pread(work->impl.descriptors[work->layer], destination,
                            remaining, offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            atomic_store_explicit(&work->valid, false, memory_order_relaxed);
            return;
        }
        destination += got; offset += got; remaining -= (size_t)got;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallReduced(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    return ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
        dispatcher, 0, experts, weights, input, output);
}

bool ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    @autoreleasepool {
        if (!dispatcher || !experts || !weights || !input || !output) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl = (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        if (layer >= impl.layerCount) {
            setError(impl, @"compact expert layer index is out of range");
            return false;
        }
        __fp16 *metalInput = impl.inputBuffer.contents;
        __fp16 *metalWeights = impl.weightsBuffer.contents;
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            metalInput[index] = (__fp16)input[index];
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
            if (experts[rank] >= ANEFieldfareGemma4ExpertCount) {
                setError(impl, @"selected expert index is out of range");
                return false;
            }
            metalWeights[rank] = (__fp16)weights[rank];
        }

        static int zeroCopyMode = -1;
        if (zeroCopyMode < 0) zeroCopyMode = getenv("ANE_FIELDFARE_METAL_STAGING") ? 0 : 1;
        static int exactCacheMode = -1;
        if (exactCacheMode < 0) exactCacheMode = exactCacheEnabled();
        uint64_t stagingStarted = mach_absolute_time();
        [impl.argumentEncoder setArgumentBuffer:impl.argumentBuffer offset:0];
        uint32_t cacheSlots[ANEFieldfareGemma4TopK] = {0};
        bool cacheUsed[MaxExactCacheSlots] = {0};
        uint64_t cacheHits = 0, cacheMisses = 0;
        ExactCacheFillWork fill = {.impl = impl, .layer = layer};
        atomic_init(&fill.valid, true);
        if (zeroCopyMode && exactCacheMode) {
            const uint32_t base = layer * impl.exactCacheSlots;
            const uint64_t clock = ++impl.exactCacheClock;
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                cacheSlots[rank] = UINT32_MAX;
                ++impl.exactCacheFrequency[
                    (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + experts[rank]];
                for (uint32_t slot = 0; slot < impl.exactCacheSlots; ++slot) {
                    if (impl.exactCacheTags[base + slot] == experts[rank]) {
                        cacheSlots[rank] = slot;
                        cacheUsed[slot] = true;
                        impl.exactCacheLastUse[base + slot] = clock;
                        ++cacheHits;
                        break;
                    }
                }
            }
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                if (cacheSlots[rank] != UINT32_MAX) continue;
                uint32_t slot = UINT32_MAX;
                uint64_t lowestFrequency = UINT64_MAX;
                uint64_t oldestUse = UINT64_MAX;
                for (uint32_t candidate = 0; candidate < impl.exactCacheSlots; ++candidate) {
                    if (cacheUsed[candidate]) continue;
                    uint32_t resident = impl.exactCacheTags[base + candidate];
                    uint64_t frequency = resident == UINT32_MAX ? 0 :
                        impl.exactCacheFrequency[
                            (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + resident];
                    uint64_t lastUse = impl.exactCacheLastUse[base + candidate];
                    if (slot == UINT32_MAX || frequency < lowestFrequency ||
                        (frequency == lowestFrequency && lastUse < oldestUse)) {
                        slot = candidate;
                        lowestFrequency = frequency;
                        oldestUse = lastUse;
                    }
                }
                if (slot == UINT32_MAX) return false;
                cacheSlots[rank] = slot; cacheUsed[slot] = true; ++cacheMisses;
                impl.exactCacheLastUse[base + slot] = clock;
                NSUInteger cacheIndex = base + slot;
                id buffer = impl.exactCacheBuffers[cacheIndex];
                if (buffer == NSNull.null) {
                    buffer = makeAlignedExactCacheBuffer(impl.device);
                    if (!buffer) { setError(impl, @"exact compact expert cache allocation failed"); return false; }
                    impl.exactCacheBuffers[cacheIndex] = buffer;
                }
                fill.slots[fill.count] = slot;
                fill.experts[fill.count] = experts[rank];
                ++fill.count;
            }
        }
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
            id<MTLBuffer> expertBuffer = nil;
            NSUInteger offset = 0;
            if (zeroCopyMode && exactCacheMode) {
                expertBuffer = impl.exactCacheBuffers[
                    (NSUInteger)layer * impl.exactCacheSlots + cacheSlots[rank]];
            } else if (zeroCopyMode) {
                expertBuffer = impl.expertBuffers[
                    (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + experts[rank]];
            } else {
                offset = (NSUInteger)rank * ExpertStride;
                memcpy((uint8_t *)impl.expertStagingBuffer.contents + offset,
                       (const uint8_t *)impl.mappings[layer] +
                           (uint64_t)experts[rank] * ExpertStride,
                       ExpertStride);
                expertBuffer = impl.expertStagingBuffer;
            }
            [impl.argumentEncoder setBuffer:expertBuffer offset:offset atIndex:rank];
        }
        static int hitSplitMode = -1;
        if (hitSplitMode < 0)
            hitSplitMode = getenv("ANE_FIELDFARE_EXPERT_HIT_SPLIT") ? 1 : 0;
        const bool splitHitsFromMisses = hitSplitMode && zeroCopyMode && exactCacheMode &&
            cacheHits > 0 && cacheMisses > 0;
        dispatch_group_t fillGroup = nil;
        if (fill.count) {
            if (splitHitsFromMisses) {
                ExactCacheFillWork *fillPointer = &fill;
                fillGroup = dispatch_group_create();
                dispatch_group_async(fillGroup,
                    dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                        dispatch_apply_f(fillPointer->count,
                            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                            fillPointer, fillExactCacheMiss);
                    });
            } else {
                dispatch_apply_f(fill.count,
                    dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                    &fill, fillExactCacheMiss);
            }
        }
        double stagingMilliseconds = zeroCopyMode ? 0.0 :
            elapsedMilliseconds(mach_absolute_time() - stagingStarted);
        double cacheFillMilliseconds = 0.0;

        id<MTLCommandBuffer> hitCommand = nil;
        uint64_t hitCommandStarted = 0;
        double hitCommandWallMilliseconds = 0.0;
        ExpertOffsets offsets = {
            GateWeightOffset, GateScaleOffset, GateBiasOffset,
            UpWeightOffset, UpScaleOffset, UpBiasOffset,
            DownWeightOffset, DownScaleOffset, DownBiasOffset,
        };
        uint32_t dimension = ANEFieldfareGemma4HiddenSize;
        uint32_t intermediate = ANEFieldfareGemma4ExpertIntermediateSize;
        uint32_t topK = ANEFieldfareGemma4TopK;

        if (splitHitsFromMisses) {
            uint32_t *activeSlots = impl.activeSlotsBuffer.contents;
            uint32_t activeCount = 0;
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                bool isMiss = false;
                for (uint32_t miss = 0; miss < fill.count; ++miss)
                    if (fill.slots[miss] == cacheSlots[rank]) { isMiss = true; break; }
                if (!isMiss) activeSlots[activeCount++] = rank;
            }
            hitCommand = [impl.queue commandBuffer];
            id<MTLComputeCommandEncoder> hitPhase = [hitCommand computeCommandEncoder];
            [hitPhase setComputePipelineState:impl.phase1Subset];
            [hitPhase setBuffer:impl.argumentBuffer offset:0 atIndex:0];
            for (uint32_t active = 0; active < activeCount; ++active) {
                uint32_t rank = activeSlots[active];
                [hitPhase useResource:impl.exactCacheBuffers[
                    (NSUInteger)layer * impl.exactCacheSlots + cacheSlots[rank]]
                                  usage:MTLResourceUsageRead];
            }
            [hitPhase setBytes:&offsets length:sizeof(offsets) atIndex:1];
            [hitPhase setBuffer:impl.inputBuffer offset:0 atIndex:2];
            [hitPhase setBuffer:impl.actsBuffer offset:0 atIndex:3];
            [hitPhase setBytes:&dimension length:sizeof(dimension) atIndex:4];
            [hitPhase setBytes:&intermediate length:sizeof(intermediate) atIndex:5];
            [hitPhase setBytes:&topK length:sizeof(topK) atIndex:6];
            [hitPhase setBuffer:impl.activeSlotsBuffer offset:0 atIndex:7];
            [hitPhase setBytes:&activeCount length:sizeof(activeCount) atIndex:8];
            [hitPhase dispatchThreadgroups:
                MTLSizeMake((activeCount * intermediate + 7) / 8, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [hitPhase endEncoding];
            hitCommandStarted = mach_absolute_time();
            [hitCommand commit];
        }

        if (fillGroup) dispatch_group_wait(fillGroup, DISPATCH_TIME_FOREVER);
        if (!atomic_load_explicit(&fill.valid, memory_order_relaxed)) {
            setError(impl, @"concurrent exact expert pread failed");
            return false;
        }
        if (fill.count) {
            const uint32_t base = layer * impl.exactCacheSlots;
            for (uint32_t miss = 0; miss < fill.count; ++miss)
                impl.exactCacheTags[base + fill.slots[miss]] = fill.experts[miss];
            cacheFillMilliseconds = elapsedMilliseconds(mach_absolute_time() - stagingStarted);
        }
        if (hitCommand) {
            [hitCommand waitUntilCompleted];
            hitCommandWallMilliseconds = elapsedMilliseconds(
                mach_absolute_time() - hitCommandStarted);
            if (hitCommand.status != MTLCommandBufferStatusCompleted) {
                setError(impl, [NSString stringWithFormat:@"Metal expert hit phase failed: %@",
                    hitCommand.error.localizedDescription ?: @"unknown"]);
                return false;
            }
        }

        id<MTLCommandBuffer> command = [impl.queue commandBuffer];

        id<MTLComputeCommandEncoder> phase1 = [command computeCommandEncoder];
        [phase1 setComputePipelineState:splitHitsFromMisses ? impl.phase1Subset : impl.phase1];
        [phase1 setBuffer:impl.argumentBuffer offset:0 atIndex:0];
        if (zeroCopyMode && exactCacheMode) {
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
                [phase1 useResource:impl.exactCacheBuffers[
                    (NSUInteger)layer * impl.exactCacheSlots + cacheSlots[rank]]
                             usage:MTLResourceUsageRead];
        } else if (zeroCopyMode) {
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
                [phase1 useResource:impl.expertBuffers[
                    (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + experts[rank]]
                             usage:MTLResourceUsageRead];
        } else [phase1 useResource:impl.expertStagingBuffer usage:MTLResourceUsageRead];
        [phase1 setBytes:&offsets length:sizeof(offsets) atIndex:1];
        [phase1 setBuffer:impl.inputBuffer offset:0 atIndex:2];
        [phase1 setBuffer:impl.actsBuffer offset:0 atIndex:3];
        [phase1 setBytes:&dimension length:sizeof(dimension) atIndex:4];
        [phase1 setBytes:&intermediate length:sizeof(intermediate) atIndex:5];
        [phase1 setBytes:&topK length:sizeof(topK) atIndex:6];
        uint32_t missCount = (uint32_t)fill.count;
        if (splitHitsFromMisses) {
            uint32_t *activeSlots = impl.activeSlotsBuffer.contents;
            for (uint32_t miss = 0; miss < fill.count; ++miss)
                for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
                    if (cacheSlots[rank] == fill.slots[miss]) {
                        activeSlots[miss] = rank;
                        break;
                    }
            [phase1 setBuffer:impl.activeSlotsBuffer offset:0 atIndex:7];
            [phase1 setBytes:&missCount length:sizeof(missCount) atIndex:8];
        }
        const uint32_t phase1Experts = splitHitsFromMisses ? missCount : topK;
        [phase1 dispatchThreadgroups:MTLSizeMake((phase1Experts * intermediate + 7) / 8, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [phase1 endEncoding];

        id<MTLComputeCommandEncoder> phase2 = [command computeCommandEncoder];
        [phase2 setComputePipelineState:impl.phase2];
        [phase2 setBuffer:impl.argumentBuffer offset:0 atIndex:0];
        if (zeroCopyMode && exactCacheMode) {
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
                [phase2 useResource:impl.exactCacheBuffers[
                    (NSUInteger)layer * impl.exactCacheSlots + cacheSlots[rank]]
                             usage:MTLResourceUsageRead];
        } else if (zeroCopyMode) {
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
                [phase2 useResource:impl.expertBuffers[
                    (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + experts[rank]]
                             usage:MTLResourceUsageRead];
        } else [phase2 useResource:impl.expertStagingBuffer usage:MTLResourceUsageRead];
        [phase2 setBytes:&offsets length:sizeof(offsets) atIndex:1];
        [phase2 setBuffer:impl.actsBuffer offset:0 atIndex:2];
        [phase2 setBuffer:impl.weightsBuffer offset:0 atIndex:3];
        [phase2 setBuffer:impl.residualBuffer offset:0 atIndex:4];
        [phase2 setBuffer:impl.outputBuffer offset:0 atIndex:5];
        [phase2 setBytes:&dimension length:sizeof(dimension) atIndex:6];
        [phase2 setBytes:&intermediate length:sizeof(intermediate) atIndex:7];
        [phase2 dispatchThreadgroups:MTLSizeMake(dimension, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [phase2 endEncoding];

        uint64_t start = mach_absolute_time();
        [command commit];
        [command waitUntilCompleted];
        double wall = elapsedMilliseconds(mach_absolute_time() - start);
        if (command.status != MTLCommandBufferStatusCompleted) {
            setError(impl, [NSString stringWithFormat:@"Metal expert dispatch failed: %@",
                                                       command.error.localizedDescription ?: @"unknown"]);
            return false;
        }
        __fp16 *metalOutput = impl.outputBuffer.contents;
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            output[index] = (float)metalOutput[index];
        @synchronized (impl) {
            ANEFieldfareMetalExpertDispatchInfo info = impl.info;
            info.dispatchCount += 1;
            info.totalCompactStagingMilliseconds += stagingMilliseconds;
            info.totalExactCacheFillMilliseconds += cacheFillMilliseconds;
            info.exactCacheHits += cacheHits;
            info.exactCacheMisses += cacheMisses;
            info.totalWallMilliseconds += wall + hitCommandWallMilliseconds;
            if (hitCommand) {
                if (hitCommand.GPUEndTime >= hitCommand.GPUStartTime)
                    info.totalGPUMilliseconds +=
                        (hitCommand.GPUEndTime - hitCommand.GPUStartTime) * 1000.0;
            }
            if (command.GPUEndTime >= command.GPUStartTime)
                info.totalGPUMilliseconds +=
                    (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            impl.info = info;
        }
        impl.lastError = nil;
        return true;
    }
}

typedef struct {
    uint32_t token;
    uint32_t slot;
    uint32_t rank;
    uint32_t reserved;
} GroupedPrefillPair;

bool ANEFieldfareMetalExpertDispatcherCallReducedPositionsAtLayer(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    uint32_t layer,
    uint32_t positionCount,
    const uint32_t *experts,
    const float *weights,
    const float *inputs,
    float *outputs) {
    @autoreleasepool {
        if (!dispatcher || layer >= 30 || !positionCount || positionCount > 64 ||
            !experts || !weights || !inputs || !outputs || !exactCacheEnabled()) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        if (layer >= impl.layerCount) return false;

        const uint32_t routeCount = positionCount * ANEFieldfareGemma4TopK;
        __fp16 *metalInputs = impl.groupedInputBuffer.contents;
        __fp16 *metalWeights = impl.groupedWeightsBuffer.contents;
        for (uint32_t token = 0; token < positionCount; ++token)
            for (uint32_t column = 0; column < ANEFieldfareGemma4HiddenSize; ++column)
                metalInputs[(uint64_t)token * ANEFieldfareGemma4HiddenSize + column] =
                    (__fp16)inputs[(uint64_t)token * ANEFieldfareGemma4HiddenSize + column];
        bool live[ANEFieldfareGemma4ExpertCount] = {0};
        uint32_t liveExperts[ANEFieldfareGemma4ExpertCount] = {0};
        uint32_t liveCount = 0;
        for (uint32_t route = 0; route < routeCount; ++route) {
            if (experts[route] >= ANEFieldfareGemma4ExpertCount) {
                setError(impl, @"grouped prefill expert index is out of range");
                return false;
            }
            metalWeights[route] = (__fp16)weights[route];
            live[experts[route]] = true;
        }
        // Keep currently resident experts first. On the second prompt tile
        // this consumes the surviving LFU entries before later misses can
        // evict them; the remainder stays in deterministic expert order.
        for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert)
            if (live[expert]) liveExperts[liveCount++] = expert;
        uint32_t cacheAware[ANEFieldfareGemma4ExpertCount] = {0};
        bool ordered[ANEFieldfareGemma4ExpertCount] = {0};
        uint32_t orderedCount = 0;
        const uint32_t layerCacheBase = layer * impl.exactCacheSlots;
        for (uint32_t slot = 0; slot < impl.exactCacheSlots; ++slot) {
            uint32_t expert = impl.exactCacheTags[layerCacheBase + slot];
            if (expert < ANEFieldfareGemma4ExpertCount && live[expert] && !ordered[expert]) {
                cacheAware[orderedCount++] = expert;
                ordered[expert] = true;
            }
        }
        for (uint32_t index = 0; index < liveCount; ++index)
            if (!ordered[liveExperts[index]]) cacheAware[orderedCount++] = liveExperts[index];
        memcpy(liveExperts, cacheAware, liveCount * sizeof(*liveExperts));

        ExpertOffsets offsets = {
            GateWeightOffset, GateScaleOffset, GateBiasOffset,
            UpWeightOffset, UpScaleOffset, UpBiasOffset,
            DownWeightOffset, DownScaleOffset, DownBiasOffset,
        };
        const uint32_t dimension = ANEFieldfareGemma4HiddenSize;
        const uint32_t intermediate = ANEFieldfareGemma4ExpertIntermediateSize;
        uint64_t totalHits = 0, totalMisses = 0, commandCount = 0;
        double totalFillMs = 0.0, totalWallMs = 0.0, totalGpuMs = 0.0;

        const uint32_t tileCapacity = MIN(16u, impl.exactCacheSlots);
        for (uint32_t tileStart = 0; tileStart < liveCount; tileStart += tileCapacity) {
            const uint32_t tileCount = MIN(tileCapacity, liveCount - tileStart);
            const uint32_t base = layer * impl.exactCacheSlots;
            bool cacheUsed[MaxExactCacheSlots] = {0};
            uint32_t cacheSlots[MaxExactCacheSlots] = {0};
            const uint64_t clock = ++impl.exactCacheClock;
            ExactCacheFillWork fill = {.impl = impl, .layer = layer};
            atomic_init(&fill.valid, true);
            uint64_t fillStarted = mach_absolute_time();

            for (uint32_t local = 0; local < tileCount; ++local) {
                const uint32_t expert = liveExperts[tileStart + local];
                ++impl.exactCacheFrequency[
                    (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + expert];
                uint32_t slot = UINT32_MAX;
                for (uint32_t candidate = 0; candidate < impl.exactCacheSlots; ++candidate) {
                    if (impl.exactCacheTags[base + candidate] == expert) {
                        slot = candidate;
                        ++totalHits;
                        break;
                    }
                }
                if (slot == UINT32_MAX) {
                    uint64_t lowestFrequency = UINT64_MAX, oldestUse = UINT64_MAX;
                    for (uint32_t candidate = 0; candidate < impl.exactCacheSlots; ++candidate) {
                        if (cacheUsed[candidate]) continue;
                        uint32_t resident = impl.exactCacheTags[base + candidate];
                        uint64_t frequency = resident == UINT32_MAX ? 0 :
                            impl.exactCacheFrequency[
                                (NSUInteger)layer * ANEFieldfareGemma4ExpertCount + resident];
                        uint64_t lastUse = impl.exactCacheLastUse[base + candidate];
                        if (slot == UINT32_MAX || frequency < lowestFrequency ||
                            (frequency == lowestFrequency && lastUse < oldestUse)) {
                            slot = candidate; lowestFrequency = frequency; oldestUse = lastUse;
                        }
                    }
                    if (slot == UINT32_MAX) return false;
                    NSUInteger cacheIndex = base + slot;
                    id buffer = impl.exactCacheBuffers[cacheIndex];
                    if (buffer == NSNull.null) {
                        buffer = makeAlignedExactCacheBuffer(impl.device);
                        if (!buffer) return false;
                        impl.exactCacheBuffers[cacheIndex] = buffer;
                    }
                    fill.slots[fill.count] = slot;
                    fill.experts[fill.count] = expert;
                    ++fill.count; ++totalMisses;
                }
                cacheSlots[local] = slot;
                cacheUsed[slot] = true;
                impl.exactCacheLastUse[base + slot] = clock;
            }
            dispatch_apply_f(fill.count,
                             dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                             &fill, fillExactCacheMiss);
            if (!atomic_load_explicit(&fill.valid, memory_order_relaxed)) {
                setError(impl, @"grouped prefill compact expert read failed");
                return false;
            }
            for (uint32_t miss = 0; miss < fill.count; ++miss)
                impl.exactCacheTags[base + fill.slots[miss]] = fill.experts[miss];
            if (fill.count) totalFillMs += elapsedMilliseconds(mach_absolute_time() - fillStarted);

            [impl.argumentEncoder setArgumentBuffer:impl.argumentBuffer offset:0];
            for (uint32_t local = 0; local < tileCount; ++local)
                [impl.argumentEncoder setBuffer:impl.exactCacheBuffers[base + cacheSlots[local]]
                                           offset:0 atIndex:local];

            GroupedPrefillPair *pairs = impl.groupedPairsBuffer.contents;
            uint32_t pairCount = 0;
            for (uint32_t local = 0; local < tileCount; ++local) {
                uint32_t expert = liveExperts[tileStart + local];
                for (uint32_t token = 0; token < positionCount; ++token)
                    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                        uint32_t route = token * ANEFieldfareGemma4TopK + rank;
                        if (experts[route] == expert)
                            pairs[pairCount++] = (GroupedPrefillPair){token, local, rank, 0};
                    }
            }
            if (!pairCount) continue;
            id<MTLCommandBuffer> command = [impl.queue commandBuffer];
            id<MTLComputeCommandEncoder> phase1 = [command computeCommandEncoder];
            [phase1 setComputePipelineState:impl.groupedPhase1];
            [phase1 setBuffer:impl.argumentBuffer offset:0 atIndex:0];
            for (uint32_t local = 0; local < tileCount; ++local)
                [phase1 useResource:impl.exactCacheBuffers[base + cacheSlots[local]]
                               usage:MTLResourceUsageRead];
            [phase1 setBytes:&offsets length:sizeof(offsets) atIndex:1];
            [phase1 setBuffer:impl.groupedInputBuffer offset:0 atIndex:2];
            [phase1 setBuffer:impl.groupedPairsBuffer offset:0 atIndex:3];
            [phase1 setBuffer:impl.groupedActsBuffer offset:0 atIndex:4];
            [phase1 setBytes:&pairCount length:sizeof(pairCount) atIndex:5];
            [phase1 setBytes:&dimension length:sizeof(dimension) atIndex:6];
            [phase1 setBytes:&intermediate length:sizeof(intermediate) atIndex:7];
            [phase1 dispatchThreadgroups:MTLSizeMake((pairCount * intermediate + 7) / 8, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [phase1 endEncoding];

            id<MTLComputeCommandEncoder> phase2 = [command computeCommandEncoder];
            [phase2 setComputePipelineState:impl.groupedPhase2];
            [phase2 setBuffer:impl.argumentBuffer offset:0 atIndex:0];
            for (uint32_t local = 0; local < tileCount; ++local)
                [phase2 useResource:impl.exactCacheBuffers[base + cacheSlots[local]]
                               usage:MTLResourceUsageRead];
            [phase2 setBytes:&offsets length:sizeof(offsets) atIndex:1];
            [phase2 setBuffer:impl.groupedPairsBuffer offset:0 atIndex:2];
            [phase2 setBuffer:impl.groupedActsBuffer offset:0 atIndex:3];
            [phase2 setBuffer:impl.groupedWeightsBuffer offset:0 atIndex:4];
            [phase2 setBuffer:impl.groupedPartialsBuffer offset:0 atIndex:5];
            [phase2 setBytes:&pairCount length:sizeof(pairCount) atIndex:6];
            [phase2 setBytes:&dimension length:sizeof(dimension) atIndex:7];
            [phase2 setBytes:&intermediate length:sizeof(intermediate) atIndex:8];
            [phase2 dispatchThreadgroups:MTLSizeMake(dimension, pairCount, 1)
                         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [phase2 endEncoding];

            if (tileStart + tileCount == liveCount) {
                id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
                [reduce setComputePipelineState:impl.groupedReduce];
                [reduce setBuffer:impl.groupedPartialsBuffer offset:0 atIndex:0];
                [reduce setBuffer:impl.groupedOutputBuffer offset:0 atIndex:1];
                [reduce setBytes:&positionCount length:sizeof(positionCount) atIndex:2];
                [reduce setBytes:&dimension length:sizeof(dimension) atIndex:3];
                [reduce dispatchThreads:MTLSizeMake(dimension, positionCount, 1)
                      threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                [reduce endEncoding];
            }
            uint64_t started = mach_absolute_time();
            [command commit];
            [command waitUntilCompleted];
            totalWallMs += elapsedMilliseconds(mach_absolute_time() - started);
            if (command.status != MTLCommandBufferStatusCompleted) {
                setError(impl, [NSString stringWithFormat:@"grouped Metal expert dispatch failed: %@",
                    command.error.localizedDescription ?: @"unknown"]);
                return false;
            }
            if (command.GPUEndTime >= command.GPUStartTime)
                totalGpuMs += (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            ++commandCount;
        }
        const __fp16 *metalOutputs = impl.groupedOutputBuffer.contents;
        for (uint32_t index = 0; index < positionCount * dimension; ++index)
            outputs[index] = (float)metalOutputs[index];
        @synchronized (impl) {
            ANEFieldfareMetalExpertDispatchInfo info = impl.info;
            info.dispatchCount += commandCount;
            info.exactCacheHits += totalHits;
            info.exactCacheMisses += totalMisses;
            info.totalExactCacheFillMilliseconds += totalFillMs;
            info.totalWallMilliseconds += totalWallMs;
            info.totalGPUMilliseconds += totalGpuMs;
            impl.info = info;
        }
        impl.lastError = nil;
        return true;
    }
}

static bool tensorOffsets(ANEFieldfareMetalExpertDispatcherImpl *impl,
                          const uint8_t *base,
                          const ANEFieldfareTensor *matrix,
                          NSUInteger *weightOffset,
                          NSUInteger *scaleOffset,
                          NSUInteger *biasOffset) {
    if (!impl.modelBuffer || !base || !matrix || !matrix->rows || !matrix->columns ||
        matrix->columns % 64) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / 64;
    if (matrix->weightBytes != (elements + 1) / 2 ||
        matrix->scaleBytes != groups * 2 || matrix->biasBytes != groups * 2) return false;
    uintptr_t origin = (uintptr_t)base;
    uintptr_t pointers[3] = {
        (uintptr_t)matrix->weights, (uintptr_t)matrix->scales, (uintptr_t)matrix->biases};
    uint64_t lengths[3] = {matrix->weightBytes, matrix->scaleBytes, matrix->biasBytes};
    NSUInteger *offsets[3] = {weightOffset, scaleOffset, biasOffset};
    for (uint32_t index = 0; index < 3; ++index) {
        if (pointers[index] < origin) return false;
        uint64_t offset = pointers[index] - origin;
        if (offset > impl.modelMappingLength || lengths[index] > impl.modelMappingLength - offset)
            return false;
        *offsets[index] = (NSUInteger)offset;
    }
    return true;
}

static bool encodeAffine(ANEFieldfareMetalExpertDispatcherImpl *impl,
                         id<MTLCommandBuffer> command,
                         const uint8_t *base,
                         const ANEFieldfareTensor *matrix,
                         id<MTLBuffer> input,
                         NSUInteger inputOffset,
                         id<MTLBuffer> output,
                         NSUInteger outputOffset) {
    NSUInteger weightOffset = 0, scaleOffset = 0, biasOffset = 0;
    if (!tensorOffsets(impl, base, matrix, &weightOffset, &scaleOffset, &biasOffset)) return false;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:getenv("ANE_FIELDFARE_AFFINE_VEC4_GROUPS")
        ? impl.affine : impl.affineScalarGroups];
    [encoder setBuffer:impl.modelBuffer offset:weightOffset atIndex:0];
    [encoder setBuffer:impl.modelBuffer offset:scaleOffset atIndex:1];
    [encoder setBuffer:impl.modelBuffer offset:biasOffset atIndex:2];
    [encoder setBuffer:input offset:inputOffset atIndex:3];
    [encoder setBuffer:output offset:outputOffset atIndex:4];
    uint32_t rows = matrix->rows, columns = matrix->columns;
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return true;
}

static bool encodeAffinePositions(ANEFieldfareMetalExpertDispatcherImpl *impl,
                                  id<MTLCommandBuffer> command,
                                  const uint8_t *base,
                                  const ANEFieldfareTensor *matrix,
                                  uint32_t positions,
                                  id<MTLBuffer> input,
                                  id<MTLBuffer> output) {
    NSUInteger weightOffset = 0, scaleOffset = 0, biasOffset = 0;
    if (!positions || positions > 64 ||
        !tensorOffsets(impl, base, matrix, &weightOffset, &scaleOffset, &biasOffset)) return false;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:impl.affineBatched];
    [encoder setBuffer:impl.modelBuffer offset:weightOffset atIndex:0];
    [encoder setBuffer:impl.modelBuffer offset:scaleOffset atIndex:1];
    [encoder setBuffer:impl.modelBuffer offset:biasOffset atIndex:2];
    [encoder setBuffer:input offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    uint32_t rows = matrix->rows, columns = matrix->columns;
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:6];
    [encoder setBytes:&positions length:sizeof(positions) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, positions, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return true;
}

bool ANEFieldfareMetalExpertDispatcherEncodeExternalAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    CFTypeRef commandBuffer,
    CFTypeRef inputBuffer,
    CFTypeRef *outputBuffer) {
    @autoreleasepool {
        if (outputBuffer) *outputBuffer = NULL;
        if (!dispatcher || !matrix || !commandBuffer || !inputBuffer || !outputBuffer ||
            matrix->rows > 8192 || matrix->columns > 8192) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)commandBuffer;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputBuffer;
        if (!encodeAffine(impl, command, tensorIndexBase, matrix,
                          input, 0, impl.denseOutput, 0)) return false;
        *outputBuffer = CFBridgingRetain(impl.denseOutput);
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCompleteExternalAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    CFTypeRef commandBuffer,
    CFTypeRef outputBuffer,
    uint64_t startedTicks,
    float *output) {
    @autoreleasepool {
        if (!dispatcher || !commandBuffer || !outputBuffer || !output) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)commandBuffer;
        id<MTLBuffer> buffer = CFBridgingRelease(outputBuffer);
        if (command.status != MTLCommandBufferStatusCompleted) {
            setError(impl, command.error.localizedDescription ?: @"external affine failed");
            return false;
        }
        memcpy(output, buffer.contents,
               ANEFieldfareGemma4HiddenSize * sizeof(float));
        @synchronized (impl) {
            ANEFieldfareMetalExpertDispatchInfo info = impl.info;
            ++info.denseDispatchCount;
            ++info.attentionOutputDispatchCount;
            double wall = elapsedMilliseconds(mach_absolute_time() - startedTicks);
            double gpu = command.GPUEndTime >= command.GPUStartTime
                ? (command.GPUEndTime - command.GPUStartTime) * 1000.0 : 0.0;
            info.totalDenseWallMilliseconds += wall;
            info.totalDenseGPUMilliseconds += gpu;
            info.attentionOutputWallMilliseconds += wall;
            info.attentionOutputGPUMilliseconds += gpu;
            impl.info = info;
        }
        return true;
    }
}

typedef enum {
    DenseCommandAttentionInput,
    DenseCommandAttentionOutput,
    DenseCommandSharedFFN,
    DenseCommandHead,
} DenseCommandKind;

static bool finishDenseCommand(ANEFieldfareMetalExpertDispatcherImpl *impl,
                               id<MTLCommandBuffer> command,
                               uint64_t started,
                               DenseCommandKind kind) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        setError(impl, [NSString stringWithFormat:@"Metal dense dispatch failed: %@",
                                                   command.error.localizedDescription ?: @"unknown"]);
        return false;
    }
    @synchronized (impl) {
        ANEFieldfareMetalExpertDispatchInfo info = impl.info;
        info.denseDispatchCount += 1;
        double wall = elapsedMilliseconds(mach_absolute_time() - started);
        double gpu = command.GPUEndTime >= command.GPUStartTime
            ? (command.GPUEndTime - command.GPUStartTime) * 1000.0 : 0.0;
        info.totalDenseWallMilliseconds += wall;
        info.totalDenseGPUMilliseconds += gpu;
        if (kind == DenseCommandAttentionInput) {
            ++info.attentionInputDispatchCount;
            info.attentionInputWallMilliseconds += wall;
            info.attentionInputGPUMilliseconds += gpu;
        } else if (kind == DenseCommandAttentionOutput) {
            ++info.attentionOutputDispatchCount;
            info.attentionOutputWallMilliseconds += wall;
            info.attentionOutputGPUMilliseconds += gpu;
        } else if (kind == DenseCommandSharedFFN) {
            ++info.sharedFFNDispatchCount;
            info.sharedFFNWallMilliseconds += wall;
            info.sharedFFNGPUMilliseconds += gpu;
        } else {
            ++info.headDispatchCount;
            info.headWallMilliseconds += wall;
            info.headGPUMilliseconds += gpu;
        }
        impl.info = info;
    }
    return true;
}

bool ANEFieldfareMetalExpertDispatcherCallAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    float *output) {
    @autoreleasepool {
        if (!dispatcher || !matrix || !input || !output || matrix->columns > 8192 ||
            matrix->rows > 262144) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        float *metalInput = impl.denseInput.contents;
        for (uint32_t column = 0; column < matrix->columns; ++column)
            metalInput[column] = input[column];
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        if (!encodeAffine(impl, command, tensorIndexBase, matrix,
                          impl.denseInput, 0, impl.denseOutput, 0)) return false;
        uint64_t started = mach_absolute_time();
        DenseCommandKind kind = matrix->rows == ANEFieldfareGemmaVocabularySize
            ? DenseCommandHead : DenseCommandAttentionOutput;
        if (!finishDenseCommand(impl, command, started, kind)) return false;
        const float *values = impl.denseOutput.contents;
        memcpy(output, values, (size_t)matrix->rows * sizeof(*output));
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallGreedyHead(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    uint32_t *token,
    float *winningRawLogit) {
    @autoreleasepool {
        if (!dispatcher || !matrix || !input || !token || !winningRawLogit ||
            !matrix->rows || matrix->rows > ANEFieldfareGemmaVocabularySize ||
            !matrix->columns || matrix->columns > 8192) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        NSUInteger weightOffset = 0, scaleOffset = 0, biasOffset = 0;
        if (!tensorOffsets(impl, tensorIndexBase, matrix,
                           &weightOffset, &scaleOffset, &biasOffset)) return false;
        memcpy(impl.denseInput.contents, input,
               (size_t)matrix->columns * sizeof(*input));
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> rows = [command computeCommandEncoder];
        [rows setComputePipelineState:impl.greedyHeadRows];
        [rows setBuffer:impl.modelBuffer offset:weightOffset atIndex:0];
        [rows setBuffer:impl.modelBuffer offset:scaleOffset atIndex:1];
        [rows setBuffer:impl.modelBuffer offset:biasOffset atIndex:2];
        [rows setBuffer:impl.denseInput offset:0 atIndex:3];
        [rows setBuffer:impl.greedyHeadSummaries offset:0 atIndex:4];
        uint32_t rowCount = matrix->rows, columnCount = matrix->columns;
        const uint32_t groupCount = (rowCount + 7) / 8;
        [rows setBytes:&rowCount length:sizeof(rowCount) atIndex:5];
        [rows setBytes:&columnCount length:sizeof(columnCount) atIndex:6];
        [rows dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [rows endEncoding];

        id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
        [reduce setComputePipelineState:impl.greedyHeadReduce];
        [reduce setBuffer:impl.greedyHeadSummaries offset:0 atIndex:0];
        [reduce setBuffer:impl.greedyHeadToken offset:0 atIndex:1];
        [reduce setBuffer:impl.greedyHeadValue offset:0 atIndex:2];
        [reduce setBytes:&groupCount length:sizeof(groupCount) atIndex:3];
        [reduce dispatchThreads:MTLSizeMake(256, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [reduce endEncoding];
        uint64_t started = mach_absolute_time();
        if (!finishDenseCommand(impl, command, started, DenseCommandHead)) return false;
        *token = *(uint32_t *)impl.greedyHeadToken.contents;
        *winningRawLogit = *(float *)impl.greedyHeadValue.contents;
        impl.lastError = nil;
        return *token < matrix->rows && isfinite(*winningRawLogit);
    }
}

bool ANEFieldfareMetalExpertDispatcherCallGreedyHeadPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t positionCount,
    const float *inputs,
    uint32_t *tokens,
    float *winningRawLogits) {
    @autoreleasepool {
        if (!dispatcher || !matrix || !positionCount || positionCount > 64 ||
            !inputs || !tokens || !winningRawLogits || !matrix->rows ||
            matrix->rows > ANEFieldfareGemmaVocabularySize || !matrix->columns ||
            matrix->columns > 8192) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        NSUInteger weightOffset = 0, scaleOffset = 0, biasOffset = 0;
        if (!tensorOffsets(impl, tensorIndexBase, matrix,
                           &weightOffset, &scaleOffset, &biasOffset)) return false;
        memcpy(impl.denseInput.contents, inputs,
               (size_t)positionCount * matrix->columns * sizeof(*inputs));
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        id<MTLComputeCommandEncoder> rows = [command computeCommandEncoder];
        [rows setComputePipelineState:impl.greedyHeadRowsPositions];
        [rows setBuffer:impl.modelBuffer offset:weightOffset atIndex:0];
        [rows setBuffer:impl.modelBuffer offset:scaleOffset atIndex:1];
        [rows setBuffer:impl.modelBuffer offset:biasOffset atIndex:2];
        [rows setBuffer:impl.denseInput offset:0 atIndex:3];
        [rows setBuffer:impl.greedyHeadSummaries offset:0 atIndex:4];
        uint32_t rowCount = matrix->rows, columnCount = matrix->columns;
        const uint32_t groupCount = (rowCount + 7) / 8;
        [rows setBytes:&rowCount length:sizeof(rowCount) atIndex:5];
        [rows setBytes:&columnCount length:sizeof(columnCount) atIndex:6];
        [rows setBytes:&positionCount length:sizeof(positionCount) atIndex:7];
        [rows dispatchThreadgroups:MTLSizeMake(groupCount, positionCount, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [rows endEncoding];
        id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
        [reduce setComputePipelineState:impl.greedyHeadReducePositions];
        [reduce setBuffer:impl.greedyHeadSummaries offset:0 atIndex:0];
        [reduce setBuffer:impl.greedyHeadToken offset:0 atIndex:1];
        [reduce setBuffer:impl.greedyHeadValue offset:0 atIndex:2];
        [reduce setBytes:&groupCount length:sizeof(groupCount) atIndex:3];
        [reduce dispatchThreadgroups:MTLSizeMake(positionCount, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [reduce endEncoding];
        uint64_t started = mach_absolute_time();
        if (!finishDenseCommand(impl, command, started, DenseCommandHead)) return false;
        memcpy(tokens, impl.greedyHeadToken.contents,
               positionCount * sizeof(*tokens));
        memcpy(winningRawLogits, impl.greedyHeadValue.contents,
               positionCount * sizeof(*winningRawLogits));
        for (uint32_t position = 0; position < positionCount; ++position)
            if (tokens[position] >= matrix->rows || !isfinite(winningRawLogits[position]))
                return false;
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallAttentionInput(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput) {
    @autoreleasepool {
        if (!dispatcher || !query || !key || (!valueEqualsKey && !value) || !input ||
            !queryOutput || !keyOutput || !valueOutput || query->columns > 8192 ||
            key->columns != query->columns || (!valueEqualsKey && value->columns != query->columns) ||
            query->rows > 8192 || key->rows > 8192 || (!valueEqualsKey && value->rows > 8192))
            return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        float *metalInput = impl.denseInput.contents;
        memcpy(metalInput, input, (size_t)query->columns * sizeof(*input));
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        bool valid = encodeAffine(impl, command, tensorIndexBase, query,
                                  impl.denseInput, 0, impl.denseA, 0) &&
            encodeAffine(impl, command, tensorIndexBase, key,
                         impl.denseInput, 0, impl.denseB, 0);
        if (valid && !valueEqualsKey)
            valid = encodeAffine(impl, command, tensorIndexBase, value,
                                 impl.denseInput, 0, impl.denseOutput, 0);
        uint64_t started = mach_absolute_time();
        if (!valid || !finishDenseCommand(
                impl, command, started, DenseCommandAttentionInput)) return false;
        memcpy(queryOutput, impl.denseA.contents, (size_t)query->rows * sizeof(float));
        memcpy(keyOutput, impl.denseB.contents, (size_t)key->rows * sizeof(float));
        if (valueEqualsKey)
            memcpy(valueOutput, keyOutput, (size_t)key->rows * sizeof(float));
        else
            memcpy(valueOutput, impl.denseOutput.contents, (size_t)value->rows * sizeof(float));
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallProjectedAttention(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t position,
    const float normalizedHidden[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    @autoreleasepool {
        if (!dispatcher || !tensorIndexBase || !weights || !cache ||
            !normalizedHidden || !output || position > UINT32_MAX) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        const ANEFieldfareAttentionDescriptor *d = &weights->descriptor;
        const uint32_t queryWidth = d->queryHeads * d->headDimension;
        const uint32_t keyWidth = d->keyValueHeads * d->headDimension;
        if (!queryWidth || queryWidth > 8192 || !keyWidth || keyWidth > 8192 ||
            weights->queryProjection.columns != ANEFieldfareGemma4HiddenSize ||
            weights->outputProjection.rows != ANEFieldfareGemma4HiddenSize ||
            weights->outputProjection.columns != queryWidth ||
            (d->kind == ANEFieldfareAttentionFull && position >= cache->capacity)) return false;
        const NSUInteger page = (NSUInteger)getpagesize();
        const NSUInteger cacheBytes = (NSUInteger)cache->capacity * keyWidth * sizeof(float);
        const NSUInteger mappedCacheBytes = (cacheBytes + page - 1) / page * page;
        if (((uintptr_t)cache->keys % page) || ((uintptr_t)cache->values % page)) {
            setError(impl, @"fused attention KV cache is not page aligned");
            return false;
        }
        id<MTLBuffer> keys = [impl.device newBufferWithBytesNoCopy:cache->keys
            length:mappedCacheBytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> values = [impl.device newBufferWithBytesNoCopy:cache->values
            length:mappedCacheBytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!keys || !values) {
            setError(impl, @"Metal rejected fused-attention KV cache");
            return false;
        }
        memcpy(impl.denseInput.contents, normalizedHidden,
               ANEFieldfareGemma4HiddenSize * sizeof(float));
        memcpy(impl.attentionQueryNorm.contents, weights->queryNormValues,
               d->headDimension * sizeof(float));
        memcpy(impl.attentionKeyNorm.contents, weights->keyNormValues,
               d->headDimension * sizeof(float));
        const uint64_t slot = d->kind == ANEFieldfareAttentionFull
            ? position : position % cache->capacity;
        cache->positions[slot] = position;
        FusedAttentionParams params = {
            d->queryHeads, d->keyValueHeads, d->headDimension, d->rotaryAngles,
            1, (uint32_t)position, (uint32_t)cache->capacity, d->slidingWindow,
            d->kind == ANEFieldfareAttentionFull, d->ropeTheta, d->rmsEpsilon,
            d->attentionScale,
        };
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        bool valid = encodeAffine(impl, command, tensorIndexBase,
                                  &weights->queryProjection,
                                  impl.denseInput, 0, impl.denseA, 0) &&
            encodeAffine(impl, command, tensorIndexBase, &weights->keyProjection,
                         impl.denseInput, 0, impl.denseB, 0);
        if (valid && !d->valueEqualsKeyProjection)
            valid = encodeAffine(impl, command, tensorIndexBase,
                                 &weights->valueProjection,
                                 impl.denseInput, 0, impl.denseOutput, 0);
        if (!valid) return false;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl.attentionPrepare];
        [encoder setBuffer:impl.denseA offset:0 atIndex:0];
        [encoder setBuffer:impl.denseB offset:0 atIndex:1];
        [encoder setBuffer:d->valueEqualsKeyProjection ? impl.denseB : impl.denseOutput
                    offset:0 atIndex:2];
        [encoder setBuffer:impl.attentionQueryNorm offset:0 atIndex:3];
        [encoder setBuffer:impl.attentionKeyNorm offset:0 atIndex:4];
        [encoder setBuffer:impl.attentionQuery offset:0 atIndex:5];
        [encoder setBuffer:keys offset:0 atIndex:6];
        [encoder setBuffer:values offset:0 atIndex:7];
        [encoder setBytes:&params length:sizeof(params) atIndex:8];
        [encoder dispatchThreads:MTLSizeMake(d->queryHeads + d->keyValueHeads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                impl.attentionPrepare.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder setComputePipelineState:impl.attentionCausal];
        [encoder setBuffer:impl.attentionQuery offset:0 atIndex:0];
        [encoder setBuffer:keys offset:0 atIndex:1];
        [encoder setBuffer:values offset:0 atIndex:2];
        [encoder setBuffer:impl.attentionCombined offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        uint64_t causal = position + 1;
        if (d->kind == ANEFieldfareAttentionSliding && causal > d->slidingWindow)
            causal = d->slidingWindow;
        [encoder setThreadgroupMemoryLength:(NSUInteger)causal * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1, d->queryHeads, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (!encodeAffine(impl, command, tensorIndexBase, &weights->outputProjection,
                          impl.attentionCombined, 0, impl.denseInput, 0)) return false;
        const uint64_t started = mach_absolute_time();
        if (!finishDenseCommand(
                impl, command, started, DenseCommandAttentionInput)) return false;
        memcpy(output, impl.denseInput.contents,
               ANEFieldfareGemma4HiddenSize * sizeof(float));
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallAffinePositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t count,
    const float *inputs,
    float *outputs) {
    @autoreleasepool {
        if (!dispatcher || !matrix || !count || count > 64 || !inputs || !outputs ||
            matrix->columns > 8192 || matrix->rows > 8192) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        size_t inputBytes = (size_t)count * matrix->columns * sizeof(*inputs);
        size_t outputBytes = (size_t)count * matrix->rows * sizeof(*outputs);
        if (inputBytes > impl.denseInput.length || outputBytes > impl.denseOutput.length)
            return false;
        memcpy(impl.denseInput.contents, inputs, inputBytes);
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        uint64_t started = mach_absolute_time();
        if (!encodeAffinePositions(impl, command, tensorIndexBase, matrix, count,
                                   impl.denseInput, impl.denseOutput) ||
            !finishDenseCommand(impl, command, started, DenseCommandAttentionOutput)) return false;
        memcpy(outputs, impl.denseOutput.contents, outputBytes);
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallAttentionInputPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    uint32_t count,
    const float *inputs,
    float *queryOutputs,
    float *keyOutputs,
    float *valueOutputs) {
    @autoreleasepool {
        if (!dispatcher || !query || !key || (!valueEqualsKey && !value) || !count || count > 64 ||
            !inputs || !queryOutputs || !keyOutputs || !valueOutputs || query->columns > 8192 ||
            key->columns != query->columns || (!valueEqualsKey && value->columns != query->columns) ||
            query->rows > 8192 || key->rows > 8192 || (!valueEqualsKey && value->rows > 8192))
            return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        size_t inputBytes = (size_t)count * query->columns * sizeof(*inputs);
        size_t queryBytes = (size_t)count * query->rows * sizeof(*queryOutputs);
        size_t keyBytes = (size_t)count * key->rows * sizeof(*keyOutputs);
        size_t valueBytes = valueEqualsKey ? keyBytes
            : (size_t)count * value->rows * sizeof(*valueOutputs);
        if (inputBytes > impl.denseInput.length || queryBytes > impl.denseA.length ||
            keyBytes > impl.denseB.length || valueBytes > impl.denseOutput.length) return false;
        memcpy(impl.denseInput.contents, inputs, inputBytes);
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        bool valid = encodeAffinePositions(impl, command, tensorIndexBase, query, count,
                                           impl.denseInput, impl.denseA) &&
            encodeAffinePositions(impl, command, tensorIndexBase, key, count,
                                  impl.denseInput, impl.denseB);
        if (valid && !valueEqualsKey)
            valid = encodeAffinePositions(impl, command, tensorIndexBase, value, count,
                                          impl.denseInput, impl.denseOutput);
        uint64_t started = mach_absolute_time();
        if (!valid || !finishDenseCommand(
                impl, command, started, DenseCommandAttentionInput)) return false;
        memcpy(queryOutputs, impl.denseA.contents, queryBytes);
        memcpy(keyOutputs, impl.denseB.contents, keyBytes);
        if (valueEqualsKey) memcpy(valueOutputs, keyOutputs, keyBytes);
        else memcpy(valueOutputs, impl.denseOutput.contents, valueBytes);
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallSharedFFN(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    @autoreleasepool {
        if (!dispatcher || !gate || !up || !down || !input || !output ||
            gate->rows != ANEFieldfareGemma4SharedIntermediateSize ||
            up->rows != gate->rows || gate->columns != ANEFieldfareGemma4HiddenSize ||
            up->columns != gate->columns || down->rows != ANEFieldfareGemma4HiddenSize ||
            down->columns != gate->rows) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        float *metalInput = impl.denseInput.contents;
        for (uint32_t column = 0; column < gate->columns; ++column)
            metalInput[column] = input[column];
        id<MTLCommandBuffer> command = [impl.queue commandBuffer];
        if (!encodeAffine(impl, command, tensorIndexBase, gate,
                          impl.denseInput, 0, impl.denseA, 0) ||
            !encodeAffine(impl, command, tensorIndexBase, up,
                          impl.denseInput, 0, impl.denseB, 0)) return false;
        id<MTLComputeCommandEncoder> activation = [command computeCommandEncoder];
        [activation setComputePipelineState:impl.swiglu];
        [activation setBuffer:impl.denseA offset:0 atIndex:0];
        [activation setBuffer:impl.denseB offset:0 atIndex:1];
        uint32_t count = gate->rows;
        [activation setBytes:&count length:sizeof(count) atIndex:2];
        [activation dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                                                    impl.swiglu.maxTotalThreadsPerThreadgroup), 1, 1)];
        [activation endEncoding];
        if (!encodeAffine(impl, command, tensorIndexBase, down,
                          impl.denseA, 0, impl.denseOutput, 0)) return false;
        uint64_t started = mach_absolute_time();
        if (!finishDenseCommand(impl, command, started, DenseCommandSharedFFN)) return false;
        const float *values = impl.denseOutput.contents;
        memcpy(output, values, (size_t)down->rows * sizeof(*output));
        impl.lastError = nil;
        return true;
    }
}

bool ANEFieldfareMetalExpertDispatcherCallSharedFFNPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    uint32_t count,
    const float *inputs,
    float *outputs) {
    @autoreleasepool {
        if (!dispatcher || !gate || !up || !down || !count || count > 64 ||
            !inputs || !outputs || gate->rows != ANEFieldfareGemma4SharedIntermediateSize ||
            up->rows != gate->rows || gate->columns != ANEFieldfareGemma4HiddenSize ||
            up->columns != gate->columns || down->rows != ANEFieldfareGemma4HiddenSize ||
            down->columns != gate->rows) return false;
        ANEFieldfareMetalExpertDispatcherImpl *impl =
            (__bridge ANEFieldfareMetalExpertDispatcherImpl *)dispatcher;
        @synchronized (impl) {
            size_t inputBytes = (size_t)count * gate->columns * sizeof(*inputs);
            size_t outputBytes = (size_t)count * down->rows * sizeof(*outputs);
            if (inputBytes > impl.denseInput.length || outputBytes > impl.denseOutput.length)
                return false;
            memcpy(impl.denseInput.contents, inputs, inputBytes);
            id<MTLCommandBuffer> command = [impl.queue commandBuffer];
            if (!encodeAffinePositions(impl, command, tensorIndexBase, gate, count,
                                       impl.denseInput, impl.denseA) ||
                !encodeAffinePositions(impl, command, tensorIndexBase, up, count,
                                       impl.denseInput, impl.denseB)) return false;
            id<MTLComputeCommandEncoder> activation = [command computeCommandEncoder];
            [activation setComputePipelineState:impl.swiglu];
            [activation setBuffer:impl.denseA offset:0 atIndex:0];
            [activation setBuffer:impl.denseB offset:0 atIndex:1];
            uint32_t activationCount = count * gate->rows;
            [activation setBytes:&activationCount length:sizeof(activationCount) atIndex:2];
            [activation dispatchThreads:MTLSizeMake(activationCount, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                                                        impl.swiglu.maxTotalThreadsPerThreadgroup), 1, 1)];
            [activation endEncoding];
            if (!encodeAffinePositions(impl, command, tensorIndexBase, down, count,
                                       impl.denseA, impl.denseOutput)) return false;
            uint64_t started = mach_absolute_time();
            if (!finishDenseCommand(impl, command, started, DenseCommandSharedFFN)) return false;
            memcpy(outputs, impl.denseOutput.contents, outputBytes);
            impl.lastError = nil;
            return true;
        }
    }
}

ANEFieldfareMetalExpertDispatchInfo ANEFieldfareMetalExpertDispatcherGetInfo(
    const ANEFieldfareMetalExpertDispatcher *dispatcher) {
    if (!dispatcher) return (ANEFieldfareMetalExpertDispatchInfo){0};
    ANEFieldfareMetalExpertDispatcherImpl *impl =
        (__bridge ANEFieldfareMetalExpertDispatcherImpl *)(void *)dispatcher;
    return impl.info;
}

const char *ANEFieldfareMetalExpertDispatcherLastError(
    const ANEFieldfareMetalExpertDispatcher *dispatcher) {
    if (!dispatcher) return "dispatcher is null";
    ANEFieldfareMetalExpertDispatcherImpl *impl =
        (__bridge ANEFieldfareMetalExpertDispatcherImpl *)(void *)dispatcher;
    return impl.lastError.UTF8String;
}

void ANEFieldfareMetalExpertDispatcherDestroy(
    ANEFieldfareMetalExpertDispatcher *dispatcher) {
    if (dispatcher) CFBridgingRelease(dispatcher);
}
