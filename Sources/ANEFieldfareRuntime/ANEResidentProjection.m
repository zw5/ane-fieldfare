#import "ANEResidentProjection.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ANEProgram.h"
#include "Attention.h"
#include "Dequant.h"

enum {
    GroupSize = 64,
    HiddenSize = 2816,
    // Every ANE program built in this repository requires a 64-wide position
    // tile; a 1-wide tile fails surface mapping with 0x1D. C1.7 swept the tile
    // and found the extra lanes free to 128 positions, with fetch collapsing
    // from 63.9 to 9.5 GB/s past 256. Decode uses lane 0 and leaves the other
    // 63 zeroed. Batched prefill now fills those lanes through the production
    // decoder seam; batch-one decode remains a measured rejection.
    PositionTile = 64,
};

// One resident program plus the identity of the checkpoint tensors baked into
// it. Dispatch matches on weight offsets within the weights file rather than on
// a layer index, because the decoder's attention seams pass tensors, not
// layers, and rather than on absolute pointers, because the decoder holds its
// own mapping of the same file.
typedef struct {
    ANEFieldfareANEProgram *program;
    uint64_t queryOffset;
    uint64_t keyOffset;
    uint64_t valueOffset;
    uint32_t inputWidth;
    uint32_t queryRows;
    uint32_t keyRows;
    uint32_t valueRows;
    uint32_t outputChannels;
    uint32_t procedureBank;
    id<MTLBuffer> directOutputBuffer;
    bool valueStacked;
    bool built;
} ResidentProgram;

struct ANEFieldfareANEResidentProjections {
    ResidentProgram *inputs;
    ResidentProgram *outputs;
    uint32_t layerCount;
    char *lastError;
    char *eventFallbackReason;
    const uint8_t *indexBase;
    id<MTLDevice> eventDevice;
    id<MTLCommandQueue> eventQueue;
    id<MTLComputePipelineState> eventPipeline;
    id<MTLSharedEvent> sharedEvent;
    id<MTLBuffer> convertedOutput;
    uint64_t nextEventValue;
    ANEFieldfareANEResidentProjectionInfo info;
};

static mach_timebase_info_data_t timebase;

static double milliseconds(uint64_t elapsed) {
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)elapsed * timebase.numer / timebase.denom / 1.0e6;
}

static void setError(ANEFieldfareANEResidentProjections *projections, const char *message) {
    if (!projections) return;
    free(projections->lastError);
    projections->lastError = message ? strdup(message) : NULL;
}

static void disableEventPath(ANEFieldfareANEResidentProjections *projections,
                             const char *reason) {
    if (!projections) return;
    projections->info.sharedEventEnabled = false;
    if (reason && !projections->eventFallbackReason)
        projections->eventFallbackReason = strdup(reason);
    projections->eventPipeline = nil;
    projections->eventQueue = nil;
    projections->sharedEvent = nil;
    projections->convertedOutput = nil;
    for (uint32_t layer = 0; layer < projections->layerCount; ++layer) {
        if (projections->inputs) projections->inputs[layer].directOutputBuffer = nil;
        if (projections->outputs) projections->outputs[layer].directOutputBuffer = nil;
    }
}

static bool prepareDirectOutputBuffer(ANEFieldfareANEResidentProjections *projections,
                                      ResidentProgram *slot) {
    if (!slot->built || slot->directOutputBuffer) return true;
    IOSurfaceRef surface = ANEFieldfareANEProgramOutput(slot->program, slot->procedureBank);
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    void *base = IOSurfaceGetBaseAddress(surface);
    size_t allocation = IOSurfaceGetAllocSize(surface);
    size_t required = (size_t)slot->outputChannels * PositionTile * sizeof(__fp16);
    long pageSize = sysconf(_SC_PAGESIZE);
    bool guarded = base && pageSize > 0 && allocation >= required &&
        ((uintptr_t)base % (uintptr_t)pageSize) == 0 &&
        allocation % (size_t)pageSize == 0;
    id<MTLBuffer> buffer = guarded ?
        [projections->eventDevice newBufferWithBytesNoCopy:base length:allocation
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil] : nil;
    bool identity = buffer && buffer.contents == base;
    IOSurfaceUnlock(surface, 0, NULL);
    if (!identity) return false;
    slot->directOutputBuffer = buffer;
    return true;
}

static void prepareEventPath(ANEFieldfareANEResidentProjections *projections) {
    projections->info.sharedEventRequested =
        getenv("ANE_FIELDFARE_ANE_METAL_EVENT") != NULL ||
        getenv("ANE_FIELDFARE_DEVICE_PREFILL") != NULL;
    if (!projections->info.sharedEventRequested || !projections->info.promotedLayers) return;
    if (getenv("ANE_FIELDFARE_ANE_METAL_EVENT_FORCE_FALLBACK")) {
        disableEventPath(projections, "forced shared-event fail-closed control");
        return;
    }
    projections->eventDevice = MTLCreateSystemDefaultDevice();
    projections->eventQueue = [projections->eventDevice newCommandQueue];
    projections->sharedEvent = [projections->eventDevice newSharedEvent];
    NSError *error = nil;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal; "
        "kernel void ane_fp16_to_fp32(const device half *input [[buffer(0)]], "
        "device float *output [[buffer(1)]], uint index [[thread_position_in_grid]]) "
        "{ output[index] = float(input[index]); }";
    id<MTLLibrary> library = [projections->eventDevice newLibraryWithSource:source
                                                                    options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"ane_fp16_to_fp32"];
    projections->eventPipeline = function ?
        [projections->eventDevice newComputePipelineStateWithFunction:function error:&error] : nil;
    projections->convertedOutput = [projections->eventDevice
        newBufferWithLength:(size_t)8192 * PositionTile * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (!projections->eventDevice || !projections->eventQueue || !projections->sharedEvent ||
        !projections->eventPipeline || !projections->convertedOutput) {
        disableEventPath(projections, "Metal shared-event companion construction failed");
        return;
    }
    for (uint32_t layer = 0; layer < projections->info.promotedLayers; ++layer) {
        if (!prepareDirectOutputBuffer(projections, &projections->inputs[layer]) ||
            !prepareDirectOutputBuffer(projections, &projections->outputs[layer])) {
            disableEventPath(projections, "ANE output IOSurface Metal-buffer guard failed");
            return;
        }
    }
    projections->nextEventValue = projections->sharedEvent.signaledValue + 1;
    projections->info.sharedEventEnabled = true;
}

// conv weight is [out_channels, in_channels, 1, 1] row major, which is exactly
// the checkpoint's row-major affine layout once each row is dequantized. This
// is the same reasoning the C1.2 gate relies on: no transpose, unlike the
// expert carrier.
typedef struct {
    const ANEFieldfareTensor *tensor;
    __fp16 *destination;
    uint32_t groupsPerRow;
} ResidentDequantWork;

static void dequantizeResidentRow(void *raw, size_t row) {
    ResidentDequantWork *work = raw;
    const ANEFieldfareTensor *tensor = work->tensor;
    ANEFieldfareDequantGroupsFP16(
        tensor->weights + (uint64_t)row * tensor->columns / 2,
        tensor->scales + (uint64_t)row * work->groupsPerRow * 2,
        tensor->biases + (uint64_t)row * work->groupsPerRow * 2,
        work->groupsPerRow,
        work->destination + (uint64_t)row * tensor->columns);
}

static void dequantizeRows(const ANEFieldfareTensor *tensor, __fp16 *destination) {
    uint32_t groupsPerRow = tensor->columns / GroupSize;
    ResidentDequantWork work = {tensor, destination, groupsPerRow};
    dispatch_apply_f(tensor->rows,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &work, dequantizeResidentRow);
}

// The C1.2 program at the minimum position tile.
static NSString *programText(uint32_t inputWidth, uint32_t outputChannels) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp16,[1,%u,1,%u]> x) {\n"
         "    string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\n"
         "    tensor<int32,[2]> st=const()[name=string(\"st\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd=const()[name=string(\"pd\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl=const()[name=string(\"dl\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr=const()[name=string(\"gr\"),val=int32(1)];\n"
         "    tensor<fp16,[%u,%u,1,1]> W=const()[name=string(\"W\"),"
         "val=tensor<fp16,[%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"),"
         "offset=uint64(64)))];\n"
         "    tensor<fp16,[1,%u,1,%u]> y=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,"
         "strides=st,weight=W,x=x)[name=string(\"y\")];\n"
         "  } -> (y);\n"
         "}\n",
        inputWidth, (uint32_t)PositionTile, outputChannels, inputWidth, outputChannels,
        inputWidth, outputChannels, (uint32_t)PositionTile];
}

static NSString *twoProcedureProgramText(uint32_t qkvInputWidth,
                                         uint32_t qkvOutputChannels,
                                         uint32_t outputInputWidth,
                                         uint32_t outputChannels) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func qkv<ios18>(tensor<fp16,[1,%u,1,%u]> x0) {\n"
         "    string pt0=const()[name=string(\"pt0\"),val=string(\"valid\")];\n"
         "    tensor<int32,[2]> st0=const()[name=string(\"st0\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd0=const()[name=string(\"pd0\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl0=const()[name=string(\"dl0\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr0=const()[name=string(\"gr0\"),val=int32(1)];\n"
         "    tensor<fp16,[%u,%u,1,1]> W0=const()[name=string(\"W0\"),val=tensor<fp16,[%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/qkv.bin\"),offset=uint64(64)))];\n"
         "    tensor<fp16,[1,%u,1,%u]> y0=conv(dilations=dl0,groups=gr0,pad=pd0,pad_type=pt0,strides=st0,weight=W0,x=x0)[name=string(\"y0\")];\n"
         "  } -> (y0);\n"
         "  func output<ios18>(tensor<fp16,[1,%u,1,%u]> x1) {\n"
         "    string pt1=const()[name=string(\"pt1\"),val=string(\"valid\")];\n"
         "    tensor<int32,[2]> st1=const()[name=string(\"st1\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd1=const()[name=string(\"pd1\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl1=const()[name=string(\"dl1\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr1=const()[name=string(\"gr1\"),val=int32(1)];\n"
         "    tensor<fp16,[%u,%u,1,1]> W1=const()[name=string(\"W1\"),val=tensor<fp16,[%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/o.bin\"),offset=uint64(64)))];\n"
         "    tensor<fp16,[1,%u,1,%u]> y1=conv(dilations=dl1,groups=gr1,pad=pd1,pad_type=pt1,strides=st1,weight=W1,x=x1)[name=string(\"y1\")];\n"
         "  } -> (y1);\n"
         "}\n",
        qkvInputWidth, (uint32_t)PositionTile,
        qkvOutputChannels, qkvInputWidth, qkvOutputChannels, qkvInputWidth,
        qkvOutputChannels, (uint32_t)PositionTile,
        outputInputWidth, (uint32_t)PositionTile,
        outputChannels, outputInputWidth, outputChannels, outputInputWidth,
        outputChannels, (uint32_t)PositionTile];
}

// Stacks `tensors` along output channels into one resident constant program.
// They must share an input width, which is what makes Q/K/V fusible: one
// convolution over the 2816-wide hidden state produces all three.
static bool buildProgram(ANEFieldfareANEResidentProjections *projections,
                         ResidentProgram *slot,
                         const ANEFieldfareTensor *tensors,
                         uint32_t tensorCount) {
    uint32_t inputWidth = tensors[0].columns;
    uint32_t outputChannels = 0;
    for (uint32_t i = 0; i < tensorCount; ++i) {
        if (tensors[i].columns != inputWidth) {
            setError(projections, "projection input width mismatch");
            return false;
        }
        if (tensors[i].columns % GroupSize) {
            setError(projections, "projection width is not group aligned");
            return false;
        }
        outputChannels += tensors[i].rows;
    }

    uint64_t elements = (uint64_t)outputChannels * inputWidth;
    uint64_t blobBytes = 0;
    void *payload = NULL;
    void *blob = ANEFieldfareANEWeightBlobCreate(elements, &blobBytes, &payload);
    if (!blob) {
        setError(projections, "resident projection weight blob allocation failed");
        return false;
    }
    __fp16 *destination = payload;
    for (uint32_t i = 0; i < tensorCount; ++i) {
        dequantizeRows(&tensors[i], destination);
        destination += (uint64_t)tensors[i].rows * inputWidth;
    }

    NSString *mil = programText(inputWidth, outputChannels);
    const char *error = NULL;
    ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithConstantWeights(
        mil.UTF8String, "w.bin", blob, blobBytes,
        (uint64_t)inputWidth * PositionTile * 2,
        (uint64_t)outputChannels * PositionTile * 2, 1, &error);
    free(blob);
    if (!program) {
        setError(projections, error ? error : "resident projection program creation failed");
        free((void *)error);
        return false;
    }
    free((void *)error);

    IOSurfaceRef carrier = ANEFieldfareANEProgramInput(program, 0);
    if (carrier && IOSurfaceLock(carrier, 0, NULL) == kIOReturnSuccess) {
        void *base = IOSurfaceGetBaseAddress(carrier);
        if (base) memset(base, 0, (size_t)inputWidth * PositionTile * 2);
        IOSurfaceUnlock(carrier, 0, NULL);
    }

    ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
    slot->program = program;
    slot->inputWidth = inputWidth;
    slot->outputChannels = outputChannels;
    slot->built = true;
    projections->info.programCount += 1;
    projections->info.compileCount += 1;
    projections->info.loadCount += 1;
    projections->info.compileMilliseconds += info.compileMilliseconds;
    projections->info.loadMilliseconds += info.loadMilliseconds;
    projections->info.constantBytes += elements * 2;
    return true;
}

static void *projectionBlob(const ANEFieldfareTensor *tensors,
                            uint32_t tensorCount,
                            uint64_t *blobBytes,
                            uint64_t *constantBytes) {
    uint32_t inputWidth = tensors[0].columns;
    uint64_t outputChannels = 0;
    for (uint32_t index = 0; index < tensorCount; ++index) {
        if (tensors[index].columns != inputWidth || tensors[index].columns % GroupSize)
            return NULL;
        outputChannels += tensors[index].rows;
    }
    uint64_t elements = outputChannels * inputWidth;
    void *payload = NULL;
    void *blob = ANEFieldfareANEWeightBlobCreate(elements, blobBytes, &payload);
    if (!blob) return NULL;
    __fp16 *destination = payload;
    for (uint32_t index = 0; index < tensorCount; ++index) {
        dequantizeRows(&tensors[index], destination);
        destination += (uint64_t)tensors[index].rows * inputWidth;
    }
    *constantBytes = elements * 2;
    return blob;
}

static bool buildTwoProcedureLayer(ANEFieldfareANEResidentProjections *projections,
                                   ResidentProgram *inputSlot,
                                   ResidentProgram *outputSlot,
                                   const ANEFieldfareTensor *inputTensors,
                                   uint32_t inputTensorCount,
                                   const ANEFieldfareTensor *outputTensor) {
    uint64_t qkvBlobBytes = 0, outputBlobBytes = 0;
    uint64_t qkvConstantBytes = 0, outputConstantBytes = 0;
    void *qkvBlob = projectionBlob(inputTensors, inputTensorCount,
                                   &qkvBlobBytes, &qkvConstantBytes);
    void *outputBlob = projectionBlob(outputTensor, 1,
                                      &outputBlobBytes, &outputConstantBytes);
    if (!qkvBlob || !outputBlob) {
        free(qkvBlob); free(outputBlob);
        setError(projections, "resident two-procedure weight blob allocation failed");
        return false;
    }
    uint32_t qkvOutputChannels = 0;
    for (uint32_t index = 0; index < inputTensorCount; ++index)
        qkvOutputChannels += inputTensors[index].rows;
    NSString *mil = twoProcedureProgramText(
        inputTensors[0].columns, qkvOutputChannels,
        outputTensor->columns, outputTensor->rows);
    const char *error = NULL;
    ANEFieldfareANEProgram *program =
        ANEFieldfareANEProgramCreateWithTwoProceduresAndConstantWeights(
            mil.UTF8String,
            "qkv.bin", qkvBlob, qkvBlobBytes,
            "o.bin", outputBlob, outputBlobBytes,
            "qkv", "output", &error);
    free(qkvBlob); free(outputBlob);
    if (!program) {
        setError(projections, error ? error : "resident two-procedure program creation failed");
        free((void *)error);
        return false;
    }
    free((void *)error);

    inputSlot->program = program;
    inputSlot->procedureBank = 0;
    inputSlot->inputWidth = inputTensors[0].columns;
    inputSlot->outputChannels = qkvOutputChannels;
    inputSlot->built = true;
    outputSlot->program = program;
    outputSlot->procedureBank = 1;
    outputSlot->inputWidth = outputTensor->columns;
    outputSlot->outputChannels = outputTensor->rows;
    outputSlot->built = true;

    ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
    projections->info.programCount += 1;
    projections->info.compileCount += 1;
    projections->info.loadCount += 1;
    projections->info.compileMilliseconds += info.compileMilliseconds;
    projections->info.loadMilliseconds += info.loadMilliseconds;
    projections->info.constantBytes += qkvConstantBytes + outputConstantBytes;
    return true;
}

static bool findLayerTensor(const ANEFieldfareTensorIndex *index,
                            uint32_t layer,
                            const char *suffix,
                            ANEFieldfareTensor *tensor) {
    char name[256];
    snprintf(name, sizeof(name), "language_model.model.layers.%u.self_attn.%s", layer, suffix);
    return ANEFieldfareTensorIndexFind(index, name, tensor);
}

ANEFieldfareANEResidentProjections *ANEFieldfareANEResidentProjectionsCreate(
    const ANEFieldfareTensorIndex *index,
    uint32_t layerCount,
    uint64_t budgetBytes,
    const char **error) {
    if (!index || !layerCount) {
        if (error) *error = strdup("resident projections require an index and a layer count");
        return NULL;
    }
    ANEFieldfareANEResidentProjections *projections = calloc(1, sizeof(*projections));
    if (!projections) {
        if (error) *error = strdup("resident projection allocation failed");
        return NULL;
    }
    projections->layerCount = layerCount;
    projections->inputs = calloc(layerCount, sizeof(*projections->inputs));
    projections->outputs = calloc(layerCount, sizeof(*projections->outputs));
    if (!projections->inputs || !projections->outputs) {
        ANEFieldfareANEResidentProjectionsDestroy(projections);
        if (error) *error = strdup("resident projection allocation failed");
        return NULL;
    }
    projections->indexBase = index->bytes;
    projections->info.requestedLayers = layerCount;
    projections->info.budgetBytes = budgetBytes;

    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        ANEFieldfareAttentionDescriptor descriptor = {0};
        if (!ANEFieldfareAttentionDescriptorForLayer(layer, &descriptor)) {
            setError(projections, "attention descriptor unavailable for layer");
            break;
        }
        // Only the full-attention layers share one projection between K and V,
        // and those layers have no v_proj tensor to find. Sliding layers carry a
        // distinct v_proj that has to be stacked. This is the same rule
        // ANEFieldfareAttentionWeightsLoad applies, deliberately mirrored rather
        // than inferred from tensor pointers.
        bool valueStacked = !descriptor.valueEqualsKeyProjection;
        ANEFieldfareTensor query = {0}, key = {0}, value = {0}, output = {0};
        if (!findLayerTensor(index, layer, "q_proj.weight", &query) ||
            !findLayerTensor(index, layer, "k_proj.weight", &key) ||
            !findLayerTensor(index, layer, "o_proj.weight", &output) ||
            (valueStacked && !findLayerTensor(index, layer, "v_proj.weight", &value))) {
            setError(projections, "attention projection tensors missing from the index");
            break;
        }
        ANEFieldfareTensor stacked[3] = {query, key, value};
        uint32_t stackedCount = valueStacked ? 3 : 2;

        uint64_t inputRows = query.rows + key.rows + (valueStacked ? value.rows : 0);
        uint64_t inputBytes = inputRows * query.columns * 2;
        uint64_t outputBytes = (uint64_t)output.rows * output.columns * 2;
        // Promote a layer only if BOTH of its programs fit. A layer with a
        // resident QKV and a Metal output projection would straddle two engines
        // inside one attention block for no measured benefit.
        if (projections->info.constantBytes + inputBytes + outputBytes > budgetBytes) break;

        bool twoProcedure = buildTwoProcedureLayer(
            projections, &projections->inputs[layer], &projections->outputs[layer],
            stacked, stackedCount, &output);
        if (!twoProcedure) {
            // Preserve the previous two-program topology as a host/OS fallback.
            // It is not the promoted result on hosts where compiler-authored
            // procedure metadata and mapping validate.
            if (!buildProgram(projections, &projections->inputs[layer], stacked, stackedCount)) break;
            if (!buildProgram(projections, &projections->outputs[layer], &output, 1)) break;
            setError(projections, NULL);
        }
        projections->inputs[layer].queryOffset = (uint64_t)(query.weights - index->bytes);
        projections->inputs[layer].keyOffset = (uint64_t)(key.weights - index->bytes);
        projections->inputs[layer].valueOffset =
            valueStacked ? (uint64_t)(value.weights - index->bytes) : 0;
        projections->inputs[layer].queryRows = query.rows;
        projections->inputs[layer].keyRows = key.rows;
        projections->inputs[layer].valueRows = value.rows;
        projections->inputs[layer].valueStacked = valueStacked;

        projections->outputs[layer].queryOffset = (uint64_t)(output.weights - index->bytes);

        projections->info.promotedLayers += 1;
    }

    prepareEventPath(projections);
    if (error) *error = NULL;
    return projections;
}

// Writes row-major positions into the compiler's channel-major 64-lane tile.
// The unused tail is cleared because a prior wider prefill may have populated
// it. Decode is the count=1 case and therefore shares the exact same program.
static bool evaluatePositions(ANEFieldfareANEResidentProjections *projections,
                              ResidentProgram *slot,
                              uint32_t count,
                              const float *inputs,
                              const void **values,
                              bool *valuesAreFP32,
                              IOSurfaceRef *outputSurface,
                              double *elapsedMs) {
    if (!count || count > PositionTile) return false;
    IOSurfaceRef surface = ANEFieldfareANEProgramInput(slot->program, slot->procedureBank);
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
        setError(projections, "resident projection input surface lock failed");
        return false;
    }
    __fp16 *carrier = IOSurfaceGetBaseAddress(surface);
    if (!carrier) {
        IOSurfaceUnlock(surface, 0, NULL);
        setError(projections, "resident projection input surface has no base address");
        return false;
    }
    for (uint32_t column = 0; column < slot->inputWidth; ++column) {
        __fp16 *lanes = carrier + (uint64_t)column * PositionTile;
        for (uint32_t position = 0; position < count; ++position)
            lanes[position] = (__fp16)inputs[(uint64_t)position * slot->inputWidth + column];
        for (uint32_t position = count; position < PositionTile; ++position)
            lanes[position] = 0;
    }
    IOSurfaceUnlock(surface, 0, NULL);

    uint64_t started = mach_absolute_time();
    if (projections->info.sharedEventEnabled && slot->directOutputBuffer) {
        uint64_t eventValue = projections->nextEventValue++;
        const char *eventError = NULL;
        bool attached = ANEFieldfareANEProgramSetOutputSignalEvent(
            slot->program, slot->procedureBank,
            (__bridge CFTypeRef)projections->sharedEvent, 0, eventValue, &eventError);
        if (!attached) {
            disableEventPath(projections, eventError ? eventError :
                "ANE request shared-event attachment failed");
            free((void *)eventError);
        } else {
            free((void *)eventError);
            NSUInteger elementCount = (NSUInteger)slot->outputChannels * PositionTile;
            id<MTLCommandBuffer> command = [projections->eventQueue commandBuffer];
            [command encodeWaitForEvent:projections->sharedEvent value:eventValue];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:projections->eventPipeline];
            [encoder setBuffer:slot->directOutputBuffer offset:0 atIndex:0];
            [encoder setBuffer:projections->convertedOutput offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(elementCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                    MIN((NSUInteger)256,
                        projections->eventPipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
            [command commit];

            const char *beginError = NULL;
            ANEFieldfareANEEvaluationTiming beginTiming = {0};
            bool began = ANEFieldfareANEProgramEvaluateBegin(
                slot->program, slot->procedureBank, &beginTiming, &beginError);
            if (!began) projections->sharedEvent.signaledValue = eventValue;
            [command waitUntilCompleted];
            projections->info.metalCompletionHostWaits += 1;
            bool metalCompleted = command.status == MTLCommandBufferStatusCompleted;

            const char *drainError = NULL;
            ANEFieldfareANEEvaluationTiming completionTiming = {0};
            bool drained = began && ANEFieldfareANEProgramEvaluateWait(
                slot->program, slot->procedureBank, 0, &completionTiming, &drainError);
            if (drained) {
                projections->info.zeroTimeoutANECompletionDrains += 1;
            } else if (began) {
                free((void *)drainError); drainError = NULL;
                drained = ANEFieldfareANEProgramEvaluateWait(
                    slot->program, slot->procedureBank, 5 * NSEC_PER_SEC,
                    &completionTiming, &drainError);
                projections->info.blockingANEHostWaits += 1;
            }
            bool eventSucceeded = began && metalCompleted && drained;
            free((void *)beginError); free((void *)drainError);
            if (eventSucceeded) {
                *elapsedMs = milliseconds(mach_absolute_time() - started);
                projections->info.sharedEventEvaluations += 1;
                projections->info.sharedEventEvaluationMilliseconds += *elapsedMs;
                *values = projections->convertedOutput.contents;
                *valuesAreFP32 = true;
                *outputSurface = NULL;
                return true;
            }
            disableEventPath(projections, !began ? "ANE shared-event submission failed" :
                (!metalCompleted ? "Metal shared-event consumer failed" :
                 "ANE completion drain after Metal signal failed"));
            // If ANE ran, its output remains authoritative and can be consumed
            // through the established host path below without re-evaluation.
            if (began && drained) {
                projections->info.sharedEventFallbacks += 1;
                IOSurfaceRef result = ANEFieldfareANEProgramOutput(
                    slot->program, slot->procedureBank);
                if (!result || IOSurfaceLock(result, kIOSurfaceLockReadOnly, NULL) !=
                                   kIOReturnSuccess) return false;
                const __fp16 *base = IOSurfaceGetBaseAddress(result);
                if (!base) {
                    IOSurfaceUnlock(result, kIOSurfaceLockReadOnly, NULL);
                    return false;
                }
                *elapsedMs = milliseconds(mach_absolute_time() - started);
                *values = base;
                *valuesAreFP32 = false;
                *outputSurface = result;
                return true;
            }
        }
    }

    if (projections->info.sharedEventRequested) projections->info.sharedEventFallbacks += 1;
    ANEFieldfareANEEvaluationTiming timing = {0};
    const char *evaluationError = NULL;
    bool evaluated = ANEFieldfareANEProgramEvaluate(
        slot->program, slot->procedureBank, 5 * NSEC_PER_SEC, &timing, &evaluationError);
    projections->info.blockingANEHostWaits += 1;
    *elapsedMs = milliseconds(mach_absolute_time() - started);
    if (!evaluated) {
        setError(projections, evaluationError ? evaluationError :
            "resident projection evaluate failed");
        free((void *)evaluationError);
        return false;
    }
    free((void *)evaluationError);

    IOSurfaceRef result = ANEFieldfareANEProgramOutput(slot->program, slot->procedureBank);
    if (!result || IOSurfaceLock(result, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        setError(projections, "resident projection output surface lock failed");
        return false;
    }
    const __fp16 *base = IOSurfaceGetBaseAddress(result);
    if (!base) {
        IOSurfaceUnlock(result, kIOSurfaceLockReadOnly, NULL);
        setError(projections, "resident projection output surface has no base address");
        return false;
    }
    *values = base;
    *valuesAreFP32 = false;
    *outputSurface = result;
    return true;
}

static float projectionValue(const void *values, bool valuesAreFP32, uint64_t index) {
    return valuesAreFP32 ? ((const float *)values)[index]
                         : (float)((const __fp16 *)values)[index];
}

static ResidentProgram *matchInputProgram(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey) {
    uint64_t queryOffset = (uint64_t)(query->weights - tensorIndexBase);
    ResidentProgram *slot = NULL;
    for (uint32_t layer = 0; layer < projections->info.promotedLayers; ++layer) {
        if (projections->inputs[layer].built &&
            projections->inputs[layer].queryOffset == queryOffset) {
            slot = &projections->inputs[layer];
            break;
        }
    }
    if (!slot || slot->keyOffset != (uint64_t)(key->weights - tensorIndexBase) ||
        (!valueEqualsKey &&
         (!value || slot->valueOffset != (uint64_t)(value->weights - tensorIndexBase))) ||
        valueEqualsKey == slot->valueStacked) return NULL;
    return slot;
}

bool ANEFieldfareANEResidentProjectionsCallAttentionInput(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput) {
    if (!projections || !tensorIndexBase || !query || !key || !input ||
        !queryOutput || !keyOutput || !valueOutput) return false;
    ResidentProgram *slot = matchInputProgram(
        projections, tensorIndexBase, query, key, value, valueEqualsKey);
    // Match the whole baked identity, not just the query tensor: a layer whose
    // key or value tensor differs from what was stacked must fall back rather
    // than read another layer's channels.
    if (!slot) {
        projections->info.fallbacks += 1;
        return false;
    }

    const void *values = NULL;
    bool valuesAreFP32 = false;
    IOSurfaceRef surface = NULL;
    double elapsed = 0;
    if (!evaluatePositions(projections, slot, 1, input, &values, &valuesAreFP32,
                           &surface, &elapsed)) return false;

    for (uint32_t row = 0; row < slot->queryRows; ++row)
        queryOutput[row] = projectionValue(values, valuesAreFP32,
                                           (uint64_t)row * PositionTile);
    for (uint32_t row = 0; row < slot->keyRows; ++row)
        keyOutput[row] = projectionValue(
            values, valuesAreFP32, (uint64_t)(slot->queryRows + row) * PositionTile);
    if (slot->valueStacked) {
        for (uint32_t row = 0; row < slot->valueRows; ++row)
            valueOutput[row] = projectionValue(
                values, valuesAreFP32,
                (uint64_t)(slot->queryRows + slot->keyRows + row) * PositionTile);
    } else {
        memcpy(valueOutput, keyOutput, (size_t)slot->keyRows * sizeof(float));
    }
    if (surface) IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);

    projections->info.inputEvaluations += 1;
    projections->info.inputPositions += 1;
    projections->info.inputEvaluationMilliseconds += elapsed;
    return true;
}

bool ANEFieldfareANEResidentProjectionsCallAttentionInputPositions(
    ANEFieldfareANEResidentProjections *projections,
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
    if (!projections || !tensorIndexBase || !query || !key || !inputs ||
        !queryOutputs || !keyOutputs || !valueOutputs || !count || count > PositionTile)
        return false;
    ResidentProgram *slot = matchInputProgram(
        projections, tensorIndexBase, query, key, value, valueEqualsKey);
    if (!slot) {
        projections->info.fallbacks += count;
        return false;
    }
    const void *values = NULL;
    bool valuesAreFP32 = false;
    IOSurfaceRef surface = NULL;
    double elapsed = 0;
    if (!evaluatePositions(projections, slot, count, inputs, &values, &valuesAreFP32,
                           &surface, &elapsed))
        return false;
    for (uint32_t position = 0; position < count; ++position) {
        for (uint32_t row = 0; row < slot->queryRows; ++row)
            queryOutputs[(uint64_t)position * slot->queryRows + row] =
                projectionValue(values, valuesAreFP32,
                                (uint64_t)row * PositionTile + position);
        for (uint32_t row = 0; row < slot->keyRows; ++row)
            keyOutputs[(uint64_t)position * slot->keyRows + row] =
                projectionValue(values, valuesAreFP32,
                    (uint64_t)(slot->queryRows + row) * PositionTile + position);
        if (slot->valueStacked) {
            for (uint32_t row = 0; row < slot->valueRows; ++row)
                valueOutputs[(uint64_t)position * slot->valueRows + row] =
                    projectionValue(values, valuesAreFP32,
                        (uint64_t)(slot->queryRows + slot->keyRows + row) *
                            PositionTile + position);
        } else {
            memcpy(valueOutputs + (uint64_t)position * slot->keyRows,
                   keyOutputs + (uint64_t)position * slot->keyRows,
                   (size_t)slot->keyRows * sizeof(float));
        }
    }
    if (surface) IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    projections->info.inputEvaluations += 1;
    projections->info.inputPositions += count;
    projections->info.inputEvaluationMilliseconds += elapsed;
    return true;
}

bool ANEFieldfareANEResidentProjectionsCallAffine(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    float *output) {
    if (!projections || !tensorIndexBase || !matrix || !input || !output) return false;
    uint64_t offset = (uint64_t)(matrix->weights - tensorIndexBase);
    ResidentProgram *slot = NULL;
    for (uint32_t layer = 0; layer < projections->info.promotedLayers; ++layer) {
        if (projections->outputs[layer].built &&
            projections->outputs[layer].queryOffset == offset) {
            slot = &projections->outputs[layer];
            break;
        }
    }
    if (!slot) {
        projections->info.fallbacks += 1;
        return false;
    }

    const void *values = NULL;
    bool valuesAreFP32 = false;
    IOSurfaceRef surface = NULL;
    double elapsed = 0;
    if (!evaluatePositions(projections, slot, 1, input, &values, &valuesAreFP32,
                           &surface, &elapsed)) return false;
    for (uint32_t row = 0; row < slot->outputChannels; ++row)
        output[row] = projectionValue(values, valuesAreFP32,
                                      (uint64_t)row * PositionTile);
    if (surface) IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);

    projections->info.outputEvaluations += 1;
    projections->info.outputPositions += 1;
    projections->info.outputEvaluationMilliseconds += elapsed;
    return true;
}

bool ANEFieldfareANEResidentProjectionsCallAffinePositions(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t count,
    const float *inputs,
    float *outputs) {
    if (!projections || !tensorIndexBase || !matrix || !inputs || !outputs ||
        !count || count > PositionTile) return false;
    uint64_t offset = (uint64_t)(matrix->weights - tensorIndexBase);
    ResidentProgram *slot = NULL;
    for (uint32_t layer = 0; layer < projections->info.promotedLayers; ++layer)
        if (projections->outputs[layer].built &&
            projections->outputs[layer].queryOffset == offset) {
            slot = &projections->outputs[layer];
            break;
        }
    if (!slot) {
        projections->info.fallbacks += count;
        return false;
    }
    const void *values = NULL;
    bool valuesAreFP32 = false;
    IOSurfaceRef surface = NULL;
    double elapsed = 0;
    if (!evaluatePositions(projections, slot, count, inputs, &values, &valuesAreFP32,
                           &surface, &elapsed))
        return false;
    for (uint32_t position = 0; position < count; ++position)
        for (uint32_t row = 0; row < slot->outputChannels; ++row)
            outputs[(uint64_t)position * slot->outputChannels + row] =
                projectionValue(values, valuesAreFP32,
                                (uint64_t)row * PositionTile + position);
    if (surface) IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    projections->info.outputEvaluations += 1;
    projections->info.outputPositions += count;
    projections->info.outputEvaluationMilliseconds += elapsed;
    return true;
}

bool ANEFieldfareANEResidentProjectionsCallDeviceAttentionPositions(
    ANEFieldfareANEResidentProjections *projections,
    ANEFieldfareMetalAttention *metalAttention,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t count,
    const float *normalizedInputs,
    float *outputs) {
    if (!projections || !metalAttention || !tensorIndexBase || !weights || !cache ||
        !normalizedInputs || !outputs || !count || count > PositionTile ||
        !projections->info.sharedEventEnabled || !projections->sharedEvent) return false;
    ResidentProgram *inputSlot = matchInputProgram(
        projections, tensorIndexBase, &weights->queryProjection, &weights->keyProjection,
        &weights->valueProjection, weights->descriptor.valueEqualsKeyProjection);
    uint64_t outputOffset = (uint64_t)(weights->outputProjection.weights - tensorIndexBase);
    ResidentProgram *outputSlot = NULL;
    for (uint32_t layer = 0; layer < projections->info.promotedLayers; ++layer)
        if (projections->outputs[layer].built &&
            projections->outputs[layer].queryOffset == outputOffset) {
            outputSlot = &projections->outputs[layer];
            break;
        }
    // A non-promoted layer is an ordinary placement fallback, not an error.
    if (!inputSlot || !outputSlot) return false;
    IOSurfaceRef qkvInput = ANEFieldfareANEProgramInput(
        inputSlot->program, inputSlot->procedureBank);
    IOSurfaceRef qkvOutput = ANEFieldfareANEProgramOutput(
        inputSlot->program, inputSlot->procedureBank);
    IOSurfaceRef projectionInput = ANEFieldfareANEProgramInput(
        outputSlot->program, outputSlot->procedureBank);
    IOSurfaceRef projectionOutput = ANEFieldfareANEProgramOutput(
        outputSlot->program, outputSlot->procedureBank);
    if (!qkvInput || !qkvOutput || !projectionInput || !projectionOutput ||
        IOSurfaceLock(qkvInput, 0, NULL) != kIOReturnSuccess) {
        setError(projections, "device pipeline ANE surface unavailable");
        return false;
    }
    __fp16 *carrier = IOSurfaceGetBaseAddress(qkvInput);
    if (!carrier) {
        IOSurfaceUnlock(qkvInput, 0, NULL);
        setError(projections, "device pipeline QKV input has no base address");
        return false;
    }
    for (uint32_t column = 0; column < inputSlot->inputWidth; ++column) {
        __fp16 *lanes = carrier + (uint64_t)column * PositionTile;
        for (uint32_t position = 0; position < count; ++position)
            lanes[position] = (__fp16)normalizedInputs[
                (uint64_t)position * inputSlot->inputWidth + column];
        for (uint32_t position = count; position < PositionTile; ++position)
            lanes[position] = 0;
    }
    IOSurfaceUnlock(qkvInput, 0, NULL);

    uint64_t qkvSignal = projections->nextEventValue++;
    uint64_t metalSignal = projections->nextEventValue++;
    const char *qkvEventError = NULL, *projectionEventError = NULL;
    bool qkvConfigured = ANEFieldfareANEProgramSetOutputSignalEvent(
        inputSlot->program, inputSlot->procedureBank,
        (__bridge CFTypeRef)projections->sharedEvent, 0, qkvSignal, &qkvEventError);
    bool projectionConfigured = qkvConfigured && ANEFieldfareANEProgramSetInputWaitEvent(
        outputSlot->program, outputSlot->procedureBank,
        (__bridge CFTypeRef)projections->sharedEvent, metalSignal, &projectionEventError);
    if (!projectionConfigured) {
        setError(projections, projectionEventError ?: qkvEventError ?:
            "device pipeline shared-event configuration failed");
        free((void *)qkvEventError); free((void *)projectionEventError);
        return false;
    }
    free((void *)qkvEventError); free((void *)projectionEventError);

    uint64_t started = mach_absolute_time();
    CFTypeRef metalCommand = NULL;
    const char *metalEncodeError = NULL;
    bool metalEncoded = ANEFieldfareMetalAttentionEncodeDirect(
        metalAttention, weights, cache, firstPosition, count, qkvOutput,
        inputSlot->queryRows, inputSlot->keyRows, inputSlot->valueStacked,
        projectionInput, (__bridge CFTypeRef)projections->sharedEvent,
        qkvSignal, metalSignal, &metalCommand, &metalEncodeError);
    if (!metalEncoded) {
        setError(projections, metalEncodeError ?: "device Metal attention encode failed");
        free((void *)metalEncodeError);
        return false;
    }
    free((void *)metalEncodeError);

    const char *outputBeginError = NULL, *qkvBeginError = NULL;
    ANEFieldfareANEEvaluationTiming outputBeginTiming = {0}, qkvBeginTiming = {0};
    bool outputBegan = ANEFieldfareANEProgramEvaluateBegin(
        outputSlot->program, outputSlot->procedureBank, &outputBeginTiming, &outputBeginError);
    bool qkvBegan = outputBegan && ANEFieldfareANEProgramEvaluateBegin(
        inputSlot->program, inputSlot->procedureBank, &qkvBeginTiming, &qkvBeginError);
    if (!qkvBegan) {
        // Release a command waiting on an ANE signal that will never arrive.
        projections->sharedEvent.signaledValue = metalSignal;
    }
    const char *outputWaitError = NULL;
    ANEFieldfareANEEvaluationTiming outputTiming = {0};
    bool outputCompleted = qkvBegan && ANEFieldfareANEProgramEvaluateWait(
        outputSlot->program, outputSlot->procedureBank, 5 * NSEC_PER_SEC,
        &outputTiming, &outputWaitError);
    projections->info.devicePipelineBlockingHostWaits += 1;
    const char *qkvDrainError = NULL;
    ANEFieldfareANEEvaluationTiming qkvTiming = {0};
    bool qkvDrained = qkvBegan && ANEFieldfareANEProgramEvaluateWait(
        inputSlot->program, inputSlot->procedureBank, 0, &qkvTiming, &qkvDrainError);
    if (!qkvDrained && qkvBegan) {
        free((void *)qkvDrainError); qkvDrainError = NULL;
        qkvDrained = ANEFieldfareANEProgramEvaluateWait(
            inputSlot->program, inputSlot->procedureBank, 5 * NSEC_PER_SEC,
            &qkvTiming, &qkvDrainError);
        projections->info.devicePipelineBlockingHostWaits += 1;
    }
    const char *metalCompleteError = NULL;
    bool metalCompleted = ANEFieldfareMetalAttentionCompleteDirect(
        metalAttention, metalCommand, started, &metalCompleteError);
    bool executed = outputBegan && qkvBegan && outputCompleted && qkvDrained && metalCompleted;
    if (!executed) setError(projections,
        outputBeginError ?: qkvBeginError ?: outputWaitError ?: qkvDrainError ?:
        metalCompleteError ?: "device attention pipeline execution failed");
    free((void *)outputBeginError); free((void *)qkvBeginError);
    free((void *)outputWaitError); free((void *)qkvDrainError);
    free((void *)metalCompleteError);
    if (!executed) return false;

    if (IOSurfaceLock(projectionOutput, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        setError(projections, "device pipeline output lock failed");
        return false;
    }
    const __fp16 *result = IOSurfaceGetBaseAddress(projectionOutput);
    if (!result) {
        IOSurfaceUnlock(projectionOutput, kIOSurfaceLockReadOnly, NULL);
        setError(projections, "device pipeline output has no base address");
        return false;
    }
    for (uint32_t position = 0; position < count; ++position)
        for (uint32_t channel = 0; channel < outputSlot->outputChannels; ++channel)
            outputs[(uint64_t)position * outputSlot->outputChannels + channel] =
                (float)result[(uint64_t)channel * PositionTile + position];
    IOSurfaceUnlock(projectionOutput, kIOSurfaceLockReadOnly, NULL);
    double elapsed = milliseconds(mach_absolute_time() - started);
    projections->info.devicePipelineEvaluations += 1;
    projections->info.devicePipelinePositions += count;
    projections->info.devicePipelineIntermediateHostCopies += 0;
    projections->info.devicePipelineMilliseconds += elapsed;
    projections->info.inputEvaluations += 1;
    projections->info.inputPositions += count;
    projections->info.outputEvaluations += 1;
    projections->info.outputPositions += count;
    return true;
}

ANEFieldfareANEResidentProjectionInfo ANEFieldfareANEResidentProjectionsGetInfo(
    const ANEFieldfareANEResidentProjections *projections) {
    if (!projections) return (ANEFieldfareANEResidentProjectionInfo){0};
    return projections->info;
}

const char *ANEFieldfareANEResidentProjectionsLastError(
    const ANEFieldfareANEResidentProjections *projections) {
    return projections && projections->lastError ? projections->lastError : "none";
}

const char *ANEFieldfareANEResidentProjectionsEventFallbackReason(
    const ANEFieldfareANEResidentProjections *projections) {
    return projections && projections->eventFallbackReason
        ? projections->eventFallbackReason : "none";
}

void ANEFieldfareANEResidentProjectionsDestroy(
    ANEFieldfareANEResidentProjections *projections) {
    if (!projections) return;
    for (uint32_t layer = 0; layer < projections->layerCount; ++layer) {
        projections->inputs[layer].directOutputBuffer = nil;
        projections->outputs[layer].directOutputBuffer = nil;
        if (projections->inputs && projections->inputs[layer].program)
            ANEFieldfareANEProgramDestroy(projections->inputs[layer].program);
        if (projections->outputs && projections->outputs[layer].program &&
            projections->outputs[layer].program != projections->inputs[layer].program)
            ANEFieldfareANEProgramDestroy(projections->outputs[layer].program);
    }
    free(projections->inputs);
    free(projections->outputs);
    free(projections->lastError);
    free(projections->eventFallbackReason);
    projections->convertedOutput = nil;
    projections->sharedEvent = nil;
    projections->eventPipeline = nil;
    projections->eventQueue = nil;
    projections->eventDevice = nil;
    free(projections);
}
