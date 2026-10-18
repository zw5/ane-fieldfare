#import "ANETiledHead.h"

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
    DecodePositions = 64,
    HeadLanes = 64,
    RowsPerHostBlock = 16,
};

struct ANEFieldfareANETiledHead {
    ANEFieldfareANEProgram *program;
    ANEFieldfareANETiledHeadInfo info;
};

static mach_timebase_info_data_t gTimebase;

static double elapsedMilliseconds(uint64_t start, uint64_t end) {
    if (!gTimebase.denom) mach_timebase_info(&gTimebase);
    return (double)(end - start) * gTimebase.numer / gTimebase.denom / 1e6;
}

static const char *retainError(NSString *message) {
    return strdup((message.length ? message : @"unknown tiled-head error").UTF8String);
}

static bool multiplyFits(uint64_t first, uint64_t second, uint64_t *product) {
    if (second && first > UINT64_MAX / second) return false;
    *product = first * second;
    return true;
}

static NSString *tiledHeadMIL(uint32_t inputWidth,
                              uint32_t tileRows,
                              uint32_t groupSize) {
    uint32_t groups = inputWidth / groupSize;
    uint32_t rowsPerLane = tileRows / HeadLanes;
    uint64_t activationElements = (uint64_t)inputWidth * DecodePositions;
    uint64_t codeElements = (uint64_t)tileRows * inputWidth;
    uint64_t parameterElements = (uint64_t)tileRows * groups;
    uint64_t scaleOffset = activationElements + codeElements;
    uint64_t biasOffset = scaleOffset + parameterElements;
    uint64_t carrierElements = biasOffset + parameterElements;
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"];
    [mil appendFormat:@"  func main<ios18>(tensor<fp16,[1,1,1,%llu]> packed) {\n",
                      carrierElements];
    [mil appendFormat:
        @"    tensor<int32,[4]> xb=const()[name=string(\"xb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[4]> xs=const()[name=string(\"xs\"),val=tensor<int32,[4]>([1,1,1,%llu])];\n"
         "    tensor<fp16,[1,1,1,%llu]> xf=slice_by_size(x=packed,begin=xb,size=xs)[name=string(\"xf\")];\n"
         "    tensor<int32,[4]> xr=const()[name=string(\"xr\"),val=tensor<int32,[4]>([1,1,%u,%u])];\n"
         "    tensor<fp16,[1,1,%u,%u]> x2=reshape(shape=xr,x=xf)[name=string(\"x2\")];\n"
         "    tensor<int32,[4]> xp=const()[name=string(\"xp\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
         "    tensor<fp16,[1,1,%u,%u]> xt0=transpose(perm=xp,x=x2)[name=string(\"xt0\")];\n"
         "    tensor<int32,[4]> xbr=const()[name=string(\"xbr\"),val=tensor<int32,[4]>([%u,1,1,%u])];\n"
         "    tensor<fp16,[%u,1,1,%u]> xt=reshape(shape=xbr,x=xt0)[name=string(\"xt\")];\n",
         activationElements, activationElements,
         inputWidth, DecodePositions, inputWidth, DecodePositions,
         DecodePositions, inputWidth,
         HeadLanes, inputWidth, HeadLanes, inputWidth];
    [mil appendFormat:
        @"    tensor<int32,[4]> qb=const()[name=string(\"qb\"),val=tensor<int32,[4]>([0,0,0,%llu])];\n"
         "    tensor<int32,[4]> qs=const()[name=string(\"qs\"),val=tensor<int32,[4]>([1,1,1,%llu])];\n"
         "    tensor<fp16,[1,1,1,%llu]> qf=slice_by_size(x=packed,begin=qb,size=qs)[name=string(\"qf\")];\n"
         "    tensor<int32,[4]> qr=const()[name=string(\"qr\"),val=tensor<int32,[4]>([%u,%u,%u,%u])];\n"
         "    tensor<fp16,[%u,%u,%u,%u]> q=reshape(shape=qr,x=qf)[name=string(\"q\")];\n"
         "    tensor<int32,[4]> sb=const()[name=string(\"sb\"),val=tensor<int32,[4]>([0,0,0,%llu])];\n"
         "    tensor<int32,[4]> ps=const()[name=string(\"ps\"),val=tensor<int32,[4]>([1,1,1,%llu])];\n"
         "    tensor<fp16,[1,1,1,%llu]> sf=slice_by_size(x=packed,begin=sb,size=ps)[name=string(\"sf\")];\n"
         "    tensor<int32,[4]> pr=const()[name=string(\"pr\"),val=tensor<int32,[4]>([%u,%u,%u,1])];\n"
         "    tensor<fp16,[%u,%u,%u,1]> scale=reshape(shape=pr,x=sf)[name=string(\"scale\")];\n"
         "    tensor<int32,[4]> bb=const()[name=string(\"bb\"),val=tensor<int32,[4]>([0,0,0,%llu])];\n"
         "    tensor<fp16,[1,1,1,%llu]> bf=slice_by_size(x=packed,begin=bb,size=ps)[name=string(\"bf\")];\n"
         "    tensor<fp16,[%u,%u,%u,1]> bias=reshape(shape=pr,x=bf)[name=string(\"bias\")];\n"
         "    tensor<fp16,[%u,%u,%u,%u]> scaled=mul(x=q,y=scale)[name=string(\"scaled\")];\n"
         "    tensor<fp16,[%u,%u,%u,%u]> dequant=add(x=scaled,y=bias)[name=string(\"dequant\")];\n",
         activationElements, codeElements, codeElements,
         HeadLanes, rowsPerLane, groups, groupSize,
         HeadLanes, rowsPerLane, groups, groupSize,
         scaleOffset, parameterElements, parameterElements,
         HeadLanes, rowsPerLane, groups, HeadLanes, rowsPerLane, groups,
         biasOffset, parameterElements, HeadLanes, rowsPerLane, groups,
         HeadLanes, rowsPerLane, groups, groupSize,
         HeadLanes, rowsPerLane, groups, groupSize];
    [mil appendFormat:
        @"    tensor<int32,[4]> wr=const()[name=string(\"wr\"),val=tensor<int32,[4]>([%u,1,%u,%u])];\n"
         "    tensor<fp16,[%u,1,%u,%u]> w=reshape(shape=wr,x=dequant)[name=string(\"w\")];\n"
         "    tensor<int32,[4]> wp=const()[name=string(\"wp\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
         "    tensor<fp16,[%u,1,%u,%u]> wt=transpose(perm=wp,x=w)[name=string(\"wt\")];\n"
         "    bool no=const()[name=string(\"no\"),val=bool(false)];\n"
         "    tensor<fp16,[%u,1,1,%u]> y=matmul(transpose_x=no,transpose_y=no,x=xt,y=wt)[name=string(\"head_tile\")];\n"
         "    tensor<int32,[4]> yp=const()[name=string(\"yp\"),val=tensor<int32,[4]>([2,3,1,0])];\n"
         "    tensor<fp16,[1,%u,1,%u]> y16=transpose(perm=yp,x=y)[name=string(\"y16\")];\n"
         "    string f32=const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%u,1,%u]> out=cast(dtype=f32,x=y16)[name=string(\"out\")];\n"
         "  } -> (out);\n"
         "}\n",
         HeadLanes, rowsPerLane, inputWidth,
         HeadLanes, rowsPerLane, inputWidth,
         HeadLanes, inputWidth, rowsPerLane,
         HeadLanes, rowsPerLane,
         rowsPerLane, HeadLanes, rowsPerLane, HeadLanes];
    return mil;
}

ANEFieldfareANETiledHead *ANEFieldfareANETiledHeadCreate(
    uint32_t inputWidth, uint32_t tileRows, uint32_t groupSize,
    uint32_t requestBanks, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!inputWidth || tileRows < HeadLanes || tileRows % HeadLanes ||
            !groupSize || inputWidth % groupSize ||
            requestBanks < 1 || requestBanks > 8) {
            if (errorMessage) *errorMessage = retainError(@"invalid tiled-head shape or bank count");
            return NULL;
        }
        uint64_t activationElements = 0, codeElements = 0, parameterElements = 0;
        if (!multiplyFits(inputWidth, DecodePositions, &activationElements) ||
            !multiplyFits(tileRows, inputWidth, &codeElements) ||
            !multiplyFits(tileRows, inputWidth / groupSize, &parameterElements) ||
            activationElements > UINT64_MAX - codeElements ||
            activationElements + codeElements > UINT64_MAX - 2 * parameterElements) {
            if (errorMessage) *errorMessage = retainError(@"tiled-head carrier size overflow");
            return NULL;
        }
        uint64_t carrierElements = activationElements + codeElements + 2 * parameterElements;
        if (carrierElements > UINT64_MAX / sizeof(__fp16) ||
            (uint64_t)tileRows > UINT64_MAX / sizeof(float)) {
            if (errorMessage) *errorMessage = retainError(@"tiled-head byte size overflow");
            return NULL;
        }
        uint64_t inputBytes = carrierElements * sizeof(__fp16);
        uint64_t outputBytes = (uint64_t)tileRows * sizeof(float);
        NSString *mil = tiledHeadMIL(inputWidth, tileRows, groupSize);
        ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreate(
            mil.UTF8String, inputBytes, outputBytes, requestBanks, errorMessage);
        if (!program) return NULL;
        ANEFieldfareANETiledHead *head = calloc(1, sizeof(*head));
        if (!head) {
            ANEFieldfareANEProgramDestroy(program);
            if (errorMessage) *errorMessage = retainError(@"tiled-head allocation failed");
            return NULL;
        }
        head->program = program;
        head->info = (ANEFieldfareANETiledHeadInfo){
            .program = ANEFieldfareANEProgramGetInfo(program),
            .inputWidth = inputWidth,
            .tileRows = tileRows,
            .groupSize = groupSize,
            .groupsPerRow = inputWidth / groupSize,
            .requestBanks = requestBanks,
            .inputBytesPerBank = inputBytes,
            .outputBytesPerBank = outputBytes,
        };
        return head;
    }
}

bool ANEFieldfareANETiledHeadEvaluateTile(
    ANEFieldfareANETiledHead *head, uint32_t bank, const float *input,
    const ANEFieldfareTensor *weights, uint32_t firstRow, uint32_t rowCount,
    float *output, ANEFieldfareANEHeadTileTiming *timing, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (timing) *timing = (ANEFieldfareANEHeadTileTiming){0};
        if (!head || !input || !weights || !output || bank >= head->info.requestBanks ||
            !rowCount || rowCount > head->info.tileRows || firstRow > weights->rows ||
            rowCount > weights->rows - firstRow ||
            weights->columns != head->info.inputWidth ||
            weights->weightBytes != ((uint64_t)weights->rows * weights->columns + 1) / 2 ||
            weights->scaleBytes != (uint64_t)weights->rows * head->info.groupsPerRow * 2 ||
            weights->biasBytes != (uint64_t)weights->rows * head->info.groupsPerRow * 2) {
            if (errorMessage) *errorMessage = retainError(@"invalid affine 4-bit head tile");
            return false;
        }

        uint64_t preparedStart = mach_absolute_time();
        IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(head->program, bank);
        IOReturn lockResult = IOSurfaceLock(inputSurface, 0, NULL);
        if (lockResult != kIOReturnSuccess) {
            if (errorMessage) *errorMessage = retainError(@"tiled-head input IOSurface lock failed");
            return false;
        }
        __fp16 *carrier = IOSurfaceGetBaseAddress(inputSurface);
        uint64_t activationElements = (uint64_t)head->info.inputWidth * DecodePositions;
        uint64_t codeElements = (uint64_t)head->info.tileRows * head->info.inputWidth;
        uint64_t parameterElements = (uint64_t)head->info.tileRows * head->info.groupsPerRow;
        __fp16 *codes = carrier + activationElements;
        __fp16 *scales = codes + codeElements;
        __fp16 *biases = scales + parameterElements;

        memset(carrier, 0, activationElements * sizeof(*carrier));
        for (uint32_t column = 0; column < head->info.inputWidth; ++column) {
            __fp16 value = (__fp16)input[column];
            __fp16 *positions = carrier + (uint64_t)column * DecodePositions;
            for (uint32_t lane = 0; lane < HeadLanes; ++lane) positions[lane] = value;
        }

        static const __fp16 codeLookup[16] = {
            (__fp16)0, (__fp16)1, (__fp16)2, (__fp16)3,
            (__fp16)4, (__fp16)5, (__fp16)6, (__fp16)7,
            (__fp16)8, (__fp16)9, (__fp16)10, (__fp16)11,
            (__fp16)12, (__fp16)13, (__fp16)14, (__fp16)15,
        };
        uint32_t blockCount = (rowCount + RowsPerHostBlock - 1) / RowsPerHostBlock;
        dispatch_apply(blockCount,
                       dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                       ^(size_t block) {
            uint32_t localBegin = (uint32_t)block * RowsPerHostBlock;
            uint32_t localEnd = MIN(localBegin + RowsPerHostBlock, rowCount);
            for (uint32_t localRow = localBegin; localRow < localEnd; ++localRow) {
                uint32_t sourceRow = firstRow + localRow;
                uint64_t sourceLinear = (uint64_t)sourceRow * head->info.inputWidth;
                __fp16 *destination = codes + (uint64_t)localRow * head->info.inputWidth;
                for (uint32_t column = 0; column < head->info.inputWidth; column += 2) {
                    uint8_t packed = weights->weights[(sourceLinear + column) >> 1];
                    destination[column] = codeLookup[packed & 0x0f];
                    destination[column + 1] = codeLookup[packed >> 4];
                }
                uint64_t sourceGroup = (uint64_t)sourceRow * head->info.groupsPerRow;
                uint64_t destinationGroup = (uint64_t)localRow * head->info.groupsPerRow;
                for (uint32_t group = 0; group < head->info.groupsPerRow; ++group) {
                    scales[destinationGroup + group] = (__fp16)ANEFieldfareBF16(
                        weights->scales + (sourceGroup + group) * 2);
                    biases[destinationGroup + group] = (__fp16)ANEFieldfareBF16(
                        weights->biases + (sourceGroup + group) * 2);
                }
            }
        });
        if (rowCount < head->info.tileRows) {
            uint64_t missingRows = head->info.tileRows - rowCount;
            memset(codes + (uint64_t)rowCount * head->info.inputWidth, 0,
                   missingRows * head->info.inputWidth * sizeof(*codes));
            memset(scales + (uint64_t)rowCount * head->info.groupsPerRow, 0,
                   missingRows * head->info.groupsPerRow * sizeof(*scales));
            memset(biases + (uint64_t)rowCount * head->info.groupsPerRow, 0,
                   missingRows * head->info.groupsPerRow * sizeof(*biases));
        }
        IOSurfaceUnlock(inputSurface, 0, NULL);
        uint64_t preparedEnd = mach_absolute_time();

        ANEFieldfareANEEvaluationTiming evaluation = {0};
        if (!ANEFieldfareANEProgramEvaluate(
                head->program, bank, 10ULL * NSEC_PER_SEC, &evaluation, errorMessage)) return false;

        uint64_t readStart = mach_absolute_time();
        IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(head->program, bank);
        lockResult = IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        if (lockResult != kIOReturnSuccess) {
            if (errorMessage) *errorMessage = retainError(@"tiled-head output IOSurface lock failed");
            return false;
        }
        const float *values = IOSurfaceGetBaseAddress(outputSurface);
        uint32_t rowsPerLane = head->info.tileRows / HeadLanes;
        for (uint32_t row = 0; row < rowCount; ++row) {
            uint32_t lane = row / rowsPerLane;
            uint32_t rowWithinLane = row % rowsPerLane;
            output[row] = values[(uint64_t)rowWithinLane * HeadLanes + lane];
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        uint64_t readEnd = mach_absolute_time();
        if (timing) *timing = (ANEFieldfareANEHeadTileTiming){
            .preparationMilliseconds = elapsedMilliseconds(preparedStart, preparedEnd),
            .enqueueMilliseconds = evaluation.enqueueMilliseconds,
            .evaluationMilliseconds = evaluation.completionMilliseconds,
            .readbackMilliseconds = elapsedMilliseconds(readStart, readEnd),
        };
        return true;
    }
}

ANEFieldfareANETiledHeadInfo ANEFieldfareANETiledHeadGetInfo(
    ANEFieldfareANETiledHead *head) {
    return head ? head->info : (ANEFieldfareANETiledHeadInfo){0};
}

void ANEFieldfareANETiledHeadDestroy(ANEFieldfareANETiledHead *head) {
    if (!head) return;
    ANEFieldfareANEProgramDestroy(head->program);
    free(head);
}
