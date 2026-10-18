#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/MemoryBudget.h"

static mach_timebase_info_data_t gTimebase;

typedef struct {
    dispatch_semaphore_t semaphore;
    _Atomic uint64_t completedAt;
} CompletionState;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * gTimebase.numer / gTimebase.denom / 1e6;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSDictionary *readJSON(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    return data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
}

static IOSurfaceRef makeSurface(NSUInteger bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static float bf16(const uint8_t *bytes) {
    uint16_t word = 0;
    memcpy(&word, bytes, sizeof(word));
    uint32_t bits = (uint32_t)word << 16;
    float value = 0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static BOOL checkedRange(NSUInteger base, NSDictionary *tensor, NSUInteger blobLength,
                         NSUInteger *size) {
    uint64_t offset = [tensor[@"offset"] unsignedLongLongValue];
    uint64_t length = [tensor[@"size"] unsignedLongLongValue];
    if (offset > NSUIntegerMax || length > NSUIntegerMax || base > NSUIntegerMax - (NSUInteger)offset) return NO;
    NSUInteger start = base + (NSUInteger)offset;
    if (start > blobLength || length > blobLength - start) return NO;
    *size = (NSUInteger)length;
    return YES;
}

typedef struct {
    const uint8_t *weights;
    const uint8_t *scales;
    const uint8_t *biases;
    NSUInteger rows;
    NSUInteger columns;
    NSUInteger weightBytes;
    NSUInteger scaleBytes;
    NSUInteger biasBytes;
} CompactMatrix;

static BOOL resolveMatrix(NSData *blob, NSUInteger expertBase, NSDictionary *tensors,
                          NSString *name, CompactMatrix *matrix) {
    NSDictionary *weight = tensors[name];
    NSDictionary *scale = tensors[[name stringByAppendingString:@"_scales"]];
    NSDictionary *bias = tensors[[name stringByAppendingString:@"_biases"]];
    NSArray *shape = weight[@"shape"];
    if (![weight[@"dtype"] isEqual:@"U32"] || [weight[@"bits"] intValue] != 4 || shape.count != 2 ||
        ![scale[@"dtype"] isEqual:@"BF16"] || ![bias[@"dtype"] isEqual:@"BF16"]) return NO;
    NSUInteger rows = [shape[0] unsignedIntegerValue];
    NSUInteger columns = [shape[1] unsignedIntegerValue];
    if (!rows || !columns || columns % 64 != 0) return NO;
    NSUInteger weightSize = 0, scaleSize = 0, biasSize = 0;
    if (!checkedRange(expertBase, weight, blob.length, &weightSize) ||
        !checkedRange(expertBase, scale, blob.length, &scaleSize) ||
        !checkedRange(expertBase, bias, blob.length, &biasSize)) return NO;
    NSUInteger weightOffset = expertBase + [weight[@"offset"] unsignedIntegerValue];
    NSUInteger scaleOffset = expertBase + [scale[@"offset"] unsignedIntegerValue];
    NSUInteger biasOffset = expertBase + [bias[@"offset"] unsignedIntegerValue];
    NSUInteger groups = rows * (columns / 64);
    if (weightSize != rows * columns / 2 || scaleSize != groups * 2 || biasSize != groups * 2) return NO;
    const uint8_t *bytes = blob.bytes;
    *matrix = (CompactMatrix){bytes + weightOffset, bytes + scaleOffset, bytes + biasOffset,
                              rows, columns, weightSize, scaleSize, biasSize};
    return YES;
}

static float compactValue(CompactMatrix matrix, NSUInteger row, NSUInteger column) {
    NSUInteger linear = row * matrix.columns + column;
    uint8_t packed = matrix.weights[linear >> 1];
    unsigned q = (linear & 1) ? packed >> 4 : packed & 0x0f;
    NSUInteger group = row * (matrix.columns / 64) + column / 64;
    return (float)q * bf16(matrix.scales + group * 2) + bf16(matrix.biases + group * 2);
}

static void fillFP16TransposedPair(__fp16 *carrier, NSUInteger carrierStride, NSUInteger offset,
                                   CompactMatrix gate, CompactMatrix up) {
    enum { Tile = 32 };
    NSUInteger rowBlocks = (gate.rows + Tile - 1) / Tile;
    dispatch_apply(rowBlocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        __fp16 gateTile[Tile][Tile];
        __fp16 upTile[Tile][Tile];
        NSUInteger rowBase = block * Tile;
        NSUInteger rowCount = MIN((NSUInteger)Tile, gate.rows - rowBase);
        for (NSUInteger columnBase = 0; columnBase < gate.columns; columnBase += Tile) {
            NSUInteger columnCount = MIN((NSUInteger)Tile, gate.columns - columnBase);
            for (NSUInteger row = 0; row < rowCount; ++row) {
                for (NSUInteger column = 0; column < columnCount; ++column) {
                    gateTile[row][column] = (__fp16)compactValue(gate, rowBase + row, columnBase + column);
                    upTile[row][column] = (__fp16)compactValue(up, rowBase + row, columnBase + column);
                }
            }
            for (NSUInteger column = 0; column < columnCount; ++column) {
                __fp16 *destination = carrier + (columnBase + column) * carrierStride + offset + rowBase;
                for (NSUInteger row = 0; row < rowCount; ++row) {
                    destination[row] = gateTile[row][column];
                    destination[gate.rows + row] = upTile[row][column];
                }
            }
        }
    });
}

static void fillFP16Down(__fp16 *carrier, NSUInteger carrierStride, NSUInteger offset,
                         CompactMatrix down) {
    enum { RowsPerBlock = 32 };
    NSUInteger blocks = (down.rows + RowsPerBlock - 1) / RowsPerBlock;
    dispatch_apply(blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        NSUInteger start = block * RowsPerBlock;
        NSUInteger end = MIN(start + RowsPerBlock, down.rows);
        for (NSUInteger row = start; row < end; ++row) {
            __fp16 *destination = carrier + row * carrierStride + offset;
            for (NSUInteger column = 0; column < down.columns; ++column)
                destination[column] = (__fp16)compactValue(down, row, column);
        }
    });
}

static NSString *expertMIL(int hidden, int intermediate, int positions, BOOL fp16Carrier) {
    int spatial = positions + 3 * intermediate;
    NSMutableString *m = [NSMutableString stringWithString:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"  func main<ios18>(tensor<%@,[1,%d,1,%d]> packed) {\n", fp16Carrier ? @"fp16" : @"fp32", hidden, spatial];
    [m appendString:@"    string f16=const()[name=string(\"f16\"),val=string(\"fp16\")];\n"];
    if (fp16Carrier)
        [m appendFormat:@"    tensor<fp16,[1,%d,1,%d]> p=identity(x=packed)[name=string(\"carrier\")];\n", hidden, spatial];
    else
        [m appendFormat:@"    tensor<fp16,[1,%d,1,%d]> p=cast(dtype=f16,x=packed)[name=string(\"cast_in\")];\n", hidden, spatial];
    [m appendString:@"    tensor<int32,[4]> pm=const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n    bool no=const()[name=string(\"no\"),val=bool(false)];\n    tensor<int32,[4]> xb=const()[name=string(\"xb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"];
    [m appendFormat:@"    tensor<int32,[4]> xs=const()[name=string(\"xs\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n    tensor<fp16,[1,%d,1,%d]> x=slice_by_size(x=p,begin=xb,size=xs)[name=string(\"x\")];\n", hidden, positions, hidden, positions];
    [m appendFormat:@"    tensor<int32,[4]> xr=const()[name=string(\"xr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n    tensor<fp16,[1,1,%d,%d]> x2=reshape(shape=xr,x=x)[name=string(\"x2\")];\n    tensor<fp16,[1,1,%d,%d]> xt=transpose(perm=pm,x=x2)[name=string(\"xt\")];\n", hidden, positions, hidden, positions, positions, hidden];
    [m appendFormat:@"    tensor<int32,[4]> ws=const()[name=string(\"ws\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n", hidden, intermediate];
    [m appendFormat:@"    tensor<int32,[4]> gb=const()[name=string(\"gb\"),val=tensor<int32,[4]>([0,0,0,%d])];\n    tensor<fp16,[1,%d,1,%d]> gw=slice_by_size(x=p,begin=gb,size=ws)[name=string(\"gw\")];\n", positions, hidden, intermediate];
    [m appendFormat:@"    tensor<int32,[4]> ub=const()[name=string(\"ub\"),val=tensor<int32,[4]>([0,0,0,%d])];\n    tensor<fp16,[1,%d,1,%d]> uw=slice_by_size(x=p,begin=ub,size=ws)[name=string(\"uw\")];\n", positions + intermediate, hidden, intermediate];
    [m appendFormat:@"    tensor<int32,[4]> db=const()[name=string(\"db\"),val=tensor<int32,[4]>([0,0,0,%d])];\n    tensor<fp16,[1,%d,1,%d]> dw=slice_by_size(x=p,begin=db,size=ws)[name=string(\"dw\")];\n", positions + 2 * intermediate, hidden, intermediate];
    [m appendFormat:@"    tensor<int32,[4]> wr=const()[name=string(\"wr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n    tensor<fp16,[1,1,%d,%d]> g2=reshape(shape=wr,x=gw)[name=string(\"g2\")];\n    tensor<fp16,[1,1,%d,%d]> u2=reshape(shape=wr,x=uw)[name=string(\"u2\")];\n", hidden, intermediate, hidden, intermediate, hidden, intermediate];
    [m appendFormat:@"    tensor<fp16,[1,1,%d,%d]> gate=matmul(transpose_x=no,transpose_y=no,x=xt,y=g2)[name=string(\"gate\")];\n    tensor<fp16,[1,1,%d,%d]> up=matmul(transpose_x=no,transpose_y=no,x=xt,y=u2)[name=string(\"up\")];\n", positions, intermediate, positions, intermediate];
    [m appendFormat:@"    tensor<fp16,[1,1,%d,%d]> gate2=mul(x=gate,y=gate)[name=string(\"gate2\")];\n    tensor<fp16,[1,1,%d,%d]> gate3=mul(x=gate2,y=gate)[name=string(\"gate3\")];\n    fp16 ck=const()[name=string(\"ck\"),val=fp16(0.044715)];\n    tensor<fp16,[1,1,%d,%d]> cubic=mul(x=gate3,y=ck)[name=string(\"cubic\")];\n    tensor<fp16,[1,1,%d,%d]> inner=add(x=gate,y=cubic)[name=string(\"inner\")];\n    fp16 tk=const()[name=string(\"tk\"),val=fp16(0.7978845608028654)];\n    tensor<fp16,[1,1,%d,%d]> scaled=mul(x=inner,y=tk)[name=string(\"scaled\")];\n    tensor<fp16,[1,1,%d,%d]> th=tanh(x=scaled)[name=string(\"th\")];\n", positions, intermediate, positions, intermediate, positions, intermediate, positions, intermediate, positions, intermediate, positions, intermediate];
    [m appendFormat:@"    fp16 one=const()[name=string(\"one\"),val=fp16(1.0)];\n    fp16 half=const()[name=string(\"half\"),val=fp16(0.5)];\n    tensor<fp16,[1,1,%d,%d]> plus=add(x=th,y=one)[name=string(\"plus\")];\n    tensor<fp16,[1,1,%d,%d]> hg=mul(x=gate,y=half)[name=string(\"hg\")];\n    tensor<fp16,[1,1,%d,%d]> gelu=mul(x=hg,y=plus)[name=string(\"gelu\")];\n    tensor<fp16,[1,1,%d,%d]> act=mul(x=gelu,y=up)[name=string(\"act\")];\n", positions, intermediate, positions, intermediate, positions, intermediate, positions, intermediate];
    [m appendFormat:@"    tensor<int32,[4]> dr=const()[name=string(\"dr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n    tensor<fp16,[1,1,%d,%d]> d2=reshape(shape=dr,x=dw)[name=string(\"d2\")];\n    tensor<fp16,[1,1,%d,%d]> dt=transpose(perm=pm,x=d2)[name=string(\"dt\")];\n    tensor<fp16,[1,1,%d,%d]> o2=matmul(transpose_x=no,transpose_y=no,x=act,y=dt)[name=string(\"down\")];\n    tensor<fp16,[1,1,%d,%d]> ot=transpose(perm=pm,x=o2)[name=string(\"ot\")];\n", hidden, intermediate, hidden, intermediate, intermediate, hidden, positions, hidden, hidden, positions];
    [m appendFormat:@"    tensor<int32,[4]> os=const()[name=string(\"os\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n    tensor<fp16,[1,%d,1,%d]> or=reshape(shape=os,x=ot)[name=string(\"or\")];\n    string f32=const()[name=string(\"f32\"),val=string(\"fp32\")];\n    tensor<fp32,[1,%d,1,%d]> out=cast(dtype=f32,x=or)[name=string(\"cast_out\")];\n  } -> (out);\n}\n", hidden, positions, hidden, positions, hidden, positions];
    return m;
}

static NSString *batchedExpertMIL(int hidden, int intermediate, int positions, int batch) {
    NSString *single = expertMIL(hidden, intermediate, positions, YES);
    NSString *batched = [single stringByReplacingOccurrencesOfString:@"[1," withString:
        [NSString stringWithFormat:@"[%d,", batch]];
    if ([NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_BATCH_NO_REDUCE"] boolValue])
        return batched;
    NSString *tail = @"  } -> (out);\n}\n";
    NSMutableString *reduction = [NSMutableString string];
    [reduction appendFormat:
        @"    tensor<int32,[4]> route_begin=const()[name=string(\"route_begin\"),val=tensor<int32,[4]>([0,0,0,1])];\n"
         "    tensor<int32,[4]> route_size=const()[name=string(\"route_size\"),val=tensor<int32,[4]>([%d,1,1,1])];\n"
         "    tensor<fp16,[%d,1,1,1]> route16=slice_by_size(x=packed,begin=route_begin,size=route_size)[name=string(\"route16\")];\n",
         batch, batch];
    for (int index = 0; index < batch; ++index) {
        [reduction appendFormat:
            @"    tensor<int32,[4]> ob%d=const()[name=string(\"ob%d\"),val=tensor<int32,[4]>([%d,0,0,0])];\n"
             "    tensor<int32,[4]> os%d=const()[name=string(\"os%d\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
             "    tensor<fp32,[1,%d,1,%d]> o%d=slice_by_size(x=out,begin=ob%d,size=os%d)[name=string(\"o%d\")];\n"
             "    tensor<int32,[4]> rb%d=const()[name=string(\"rb%d\"),val=tensor<int32,[4]>([%d,0,0,0])];\n"
             "    tensor<int32,[4]> rs%d=const()[name=string(\"rs%d\"),val=tensor<int32,[4]>([1,1,1,1])];\n"
             "    tensor<fp16,[1,1,1,1]> r16_%d=slice_by_size(x=route16,begin=rb%d,size=rs%d)[name=string(\"r16_%d\")];\n"
             "    tensor<fp32,[1,1,1,1]> r%d=cast(dtype=f32,x=r16_%d)[name=string(\"r%d\")];\n"
             "    tensor<fp32,[1,%d,1,%d]> w%d=mul(x=o%d,y=r%d)[name=string(\"w%d\")];\n",
             index,index,index,
             index,index,hidden,positions,
             hidden,positions,index,index,index,index,
             index,index,index,
             index,index,
             index,index,index,index,
             index,index,index,
             hidden,positions,index,index,index,index];
        if (index == 1)
            [reduction appendFormat:@"    tensor<fp32,[1,%d,1,%d]> sum1=add(x=w0,y=w1)[name=string(\"sum1\")];\n", hidden, positions];
        else if (index > 1)
            [reduction appendFormat:@"    tensor<fp32,[1,%d,1,%d]> sum%d=add(x=sum%d,y=w%d)[name=string(\"sum%d\")];\n",
             hidden, positions, index, index - 1, index, index];
    }
    [reduction appendFormat:@"  } -> (sum%d);\n}\n", batch - 1];
    return [batched stringByReplacingOccurrencesOfString:tail withString:reduction];
}

static NSString *gateOnlyMIL(int hidden, int fullIntermediate, int tileWidth, int tileOffset, int positions) {
    int spatial = positions + 3 * fullIntermediate;
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1,%d,1,%d]> packed) {\n"
         "    string f16 = const()[name=string(\"f16\"),val=string(\"fp16\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> p = cast(dtype=f16,x=packed)[name=string(\"cast_in\")];\n"
         "    tensor<int32,[4]> xb = const()[name=string(\"xb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[4]> xs = const()[name=string(\"xs\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> x = slice_by_size(x=p,begin=xb,size=xs)[name=string(\"x\")];\n"
         "    tensor<int32,[4]> wb = const()[name=string(\"wb\"),val=tensor<int32,[4]>([0,0,0,%d])];\n"
         "    tensor<int32,[4]> ws = const()[name=string(\"ws\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> w = slice_by_size(x=p,begin=wb,size=ws)[name=string(\"w\")];\n"
         "    tensor<int32,[4]> xr = const()[name=string(\"xr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "    tensor<fp16,[1,1,%d,%d]> x2 = reshape(shape=xr,x=x)[name=string(\"x2\")];\n"
         "    tensor<int32,[4]> pm = const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
         "    tensor<fp16,[1,1,%d,%d]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];\n"
         "    tensor<int32,[4]> wr = const()[name=string(\"wr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "    tensor<fp16,[1,1,%d,%d]> w2 = reshape(shape=wr,x=w)[name=string(\"w2\")];\n"
         "    bool no = const()[name=string(\"no\"),val=bool(false)];\n"
         "    tensor<fp16,[1,1,%d,%d]> y2 = matmul(transpose_x=no,transpose_y=no,x=xt,y=w2)[name=string(\"projection\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> yt = transpose(perm=pm,x=y2)[name=string(\"yt\")];\n"
         "    tensor<int32,[4]> yr = const()[name=string(\"yr\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> y16 = reshape(shape=yr,x=yt)[name=string(\"y16\")];\n"
         "    string f32 = const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%d,1,%d]> out = cast(dtype=f32,x=y16)[name=string(\"cast_out\")];\n"
         "  } -> (out);\n"
         "}\n",
         hidden, spatial,
         hidden, spatial,
         hidden,
         positions,
         hidden, positions,
         positions + tileOffset,
         hidden, tileWidth,
         hidden, tileWidth,
         hidden,
         positions,
         hidden, positions,
         positions, hidden,
         hidden, tileWidth,
         hidden, tileWidth,
         positions, tileWidth,
         tileWidth, positions,
         tileWidth, positions,
         tileWidth, positions,
         tileWidth, positions];
}

static void fillCarrier(IOSurfaceRef surface, CompactMatrix gate, CompactMatrix up, CompactMatrix down,
                        NSUInteger positions, BOOL fp16Carrier) {
    const NSUInteger hidden = gate.columns;
    const NSUInteger intermediate = gate.rows;
    const NSUInteger stride = positions + 3 * intermediate;
    IOSurfaceLock(surface, 0, NULL);
    if (fp16Carrier) {
        __fp16 *values = IOSurfaceGetBaseAddress(surface);
        memset(values, 0, hidden * stride * sizeof(__fp16));
        for (NSUInteger column = 0; column < hidden; ++column) {
            values[column * stride] = (__fp16)(0.03125f * sinf((float)(column + 1) * 0.017f));
        }
        fillFP16TransposedPair(values, stride, positions, gate, up);
        fillFP16Down(values, stride, positions + 2 * intermediate, down);
    } else {
        float *values = IOSurfaceGetBaseAddress(surface);
        memset(values, 0, hidden * stride * sizeof(float));
        for (NSUInteger column = 0; column < hidden; ++column) {
            values[column * stride] = 0.03125f * sinf((float)(column + 1) * 0.017f);
            for (NSUInteger row = 0; row < intermediate; ++row) {
                values[column * stride + positions + row] = compactValue(gate, row, column);
                values[column * stride + positions + intermediate + row] = compactValue(up, row, column);
                values[column * stride + positions + 2 * intermediate + row] = compactValue(down, column, row);
            }
        }
    }
    IOSurfaceUnlock(surface, 0, NULL);
}

static void fillFP16CarrierMemory(__fp16 *values, CompactMatrix gate, CompactMatrix up,
                                  CompactMatrix down, NSUInteger positions, float routeWeight) {
    NSUInteger hidden = gate.columns, intermediate = gate.rows;
    NSUInteger stride = positions + 3 * intermediate;
    memset(values, 0, hidden * stride * sizeof(__fp16));
    for (NSUInteger column = 0; column < hidden; ++column)
        values[column * stride] = (__fp16)(0.03125f * sinf((float)(column + 1) * 0.017f));
    values[1] = (__fp16)routeWeight;
    fillFP16TransposedPair(values, stride, positions, gate, up);
    fillFP16Down(values, stride, positions + 2 * intermediate, down);
}

static NSMutableData *cpuOracle(CompactMatrix gate, CompactMatrix up, CompactMatrix down) {
    NSUInteger hidden = gate.columns, intermediate = gate.rows;
    NSMutableData *xData = [NSMutableData dataWithLength:hidden * sizeof(__fp16)];
    NSMutableData *activatedData = [NSMutableData dataWithLength:intermediate * sizeof(__fp16)];
    NSMutableData *outData = [NSMutableData dataWithLength:hidden * sizeof(float)];
    __fp16 *x = xData.mutableBytes, *activated = activatedData.mutableBytes;
    float *out = outData.mutableBytes;
    for (NSUInteger c = 0; c < hidden; ++c) x[c] = (__fp16)(0.03125f * sinf((float)(c + 1) * 0.017f));
    for (NSUInteger row = 0; row < intermediate; ++row) {
        float gateSum = 0, upSum = 0;
        for (NSUInteger c = 0; c < hidden; ++c) {
            gateSum += (float)x[c] * (float)(__fp16)compactValue(gate, row, c);
            upSum += (float)x[c] * (float)(__fp16)compactValue(up, row, c);
        }
        float g = (float)(__fp16)gateSum;
        float u = (float)(__fp16)upSum;
        float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
        activated[row] = (__fp16)(gelu * u);
    }
    for (NSUInteger row = 0; row < hidden; ++row) {
        float sum = 0;
        for (NSUInteger c = 0; c < intermediate; ++c)
            sum += (float)activated[c] * (float)(__fp16)compactValue(down, row, c);
        out[row] = (float)(__fp16)sum;
    }
    return outData;
}

static NSMutableData *cpuGateOracle(CompactMatrix gate, NSUInteger rowOffset, NSUInteger rowCount) {
    NSMutableData *outData = [NSMutableData dataWithLength:rowCount * sizeof(float)];
    float *out = outData.mutableBytes;
    for (NSUInteger localRow = 0; localRow < rowCount; ++localRow) {
        NSUInteger row = rowOffset + localRow;
        float sum = 0;
        for (NSUInteger column = 0; column < gate.columns; ++column) {
            __fp16 x = (__fp16)(0.03125f * sinf((float)(column + 1) * 0.017f));
            __fp16 w = (__fp16)compactValue(gate, row, column);
            sum += (float)x * (float)w;
        }
        out[localRow] = (float)(__fp16)sum;
    }
    return outData;
}

static NSDictionary *validateOutput(IOSurfaceRef output, NSData *oracle,
                                    int outputCount, int positions, NSUInteger valueOffset, BOOL evaluated) {
    double maxAbs = 0, maxRel = 0, rmse = 0, actualMax = 0, expectedMax = 0;
    NSUInteger finite = 0, mismatches = 0, nonzero = 0;
    uint64_t checksum = 1469598103934665603ULL;
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const float *actual = IOSurfaceGetBaseAddress(output);
    const float *expected = oracle.bytes;
    for (int i = 0; i < outputCount; ++i) {
        float actualValue = actual[valueOffset + i * positions];
        double difference = fabs((double)actualValue - expected[i]);
        maxAbs = fmax(maxAbs, difference);
        maxRel = fmax(maxRel, difference / fmax(1e-4, fabs((double)expected[i])));
        actualMax = fmax(actualMax, fabs((double)actualValue));
        expectedMax = fmax(expectedMax, fabs((double)expected[i]));
        rmse += difference * difference;
        if (isfinite(actualValue)) finite++;
        if (actualValue != 0.0f) nonzero++;
        if (difference > 0.05 + 0.03 * fabs((double)expected[i])) mismatches++;
        uint32_t bits = 0;
        memcpy(&bits, &actualValue, sizeof(bits));
        checksum ^= bits;
        checksum *= 1099511628211ULL;
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    rmse = sqrt(rmse / outputCount);
    BOOL validated = evaluated && finite == (NSUInteger)outputCount && nonzero > 0 && mismatches == 0;
    return @{
        @"status": validated ? @"validated" : (evaluated ? @"numerical_mismatch" : @"rejected"),
        @"validated": @(validated),
        @"finite_outputs": @(finite),
        @"nonzero_outputs": @(nonzero),
        @"mismatches": @(mismatches),
        @"actual_max_absolute_value": @(actualMax),
        @"oracle_max_absolute_value": @(expectedMax),
        @"max_absolute_error": @(maxAbs),
        @"max_relative_error": @(maxRel),
        @"rmse": @(rmse),
        @"output_fnv1a64": [NSString stringWithFormat:@"0x%016llx", checksum],
    };
}

static NSDictionary *expertForID(NSDictionary *layerRecord, NSInteger expertID) {
    for (NSDictionary *candidate in layerRecord[@"experts"])
        if ([candidate[@"expert"] integerValue] == expertID) return candidate;
    return nil;
}

static NSArray<NSNumber *> *expertSequence(NSInteger defaultExpert, NSInteger expertCount,
                                           BOOL allowSequence, NSString **error) {
    NSString *spec = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXPERT_SEQUENCE"];
    if (!allowSequence || spec.length == 0) return @[@(defaultExpert)];
    NSMutableArray<NSNumber *> *result = [NSMutableArray array];
    for (NSString *component in [spec componentsSeparatedByString:@","]) {
        NSScanner *scanner = [NSScanner scannerWithString:component];
        NSInteger value = -1;
        if (![scanner scanInteger:&value] || !scanner.isAtEnd || value < 0 || value >= expertCount) {
            if (error) *error = [NSString stringWithFormat:@"invalid expert sequence component: %@", component];
            return nil;
        }
        [result addObject:@(value)];
    }
    if (result.count == 0 || result.count > 129) {
        if (error) *error = @"expert sequence must contain 1...129 ids";
        return nil;
    }
    return result;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&gTimebase);
        if (argc < 2 || argc > 4) {
            fprintf(stderr, "usage: ane-fieldfare-expert-gate MODEL_DIRECTORY [LAYER] [EXPERT]\n");
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSInteger selectedLayer = argc >= 3 ? atoi(argv[2]) : 0;
        NSInteger selectedExpert = argc >= 4 ? atoi(argv[3]) : 0;
        NSDictionary *manifest = readJSON([root stringByAppendingPathComponent:@"manifest.json"]);
        NSDictionary *layout = readJSON([root stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *arch = manifest[@"arch"];
        int hidden = [arch[@"hiddenSize"] intValue];
        int intermediate = [arch[@"moeIntermediateSize"] intValue];
        NSString *construction = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXPERT_CONSTRUCTION"] ?: @"fused";
        BOOL gateOnly = [construction isEqual:@"gate_only"] || [construction isEqual:@"gate_tile"];
        BOOL fp16Carrier = [construction isEqual:@"fused_fp16"];
        BOOL batched8 = [construction isEqual:@"batched8_fp16"];
        BOOL batchNoReduce = batched8 && [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_BATCH_NO_REDUCE"] boolValue];
        int expertBatch = batched8 ? [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXPERT_BATCH"] intValue] : 1;
        if (batched8 && expertBatch == 0) expertBatch = 8;
        BOOL gateTile = [construction isEqual:@"gate_tile"];
        int tileOffset = gateTile ? [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXPERT_TILE_OFFSET"] intValue] : 0;
        int tileWidth = gateTile ? 64 : intermediate;
        if (!gateOnly && ![construction isEqual:@"fused"] && !fp16Carrier && !batched8) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"arguments"});
            return 2;
        }
        if (tileOffset < 0 || tileOffset + tileWidth > intermediate) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"tile_arguments"});
            return 2;
        }
        if (!manifest || !layout || hidden != 2816 || intermediate != 704 ||
            selectedLayer < 0 || selectedLayer >= [layout[@"numLayers"] integerValue] ||
            selectedExpert < 0 || selectedExpert >= [layout[@"expertsPerLayer"] integerValue]) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"installation_contract"});
            return 1;
        }
        NSDictionary *layerRecord = nil;
        for (NSDictionary *candidate in layout[@"layers"])
            if ([candidate[@"layer"] integerValue] == selectedLayer) layerRecord = candidate;
        NSString *sequenceError = nil;
        NSArray<NSNumber *> *sequence = expertSequence(
            selectedExpert, [layout[@"expertsPerLayer"] integerValue], !gateOnly, &sequenceError);
        if (batched8 && NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXPERT_SEQUENCE"].length == 0) {
            NSMutableArray *defaults = [NSMutableArray array];
            for (int index = 0; index < expertBatch; ++index) [defaults addObject:@(index)];
            sequence = defaults;
        }
        if (batched8 && (expertBatch < 2 || expertBatch > 8 || sequence.count != (NSUInteger)expertBatch)) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected",
                   @"stage": @"expert_sequence", @"detail": @"batched8_fp16 requires batch 2...8 and one expert id per batch row"});
            return 2;
        }
        if (!sequence) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected",
                   @"stage": @"expert_sequence", @"detail": sequenceError ?: @"invalid sequence"});
            return 2;
        }
        NSDictionary *expertRecord = expertForID(layerRecord, sequence.firstObject.integerValue);
        NSString *file = [root stringByAppendingPathComponent:[@"packed_experts" stringByAppendingPathComponent:layerRecord[@"file"]]];
        NSData *layerData = [NSData dataWithContentsOfFile:file options:NSDataReadingMappedIfSafe error:nil];
        NSUInteger expertBase = [expertRecord[@"offset"] unsignedIntegerValue];
        CompactMatrix gate = {0}, up = {0}, down = {0};
        if (!layerData || !expertRecord || !resolveMatrix(layerData, expertBase, expertRecord[@"tensors"], @"gate", &gate) ||
            !resolveMatrix(layerData, expertBase, expertRecord[@"tensors"], @"up", &up) ||
            !resolveMatrix(layerData, expertBase, expertRecord[@"tensors"], @"down", &down) ||
            gate.rows != (NSUInteger)intermediate || gate.columns != (NSUInteger)hidden ||
            up.rows != gate.rows || up.columns != gate.columns ||
            down.rows != (NSUInteger)hidden || down.columns != (NSUInteger)intermediate) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"compact_layout"});
            return 1;
        }

        uint64_t footprint = 0;
        int positions = 64;
        if (batched8) fp16Carrier = YES;
        uint64_t carrierElementBytes = fp16Carrier ? sizeof(__fp16) : sizeof(float);
        uint64_t singleCarrierBytes = (uint64_t)hidden * (positions + 3 * intermediate) * carrierElementBytes;
        uint64_t carrierBytes = singleCarrierBytes * (batched8 ? expertBatch : 1);
        NSUInteger runtimeBankCount = !batched8 && sequence.count > 1 ? 2 : 1;
        int outputCount = gateOnly ? tileWidth : hidden;
        uint64_t outputBytes = (uint64_t)outputCount * positions * sizeof(float) * (batchNoReduce ? expertBatch : 1);
        uint64_t requested = carrierBytes * runtimeBankCount + outputBytes * runtimeBankCount +
                             (uint64_t)hidden * sizeof(float) * 4 +
                             (uint64_t)intermediate * sizeof(float) * 4 + [expertRecord[@"size"] unsignedLongLongValue];
        ANEFieldfareMemoryPolicy memoryPolicy = ANEFieldfareDefaultMemoryPolicy();
        footprint = ANEFieldfarePhysicalFootprint();
        ANEFieldfareMemoryDecision memoryDecision = ANEFieldfareMemoryBudgetDecision(
            memoryPolicy, footprint, requested);
        if (memoryDecision >= ANEFieldfareMemoryDecisionReject) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"memory_admission",
                   @"current_physical_footprint_bytes": @(footprint), @"requested_residency_bytes": @(requested),
                   @"target_bytes": @(memoryPolicy.targetBytes), @"hard_limit_bytes": @(memoryPolicy.hardLimitBytes),
                   @"memory_decision": @(memoryDecision)});
            return 1;
        }

        IOSurfaceRef input = makeSurface((NSUInteger)carrierBytes);
        IOSurfaceRef output = makeSurface((NSUInteger)outputBytes);
        IOSurfaceRef secondInput = runtimeBankCount == 2 ? makeSurface((NSUInteger)carrierBytes) : NULL;
        IOSurfaceRef secondOutput = runtimeBankCount == 2 ? makeSurface((NSUInteger)outputBytes) : NULL;
        if (!input || !output || (runtimeBankCount == 2 && (!secondInput || !secondOutput))) return 1;

        void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass) return 1;
        NSString *milText = gateOnly ? gateOnlyMIL(hidden, intermediate, tileWidth, tileOffset, positions) :
            (batched8 ? batchedExpertMIL(hidden, intermediate, positions, expertBatch) : expertMIL(hidden, intermediate, positions, fp16Carrier));
        NSData *mil = [milText dataUsingEncoding:NSUTF8StringEncoding];
        NSString *dumpPath = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_DUMP_MIL"];
        if (dumpPath.length) [mil writeToFile:dumpPath atomically:YES];
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) return 1;
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSError *error = nil;
        uint64_t start = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - start);
        if (!ok) {
            emit(@{@"schema": @"ane-fieldfare.expert-gate.v1", @"status": @"rejected", @"stage": @"compile", @"detail": error.localizedDescription ?: @"unknown"});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }
        start = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        BOOL loaded = ok;
        double loadMs = milliseconds(mach_absolute_time() - start);
        int8_t queueDepth = loaded && [model respondsToSelector:@selector(queueDepth)]
            ? ((int8_t (*)(id, SEL))objc_msgSend)(model, @selector(queueDepth)) : 0;
        IOSurfaceRef inputBanks[2] = {input, secondInput};
        IOSurfaceRef outputBanks[2] = {output, secondOutput};
        id requests[2] = {nil, nil};
        CompletionState completionStates[2] = {0};
        BOOL surfacesMapped[2] = {NO, NO};
        BOOL explicitSurfaceMapping = ![NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_DISABLE_EXPLICIT_MAP"] boolValue];
        BOOL requestSetupOK = loaded;
        for (NSUInteger bank = 0; bank < runtimeBankCount; ++bank) {
            id wrappedInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), inputBanks[bank]);
            id wrappedOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), outputBanks[bank]);
            requests[bank] = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wrappedInput], @[@0], @[wrappedOutput], @[@0], nil, nil, @0);
            completionStates[bank].semaphore = dispatch_semaphore_create(0);
            CompletionState *state = &completionStates[bank];
            void (^completionHandler)(void) = ^{
                atomic_store_explicit(&state->completedAt, mach_absolute_time(), memory_order_release);
                dispatch_semaphore_signal(state->semaphore);
            };
            ((void (*)(id, SEL, id))objc_msgSend)(
                requests[bank], @selector(setCompletionHandler:), completionHandler);
            if (explicitSurfaceMapping) {
                NSError *mapError = nil;
                surfacesMapped[bank] = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                    model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                    requests[bank], YES, &mapError);
                requestSetupOK = requestSetupOK && surfacesMapped[bank];
                if (!surfacesMapped[bank] && !error) error = mapError;
            }
            requestSetupOK = requestSetupOK && requests[bank] != nil;
        }
        id request = requests[0];
        NSMutableArray *runs = [NSMutableArray array];
        double totalPreparationMs = 0, totalOracleMs = 0, totalEvaluationMs = 0, totalEnqueueMs = 0;
        double streamMakespanMs = 0;
        BOOL validated = requestSetupOK;
        if (batched8) {
            CompactMatrix gates[8] = {0}, ups[8] = {0}, downs[8] = {0};
            NSMutableArray<NSData *> *oracles = [NSMutableArray arrayWithCapacity:(NSUInteger)expertBatch];
            uint64_t oracleStart = mach_absolute_time();
            for (NSUInteger index = 0; index < (NSUInteger)expertBatch; ++index) {
                NSDictionary *runExpert = expertForID(layerRecord, sequence[index].integerValue);
                NSUInteger runBase = [runExpert[@"offset"] unsignedIntegerValue];
                BOOL layoutOK = runExpert &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"gate", &gates[index]) &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"up", &ups[index]) &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"down", &downs[index]);
                if (!layoutOK) { validated = NO; break; }
                [oracles addObject:cpuOracle(gates[index], ups[index], downs[index])];
            }
            totalOracleMs = milliseconds(mach_absolute_time() - oracleStart);
            NSMutableData *reducedOracle = [NSMutableData dataWithLength:(NSUInteger)hidden * sizeof(float)];
            float *reduced = reducedOracle.mutableBytes;
            for (NSUInteger index = 0; index < oracles.count; ++index) {
                float denominator = (float)(expertBatch * (expertBatch + 1) / 2);
                float weight = (float)(index + 1) / denominator;
                const float *values = oracles[index].bytes;
                for (int element = 0; element < hidden; ++element)
                    reduced[element] += values[element] * weight;
            }
            uint64_t preparationStart = mach_absolute_time();
            IOSurfaceLock(input, 0, NULL);
            __fp16 *base = IOSurfaceGetBaseAddress(input);
            for (NSUInteger index = 0; index < (NSUInteger)expertBatch; ++index) {
                float denominator = (float)(expertBatch * (expertBatch + 1) / 2);
                float weight = (float)(index + 1) / denominator;
                fillFP16CarrierMemory(base + index * (singleCarrierBytes / sizeof(__fp16)),
                                      gates[index], ups[index], downs[index], (NSUInteger)positions, weight);
            }
            IOSurfaceUnlock(input, 0, NULL);
            totalPreparationMs = milliseconds(mach_absolute_time() - preparationStart);
            IOSurfaceLock(output, 0, NULL);
            memset(IOSurfaceGetBaseAddress(output), 0, (NSUInteger)outputCount * positions * sizeof(float) * (batchNoReduce ? expertBatch : 1));
            IOSurfaceUnlock(output, 0, NULL);
            error = nil;
            uint64_t evaluationStart = mach_absolute_time();
            atomic_store_explicit(&completionStates[0].completedAt, 0, memory_order_relaxed);
            BOOL evaluated = loaded && validated && ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
            double enqueueMs = milliseconds(mach_absolute_time() - evaluationStart);
            BOOL completed = evaluated && dispatch_semaphore_wait(
                completionStates[0].semaphore,
                dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0;
            uint64_t completedAt = atomic_load_explicit(&completionStates[0].completedAt, memory_order_acquire);
            evaluated = evaluated && completed && completedAt >= evaluationStart;
            totalEnqueueMs = enqueueMs;
            totalEvaluationMs = evaluated ? milliseconds(completedAt - evaluationStart) : 0;
            NSMutableDictionary *run = [validateOutput(output, reducedOracle, outputCount, positions, 0, evaluated) mutableCopy];
            if (batchNoReduce) {
                [runs removeAllObjects];
                validated = evaluated;
                for (NSUInteger index = 0; index < (NSUInteger)expertBatch; ++index) {
                    NSMutableDictionary *batchRun = [validateOutput(
                        output, oracles[index], outputCount, positions,
                        index * (NSUInteger)outputCount * positions, evaluated) mutableCopy];
                    batchRun[@"expert"] = sequence[index];
                    [runs addObject:batchRun];
                    validated = validated && [batchRun[@"validated"] boolValue];
                }
            } else {
                run[@"experts"] = sequence;
                NSMutableArray *routingWeights = [NSMutableArray array];
                double denominator = expertBatch * (expertBatch + 1) / 2.0;
                for (int index = 0; index < expertBatch; ++index)
                    [routingWeights addObject:@((index + 1) / denominator)];
                run[@"routing_weights"] = routingWeights;
                run[@"preparation_ms"] = @(totalPreparationMs);
                run[@"cpu_oracle_ms"] = @(totalOracleMs);
                run[@"evaluation_ms"] = @(totalEvaluationMs);
                run[@"enqueue_ms"] = @(enqueueMs);
                run[@"evaluation_error"] = error.localizedDescription ?: @"none";
                [runs addObject:run];
                validated = validated && [run[@"validated"] boolValue];
            }
        } else {
            NSUInteger runCount = sequence.count;
            CompactMatrix *runGates = calloc(runCount, sizeof(*runGates));
            CompactMatrix *runUps = calloc(runCount, sizeof(*runUps));
            CompactMatrix *runDowns = calloc(runCount, sizeof(*runDowns));
            NSMutableArray<NSData *> *oracles = [NSMutableArray arrayWithCapacity:runCount];
            NSMutableArray<NSNumber *> *oracleTimes = [NSMutableArray arrayWithCapacity:runCount];
            NSMutableArray<NSNumber *> *preparationTimes = [NSMutableArray arrayWithCapacity:runCount];
            for (NSUInteger index = 0; index < runCount; ++index) {
                NSInteger expertID = sequence[index].integerValue;
                NSDictionary *runExpert = expertForID(layerRecord, expertID);
                NSUInteger runBase = [runExpert[@"offset"] unsignedIntegerValue];
                BOOL layoutOK = runExpert &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"gate", &runGates[index]) &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"up", &runUps[index]) &&
                    resolveMatrix(layerData, runBase, runExpert[@"tensors"], @"down", &runDowns[index]);
                if (!layoutOK) {
                    [runs addObject:@{@"expert": @(expertID), @"status": @"rejected", @"stage": @"compact_layout"}];
                    validated = NO;
                    break;
                }
                uint64_t oracleStart = mach_absolute_time();
                NSData *runOracle = gateOnly
                    ? cpuGateOracle(runGates[index], (NSUInteger)tileOffset, (NSUInteger)tileWidth)
                    : cpuOracle(runGates[index], runUps[index], runDowns[index]);
                double oracleMs = milliseconds(mach_absolute_time() - oracleStart);
                [oracles addObject:runOracle];
                [oracleTimes addObject:@(oracleMs)];
                totalOracleMs += oracleMs;
            }
            uint64_t streamStart = mach_absolute_time();
            if (validated && runCount) {
                uint64_t preparationStart = mach_absolute_time();
                fillCarrier(inputBanks[0], runGates[0], runUps[0], runDowns[0], (NSUInteger)positions, fp16Carrier);
                double preparationMs = milliseconds(mach_absolute_time() - preparationStart);
                [preparationTimes addObject:@(preparationMs)];
                totalPreparationMs += preparationMs;
                IOSurfaceLock(outputBanks[0], 0, NULL);
                memset(IOSurfaceGetBaseAddress(outputBanks[0]), 0, (NSUInteger)outputBytes);
                IOSurfaceUnlock(outputBanks[0], 0, NULL);
            }
            for (NSUInteger index = 0; validated && index < runCount; ++index) {
                NSUInteger bank = index % runtimeBankCount;
                NSError *runError = nil;
                atomic_store_explicit(&completionStates[bank].completedAt, 0, memory_order_relaxed);
                uint64_t evaluationStart = mach_absolute_time();
                BOOL evaluated = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, requests[bank], &runError);
                double enqueueMs = milliseconds(mach_absolute_time() - evaluationStart);
                totalEnqueueMs += enqueueMs;

                if (evaluated && index + 1 < runCount) {
                    NSUInteger next = index + 1;
                    NSUInteger nextBank = next % runtimeBankCount;
                    uint64_t preparationStart = mach_absolute_time();
                    fillCarrier(inputBanks[nextBank], runGates[next], runUps[next], runDowns[next],
                                (NSUInteger)positions, fp16Carrier);
                    double preparationMs = milliseconds(mach_absolute_time() - preparationStart);
                    [preparationTimes addObject:@(preparationMs)];
                    totalPreparationMs += preparationMs;
                    IOSurfaceLock(outputBanks[nextBank], 0, NULL);
                    memset(IOSurfaceGetBaseAddress(outputBanks[nextBank]), 0, (NSUInteger)outputBytes);
                    IOSurfaceUnlock(outputBanks[nextBank], 0, NULL);
                }

                BOOL completed = evaluated && dispatch_semaphore_wait(
                    completionStates[bank].semaphore,
                    dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0;
                uint64_t completedAt = atomic_load_explicit(
                    &completionStates[bank].completedAt, memory_order_acquire);
                evaluated = evaluated && completed && completedAt >= evaluationStart;
                double evaluationMs = evaluated ? milliseconds(completedAt - evaluationStart) : 0;
                totalEvaluationMs += evaluationMs;
                NSMutableDictionary *run = [validateOutput(
                    outputBanks[bank], oracles[index], outputCount, positions, 0, evaluated) mutableCopy];
                run[@"expert"] = sequence[index];
                run[@"bank"] = @(bank);
                run[@"preparation_ms"] = preparationTimes[index];
                run[@"cpu_oracle_ms"] = oracleTimes[index];
                run[@"enqueue_ms"] = @(enqueueMs);
                run[@"completion_ms"] = @(evaluationMs);
                run[@"evaluation_ms"] = @(evaluationMs);
                run[@"evaluation_error"] = runError.localizedDescription ?: (completed ? @"none" : @"completion timeout");
                [runs addObject:run];
                validated = [run[@"validated"] boolValue];
            }
            streamMakespanMs = milliseconds(mach_absolute_time() - streamStart);
            free(runGates);
            free(runUps);
            free(runDowns);
        }
        BOOL returnedToA = runs.count >= 3 &&
            [sequence.firstObject isEqual:sequence.lastObject] &&
            [runs.firstObject[@"output_fnv1a64"] isEqual:runs.lastObject[@"output_fnv1a64"]];
        if ([sequence.firstObject isEqual:sequence.lastObject] && sequence.count >= 3) validated = validated && returnedToA;
        NSDictionary *representative = runs.firstObject ?: @{};
        NSUInteger evaluationCount = batched8 ? (loaded ? 1 : 0) : runs.count;
        double evaluationMs = evaluationCount ? totalEvaluationMs / evaluationCount : 0;
        double cpuMs = runs.count ? totalOracleMs / runs.count : 0;
        uint64_t finalFootprint = ANEFieldfarePhysicalFootprint();
        for (NSUInteger bank = 0; bank < runtimeBankCount; ++bank) {
            if (surfacesMapped[bank])
                ((void (*)(id, SEL, id))objc_msgSend)(
                    model, @selector(unmapIOSurfacesWithRequest:), requests[bank]);
            if (requests[bank])
                ((void (*)(id, SEL, id))objc_msgSend)(
                    requests[bank], @selector(setCompletionHandler:), nil);
        }
        if (loaded) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &error);
        [files removeItemAtPath:temporary error:nil];
        CFRelease(input); CFRelease(output);
        if (secondInput) CFRelease(secondInput);
        if (secondOutput) CFRelease(secondOutput);

        NSProcessInfo *process = NSProcessInfo.processInfo;
        emit(@{
            @"schema": @"ane-fieldfare.expert-gate.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"numerical_oracle" : @"expert_stream",
            @"gate": gateOnly ? @"checkpoint_gate_projection_via_runtime_input" :
                (batched8 ? @"batched_checkpoint_experts_via_runtime_input" :
                 (sequence.count > 1 ? @"streamed_checkpoint_experts_via_runtime_input" : @"exact_checkpoint_expert_via_runtime_input")),
            @"construction": construction,
            @"carrier_dtype": fp16Carrier ? @"fp16" : @"fp32",
            @"tile": gateOnly ? @{@"output_offset": @(tileOffset), @"output_count": @(tileWidth)} : @{},
            @"model_id": manifest[@"modelID"] ?: @"unknown",
            @"layer": @(selectedLayer), @"expert": sequence.firstObject,
            @"expert_sequence": sequence,
            @"shape": @{@"hidden": @(hidden), @"intermediate": @(intermediate), @"positions": @(positions)},
            @"compact_expert_bytes": expertRecord[@"size"],
            @"compact_batch_bytes": @([expertRecord[@"size"] unsignedLongLongValue] * (batched8 ? (uint64_t)expertBatch : 1)),
            @"expanded_carrier_bytes": @(carrierBytes),
            @"compile_count": @1, @"load_count": loaded ? @1 : @0, @"evaluation_count": @(evaluationCount),
            @"compile_ms": @(compileMs), @"load_ms": @(loadMs),
            @"evaluation_ms": @(evaluationMs), @"evaluation_total_ms": @(totalEvaluationMs),
            @"enqueue_ms": @(evaluationCount ? totalEnqueueMs / evaluationCount : 0),
            @"enqueue_total_ms": @(totalEnqueueMs),
            @"stream_makespan_ms": @(streamMakespanMs),
            @"serial_preparation_plus_completion_ms": @(totalPreparationMs + totalEvaluationMs),
            @"measured_overlap_savings_ms": @(MAX(0.0, totalPreparationMs + totalEvaluationMs - streamMakespanMs)),
            @"runtime": @{@"request_bank_count": @(runtimeBankCount),
                           @"completion_backed": @YES,
                           @"explicit_iosurface_mapping": @(explicitSurfaceMapping),
                           @"mapped_request_count": @((surfacesMapped[0] ? 1 : 0) + (surfacesMapped[1] ? 1 : 0)),
                           @"reported_queue_depth": @(queueDepth)},
            @"preparation_ms": @(batched8 ? totalPreparationMs : (runs.count ? totalPreparationMs / runs.count : 0)),
            @"preparation_total_ms": @(totalPreparationMs),
            @"cpu_oracle_ms": @(cpuMs), @"cpu_oracle_total_ms": @(totalOracleMs),
            @"finite_outputs": representative[@"finite_outputs"] ?: @0,
            @"nonzero_outputs": representative[@"nonzero_outputs"] ?: @0,
            @"mismatches": representative[@"mismatches"] ?: @0,
            @"actual_max_absolute_value": representative[@"actual_max_absolute_value"] ?: @0,
            @"oracle_max_absolute_value": representative[@"oracle_max_absolute_value"] ?: @0,
            @"max_absolute_error": representative[@"max_absolute_error"] ?: @0,
            @"max_relative_error": representative[@"max_relative_error"] ?: @0,
            @"rmse": representative[@"rmse"] ?: @0,
            @"output_fnv1a64": representative[@"output_fnv1a64"] ?: @"none",
            @"a_b_a_checksum_restored": @(returnedToA),
            @"runs": runs,
            @"memory": @{@"target_bytes": @(memoryPolicy.targetBytes), @"hard_limit_bytes": @(memoryPolicy.hardLimitBytes),
                           @"admission_state": memoryDecision == ANEFieldfareMemoryDecisionAdmit ? @"within_target" : @"within_tolerance",
                           @"admission_headroom_bytes": @(memoryPolicy.requiredHeadroomBytes),
                           @"initial_physical_footprint_bytes": @(footprint), @"requested_residency_bytes": @(requested),
                           @"post_evaluation_physical_footprint_bytes": @(finalFootprint)},
            @"host": process.hostName ?: @"unknown", @"os": process.operatingSystemVersionString ?: @"unknown",
            @"claim_boundary": gateOnly
                ? @"One authentic checkpoint expert gate projection at [704,2816], dequantized once into an ordinary FP32 ANE input carrier and evaluated against an independent CPU oracle. Up, activation, down, a complete expert, compiler-declared mutable weights, compact prepared execution, routing, and generation are not claimed."
                : (validated
                    ? (batched8
                        ? @"One compiled and loaded batch graph evaluated eight authentic compact-checkpoint experts in one ANE request, returning eight independent outputs that each match their CPU oracle. The measured request is slower than eight warmed single-expert dispatches, so this construction is a validated scheduling negative rather than the production routed-layer path. It does not validate router selection, weighted reduction, a full layer, logits, or token generation."
                        : (sequence.count > 1
                        ? @"One compiled and loaded real-shape ANE program reused across a completion-backed two-bank sequence of authentic compact-checkpoint experts. Each expert executes gate/up, gelu_pytorch_tanh product, and down after host expansion into a reusable ordinary runtime-input carrier of the recorded dtype, with position zero checked against an independent CPU oracle. This validates mapped request reuse and host-preparation overlap, not compact prepared weights, router-selected top-8, a full layer, logits, or token generation."
                        : @"One authentic compact checkpoint routed expert executed end to end as gate/up, gelu_pytorch_tanh product, and down through an ordinary ANE runtime-input carrier of the recorded dtype, with position zero checked against an independent CPU oracle. This is real expert execution, but not compiler-declared mutable weights, compact prepared execution, routed top-8, a full layer, logits, token generation, or streaming."))
                    : @"One authentic compact checkpoint routed-expert fused construction attempted through an ordinary FP32 ANE input carrier and checked against an independent CPU oracle. A rejected evaluation is not expert execution. This is not compiler-declared mutable weight execution, compact prepared execution, routed top-8, a full layer, logits, token generation, or streaming.")
        });
        return validated ? 0 : 1;
    }
}
