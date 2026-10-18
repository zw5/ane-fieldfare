#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *root = argc > 1 ? [NSString stringWithUTF8String:argv[1]] :
            @"/System/Library/PrivateFrameworks/EmbeddingCore.framework/Versions/A/Resources/cross_encoder_v130_ane_8bit_combined.mlmodelc";
        NSString *milPath = [root stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [root stringByAppendingPathComponent:@"weights/weight.bin"];
        NSData *mil = [NSData dataWithContentsOfFile:milPath];
        NSData *weightsData = [NSData dataWithContentsOfFile:weightPath];
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        NSError *error = nil;
        BOOL compiled = NO, loaded = NO, unloaded = NO;
        double compileMs = 0, loadMs = 0;
        NSString *failure = @"none";
        id model = nil;
        NSString *temporary = nil;
        if (!framework || !mil || !weightsData || !descriptorClass || !modelClass) {
            failure = @"framework, model MIL, or weights unavailable";
        } else {
            NSDictionary *weights = @{@"@model_path/weights/weight.bin":
                @{ @"offset": @0, @"data": weightsData }};
            id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
                descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weights, nil);
            model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
                modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
            if (!model) failure = @"in-memory model construction failed";
            else {
                NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
                temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
                [NSFileManager.defaultManager createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
                                         withIntermediateDirectories:YES attributes:nil error:nil];
                [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                [weightsData writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
                uint64_t start = mach_continuous_time();
                compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
                mach_timebase_info_data_t tb; mach_timebase_info(&tb);
                compileMs = (double)(mach_continuous_time() - start) * tb.numer / tb.denom / 1e6;
                if (!compiled) failure = error.localizedDescription ?: @"compile failed";
                if (compiled) {
                    start = mach_continuous_time();
                    loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                        model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
                    loadMs = (double)(mach_continuous_time() - start) * tb.numer / tb.denom / 1e6;
                    if (!loaded) failure = error.localizedDescription ?: @"load failed";
                }
                if (loaded) unloaded = ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                    model, @selector(unloadWithQoS:error:), 21, &error);
            }
        }
        if (temporary) [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.system-compressed-model.v1",
            @"source_model": root,
            @"mil_bytes": @(mil.length),
            @"weight_blob_bytes": @(weightsData.length),
            @"contains_blockwise_int8": @(mil && [[NSString alloc] initWithData:mil encoding:NSUTF8StringEncoding] &&
                [[[NSString alloc] initWithData:mil encoding:NSUTF8StringEncoding] containsString:@"constexpr_blockwise_shift_scale(data = tensor<int8"]),
            @"compiled": @(compiled),
            @"loaded": @(loaded),
            @"unloaded": @(unloaded),
            @"compile_ms": @(compileMs),
            @"load_ms": @(loadMs),
            @"error": failure,
            @"claim_boundary": @"Compile/load lifecycle for Apple's shipped int8 blockwise model through the same _ANEInMemoryModel route. No inference or numerical oracle; the source MIL and stored blob establish compressed source storage, while successful ANE load establishes route acceptance, not per-layer speed.",
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        return loaded && unloaded ? 0 : 1;
    }
}
