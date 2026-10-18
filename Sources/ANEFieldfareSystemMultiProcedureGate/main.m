#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>

static id callObject(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((id (*)(id, SEL))objc_msgSend)(object, selector) : nil;
}

static uint64_t callUInt64(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((uint64_t (*)(id, SEL))objc_msgSend)(object, selector) : 0;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

int main(void) {
    @autoreleasepool {
        NSString *path = @"/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Speech_AutomaticSpeechRecognition/purpose_auto/971265b62793d7941762a4aed871a2aa6310c020.asset/.AssetData/VAD.mlmodelc/model.specialization.bundle/H17G.bundle/context_16/multiprocedure";
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class modelClass = NSClassFromString(@"_ANEModel");
        Class clientClass = NSClassFromString(@"_ANEClient");
        NSURL *url = [NSURL fileURLWithPath:path isDirectory:YES];
        id model = framework && modelClass ?
            ((id (*)(Class, SEL, id, id))objc_msgSend)(
                modelClass, @selector(modelAtURL:key:), url, nil) : nil;
        id client = clientClass ? ((id (*)(Class, SEL))objc_msgSend)(
            clientClass, @selector(sharedConnection)) : nil;
        NSError *error = nil;
        BOOL loaded = model && client && ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(loadModel:options:qos:error:), model, @{}, 21, &error);
        NSDictionary *attributes = callObject(model, @selector(modelAttributes));
        NSArray *attributeProcedures = attributes[@"ANEFModelDescription"][@"ANEFModelProcedures"];
        unsigned int procedureCount = [model respondsToSelector:@selector(proceduresCount)]
            ? ((unsigned int (*)(id, SEL))objc_msgSend)(model, @selector(proceduresCount)) : 0;
        NSMutableArray *procedures = [NSMutableArray array];
        NSUInteger count = MAX((NSUInteger)procedureCount, attributeProcedures.count);
        for (NSUInteger index = 0; index < count; ++index) {
            id info = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                model, @selector(procedureInfoForProcedureIndex:), (unsigned int)index);
            id inputs = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                model, @selector(inputSymbolIndicesForProcedureIndex:), (unsigned int)index);
            id outputs = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                model, @selector(outputSymbolIndicesForProcedureIndex:), (unsigned int)index);
            [procedures addObject:@{@"index": @(index), @"info": [info description] ?: @"none",
                                    @"inputs": [inputs description] ?: @"none",
                                    @"outputs": [outputs description] ?: @"none"}];
        }
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.system-multiprocedure.v1",
            @"status": loaded && count >= 2 ? @"accepted" : @"rejected",
            @"source": path, @"model_created": @(model != nil), @"loaded": @(loaded),
            @"error": error.localizedDescription ?: @"none",
            @"program_handle": @(callUInt64(model, @selector(programHandle))),
            @"procedures_count": @(procedureCount),
            @"attribute_procedure_count": @(attributeProcedures.count),
            @"num_inputs": @(((unsigned int (*)(id, SEL))objc_msgSend)(model, @selector(numInputs))),
            @"num_outputs": @(((unsigned int (*)(id, SEL))objc_msgSend)(model, @selector(numOutputs))),
            @"procedures": procedures,
            @"model_attributes": attributes ?: @{},
            @"claim_boundary": @"Load and compiler-authored procedure-table inspection for Apple's shipped multiprocedure hardware binary. Acceptance proves multiple procedures coexist behind one loaded program handle; no numerical inference oracle is included."
        };
        emit(receipt);
        if (loaded) {
            NSError *unloadError = nil;
            ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                client, @selector(unloadModel:options:qos:error:), model, @{}, 21, &unloadError);
        }
        return loaded && count >= 2 ? 0 : 1;
    }
}
