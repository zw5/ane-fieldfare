#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSDictionary *check(Class cls, SEL selector, BOOL classMethod, NSString *expected) {
    Method method = classMethod ? class_getClassMethod(cls, selector) : class_getInstanceMethod(cls, selector);
    const char *raw = method ? method_getTypeEncoding(method) : NULL;
    NSString *actual = raw ? [NSString stringWithUTF8String:raw] : @"missing";
    return @{
        @"class": cls ? NSStringFromClass(cls) : @"missing",
        @"selector": NSStringFromSelector(selector),
        @"method_kind": classMethod ? @"class" : @"instance",
        @"expected_encoding": expected,
        @"actual_encoding": actual,
        @"matches": @([actual isEqual:expected]),
    };
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class model = NSClassFromString(@"_ANEInMemoryModel");
        Class request = NSClassFromString(@"_ANERequest");
        Class surface = NSClassFromString(@"_ANEIOSurfaceObject");
        NSArray *checks = @[
            check(descriptor, @selector(modelWithMILText:weights:optionsPlist:), YES, @"@40@0:8@16@24@32"),
            check(model, @selector(inMemoryModelWithDescriptor:), YES, @"@24@0:8@16"),
            check(model, @selector(compileWithQoS:options:error:), NO, @"B36@0:8I16@20^@28"),
            check(model, @selector(loadWithQoS:options:error:), NO, @"B36@0:8I16@20^@28"),
            check(model, @selector(evaluateWithQoS:options:request:error:), NO, @"B44@0:8I16@20@28^@36"),
            check(model, @selector(mapIOSurfacesWithRequest:cacheInference:error:), NO, @"B36@0:8@16B24^@28"),
            check(model, @selector(unmapIOSurfacesWithRequest:), NO, @"v24@0:8@16"),
            check(model, @selector(queueDepth), NO, @"c16@0:8"),
            check(model, @selector(unloadWithQoS:error:), NO, @"B28@0:8I16^@20"),
            check(surface, @selector(objectWithIOSurface:), YES, @"@24@0:8^{__IOSurface=}16"),
            check(request,
                  @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                  YES, @"@72@0:8@16@24@32@40@48@56@64"),
            check(request, @selector(setCompletionHandler:), NO, @"v24@0:8@?16"),
        ];
        BOOL valid = framework != NULL;
        for (NSDictionary *record in checks) valid = valid && [record[@"matches"] boolValue];
        NSProcessInfo *process = NSProcessInfo.processInfo;
        emit(@{
            @"schema": @"ane-fieldfare.api-gate.v2",
            @"status": valid ? @"validated" : @"rejected",
            @"framework_loaded": @(framework != NULL),
            @"checks": checks,
            @"host": process.hostName ?: @"unknown",
            @"os": process.operatingSystemVersionString ?: @"unknown",
            @"claim_boundary": @"Exact Objective-C runtime surface and type-encoding lock for the asynchronous in-memory ANE lifecycle, explicit IOSurface mapping, and completion-backed request ring used by this repository; selector presence alone does not validate functional execution.",
        });
        return valid ? 0 : 1;
    }
}
