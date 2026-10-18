#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static BOOL interesting(NSString *name) {
    NSArray<NSString *> *needles = @[
        @"weight", @"mutable", @"buffer", @"procedure", @"request", @"queue",
        @"state", @"surface", @"map", @"cache", @"compile", @"load", @"evaluate",
        @"performance", @"intermediate", @"memory", @"synchron"
        , @"connection", @"controller", @"attribute", @"model", @"handle", @"option"
    ];
    NSString *lower = name.lowercaseString;
    for (NSString *needle in needles) if ([lower containsString:needle]) return YES;
    return NO;
}

static NSArray *methodsForClass(Class cls, BOOL classMethods) {
    Class target = classMethods ? object_getClass(cls) : cls;
    unsigned int count = 0;
    Method *methods = class_copyMethodList(target, &count);
    NSMutableArray *records = [NSMutableArray array];
    for (unsigned int index = 0; index < count; ++index) {
        NSString *selector = NSStringFromSelector(method_getName(methods[index]));
        if (!interesting(selector)) continue;
        const char *encoding = method_getTypeEncoding(methods[index]);
        [records addObject:@{
            @"selector": selector,
            @"encoding": encoding ? [NSString stringWithUTF8String:encoding] : @"unknown",
            @"kind": classMethods ? @"class" : @"instance"
        }];
    }
    free(methods);
    return [records sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *left, NSDictionary *right) {
        return [left[@"selector"] compare:right[@"selector"]];
    }];
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        void *compilerFramework = dlopen(
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
            RTLD_NOW | RTLD_LOCAL);
        void *milFramework = dlopen(
            "/System/Library/PrivateFrameworks/MilAneflow.framework/MilAneflow",
            RTLD_NOW | RTLD_LOCAL);
        NSArray<NSString *> *names = @[
            @"_ANEInMemoryModelDescriptor", @"_ANEInMemoryModel", @"_ANEProgramForEvaluation",
            @"_ANERequest", @"_ANEIOSurfaceObject", @"_ANEModel", @"_ANEClient",
            @"_ANEBuffer", @"_ANEWeight", @"_ANEProcedureData", @"_ANEProgramProcedurePriv",
            @"_ANEChainingRequest", @"_ANEProgramIOSurfacesMapper"
        ];
        NSMutableArray *classes = [NSMutableArray array];
        for (NSString *name in names) {
            Class cls = NSClassFromString(name);
            if (!cls) continue;
            [classes addObject:@{
                @"class": name,
                @"instance_methods": methodsForClass(cls, NO),
                @"class_methods": methodsForClass(cls, YES)
            }];
        }
        unsigned int runtimeCount = 0;
        Class *runtimeClasses = objc_copyClassList(&runtimeCount);
        NSMutableArray *discovered = [NSMutableArray array];
        if (runtimeClasses) {
            for (unsigned int index = 0; index < runtimeCount; ++index) {
                NSString *name = NSStringFromClass(runtimeClasses[index]);
                NSString *lower = name.lowercaseString;
                if (([name hasPrefix:@"_ANE"] || [lower containsString:@"ane"] ||
                     [lower containsString:@"mil"]) &&
                    ([lower containsString:@"weight"] || [lower containsString:@"buffer"] ||
                     [lower containsString:@"procedure"] || [lower containsString:@"state"] ||
                     [lower containsString:@"request"] || [lower containsString:@"program"] ||
                     [lower containsString:@"compiler"] || [lower containsString:@"option"]))
                    [discovered addObject:name];
            }
            free(runtimeClasses);
        }
        [discovered sortUsingSelector:@selector(compare:)];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.capability-surface.v1",
            @"status": framework && classes.count ? @"present" : @"rejected",
            @"framework_loaded": @(framework != NULL),
            @"compiler_framework_loaded": @(compilerFramework != NULL),
            @"mil_framework_loaded": @(milFramework != NULL),
            @"classes": classes,
            @"discovered_weight_state_request_classes": discovered,
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"claim_boundary": @"Live Objective-C method inventory for ANE weight, state, request, mapping, queue, and evaluation surfaces. Presence is not functional acceptance or numerical validation."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return framework && classes.count ? 0 : 1;
    }
}
