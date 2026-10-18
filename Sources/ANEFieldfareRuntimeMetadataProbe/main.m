#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static BOOL isRelevant(NSString *name) {
    static NSArray<NSString *> *needles;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        needles = @[@"async", @"chain", @"completion", @"counter", @"event",
                    @"intermediate", @"map", @"mutable", @"perf", @"procedure",
                    @"queue", @"state", @"sync", @"transaction", @"weight"];
    });
    NSString *lower = name.lowercaseString;
    for (NSString *needle in needles) if ([lower containsString:needle]) return YES;
    return NO;
}

static NSArray *methods(Class cls, BOOL classMethods, BOOL relevantOnly) {
    Class owner = classMethods ? object_getClass(cls) : cls;
    unsigned int count = 0;
    Method *list = class_copyMethodList(owner, &count);
    NSMutableArray *out = [NSMutableArray array];
    for (unsigned int i = 0; i < count; ++i) {
        NSString *selector = NSStringFromSelector(method_getName(list[i]));
        if (relevantOnly && !isRelevant(selector)) continue;
        const char *encoding = method_getTypeEncoding(list[i]);
        [out addObject:@{
            @"kind": classMethods ? @"class" : @"instance",
            @"selector": selector,
            @"encoding": encoding ? @(encoding) : @"",
        }];
    }
    free(list);
    [out sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [a[@"selector"] compare:b[@"selector"]];
    }];
    return out;
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        NSArray<NSString *> *names = @[
            @"_ANEInMemoryModel", @"_ANEProgramForEvaluation", @"_ANERequest",
            @"_ANEClient", @"_ANEModel", @"_ANEProcedureData",
            @"_ANEProgramProcedurePriv", @"_ANEChainingRequest",
            @"_ANEInputBuffersReady", @"_ANEOutputSetEnqueue",
            @"_ANEIOSurfaceOutputSets", @"_ANEProgramIOSurfacesMapper"
        ];
        NSMutableArray *classes = [NSMutableArray array];
        for (NSString *name in names) {
            Class cls = NSClassFromString(name);
            if (!cls) continue;
            [classes addObject:@{
                @"class": name,
                @"instance_methods": methods(cls, NO, YES),
                @"class_methods": methods(cls, YES, YES),
            }];
        }
        int runtimeClassCount = objc_getClassList(NULL, 0);
        __unsafe_unretained Class *runtimeClasses = runtimeClassCount > 0
            ? (__unsafe_unretained Class *)calloc((size_t)runtimeClassCount, sizeof(Class)) : NULL;
        if (runtimeClasses) runtimeClassCount = objc_getClassList(runtimeClasses, runtimeClassCount);
        NSMutableArray *frameworkClassNames = [NSMutableArray array];
        NSMutableArray *frameworkClasses = [NSMutableArray array];
        NSMutableArray *discoveredRelevantClasses = [NSMutableArray array];
        for (int index = 0; index < runtimeClassCount; ++index) {
            Class cls = runtimeClasses[index];
            const char *image = class_getImageName(cls);
            if (!image || !strstr(image, "AppleNeuralEngine.framework")) continue;
            NSString *name = NSStringFromClass(cls);
            [frameworkClassNames addObject:name];
            NSArray *allInstanceMethods = methods(cls, NO, NO);
            NSArray *allClassMethods = methods(cls, YES, NO);
            NSArray *relevantInstanceMethods = methods(cls, NO, YES);
            NSArray *relevantClassMethods = methods(cls, YES, YES);
            [frameworkClasses addObject:@{
                @"class": name,
                @"instance_methods": allInstanceMethods,
                @"class_methods": allClassMethods,
            }];
            if (isRelevant(name) || relevantInstanceMethods.count || relevantClassMethods.count) {
                [discoveredRelevantClasses addObject:@{
                    @"class": name,
                    @"instance_methods": allInstanceMethods,
                    @"class_methods": allClassMethods,
                }];
            }
        }
        free(runtimeClasses);
        [frameworkClassNames sortUsingSelector:@selector(compare:)];
        [frameworkClasses sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
            return [a[@"class"] compare:b[@"class"]];
        }];
        [discoveredRelevantClasses sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
            return [a[@"class"] compare:b[@"class"]];
        }];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.runtime-metadata.v2",
            @"status": framework && classes.count ? @"present" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"command": @".build/ane-fieldfare-runtime-metadata-probe",
            @"framework_loaded": @(framework != NULL),
            @"classes": classes,
            @"framework_class_count": @(frameworkClassNames.count),
            @"framework_class_names": frameworkClassNames,
            @"framework_classes": frameworkClasses,
            @"discovered_relevant_classes": discoveredRelevantClasses,
            @"claim_boundary": @"Live Objective-C selector and type-encoding inventory only. Presence does not establish call acceptance, execution, synchronization, persistence, counters, concurrency, or numerical correctness."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return framework && classes.count ? 0 : 1;
    }
}
