#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static BOOL relevant(NSString *text) {
    NSString *lower = text.lowercaseString;
    return [lower containsString:@"pool"] ||
           [lower containsString:@"wire"] ||
           [lower containsString:@"unwir"] ||
           [lower containsString:@"instance"] ||
           [lower containsString:@"load"] ||
           [lower containsString:@"client"] ||
           [lower containsString:@"quota"];
}

static NSArray *methodsForClass(Class cls, BOOL classMethods) {
    Class owner = classMethods ? object_getClass(cls) : cls;
    NSString *className = NSStringFromClass(cls);
    BOOL includeAll = [className isEqualToString:@"_ANEInMemoryModel"] ||
                      [className isEqualToString:@"_ANEClient"] ||
                      [className isEqualToString:@"_ANEModel"] ||
                      [className isEqualToString:@"_ANERequest"] ||
                      [className isEqualToString:@"_ANEProgramIOSurfacesMapper"] ||
                      [className isEqualToString:@"_ANEProgramForEvaluation"] ||
                      [className isEqualToString:@"_ANEProgramProcedurePriv"] ||
                      [className isEqualToString:@"_ANEModelInstanceParameters"] ||
                      [className isEqualToString:@"_ANEProcedureData"] ||
                      [className isEqualToString:@"_ANEWeight"] ||
                      [className isEqualToString:@"_ANECloneHelper"];
    unsigned int count = 0;
    Method *methods = class_copyMethodList(owner, &count);
    NSMutableArray *records = [NSMutableArray array];
    for (unsigned int i = 0; i < count; ++i) {
        SEL selector = method_getName(methods[i]);
        NSString *name = NSStringFromSelector(selector);
        if (!includeAll && !relevant(name)) continue;
        const char *encoding = method_getTypeEncoding(methods[i]);
        [records addObject:@{
            @"kind": classMethods ? @"class" : @"instance",
            @"selector": name,
            @"encoding": encoding ? [NSString stringWithUTF8String:encoding] : @"",
        }];
    }
    free(methods);
    return records;
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        int count = objc_getClassList(NULL, 0);
        Class *classes = count > 0 ? (__unsafe_unretained Class *)calloc((size_t)count, sizeof(Class)) : NULL;
        if (classes) count = objc_getClassList(classes, count);
        NSMutableArray *classRecords = [NSMutableArray array];
        for (int i = 0; i < count; ++i) {
            NSString *name = NSStringFromClass(classes[i]);
            if (![name hasPrefix:@"_ANE"] && ![name hasPrefix:@"ANE"]) continue;
            NSArray *instance = methodsForClass(classes[i], NO);
            NSArray *classMethods = methodsForClass(classes[i], YES);
            if (!instance.count && !classMethods.count && !relevant(name)) continue;
            unsigned int ivarCount = 0;
            Ivar *ivars = class_copyIvarList(classes[i], &ivarCount);
            NSMutableArray *ivarRecords = [NSMutableArray array];
            for (unsigned int j = 0; j < ivarCount; ++j) {
                const char *ivarName = ivar_getName(ivars[j]);
                NSString *ivar = ivarName ? [NSString stringWithUTF8String:ivarName] : @"";
                if (!relevant(ivar)) continue;
                const char *type = ivar_getTypeEncoding(ivars[j]);
                [ivarRecords addObject:@{
                    @"name": ivar,
                    @"type": type ? [NSString stringWithUTF8String:type] : @"",
                    @"offset": @(ivar_getOffset(ivars[j])),
                }];
            }
            free(ivars);
            [classRecords addObject:@{
                @"class": name,
                @"methods": [instance arrayByAddingObjectsFromArray:classMethods],
                @"ivars": ivarRecords,
            }];
        }
        free(classes);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.private-surface.v1",
            @"framework_loaded": @(framework != NULL),
            @"classes": classRecords,
            @"claim_boundary": @"Runtime method and ivar inventory only. Presence and encodings do not establish accepted or validated behavior.",
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        return framework ? 0 : 1;
    }
}
