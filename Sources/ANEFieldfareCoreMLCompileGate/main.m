#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Core ML is used here only to invoke Apple's closed package compiler and
// retain its output for inspection.  Successful compilation is not ANE device
// assignment, execution, or residency evidence.
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s <input.mlpackage> <output.mlmodelc>\n", argv[0]);
            return 2;
        }
        NSURL *input = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        NSURL *output = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
        NSError *error = nil;
        NSURL *temporary = [MLModel compileModelAtURL:input error:&error];
        BOOL compiled = temporary != nil;
        BOOL retained = NO;
        if (compiled) {
            NSFileManager *files = NSFileManager.defaultManager;
            if ([files fileExistsAtPath:output.path] && ![files removeItemAtURL:output error:&error]) {
                retained = NO;
            } else {
                retained = [files copyItemAtURL:temporary toURL:output error:&error];
            }
        }
        NSMutableArray *hwxFiles = [NSMutableArray array];
        if (retained) {
            NSDirectoryEnumerator *enumerator = [NSFileManager.defaultManager
                enumeratorAtURL:output
     includingPropertiesForKeys:nil
                        options:0
                   errorHandler:nil];
            for (NSURL *file in enumerator)
                if ([file.pathExtension.lowercaseString isEqualToString:@"hwx"])
                    [hwxFiles addObject:[file.path substringFromIndex:output.path.length + 1]];
        }
        BOOL multiprocedure = [hwxFiles indexOfObjectPassingTest:
            ^BOOL(NSString *path, NSUInteger index, BOOL *stop) {
                (void)index; (void)stop;
                return [path containsString:@"multiprocedure/model.hwx"];
            }] != NSNotFound;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.coreml-compile-artifact.v1",
            @"status": retained ? @"accepted" : @"rejected",
            @"input": input.path,
            @"output": output.path,
            @"compile_count": @(compiled ? 1 : 0),
            @"load_count": @0,
            @"numerical_result": @"not_run",
            @"hwx_files": hwxFiles,
            @"multiprocedure_hwx_present": @(multiprocedure),
            @"error": error.localizedDescription ?: @"none",
            @"claim_boundary": @"Retains the output of the public Core ML package compiler for static artifact inspection only. Compilation does not establish ANE assignment, private-ANE loading, hardware execution, multiprocedure residency, or a numerical oracle."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
        const char *receiptPath = getenv("ANE_FIELDFARE_RECEIPT_PATH");
        if (receiptPath) [json writeToFile:[NSString stringWithUTF8String:receiptPath] atomically:YES];
        return retained ? 0 : 1;
    }
}
