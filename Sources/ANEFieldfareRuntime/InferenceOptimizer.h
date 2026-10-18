#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Training-free execution-field lowerer. Model adapters supply logical shape;
// backends supply explicit capabilities and measurements. The lowerer contains
// no device thresholds or cache-size policy. Candidate plans are not trusted
// until an external reference/candidate admission writes a validated receipt.
@interface ANEFieldfareInferenceOptimizer : NSObject

@property(nonatomic, readonly) NSDictionary<NSString *, NSNumber *> *architecture;
@property(nonatomic, readonly) NSString *architectureSignature;

+ (nullable instancetype)optimizerForModelDirectory:(NSString *)modelDirectory
                                               error:(NSString * _Nullable * _Nullable)error;

// Generates a cheap in-memory candidate. This performs no weight training and
// no model compilation. It describes a schedule using the backend capabilities
// currently implemented by the runtime.
- (NSDictionary *)candidatePlanForPositionCount:(NSUInteger)positionCount
                           availableMemoryBytes:(uint64_t)availableMemoryBytes
                            backendCapabilities:(NSDictionary *)capabilities
                               costObservations:(NSDictionary *)observations;

// Loads only a plan whose receipt is validated, whose architecture signature
// matches, and whose declared position range contains positionCount.
- (nullable NSDictionary *)loadAdmittedPlanAtPath:(NSString *)path
                                     positionCount:(NSUInteger)positionCount
                                             error:(NSString * _Nullable * _Nullable)error;

@end

NS_ASSUME_NONNULL_END
