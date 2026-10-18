import Metal
import Testing

import ANEFieldfareHotpath

@Suite struct PublicAPISignatureCompatibilityTests {
    @Test func originalDescriptorFreeFunctionTypesRemainAvailable() {
        let _: (StreamLayout, MTLDevice, Int, ExpertCachePolicy) throws -> PreadExpertStreamer =
            PreadExpertStreamer.init(layout:device:slotCount:cachePolicy:)
    }
}
