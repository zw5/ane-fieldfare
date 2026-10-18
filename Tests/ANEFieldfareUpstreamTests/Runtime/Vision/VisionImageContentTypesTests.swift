import Foundation
import Testing
import UniformTypeIdentifiers
@testable import ANEFieldfareHotpath

/// `ImageMetadataReader` accepts only the formats the pack can decode, a far
/// narrower set than `public.image`.
@Suite struct VisionImageContentTypesTests {
    @Test func formatsThePackCannotDecodeAreOutsideTheList() {
        let limits = VisionImageLimits()
        for excluded in [UTType("com.adobe.raw-image"),
                         UTType("public.camera-raw-image"),
                         UTType("com.ilm.openexr-image"),
                         UTType.webP,
                         UTType.bmp,
                         UTType.pdf] {
            guard let excluded else { continue }
            #expect(!limits.allowedTypeIdentifiers.contains(excluded.identifier),
                    "\(excluded.identifier) is admitted but cannot be decoded")
        }
        for expected in [UTType.jpeg, .png, .heic, .tiff, .gif] {
            #expect(limits.allowedTypeIdentifiers.contains(expected.identifier),
                    "\(expected.identifier) is decodable but not admitted")
        }
    }
}
