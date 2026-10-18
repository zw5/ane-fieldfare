import Foundation

public enum MultimodalContentPart: Sendable, Equatable {
    case text(String)
    case image(id: UUID)
}

public struct MultimodalMessage: Sendable, Equatable {
    public let role: GFTokenizer.Role
    public let content: [MultimodalContentPart]

    public init(role: GFTokenizer.Role, content: [MultimodalContentPart]) {
        self.role = role
        self.content = content
    }
}

public enum MultimodalPromptRendererError: Error, Equatable {
    case emptyMessages
    case emptyContent
    case reservedImageMarker
    case missingImage(UUID)
    case unexpectedImage(UUID)
    case placeholderMismatch
}

public enum MultimodalPromptRenderer {
    public static let placeholder = "<|image|>"
    public static let imageTokenID = Int32(258_880)
    public static let beginImageTokenID = Int32(255_999)
    public static let endImageTokenID = Int32(258_882)

    public static func render(
        messages: [MultimodalMessage],
        featuresByID: [UUID: VisionFeatures],
        tokenizer: GFTokenizer
    ) throws -> MultimodalPrefillInput {
        guard !messages.isEmpty else { throw MultimodalPromptRendererError.emptyMessages }
        var orderedImages: [(UUID, VisionFeatures)] = []
        let tokenizerMessages = try messages.map { message in
            guard !message.content.isEmpty else {
                throw MultimodalPromptRendererError.emptyContent
            }
            var text = ""
            for part in message.content {
                switch part {
                case .text(let value):
                    guard !value.contains(placeholder) else {
                        throw MultimodalPromptRendererError.reservedImageMarker
                    }
                    text += value
                case .image(let id):
                    guard let features = featuresByID[id] else {
                        throw MultimodalPromptRendererError.missingImage(id)
                    }
                    text += placeholder
                    orderedImages.append((id, features))
                }
            }
            return GFTokenizer.Message(role: message.role, content: text)
        }
        guard Set(orderedImages.map(\.0)) == Set(featuresByID.keys) else {
            let used = Set(orderedImages.map(\.0))
            throw MultimodalPromptRendererError.unexpectedImage(
                featuresByID.keys.first { !used.contains($0) }!)
        }

        let usesToolTemplate = tokenizerMessages.contains {
            $0.role == .developer || $0.role == .tool
        }
        let templateTokens: [Int32]
        if usesToolTemplate {
            templateTokens = try tokenizer.encodeToolChat(messages: tokenizerMessages)
        } else {
            let rendered = try tokenizer.applyChatTemplate(tokenizerMessages)
            templateTokens = tokenizer.encode(rendered, addBOS: false)
        }
        let placeholders = templateTokens.indices.filter {
            templateTokens[$0] == imageTokenID
        }
        guard placeholders.count == orderedImages.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var spans: [MultimodalImageSpan] = []
        effective.reserveCapacity(
            templateTokens.count + orderedImages.reduce(0) { $0 + $1.1.tokenCount + 1 })
        embedding.reserveCapacity(effective.capacity)
        var imageIndex = 0
        for token in templateTokens {
            guard token == imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let features = orderedImages[imageIndex].1
            effective.append(beginImageTokenID)
            embedding.append(beginImageTokenID)
            let lower = effective.count
            effective.append(contentsOf: repeatElement(imageTokenID, count: features.tokenCount))
            embedding.append(contentsOf: repeatElement(Int32(0), count: features.tokenCount))
            spans.append(MultimodalImageSpan(
                tokenRange: lower..<effective.count,
                features: features))
            effective.append(endImageTokenID)
            embedding.append(endImageTokenID)
            imageIndex += 1
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: spans)
    }
}
