import Foundation

@frozen
public struct PrefillMoEGroup: Equatable, Sendable {
    public var expert: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(expert: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.expert = expert
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

@frozen
public struct PrefillMoETile: Equatable, Sendable {
    public var groupStart: UInt32
    public var groupCount: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(groupStart: UInt32, groupCount: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.groupStart = groupStart
        self.groupCount = groupCount
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

public struct PrefillMoEGroupedRoutes: Equatable, Sendable {
    public let sortedPairs: [PrefillTokenExpertPair]
    public let groups: [PrefillMoEGroup]
    public let tiles: [PrefillMoETile]
}

enum PrefillMoEGroupingError: Error, Equatable, CustomStringConvertible {
    case expertOutOfRange(UInt32)

    public var description: String {
        switch self {
        case .expertOutOfRange(let value):
            return "route expert \(value) is out of range"
        }
    }
}

enum PrefillMoEGrouping {
    static func groupTokenExpertPairs(
        _ pairs: [PrefillTokenExpertPair],
        queryCount: Int,
        topK: Int,
        numExperts: Int,
        tileExpertCount: Int,
        expertSortKeys: [UInt64]? = nil
    ) throws -> PrefillMoEGroupedRoutes {
        precondition(queryCount >= 0 && topK > 0 && numExperts > 0 && (1...16).contains(tileExpertCount))
        precondition(pairs.count <= queryCount * topK)
        precondition(expertSortKeys.map { $0.count == numExperts } ?? true)
        // The experts come from the GPU router; everything else was built on
        // the host from them.
        if let bad = pairs.first(where: { $0.expert >= UInt32(numExperts) }) {
            throw PrefillMoEGroupingError.expertOutOfRange(bad.expert)
        }

        let sortedPairs = pairs.sorted {
            if let expertSortKeys {
                let lhsKey = expertSortKeys[Int($0.expert)]
                let rhsKey = expertSortKeys[Int($1.expert)]
                if lhsKey != rhsKey { return lhsKey < rhsKey }
            }
            if $0.expert != $1.expert { return $0.expert < $1.expert }
            if $0.token != $1.token { return $0.token < $1.token }
            return $0.rank < $1.rank
        }

        var groups: [PrefillMoEGroup] = []
        groups.reserveCapacity(min(numExperts, sortedPairs.count))

        var i = 0
        while i < sortedPairs.count {
            let expert = sortedPairs[i].expert
            let start = i
            while i < sortedPairs.count, sortedPairs[i].expert == expert {
                i += 1
            }
            groups.append(PrefillMoEGroup(expert: expert,
                                          pairStart: UInt32(start),
                                          pairCount: UInt32(i - start)))
        }

        var tiles: [PrefillMoETile] = []
        var groupStart = 0
        while groupStart < groups.count {
            let groupEnd = min(groups.count, groupStart + tileExpertCount)
            let first = groups[groupStart]
            let last = groups[groupEnd - 1]
            let pairStart = first.pairStart
            let pairEnd = last.pairStart + last.pairCount
            tiles.append(PrefillMoETile(groupStart: UInt32(groupStart),
                                        groupCount: UInt32(groupEnd - groupStart),
                                        pairStart: pairStart,
                                        pairCount: pairEnd - pairStart))
            groupStart = groupEnd
        }

        return PrefillMoEGroupedRoutes(sortedPairs: sortedPairs,
                                       groups: groups,
                                       tiles: tiles)
    }
}
