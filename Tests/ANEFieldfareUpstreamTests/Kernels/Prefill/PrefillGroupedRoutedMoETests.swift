import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct PrefillGroupedRoutedMoETests {
    static func pair(token: UInt32, expert: UInt32, rank: UInt32) -> PrefillTokenExpertPair {
        PrefillTokenExpertPair(token: token,
                               expert: expert,
                               rank: rank,
                               weight: Float16(0.125 + Float(rank) * 0.0625))
    }

    static func fakeTensorViews(device: any MTLDevice, count: Int) throws -> [TensorView] {
        guard let buffer = device.makeBuffer(length: max(count, 1) * 64,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("fake tensor view buffer")
        }
        return (0..<count).map { index in
            TensorView(buffer: buffer,
                       offset: UInt64(index * 64),
                       length: 64,
                       scaleOffset: 0,
                       scaleLength: 0,
                       biasOffset: 0,
                       biasLength: 0,
                       shape: (0, UInt32(index), 0, 0),
                       dtype: 0)
        }
    }

    static func streamedViewsWithNonzeroOffsets(device: any MTLDevice,
                                                        pool: SyntheticExpertPool,
                                                        expertIDs: [Int]) throws -> [TensorView] {
        try expertIDs.enumerated().map { index, expertID in
            let start = expertID * pool.stride
            let end = start + pool.stride
            let prefix = 64 + index * 16
            let suffix = 32
            var bytes = [UInt8](repeating: 0xA5, count: prefix)
            bytes.append(contentsOf: pool.bytes[start..<end])
            bytes.append(contentsOf: repeatElement(UInt8(0x5A), count: suffix))
            guard let buffer = unsafe device.makeBuffer(bytes: bytes,
                                                 length: bytes.count,
                                                 options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("streamed expert \(expertID)")
            }
            return TensorView(buffer: buffer,
                              offset: UInt64(prefix),
                              length: UInt64(pool.stride),
                              scaleOffset: 0,
                              scaleLength: 0,
                              biasOffset: 0,
                              biasLength: 0,
                              shape: (0, UInt32(expertID), 0, 0),
                              dtype: 0)
        }
    }

    struct SyntheticExpertPool {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
        let stride: Int
    }

    static func makeSyntheticExpertPool(numExperts: Int, d: Int, f: Int) -> SyntheticExpertPool {
        var allBytes: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for expert in 0..<numExperts {
            var bytes: [UInt8] = []
            // Packed weights, scales, then biases for gate, up and down.
            var o: [UInt32] = []
            for (role, shape) in [(rows: f, cols: d), (rows: f, cols: d), (rows: d, cols: f)].enumerated() {
                let rows = Self.syntheticRows(rows: shape.rows, cols: shape.cols, expert: expert, role: role)
                    .map { Quantization.quantizeInt4Affine($0) }
                o.append(UInt32(bytes.count))
                for row in rows { bytes.append(contentsOf: row.packed) }
                o.append(UInt32(bytes.count))
                for row in rows { Self.appendU16(row.scales, to: &bytes) }
                o.append(UInt32(bytes.count))
                for row in rows { Self.appendU16(row.biases, to: &bytes) }
            }
            let currentOffsets = MoEExpertOffsets(gateWOff: o[0], gateSOff: o[1], gateBOff: o[2],
                                                  upWOff: o[3], upSOff: o[4], upBOff: o[5],
                                                  downWOff: o[6], downSOff: o[7], downBOff: o[8])
            if offsets == nil {
                offsets = currentOffsets
                stride = bytes.count
            } else {
                #expect(stride == bytes.count)
                #expect(offsets!.gateWOff == currentOffsets.gateWOff)
                #expect(offsets!.downBOff == currentOffsets.downBOff)
            }
            allBytes.append(contentsOf: bytes)
        }
        return SyntheticExpertPool(bytes: allBytes, offsets: offsets!, stride: stride)
    }

    static func syntheticRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
        (0..<rows).map { row in
            (0..<cols).map { col in
                Float(expert + 1) * 0.001
                    + Float(role + 1) * 0.003
                    + Float((row % 7) - 3) * 0.0004
                    + Float((col % 11) - 5) * 0.0002
            }
        }
    }

    static func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    static func readU16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    static func cpuSyntheticRoutePartials(routes: PrefillMoEGroupedRoutes,
                                                  queryCount: Int,
                                                  hidden: [Float16],
                                                  hiddenStride: Int,
                                                  pool: SyntheticExpertPool,
                                                  topK: Int,
                                                  d: Int,
                                                  f: Int) -> [Float16] {
        var out = [Float16](repeating: -99, count: queryCount * topK * d)
        let o = pool.offsets
        for pair in routes.sortedPairs {
            let expertBase = Int(pair.expert) * pool.stride
            func dot(_ w: UInt32, _ s: UInt32, _ b: UInt32, row: Int, n: Int, x: [Float]) -> Float {
                Self.cpuInt4Dot(bytes: pool.bytes, base: expertBase,
                                wOff: Int(w), sOff: Int(s), bOff: Int(b), row: row, n: n, x: x)
            }
            let xBase = Int(pair.token) * hiddenStride
            let x = (0..<d).map { Float(hidden[xBase + $0]) }
            let act = (0..<f).map { row -> Float in
                let gate = dot(o.gateWOff, o.gateSOff, o.gateBOff, row: row, n: d, x: x)
                let up = dot(o.upWOff, o.upSOff, o.upBOff, row: row, n: d, x: x)
                return Float(Float16(MoeRef.geluTanh([gate])[0] * up))
            }
            let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
            for row in 0..<d {
                out[outBase + row] = Float16(dot(o.downWOff, o.downSOff, o.downBOff, row: row, n: f, x: act))
            }
        }
        return out
    }

    static func cpuInt4Dot(bytes: [UInt8],
                                   base: Int,
                                   wOff: Int,
                                   sOff: Int,
                                   bOff: Int,
                                   row: Int,
                                   n: Int,
                                   x: [Float]) -> Float {
        let groups = n / Quantization.groupSize
        let rowBytes = n / 2
        let wRow = base + wOff + row * rowBytes
        let sRow = base + sOff + row * groups * MemoryLayout<UInt16>.stride
        let bRow = base + bOff + row * groups * MemoryLayout<UInt16>.stride
        var acc: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(Self.readU16(bytes, sRow + group * 2))
            let bias = Quantization.bf16ToFloat(Self.readU16(bytes, bRow + group * 2))
            for k in 0..<Quantization.groupSize {
                let col = group * Quantization.groupSize + k
                let packed = bytes[wRow + col / 2]
                let q = (k & 1) == 0 ? Float(packed & 0x0F) : Float(packed >> 4)
                acc += (q * scale + bias) * x[col]
            }
        }
        return acc
    }
}
