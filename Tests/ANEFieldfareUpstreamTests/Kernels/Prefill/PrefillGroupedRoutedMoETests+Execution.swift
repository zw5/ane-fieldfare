import Metal
import Testing
import ANEFieldfareValidationSupport

@testable import ANEFieldfareHotpath

extension PrefillGroupedRoutedMoETests {
  @Test func streamedBatchedMatchesReferenceAcrossPartialMicrobatch() throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        Self.pair(token: 0, expert: 2, rank: 0),
        Self.pair(token: 0, expert: 0, rank: 1),
        Self.pair(token: 1, expert: 1, rank: 0),
        Self.pair(token: 1, expert: 2, rank: 1),
        Self.pair(token: 2, expert: 0, rank: 0),
        Self.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 16, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 17) - 8) * 0.01)
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      queryCount: rows,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = unsafe ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 2 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = grouped.makeStreamedArgumentBuffer(binding: binding)
    try grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      actScratch: activationScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(maxAbsoluteError <= 0.0015, "maxAbsoluteError=\(maxAbsoluteError)")
    #expect(binding.views.allSatisfy { $0.offset > 0 })
  }

}


extension PrefillGroupedRoutedMoETests {
  /// Runs longer than a 32-pair tile, a second microbatch, Gemma's
  /// 704-wide intermediate and activations past FP16's range, against an
  /// FP32 reference.
  @Test func expertsMatchFP32ReferenceWithLargeActivations() throws {
    let d = 256, f = 704, rows = 40, topK = 2
    // Expert 0 takes every token's rank 0 (40 pairs: tiles of 32 and 8);
    // expert 1 rank 1 of the first 7, expert 2 rank 1 of the other 33.
    var pairs: [PrefillTokenExpertPair] = []
    for t in 0..<rows {
      pairs.append(Self.pair(token: UInt32(t), expert: 0, rank: 0))
      pairs.append(Self.pair(token: UInt32(t), expert: t < 7 ? 1 : 2, rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs, queryCount: rows, topK: topK, numExperts: 16, tileExpertCount: 16)

    // One expert: [gate W, S, B, up W, S, B, down W, S, B], tensors 16-aligned.
    struct Matrix { var q: [[UInt8]]; var s: [[UInt16]]; var b: [[UInt16]] }
    func matrix(rows n: Int, cols k: Int, seed: Int, scale: (Int) -> Float, bias: (Int) -> Float) -> Matrix {
      var m = Matrix(q: [], s: [], b: [])
      for r in 0..<n {
        m.q.append((0..<k).map { UInt8((r * 7 + $0 * 3 + seed * 5 + ($0 / 5)) % 16) })
        m.s.append((0..<(k / 64)).map { Quantization.bf16Bits(scale(r) * (1 + 0.1 * Float($0 % 3))) })
        m.b.append((0..<(k / 64)).map { _ in Quantization.bf16Bits(bias(r)) })
      }
      return m
    }
    func pack4(_ q: [UInt8]) -> [UInt8] {
      stride(from: 0, to: q.count, by: 2).map { q[$0] | (q[$0 + 1] << 4) }
    }
    func dequant(_ m: Matrix, _ r: Int, _ c: Int) -> Float {
      Float(m.q[r][c]) * Quantization.bf16ToFloat(m.s[r][c / 64]) + Quantization.bf16ToFloat(m.b[r][c / 64])
    }
    var experts: [[Matrix]] = []
    var bytesPerExpert: [[UInt8]] = []
    var o: [UInt32] = []
    for e in 0..<16 {
      // Every ninth gate and up row is large, so some activations pass 1e5.
      let big: (Int) -> Float = { $0 % 9 == 0 ? 0.08 : 0.002 }
      let gate = matrix(rows: f, cols: d, seed: e, scale: big, bias: { $0 % 9 == 0 ? 0.02 : -0.012 })
      let up = matrix(rows: f, cols: d, seed: e + 3, scale: big, bias: { $0 % 9 == 0 ? 0.03 : -0.013 })
      let down = matrix(rows: d, cols: f, seed: e + 6, scale: { _ in 5e-6 }, bias: { _ in -3.6e-5 })
      experts.append([gate, up, down])
      var bytes: [UInt8] = []
      var offsets: [UInt32] = []
      func align() { while bytes.count % 16 != 0 { bytes.append(0) } }
      for m in [gate, up, down] {
        align(); offsets.append(UInt32(bytes.count)); for r in m.q { bytes += pack4(r) }
        align(); offsets.append(UInt32(bytes.count)); for r in m.s { Self.appendU16(r, to: &bytes) }
        align(); offsets.append(UInt32(bytes.count)); for r in m.b { Self.appendU16(r, to: &bytes) }
      }
      align()
      o = offsets
      bytesPerExpert.append(bytes)
    }
    let pool = SyntheticExpertPool(
      bytes: bytesPerExpert.flatMap { $0 },
      offsets: MoEExpertOffsets(gateWOff: o[0], gateSOff: o[1], gateBOff: o[2],
                                upWOff: o[3], upSOff: o[4], upBOff: o[5],
                                downWOff: o[6], downSOff: o[7], downBOff: o[8]),
      stride: bytesPerExpert[0].count)
    let hidden = (0..<(rows * d)).map { i in Float16(1.5 + Float((i * 13) % 29) * 0.05 - Float(i % d) * 0.002) }

    var expected = [Float](repeating: 0, count: rows * topK * d)
    var peakActivation: Float = 0
    for pair in routes.sortedPairs {
      let m = experts[Int(pair.expert)]
      let x = (0..<d).map { Float(hidden[Int(pair.token) * d + $0]) }
      let act = (0..<f).map { r -> Float in
        var g: Float = 0, u: Float = 0
        for c in 0..<d { g += dequant(m[0], r, c) * x[c]; u += dequant(m[1], r, c) * x[c] }
        return MoeRef.geluTanh([g])[0] * u
      }
      peakActivation = max(peakActivation, act.map(abs).max() ?? 0)
      for r in 0..<d {
        var acc: Float = 0
        for c in 0..<f { acc += dequant(m[2], r, c) * act[c] }
        expected[(Int(pair.token) * topK + Int(pair.rank)) * d + r] = acc
      }
    }
    #expect(peakActivation > 100_000, "peakActivation=\(peakActivation)")

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    let microbatch = 64
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = unsafe ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 2 * microbatch * f * MemoryLayout<Float16>.stride, options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }
    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(device: ctx.device, pool: pool, expertIDs: expertIDs))
    try grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      actScratch: activationScratch,
      argumentBuffer: grouped.makeStreamedArgumentBuffer(binding: binding),
      binding: binding,
      params: PrefillGroupedRoutedMoEStreamedParams(
        pairStart: 0, pairCount: UInt32(routes.sortedPairs.count), d: UInt32(d),
        routedIntermediate: UInt32(f), topK: UInt32(topK), hiddenStrideElements: UInt32(d),
        binding: binding, offsets: pool.offsets),
      pairMicrobatchRows: microbatch)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d).map { Float($0) }
    let peak = expected.map(abs).max() ?? 1
    let maxError = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
    let rmsError = (zip(actual, expected).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Float(expected.count)).squareRoot()
    #expect(peak > 1, "peak=\(peak)")
    #expect(maxError <= 1.5e-3 * peak, "maxError=\(maxError) peak=\(peak)")
    #expect(rmsError <= 5e-4 * peak, "rmsError=\(rmsError) peak=\(peak)")
  }
}
