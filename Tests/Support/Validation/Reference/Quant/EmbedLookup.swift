import Foundation
import ANEFieldfareHotpath

public enum EmbedLookupRef {
    /// FP32 reference for `embed_lookup_int4`.
    ///
    /// `tablePacked` carries `V * D/2` bytes (low nibble = even index inside a
    /// row, high nibble = odd index). Output is multiplied by `outScale` —
    /// pass `sqrt(hidden_size)` to match Gemma 4's post-embedding scale, or
    /// `1.0` for the raw dequant.
    public static func applyInt4(
        tablePacked: [UInt8],
        tableScales: [UInt16],
        tableBiases: [UInt16],
        tokenId: Int,
        d: Int,
        outScale: Float
    ) -> [Float] {
        precondition(d % Quantization.groupSize == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        let groupsPerRow = d / Quantization.groupSize
        let rowBytes  = d / 2
        let packBase  = tokenId * rowBytes
        let scaleBase = tokenId * groupsPerRow

        precondition(packBase + rowBytes <= tablePacked.count, "token out of range")
        precondition(scaleBase + groupsPerRow <= tableScales.count, "scales out of range")
        precondition(scaleBase + groupsPerRow <= tableBiases.count, "biases out of range")

        var out = [Float](repeating: 0, count: d)
        for i in 0..<d {
            let byte = tablePacked[packBase + (i >> 1)]
            let q    = (i & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
            let g    = i / Quantization.groupSize
            let scale = Quantization.bf16ToFloat(tableScales[scaleBase + g])
            let bias  = Quantization.bf16ToFloat(tableBiases[scaleBase + g])
            out[i] = (Float(q) * scale + bias) * outScale
        }
        return out
    }
}
