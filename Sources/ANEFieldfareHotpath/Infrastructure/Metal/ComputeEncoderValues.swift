public import Metal

extension MTLComputeCommandEncoder {
    /// Binds `value` as constant data at `index`. The length is the type's own
    /// stride, so it can never disagree with the value it describes.
    public func setValue<Value: BitwiseCopyable>(_ value: Value, index: Int) {
        // SAFETY: a one-element temporary of exactly `Value`'s stride, alive
        // for the call; `setBytes` copies the bytes before returning.
        withUnsafeTemporaryAllocation(of: Value.self, capacity: 1) { storage in
            guard let base = storage.baseAddress else { return }
            unsafe base.initialize(to: value)
            unsafe setBytes(base, length: MemoryLayout<Value>.stride, index: index)
        }
    }

    /// Binds `values` as one constant array at `index`.
    public func setValues<Value: BitwiseCopyable>(_ values: [Value], index: Int) {
        // SAFETY: the array's contiguous storage is alive for the call, and
        // `setBytes` copies it before returning.
        values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            unsafe setBytes(base, length: bytes.count, index: index)
        }
    }
}
