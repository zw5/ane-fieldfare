import Foundation
import Metal

// Guest contracts. A guest executes one placed operation for one layer and
// writes into buffers the Metal host owns. Guests never encode attention,
// routing, layer tails or logits; they only replace the routed-expert or
// shared-FFN arithmetic, and the host rail keeps a complete implementation of
// each so a faulted guest can be evicted without losing the request.

/// Routed top-8 prefill for one layer. On return every routed
/// `(token, rank)` row named by `pairs` has been written to `partials`. On
/// throw the guest has drained its own in-flight work, so the host may
/// overwrite `partials` immediately.
protocol RoutedPrefillGuest: AnyObject {
    func runLayer(model: Model, layer: Int, hidden: MTLBuffer, partials: MTLBuffer,
                  routes: PrefillMoEGroupedRoutes, pairs: MTLBuffer) async throws
    var profile: [String: Any] { get }
}

/// One layer's shared FFN on a guest rail. The lifecycle is
/// `encodeInput -> (producer completes) -> begin -> encodeOutput -> finish`.
/// `awaitOutcome` lets the supervisor learn a failure before the host
/// consumes the output; `abandon` releases a settled request whose output the
/// host will replace.
protocol SharedFFNGuestLayer: AnyObject {
    func encodeInput(_ cb: MTLCommandBuffer, source: MTLBuffer, rows: Int) throws
    func begin() throws
    func awaitOutcome() -> String?
    func encodeOutput(_ cb: MTLCommandBuffer, destination: MTLBuffer) throws
    func finish() throws
    func abandon()
}

protocol SharedFFNGuest: AnyObject {
    func layer(_ index: Int, prefill: Bool) -> SharedFFNGuestLayer?
    var profile: [String: Any] { get }
}

/// Routed top-8 decode for one token. Synchronous: on return `output` holds
/// the weighted expert sum for `input`.
protocol RoutedDecodeGuest: AnyObject {
    func call(layer: Int, experts: [Int], weights: [Float16],
              input: UnsafePointer<Float16>, output: UnsafeMutablePointer<Float16>) throws
    var profile: [String: Any] { get }
}

extension ANERoutedPrefillBackend: RoutedPrefillGuest {}

extension ANESharedFFNBackend: SharedFFNGuest {}

extension ANESharedFFNLayer: SharedFFNGuestLayer {}

extension AOTExpertBackend: RoutedDecodeGuest {
    var profile: [String: Any] {
        ["claim_boundary": "Compiler-authored named expert procedures; experimental decode rail."]
    }
}
