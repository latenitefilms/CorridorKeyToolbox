//
//  InferenceCoordinator.swift
//  CorridorKey by LateNite
//
//  The inference adapter for the analyser path. Every analyse-time call
//  goes through MLX — there is intentionally **no** fallback engine, so
//  the cached matte sequence is always 100% neural-network output. If
//  MLX isn't warm yet, `runInference` blocks (via the registry's
//  `waitForReady`) until the bridge is ready or the warm-up fails.
//  Engines live in `SharedMLXBridgeRegistry` so multiple plug-in
//  instances share a single warmed bridge per `(device, rung)`.
//
//  Note: the green-bias "rough matte" still exists, but as a *hint*
//  fed into MLX as the 4th input channel by the pre-inference pass
//  (`extractHint` / `combineAndNormaliseIntoBuffer`). It is never used
//  as a final output matte.
//
//  A single-frame MLX output cache is layered on top: if the user
//  changes a post-process parameter (output mode, despill, matte
//  sliders) without moving the playhead or swapping screen colour, the
//  coordinator returns the previous MLX result instead of paying for
//  another ~500 ms inference.
//

import Foundation
import Metal
import CoreMedia
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

/// Cache key for the single-frame MLX output cache. Anything that changes the
/// neural model's input must participate in the hash; anything purely
/// post-process (output mode, despill strength, etc.) must not.
struct InferenceCacheKey: Hashable, Sendable {
    let frameTimeValue: Int64
    let frameTimeTimescale: Int32
    let screenColorRaw: Int
    let inferenceResolution: Int
    let cacheEntryID: ObjectIdentifier

    init(
        frameTime: CMTime,
        screenColorRaw: Int,
        inferenceResolution: Int,
        cacheEntry: MetalDeviceCacheEntry
    ) {
        self.frameTimeValue = frameTime.value
        self.frameTimeTimescale = frameTime.timescale
        self.screenColorRaw = screenColorRaw
        self.inferenceResolution = inferenceResolution
        self.cacheEntryID = ObjectIdentifier(cacheEntry)
    }
}

/// Result bundle from `runInference`. The `engineDescription` records
/// **which engine actually processed this frame** (MLX or the rough-matte
/// fallback), not whichever engine happens to be ready when a log line
/// fires. Without this distinction the analyse log misreports rough-matte
/// frames as MLX, which masked the "first analysis is mostly rough-matte
/// because MLX is still warming up" diagnosis for ages.
struct InferenceRunResult {
    let output: KeyingInferenceOutput
    let engineDescription: String
}

final class InferenceCoordinator: @unchecked Sendable {

    /// Master switch for the MLX bridge. Always `true` in shipping
    /// builds — the analyser refuses to run without MLX. Kept as a
    /// constant so a future development build can flip it for debugging
    /// the pre-/post-inference pipeline without paying for the warm-up.
    private static let mlxEnabled = true

    private let stateLock = NSLock()

    /// The `(device, rung, screenColor)` this coordinator most recently
    /// asked the registry to warm. Used by `warmupStatus` and
    /// `cancelWarmup` to target the right key.
    private var trackedDeviceRegistryID: UInt64 = 0
    private var trackedRung: Int = 0
    private var trackedScreenColor: ScreenColor = .green

    /// Small LRU of recent MLX inference outputs. The single-entry
    /// cache covered the slider-tweak case at the same play-head, but
    /// scrubbing back over a window the user has already analysed used
    /// to miss every frame even though the work was a moment ago. With
    /// the LRU, short-range scrubbing inside the window stays free
    /// until the cache evicts. Capacity is small on purpose: each entry
    /// pins an alpha (`r32Float`) + foreground (`rgba16Float`) texture
    /// at the inference resolution, so 8 × 2048 ≈ 640 MB upper bound on
    /// the maxed-out tier — well within the unified-memory budget the
    /// bridge LRU already accounts for.
    private static let cacheCapacity = 8
    private var cacheEntriesByKey: [InferenceCacheKey: KeyingInferenceOutput] = [:]
    /// Insertion order for LRU eviction. Most-recently used appears at
    /// the end; the head is dropped when the cache is full.
    private var cacheLRUOrder: [InferenceCacheKey] = []

    /// Human-readable backend summary for diagnostics.
    var backendDescription: String {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        let color = trackedScreenColor
        stateLock.unlock()
        if rung > 0, let engine = SharedMLXBridgeRegistry.shared.readyEngine(
            deviceRegistryID: device,
            rung: rung,
            screenColor: color
        ) {
            return engine.backendDisplayName
        }
        return "Idle"
    }

    /// Current warm-up status. The inspector bridge consumes this to render
    /// the "Loading neural model…" badge during the first-play stall.
    var warmupStatus: WarmupStatus {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        let color = trackedScreenColor
        stateLock.unlock()
        guard rung > 0 else { return .cold }
        return SharedMLXBridgeRegistry.shared.status(
            deviceRegistryID: device,
            rung: rung,
            screenColor: color
        )
    }

    /// Cancels the in-flight MLX warm-up tracked by this coordinator.
    /// Safe when no warm-up is running — it's a no-op in that case.
    func cancelWarmup() {
        stateLock.lock()
        let device = trackedDeviceRegistryID
        let rung = trackedRung
        let color = trackedScreenColor
        stateLock.unlock()
        guard rung > 0 else { return }
        SharedMLXBridgeRegistry.shared.cancelWarmup(
            deviceRegistryID: device,
            rung: rung,
            screenColor: color
        )
    }

    /// Runs inference for a single frame. Uses MLX when the shared
    /// registry has a warm engine for the requested `(device, rung)`;
    /// otherwise synchronously produces a rough matte. Warm-up is
    /// requested in the background so subsequent frames can use MLX.
    ///
    /// Passes `cacheKey` so we can short-circuit repeated inferences at
    /// the same play-head when only post-process parameters have
    /// changed.
    func runInference(
        request: KeyingInferenceRequest,
        screenColor: ScreenColor,
        cacheEntry: MetalDeviceCacheEntry,
        cacheKey: InferenceCacheKey
    ) throws -> InferenceRunResult {
        if let cached = cachedOutput(for: cacheKey) {
            return InferenceRunResult(output: cached, engineDescription: "MLX cache hit")
        }

        guard Self.mlxEnabled else {
            throw KeyingInferenceError.modelUnavailable("MLX is disabled in this build.")
        }

        let deviceRegistryID = cacheEntry.device.registryID
        trackRequestedBridge(
            deviceRegistryID: deviceRegistryID,
            rung: request.inferenceResolution,
            screenColor: screenColor
        )

        // Block until the MLX bridge is warm. Earlier builds fell
        // through to a green-bias rough-matte engine when MLX wasn't
        // ready yet, which silently degraded the first ~15 s of every
        // analysis to a shape-only matte. We now wait — the green
        // hint is still used inside MLX (as the 4th input channel via
        // `extractHint` in pre-inference), but the *output* matte is
        // always the neural one.
        let engine: MLXKeyingEngine
        do {
            engine = try SharedMLXBridgeRegistry.shared.waitForReady(
                deviceRegistryID: deviceRegistryID,
                rung: request.inferenceResolution,
                screenColor: screenColor,
                cacheEntry: cacheEntry
            )
        } catch {
            PluginLog.error("MLX bridge unavailable: \(error.localizedDescription)")
            throw error
        }

        let output = try makeOutputTextures(request: request, cacheEntry: cacheEntry)
        try engine.run(request: request, output: output)
        storeCachedOutput(output, for: cacheKey)
        return InferenceRunResult(output: output, engineDescription: engine.backendDisplayName)
    }

    /// Asks the shared registry to start warm-up for `(device, rung,
    /// screenColor)` without waiting. Used by
    /// `CorridorKeyToolboxPlugIn.init` to pre-warm the default bridge the
    /// moment the effect is applied.
    func requestEagerWarmup(rung: Int, screenColor: ScreenColor, cacheEntry: MetalDeviceCacheEntry) {
        guard Self.mlxEnabled else { return }
        let deviceRegistryID = cacheEntry.device.registryID
        trackRequestedBridge(deviceRegistryID: deviceRegistryID, rung: rung, screenColor: screenColor)
        SharedMLXBridgeRegistry.shared.beginWarmup(
            deviceRegistryID: deviceRegistryID,
            rung: rung,
            screenColor: screenColor,
            cacheEntry: cacheEntry
        )
    }

    /// Releases the cached MLX output and asks MLX to drain its internal
    /// Metal buffer cache. Called by the analyser at the end of an
    /// Analyse Clip pass so memory doesn't ramp across long editing
    /// sessions. The MLX cache is documented to grow unboundedly across
    /// successive inferences with different shapes — clearing it once
    /// per session strikes the right balance: zero per-frame thrashing,
    /// memory back to baseline once the user is done analysing.
    func releaseCacheBetweenSessions() {
        stateLock.lock()
        cacheEntriesByKey.removeAll()
        cacheLRUOrder.removeAll()
        let deviceRegistryID = trackedDeviceRegistryID
        let rung = trackedRung
        let color = trackedScreenColor
        stateLock.unlock()

        if rung > 0,
           let mlx = SharedMLXBridgeRegistry.shared.readyEngine(
               deviceRegistryID: deviceRegistryID,
               rung: rung,
               screenColor: color
           ) {
            mlx.clearMLXCache()
        }
    }

    // MARK: - MLX frame cache

    private func cachedOutput(for key: InferenceCacheKey) -> KeyingInferenceOutput? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard let cached = cacheEntriesByKey[key] else { return nil }
        // Bump recency on hit so an entry the user keeps revisiting
        // doesn't get evicted by a one-off scrub elsewhere.
        if let index = cacheLRUOrder.firstIndex(of: key) {
            cacheLRUOrder.remove(at: index)
        }
        cacheLRUOrder.append(key)
        return cached
    }

    private func storeCachedOutput(_ output: KeyingInferenceOutput, for key: InferenceCacheKey) {
        stateLock.lock()
        if let index = cacheLRUOrder.firstIndex(of: key) {
            cacheLRUOrder.remove(at: index)
        }
        cacheEntriesByKey[key] = output
        cacheLRUOrder.append(key)
        while cacheLRUOrder.count > Self.cacheCapacity {
            let evicted = cacheLRUOrder.removeFirst()
            cacheEntriesByKey.removeValue(forKey: evicted)
        }
        stateLock.unlock()
    }

    // MARK: - Tracked bridge target

    private func trackRequestedBridge(deviceRegistryID: UInt64, rung: Int, screenColor: ScreenColor) {
        stateLock.lock()
        if trackedDeviceRegistryID != deviceRegistryID
            || trackedRung != rung
            || trackedScreenColor != screenColor {
            // Switching device, rung, or screen colour invalidates the
            // entire cache — entries are pinned to the prior tracked
            // bridge and the textures are sized for that rung.
            cacheEntriesByKey.removeAll()
            cacheLRUOrder.removeAll()
        }
        trackedDeviceRegistryID = deviceRegistryID
        trackedRung = rung
        trackedScreenColor = screenColor
        stateLock.unlock()
    }

    // MARK: - Output textures

    private func makeOutputTextures(
        request: KeyingInferenceRequest,
        cacheEntry: MetalDeviceCacheEntry
    ) throws -> KeyingInferenceOutput {
        guard let alpha = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        // Foreground rides through the compose pass as `texture2d<float>`,
        // which auto-converts the underlying half-precision storage to
        // float for the shader. fp16 here halves bandwidth on the
        // writeback + compose without any precision impact (foreground
        // RGB is sigmoid output in [0,1], well within fp16 range).
        guard let foreground = cacheEntry.makeIntermediateTexture(
            width: request.inferenceResolution,
            height: request.inferenceResolution,
            pixelFormat: .rgba16Float,
            storageMode: .shared
        ) else {
            throw KeyingInferenceError.deviceUnavailable
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }
}
