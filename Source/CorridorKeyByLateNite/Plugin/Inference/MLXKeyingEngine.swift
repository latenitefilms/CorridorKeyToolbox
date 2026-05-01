//
//  MLXKeyingEngine.swift
//  CorridorKey by LateNite
//
//  Loads a CorridorKey `.mlxfn` bridge via mlx-swift's public
//  `ImportedFunction` API and runs inference for one frame at a time. The
//  bridge file is a pre-compiled MLX graph produced upstream by the
//  CorridorKey training pipeline, so this engine is a thin adapter between
//  the render pipeline's Metal textures and MLX's tensor API.
//
//  Apple Silicon's unified memory means the Metal↔MLX hand-off is mostly a
//  pointer copy; the GPU work itself is scheduled by MLX against the Neural
//  Engine or the GPU as appropriate for the compiled graph.
//

import Foundation
import Metal
import MLX
import simd
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

/// Precision the `.mlxfn` bridge expects on its inputs/outputs. Bridges
/// exported with `Source/scripts/export_mlxfn.py --fp16` use
/// `.float16`; legacy bridges produced before that script existed use
/// `.float32`. The engine probes the bundle for the fp16 variant first
/// (its filename has an `_fp16` suffix) and falls back to fp32 when the
/// fp16 file isn't available, so neither developer machines without the
/// re-exported pack nor older test fixtures break.
enum BridgePrecision: Sendable {
    case float32
    case float16

    /// Bytes per scalar in the i/o tensors. Used by the Metal-side
    /// buffer allocator to size the input ring.
    var elementBytes: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        }
    }

    /// Human-readable label for logs.
    var displayName: String {
        switch self {
        case .float32: return "fp32"
        case .float16: return "fp16"
        }
    }
}

/// Names of the bundled `.mlxfn` artefacts. Matches CorridorKey-Runtime's
/// `{corridorkey,corridorkeyblue}_mlx_bridge_{N}.mlxfn` convention so the
/// same Hugging Face / `Source/scripts/export_mlxfn.py` outputs can be
/// consumed unmodified — the only difference between the two colour
/// packs is the trained weights, the architecture is identical.
///
/// **Tiled inference** (deferred — needs upstream model export):
/// The reference `corridorkey-mlx` Python implementation supports
/// 512 px tiles with 64 px overlap so users on a 32 GB Mac can run
/// effective resolutions up to 4K without exceeding RAM. To enable
/// the same path here we'd need:
///
/// 1. An upstream `.mlxfn` exported at a "tile" shape (e.g. fixed
///    512 px with shape-flexibility on the batch axis) — the current
///    bundles are exported per-rung at fixed shape because Hiera's
///    shape-dependent reshapes don't compile shape-flexibly.
/// 2. A pre/post tiling helper in `MLXKeyingEngine.run` that
///    chunks the input, runs N inferences, and Lanczos-blends the
///    overlap. Without the upstream model this can't be tested
///    end-to-end so it's left as a TODO.
///
/// Once the upstream model lands, the change is local to
/// `MLXKeyingEngine`; nothing in the renderer assumes single-tile
/// inference.
enum MLXBridgeArtifact {

    /// Supported bridge resolutions, in preference order from lowest to
    /// highest. `closestSupportedResolution` walks this list. Rungs ≤ 2048
    /// run a full Hiera encoder at the rung size; the 4096 rung ships
    /// only as an `_fp16.mlxfn` and is a *hybrid* bridge — input arrives
    /// at 4096, the bridge downsamples internally to 1024 for the Hiera
    /// encoder/decoder, then upsamples logits back to 4096 and runs the
    /// CNN refiner at full target size. Mirrors the closed-source
    /// approach for fast 4K keying without retraining: encoder grid drops
    /// from 4 M tokens to 65 k tokens with no learned-weight changes.
    static let ladder: [Int] = [512, 768, 1024, 1536, 2048, 4096]

    static func filename(forResolution resolution: Int, screenColor: ScreenColor) -> String {
        filename(forResolution: resolution, screenColor: screenColor, precision: .float32)
    }

    /// Precision-aware filename. The fp16 variants append `_fp16` to
    /// the rung so an installed bundle can ship both alongside each
    /// other and `BridgeVariantResolver` can pick the right one.
    static func filename(forResolution resolution: Int, screenColor: ScreenColor, precision: BridgePrecision) -> String {
        let suffix = precision == .float16 ? "_fp16" : ""
        return "\(screenColor.bridgeFilenamePrefix)_\(resolution)\(suffix).mlxfn"
    }

    /// Returns the ladder rung that is at least as large as `requested`,
    /// falling back to the maximum if nothing larger exists.
    static func closestSupportedResolution(forRequested requested: Int) -> Int? {
        ladder.first(where: { $0 >= requested }) ?? ladder.last
    }
}

/// Resolves the best available bridge file in the bundle for a given
/// `(rung, screenColor)`. Prefers the fp16 variant when present, falls
/// back to fp32. Centralised so call sites never have to re-implement
/// the prefer-fp16 policy.
enum BridgeVariantResolver {
    static func resolve(rung: Int, screenColor: ScreenColor) -> (URL, BridgePrecision)? {
        let halfFilename = MLXBridgeArtifact.filename(
            forResolution: rung, screenColor: screenColor, precision: .float16
        )
        if let url = MLXBridgeResourceLocator.url(for: halfFilename) {
            return (url, .float16)
        }
        let floatFilename = MLXBridgeArtifact.filename(
            forResolution: rung, screenColor: screenColor, precision: .float32
        )
        if let url = MLXBridgeResourceLocator.url(for: floatFilename) {
            return (url, .float32)
        }
        return nil
    }
}

/// Lazy bundled-resource lookup that works from any of the three
/// processes that may host the engine: the FxPlug XPC service (the
/// .mlxfn files sit in its own bundle), the standalone wrapper app
/// (which reaches into the embedded renderer plugin so we don't have
/// to duplicate ~600 MB of model artefacts in two places), and the
/// SPM test runner (which loads the smallest bridge from its test
/// resources).
private enum MLXBridgeResourceLocator {
    static func url(for filename: String) -> URL? {
        let fileManager = FileManager.default
        let filenameStem = (filename as NSString).deletingPathExtension
        let filenameExtension = (filename as NSString).pathExtension

        // 1. The simplest case — the .mlxfn lives inside whatever
        //    bundle this process's main executable was loaded from.
        //    True for the FxPlug XPC service and for SPM test runners
        //    that bundle the test resource directly.
        if let url = Bundle.main.url(forResource: filenameStem, withExtension: filenameExtension) {
            return url
        }

        // 2. The wrapper app: the renderer plugin sits inside the
        //    .app at Contents/PlugIns/CorridorKey by LateNite Renderer.pluginkit
        //    and owns the .mlxfn resources. Looking them up here lets
        //    the standalone editor reuse the same on-disk artefacts
        //    the renderer ships, instead of bundling a second copy.
        if let pluginsURL = Bundle.main.builtInPlugInsURL {
            do {
                let plugins = try fileManager.contentsOfDirectory(
                    at: pluginsURL,
                    includingPropertiesForKeys: nil
                )
                for plugin in plugins {
                    let candidate = plugin
                        .appending(path: "Contents/Resources/\(filename)")
                    if fileManager.fileExists(atPath: candidate.path) {
                        return candidate
                    }
                }
            } catch {
                // Plugins folder absent or unreadable — fall through to
                // the bundle walk below; nothing to surface to the user.
            }
        }

        // 3. Walk every loaded bundle in case a host loaded ours
        //    indirectly (dev builds, unit tests, etc).
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: filenameStem, withExtension: filenameExtension) {
                return url
            }
            if let resourceURL = bundle.resourceURL {
                let candidate = resourceURL.appending(path: filename)
                if fileManager.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }
}

final class MLXKeyingEngine: KeyingInferenceEngine, @unchecked Sendable {

    /// Strategy for handing the normalised input tensor to MLX. Switched
    /// at engine init by the production code; the test suite can override
    /// per-instance via `testOverrideInputStrategy(_:)` to cross-check
    /// parity between the two paths.
    ///
    /// * `.zeroCopy` — aliases the shared MTLBuffer directly via
    ///   `MLXArray(rawPointer:)`. Apple Silicon's unified memory makes
    ///   this near-free — no copy, MLX reads the same bytes the Metal
    ///   normalise kernel just wrote. The shipping default.
    /// * `.cpuStaging` — copies the buffer into a reusable Swift
    ///   `[Float]` and constructs an `MLXArray` that owns its memory.
    ///   Was tried as a workaround for a slow-eval symptom on one clip
    ///   but proved unreliable in production (matte quality regressions).
    ///   Kept available so unit tests can validate parity with the
    ///   shipping path.
    enum InputStrategy: Sendable {
        case zeroCopy
        case cpuStaging
    }
    private static let defaultInputStrategy: InputStrategy = .zeroCopy
    private var inputStrategy: InputStrategy = MLXKeyingEngine.defaultInputStrategy

    /// Test-only entry point. Production callers must not flip the
    /// strategy mid-flight — the override exists so parity tests can
    /// run both paths against the same input within a single process.
    func testOverrideInputStrategy(_ strategy: InputStrategy) {
        runLock.lock()
        inputStrategy = strategy
        runLock.unlock()
    }

    let backendDisplayName: String
    var guideSourceDescription: String

    private let cacheEntry: MetalDeviceCacheEntry
    /// Guards `importedFunction` / `loadedResolution` / `loadedScreenColor`
    /// reads and writes. Held briefly so warm-up and per-frame renders never
    /// deadlock each other.
    private let stateLock = NSLock()
    /// Serialises the entire `run(...)` path. FxPlug calls us from multiple
    /// render threads concurrently and `ImportedFunction` is not documented as
    /// thread-safe — funnel everything through a single in-flight inference.
    private let runLock = NSLock()
    private var importedFunction: ImportedFunction?
    private var loadedResolution: Int = 0
    /// Screen colour the currently-loaded bridge was exported for. Tracked
    /// alongside `loadedResolution` so `alreadyLoaded` rejects a request
    /// that asks for the same rung at a different colour and forces a
    /// fresh `ImportedFunction` for the new bridge file.
    private var loadedScreenColor: ScreenColor = .green
    /// Precision of the currently-loaded bridge. Determines which Metal
    /// kernel pair the writeback path uses and what dtype the engine
    /// constructs the input MLXArray with.
    private var loadedPrecision: BridgePrecision = .float32

    /// Reusable scratch buffer for the `.cpuStaging` strategy. Sized at
    /// warm-up so per-frame inference doesn't pay the 67 MB allocation
    /// cost on every call. `runLock` serialises access; no extra guard.
    private var inputScratch: [Float] = []

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "MLX on \(cacheEntry.device.name)"
        self.guideSourceDescription = "Auto rough fallback"
    }

    /// Releases MLX's internal buffer cache. Called by the analyser at the
    /// end of an Analyse Clip pass so memory doesn't ramp across long
    /// editing sessions. Cheap during normal inference (we don't call it
    /// per-frame because forcing reallocation between frames thrashes the
    /// allocator and made wall-time go from 41 ms to 15 s in field tests).
    ///
    /// The unit-test `MLXMemoryTests` verifies the cache settles after
    /// `clearCache` runs — without this hook, MLX cache observed at 4.4 GB
    /// after 30× 512 inferences scales to ~70 GB at 2048, matching the
    /// 42 GB symptom Final Cut Pro hit during a 26-frame Analyse pass.
    func clearMLXCache() {
        let beforeBytes = MLX.Memory.cacheMemory
        MLX.Memory.clearCache()
        let afterBytes = MLX.Memory.cacheMemory
        let releasedMB = (beforeBytes - afterBytes) / (1024 * 1024)
        if releasedMB > 0 {
            PluginLog.notice("MLX cache cleared: released \(releasedMB) MB.")
        }
    }

    /// Blocks the calling thread until MLX's GPU stream has retired
    /// every operation it had outstanding. The Standalone Editor's
    /// quit path calls this so the global compute pipeline state
    /// objects can't be released while MLX still has inference work
    /// queued on its private command queue. Without it, Quit during
    /// analysis trips Metal's API Validation
    /// `notifyExternalReferencesNonZeroOnDealloc` assertion in Debug
    /// builds (the only place Validation is enabled by default);
    /// release builds wouldn't crash but would still race the same
    /// release/deinit, so we guard both.
    static func synchronizeMLXGPUStream() {
        Stream.gpu.synchronize()
    }

    func supports(resolution: Int, screenColor: ScreenColor) -> Bool {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution) else {
            return false
        }
        return BridgeVariantResolver.resolve(rung: rung, screenColor: screenColor) != nil
    }

    /// Precision the currently-loaded bridge expects on its input/output
    /// tensors. `.float32` until a successful `prepare` call lands. The
    /// pre-inference pipeline asks for this so it can size the input
    /// buffer and pick the right normalise kernel.
    func currentPrecision() -> BridgePrecision {
        stateLock.lock()
        defer { stateLock.unlock() }
        return loadedPrecision
    }

    func prepare(resolution: Int, screenColor: ScreenColor) async throws {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution) else {
            throw KeyingInferenceError.modelUnavailable(
                "No MLX bridge file bundled for \(resolution)px."
            )
        }
        guard let (bridgeURL, precision) = BridgeVariantResolver.resolve(
            rung: rung, screenColor: screenColor
        ) else {
            throw KeyingInferenceError.modelUnavailable(
                "No \(screenColor.displayName) MLX bridge bundled for \(resolution)px."
            )
        }
        try await prepare(bridgeURL: bridgeURL, rung: rung, screenColor: screenColor, precision: precision)
    }

    /// Test entry point that loads the bridge from an explicit URL. Useful
    /// from SPM unit tests where the `.mlxfn` lives in the test target's
    /// resources bundle (which is not enumerated by `Bundle.allBundles`).
    /// Production callers should use `prepare(resolution:screenColor:)`.
    /// `precision` defaults to `.float32` for backward compatibility with
    /// tests that hand a legacy fp32 bridge directly; the production
    /// `prepare(resolution:screenColor:)` path resolves the precision from
    /// the bundle and forwards it through.
    func prepare(
        bridgeURL: URL,
        rung: Int,
        screenColor: ScreenColor,
        precision: BridgePrecision = .float32
    ) async throws {
        if alreadyLoaded(rung: rung, screenColor: screenColor, precision: precision) { return }

        PluginLog.notice(
            "Loading MLX bridge from \(bridgeURL.path) (\(screenColor.displayName) screen, \(precision.displayName))."
        )
        let function: ImportedFunction
        do {
            function = try ImportedFunction(url: bridgeURL)
        } catch {
            throw KeyingInferenceError.modelUnavailable(
                "MLX could not load \(bridgeURL.lastPathComponent): \(error.localizedDescription)"
            )
        }

        // Drive one zero-filled inference to trigger MLX's JIT compilation and
        // allocate the Metal buffer pool. Finishing this before we advertise
        // the engine as loaded means the first real render frame never pays a
        // multi-second stall while MLX compiles on demand.
        await warmJIT(function: function, rung: rung, precision: precision)
        storeFunction(function, rung: rung, screenColor: screenColor, precision: precision)
        prewarmVisionHint()
    }

    /// Issues a tiny synthetic Vision request so the foreground-instance
    /// model is loaded onto the Neural Engine before the user's first
    /// analyse frame asks for it. The first `perform` of a session is
    /// the slow one (~50–100 ms cold); priming here hides it. Runs on
    /// the same warm-up task so callers don't see this cost either.
    private func prewarmVisionHint() {
        guard #available(macOS 14.0, *) else { return }
        guard let engine = cacheEntry.visionHintEngine() as? VisionHintEngine else { return }
        engine.prewarm()
    }

    /// Runs a throwaway inference on a zero tensor so MLX compiles the graph
    /// and warms the Metal buffer cache. Any failure here is non-fatal — the
    /// real inference will surface the same error to the caller later.
    private func warmJIT(function: ImportedFunction, rung: Int, precision: BridgePrecision) async {
        let warmupStart = Date()
        let dtype: DType = precision == .float16 ? .float16 : .float32
        let input = MLXArray.zeros([1, rung, rung, 4], dtype: dtype)
        do {
            let outputs = try withError {
                try function(input)
            }
            try checkedEval(outputs)
        } catch {
            PluginLog.error("MLX JIT warm-up failed (non-fatal): \(error.localizedDescription)")
            return
        }
        let elapsedSeconds = Date().timeIntervalSince(warmupStart)
        PluginLog.notice(
            "MLX JIT warm-up finished in \(String(format: "%.2f", elapsedSeconds))s for \(rung)px (\(precision.displayName))."
        )
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        runLock.lock()
        defer { runLock.unlock() }
        // Per-call autoreleasepool drops the local MLXArrays + their
        // captured retain-boxes the moment we exit this method.
        // Without it, the autoreleased Obj-C wrappers MLX returns
        // hang around in the calling thread's outer pool — Final Cut
        // Pro's analyse loop never drains that pool, which is what
        // turned the cache into a 40+ GB sink in production.
        let runStart = ContinuousClock.now
        try autoreleasepool {
            try runBody(request: request, output: output)
        }
        let elapsed = ContinuousClock.now - runStart
        let elapsedSeconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        // Feed the device capability cache so subsequent `automatic`
        // ceiling decisions reflect real measurements, not the static
        // RAM-tier heuristic.
        let (_, rung, _) = loadedState()
        if rung > 0 {
            DeviceCapabilityCache.shared.record(
                deviceRegistryID: cacheEntry.device.registryID,
                rung: rung,
                milliseconds: elapsedSeconds * 1000
            )
        }
    }

    private func runBody(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        let (function, rung, precision) = loadedState()
        guard let function, rung > 0 else {
            throw KeyingInferenceError.modelUnavailable("MLX bridge not prepared.")
        }
        _ = request.rawSourceTexture
        let expectedBytes = rung * rung * 4 * precision.elementBytes
        guard request.normalisedInputBuffer.length >= expectedBytes else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX input buffer is \(request.normalisedInputBuffer.length) bytes; expected ≥ \(expectedBytes)."
            )
        }

        // Step 1: build the MLX input. Two strategies live here; see
        // `InputStrategy` for the trade-off and the wall-time data.
        let inputBuffer = request.normalisedInputBuffer
        let inputArray: MLXArray
        let mlxDType: DType = precision == .float16 ? .float16 : .float32
        switch inputStrategy {
        case .zeroCopy:
            inputArray = MLXArray(
                rawPointer: inputBuffer.contents(),
                [1, rung, rung, 4],
                dtype: mlxDType,
                finalizer: { _ = inputBuffer }
            )
        case .cpuStaging:
            // CPU staging is fp32-only: it goes through a Swift `[Float]`
            // scratch. fp16 bridges require the zero-copy path because we
            // don't ship a Swift `Float16` scratch and the strategy is
            // only kept for parity testing with the legacy fp32 bridges.
            guard precision == .float32 else {
                throw KeyingInferenceError.modelUnavailable(
                    "cpuStaging input strategy is only supported for fp32 bridges; current bridge is \(precision.displayName)."
                )
            }
            let expectedCount = rung * rung * 4
            if inputScratch.count != expectedCount {
                inputScratch = [Float](repeating: 0, count: expectedCount)
            }
            // Copy 67 MB once into the Swift scratch. On Apple Silicon's
            // unified memory both source and destination are CPU-visible,
            // so this is just a memcpy with no GPU sync. Lets MLX own the
            // input layout end-to-end, which empirically keeps MLX
            // evaluation on the fast path.
            inputScratch.withUnsafeMutableBufferPointer { destination in
                guard let destinationBase = destination.baseAddress else { return }
                memcpy(destinationBase, inputBuffer.contents(), expectedCount * MemoryLayout<Float>.size)
            }
            inputArray = MLXArray(inputScratch, [1, rung, rung, 4])
        }

        // Step 2: invoke the imported function. The graph returns
        // `(alpha, foreground)` per CorridorKey's bridge exporter.
        let results: [MLXArray]
        do {
            results = try withError {
                try function(inputArray)
            }
        } catch {
            throw KeyingInferenceError.modelUnavailable(
                "MLX apply failed: \(error.localizedDescription)"
            )
        }
        guard results.count >= 2 else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX bridge returned \(results.count) outputs; expected 2."
            )
        }
        try assertOutputShapes(alpha: results[0], foreground: results[1], rung: rung)

        // Step 3: force evaluation so the backing buffers are populated.
        do {
            try checkedEval(results[0], results[1])
        } catch {
            throw KeyingInferenceError.modelUnavailable(
                "MLX eval failed: \(error.localizedDescription)"
            )
        }

        // Step 4: alias MLX's own output storage as MTLBuffers (still on
        // the GPU, no CPU copy) and encode a compute pass that reads them
        // into the output textures the caller provided. A completion
        // handler retains the MLXArrays so their backing memory survives
        // until the kernel finishes reading.
        let device = cacheEntry.device
        guard let alphaMLXBuffer = results[0].asMTLBuffer(device: device, noCopy: true) else {
            throw KeyingInferenceError.modelUnavailable("MLX alpha output could not be exposed as MTLBuffer.")
        }
        guard let foregroundMLXBuffer = results[1].asMTLBuffer(device: device, noCopy: true) else {
            throw KeyingInferenceError.modelUnavailable("MLX foreground output could not be exposed as MTLBuffer.")
        }

        guard let commandQueue = cacheEntry.borrowCommandQueue() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        defer { cacheEntry.returnCommandQueue(commandQueue) }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        commandBuffer.label = "CorridorKey by LateNite MLX Writeback"

        try RenderStages.writeMLXOutputsToTextures(
            alphaBuffer: alphaMLXBuffer,
            foregroundBuffer: foregroundMLXBuffer,
            alphaDestination: output.alphaTexture,
            foregroundDestination: output.foregroundTexture,
            entry: cacheEntry,
            commandBuffer: commandBuffer,
            precision: precision
        )

        // Retain the MLXArrays (and the MTLBuffer aliases they back) until
        // the GPU is done reading them. `asMTLBuffer(noCopy:true)` requires
        // the MLXArray to outlive the MTLBuffer; `addCompletedHandler`
        // fires after the kernels are done, so keeping a strong
        // reference here is the simplest lifetime extension that works.
        // The captures cross a `@Sendable` boundary — MLXArray and
        // MTLBuffer aren't Sendable-typed, but they're safe to capture
        // because we only use them to keep allocations alive, not mutate
        // them. Wrap in a small `@unchecked Sendable` box to satisfy the
        // concurrency checker.
        final class RetainBox: @unchecked Sendable {
            let alphaArray: MLXArray
            let foregroundArray: MLXArray
            let alphaBuffer: any MTLBuffer
            let foregroundBuffer: any MTLBuffer
            init(alphaArray: MLXArray, foregroundArray: MLXArray, alphaBuffer: any MTLBuffer, foregroundBuffer: any MTLBuffer) {
                self.alphaArray = alphaArray
                self.foregroundArray = foregroundArray
                self.alphaBuffer = alphaBuffer
                self.foregroundBuffer = foregroundBuffer
            }
        }
        let retainBox = RetainBox(
            alphaArray: results[0],
            foregroundArray: results[1],
            alphaBuffer: alphaMLXBuffer,
            foregroundBuffer: foregroundMLXBuffer
        )
        commandBuffer.addCompletedHandler { _ in
            _ = retainBox
        }

        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Lock-guarded state helpers

    private func alreadyLoaded(rung: Int, screenColor: ScreenColor, precision: BridgePrecision) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return importedFunction != nil
            && loadedResolution == rung
            && loadedScreenColor == screenColor
            && loadedPrecision == precision
    }

    private func loadedState() -> (ImportedFunction?, Int, BridgePrecision) {
        stateLock.lock()
        defer { stateLock.unlock() }
        return (importedFunction, loadedResolution, loadedPrecision)
    }

    private func storeFunction(
        _ function: ImportedFunction,
        rung: Int,
        screenColor: ScreenColor,
        precision: BridgePrecision
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }
        if importedFunction != nil,
           loadedResolution == rung,
           loadedScreenColor == screenColor,
           loadedPrecision == precision {
            return
        }
        importedFunction = function
        loadedResolution = rung
        loadedScreenColor = screenColor
        loadedPrecision = precision
    }

    /// Validates that the loaded bridge's outputs have the expected NHWC layout.
    /// Fails loudly on mismatch so a misbuilt or misplaced `.mlxfn` doesn't
    /// silently corrupt downstream writes.
    private func assertOutputShapes(alpha: MLXArray, foreground: MLXArray, rung: Int) throws {
        let expectedAlpha = [1, rung, rung, 1]
        let expectedForeground = [1, rung, rung, 3]
        if alpha.shape != expectedAlpha || foreground.shape != expectedForeground {
            PluginLog.error(
                "MLX bridge returned unexpected shapes: alpha=\(alpha.shape) foreground=\(foreground.shape); expected \(expectedAlpha) and \(expectedForeground)."
            )
            throw KeyingInferenceError.modelUnavailable(
                "MLX bridge returned unexpected output shapes."
            )
        }
    }

}
