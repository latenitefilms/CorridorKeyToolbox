//
//  MLXKeyingEngine.swift
//  Corridor Key Toolbox
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

/// Names of the bundled `.mlxfn` artefacts. Matches CorridorKey-Runtime's
/// `corridorkey_mlx_bridge_{N}.mlxfn` convention so the same Hugging Face
/// release can be used unmodified.
enum MLXBridgeArtifact {
    static let filenameStem = "corridorkey_mlx_bridge"

    /// Supported bridge resolutions, in preference order from lowest to
    /// highest. `closestSupportedResolution` walks this list.
    static let ladder: [Int] = [512, 768, 1024, 1536, 2048]

    static func filename(forResolution resolution: Int) -> String {
        "\(filenameStem)_\(resolution).mlxfn"
    }

    /// Returns the ladder rung that is at least as large as `requested`,
    /// falling back to the maximum if nothing larger exists.
    static func closestSupportedResolution(forRequested requested: Int) -> Int? {
        ladder.first(where: { $0 >= requested }) ?? ladder.last
    }
}

/// Lazy bundled-resource lookup that works from either the XPC service
/// bundle or its host app bundle, whichever contains the `.mlxfn` files.
private enum MLXBridgeResourceLocator {
    static func url(for filename: String) -> URL? {
        let fileManager = FileManager.default

        // Bundle.main.url(forResource:…) is the simplest path — it resolves
        // to the service bundle's Resources folder when FCP loads us.
        let filenameStem = (filename as NSString).deletingPathExtension
        let filenameExtension = (filename as NSString).pathExtension
        if let url = Bundle.main.url(forResource: filenameStem, withExtension: filenameExtension) {
            return url
        }

        // Also walk `Bundle.allBundles` in case the file lives inside the
        // wrapper app's Resources folder (for a dev build where the pluginkit
        // is copied in after Xcode bundles the mlxfn into the outer app).
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
    let backendDisplayName: String
    var guideSourceDescription: String

    private let cacheEntry: MetalDeviceCacheEntry
    /// Guards `importedFunction` / `loadedResolution` reads and writes. Held
    /// briefly so warm-up and per-frame renders never deadlock each other.
    private let stateLock = NSLock()
    /// Serialises the entire `run(...)` path. FxPlug calls us from multiple
    /// render threads concurrently and `ImportedFunction` is not documented as
    /// thread-safe — funnel everything through a single in-flight inference.
    private let runLock = NSLock()
    private var importedFunction: ImportedFunction?
    private var loadedResolution: Int = 0

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "MLX on \(cacheEntry.device.name)"
        self.guideSourceDescription = "Auto rough fallback"
    }

    func supports(resolution: Int) -> Bool {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution) else {
            return false
        }
        return MLXBridgeResourceLocator.url(for: MLXBridgeArtifact.filename(forResolution: rung)) != nil
    }

    func prepare(resolution: Int) async throws {
        guard let rung = MLXBridgeArtifact.closestSupportedResolution(forRequested: resolution),
              let bridgeURL = MLXBridgeResourceLocator.url(for: MLXBridgeArtifact.filename(forResolution: rung))
        else {
            throw KeyingInferenceError.modelUnavailable(
                "No MLX bridge file bundled for \(resolution)px."
            )
        }

        if alreadyLoaded(rung: rung) { return }

        PluginLog.notice("Loading MLX bridge from \(bridgeURL.path).")
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
        await warmJIT(function: function, rung: rung)
        storeFunction(function, rung: rung)
    }

    /// Runs a throwaway inference on a zero tensor so MLX compiles the graph
    /// and warms the Metal buffer cache. Any failure here is non-fatal — the
    /// real inference will surface the same error to the caller later.
    private func warmJIT(function: ImportedFunction, rung: Int) async {
        let warmupStart = Date()
        let zeros = [Float](repeating: 0, count: rung * rung * 4)
        let input = MLXArray(zeros, [1, rung, rung, 4])
        do {
            let outputs = try function(input)
            eval(outputs)
        } catch {
            PluginLog.error("MLX JIT warm-up failed (non-fatal): \(error.localizedDescription)")
            return
        }
        let elapsedSeconds = Date().timeIntervalSince(warmupStart)
        PluginLog.notice("MLX JIT warm-up finished in \(String(format: "%.2f", elapsedSeconds))s for \(rung)px.")
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        runLock.lock()
        defer { runLock.unlock() }

        let (function, rung) = loadedState()
        guard let function, rung > 0 else {
            throw KeyingInferenceError.modelUnavailable("MLX bridge not prepared.")
        }
        _ = request.rawSourceTexture
        let expectedBytes = rung * rung * 4 * MemoryLayout<Float>.size
        guard request.normalisedInputBuffer.length >= expectedBytes else {
            throw KeyingInferenceError.modelUnavailable(
                "MLX input buffer is \(request.normalisedInputBuffer.length) bytes; expected ≥ \(expectedBytes)."
            )
        }

        // Step 1: wrap the shared MTLBuffer directly as an MLXArray. On
        // Apple Silicon's unified memory this is a zero-copy alias — MLX's
        // backing storage IS the same bytes the Metal kernel wrote. The
        // finalizer holds a strong reference to the buffer so it can't be
        // freed while MLX still owns it.
        let inputBuffer = request.normalisedInputBuffer
        let inputArray = MLXArray(
            rawPointer: inputBuffer.contents(),
            [1, rung, rung, 4],
            dtype: .float32,
            finalizer: { _ = inputBuffer }
        )

        // Step 2: invoke the imported function. The graph returns
        // `(alpha, foreground)` per CorridorKey's bridge exporter.
        let results: [MLXArray]
        do {
            results = try function(inputArray)
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
        eval(results[0], results[1])

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
        commandBuffer.label = "Corridor Key Toolbox MLX Writeback"

        try RenderStages.writeAlphaBufferToTexture(
            buffer: alphaMLXBuffer,
            destination: output.alphaTexture,
            entry: cacheEntry,
            commandBuffer: commandBuffer
        )
        try RenderStages.writeForegroundBufferToTexture(
            buffer: foregroundMLXBuffer,
            destination: output.foregroundTexture,
            entry: cacheEntry,
            commandBuffer: commandBuffer
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

    private func alreadyLoaded(rung: Int) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return importedFunction != nil && loadedResolution == rung
    }

    private func loadedState() -> (ImportedFunction?, Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        return (importedFunction, loadedResolution)
    }

    private func storeFunction(_ function: ImportedFunction, rung: Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        if importedFunction != nil, loadedResolution == rung { return }
        importedFunction = function
        loadedResolution = rung
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
