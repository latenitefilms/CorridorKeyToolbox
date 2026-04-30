//
//  InferenceTestHarness.swift
//  CorridorKeyToolboxInferenceTests
//
//  Builds a usable `MetalDeviceCacheEntry` for the SPM inference tests.
//  Mirrors the `TestHarness` shape used by the Metal-stages tests but is
//  duplicated here so the inference target can run independently of the
//  Metal-stages test bundle.
//

import Foundation
import Metal
@testable import CorridorKeyToolboxMetalStages

enum InferenceTestHarness {

    enum InputPattern {
        case linearRamp
        case structuredKey
    }

    /// Skippable failure for CI runners without a Metal device.
    struct MetalUnavailable: Error, CustomStringConvertible {
        let reason: String
        var description: String { reason }
    }

    /// Returns a fully-initialised entry whose Metal library is compiled
    /// from the bundled `CorridorKeyShaders.metal` source. Throws
    /// `MetalUnavailable` when the host has no GPU or the shaders fail to
    /// compile so tests can early-return cleanly.
    static func makeEntry() throws -> MetalDeviceCacheEntry {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalUnavailable(reason: "No Metal device available on this host.")
        }
        let library = try makeShaderLibrary(device: device)
        return try MetalDeviceCacheEntry(device: device, library: library)
    }

    static var allBridgeRungs: [Int] {
        MLXBridgeArtifact.ladder
    }

    /// Locates an MLX bridge by resolution. The 512px bridge is copied
    /// into the inference test bundle for fast CI runs; the larger
    /// production bridges stay in the app resources directory so we do
    /// not duplicate another ~1.5 GB into SwiftPM build products.
    static func bridgeURL(forRung rung: Int) throws -> URL {
        let filename = MLXBridgeArtifact.filename(forResolution: rung)
        let filenameURL = URL(fileURLWithPath: filename)
        let filenameStem = filenameURL.deletingPathExtension().lastPathComponent
        let filenameExtension = filenameURL.pathExtension
        if let bundled = Bundle.module.url(
            forResource: filenameStem,
            withExtension: filenameExtension
        ) {
            return bundled
        }

        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let productionURL = sourceRoot
            .appending(path: "CorridorKeyByLateNite/Plugin/Resources/MLX Models")
            .appending(path: filename)
        if FileManager.default.fileExists(atPath: productionURL.path) {
            return productionURL
        }

        throw MetalUnavailable(reason: "\(filename) missing from the test bundle and production resources.")
    }

    /// Locates the bundled 512px MLX bridge inside the test target's
    /// resources bundle. `MLXBridgeResourceLocator` walks `Bundle.main` /
    /// `Bundle.allBundles`, neither of which reliably picks up SPM test
    /// resource bundles, so tests pass the URL explicitly.
    static func bridgeURL512() throws -> URL {
        try bridgeURL(forRung: 512)
    }

    static func readAlpha(output: KeyingInferenceOutput) -> [Float] {
        let width = output.alphaTexture.width
        let height = output.alphaTexture.height
        var pixels = [Float](repeating: 0, count: width * height)
        let bytesPerRow = width * MemoryLayout<Float>.size
        pixels.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                output.alphaTexture.getBytes(
                    base,
                    bytesPerRow: bytesPerRow,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        return pixels
    }

    /// Allocates a normalised-input MTLBuffer matching the shape MLX
    /// expects (1 × rung × rung × 4 floats), filled with deterministic
    /// input so each test exercises a real graph rather than all-zero
    /// tensors that MLX could constant-fold.
    static func makeRequest(
        rung: Int,
        entry: MetalDeviceCacheEntry,
        pattern: InputPattern = .structuredKey
    ) throws -> KeyingInferenceRequest {
        guard let buffer = entry.normalizedInputBuffer(forRung: rung) else {
            throw MetalUnavailable(reason: "Could not allocate normalised input buffer.")
        }
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        switch pattern {
        case .linearRamp:
            let elementCount = rung * rung * 4
            for index in 0..<elementCount {
                pointer[index] = Float(index % 256) / 255.0
            }
        case .structuredKey:
            for row in 0..<rung {
                for column in 0..<rung {
                    let dx = Float(column) - Float(rung) / 2
                    let dy = Float(row) - Float(rung) / 2
                    let radius = sqrt(dx * dx + dy * dy) / Float(rung) * 2
                    let isSubject = radius < 0.4
                    let baseIndex = (row * rung + column) * 4
                    if isSubject {
                        pointer[baseIndex + 0] = 0.85
                        pointer[baseIndex + 1] = 0.30
                        pointer[baseIndex + 2] = 0.50
                        pointer[baseIndex + 3] = 0.0
                    } else {
                        pointer[baseIndex + 0] = 0.10
                        pointer[baseIndex + 1] = 0.85
                        pointer[baseIndex + 2] = 0.10
                        pointer[baseIndex + 3] = 1.0
                    }
                }
            }
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: rung,
            height: rung,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let rawSource = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not allocate raw source texture.")
        }
        return KeyingInferenceRequest(
            normalisedInputBuffer: buffer,
            rawSourceTexture: rawSource,
            inferenceResolution: rung
        )
    }

    /// Allocates the alpha + foreground destination textures the engine
    /// writes into. Both are `.shared` so the production code can read
    /// them back; that lifecycle matches `InferenceCoordinator`.
    static func makeOutput(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceOutput {
        guard let alpha = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate alpha texture.")
        }
        guard let foreground = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate foreground texture.")
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }

    /// Compiles the bundled shader source at run-time so the SPM build
    /// doesn't need Xcode's `default.metallib` step. The Metal-stages
    /// `TestHarness` uses the same trick.
    private static func makeShaderLibrary(device: any MTLDevice) throws -> any MTLLibrary {
        let bundle = Bundle.module
        guard let shaderURL = bundle.url(forResource: "CorridorKeyShaders", withExtension: "metal") else {
            // The shader source travels with `CorridorKeyToolboxMetalStages`,
            // not with the inference test target — fall back to its bundle.
            let metalBundle = MetalStagesResourceBundle.bundle
            guard let stagesShaderURL = metalBundle.url(forResource: "CorridorKeyShaders", withExtension: "metal") else {
                throw MetalUnavailable(reason: "Could not locate CorridorKeyShaders.metal.")
            }
            return try compileLibrary(device: device, shaderURL: stagesShaderURL, headerBundle: metalBundle)
        }
        return try compileLibrary(device: device, shaderURL: shaderURL, headerBundle: bundle)
    }

    private static func compileLibrary(
        device: any MTLDevice,
        shaderURL: URL,
        headerBundle: Bundle
    ) throws -> any MTLLibrary {
        guard let headerURL = headerBundle.url(forResource: "CorridorKeyShaderTypes", withExtension: "h") else {
            throw MetalUnavailable(reason: "Could not locate CorridorKeyShaderTypes.h.")
        }
        let shaderSource = try String(contentsOf: shaderURL, encoding: .utf8)
        let headerSource = try String(contentsOf: headerURL, encoding: .utf8)
        let combined = headerSource
            .replacing("#ifndef CorridorKeyShaderTypes_h", with: "")
            .replacing("#define CorridorKeyShaderTypes_h", with: "")
            .replacing("#endif /* CorridorKeyShaderTypes_h */", with: "")
            + "\n"
            + shaderSource.replacing(#"#include "CorridorKeyShaderTypes.h""#, with: "")
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        do {
            return try device.makeLibrary(source: combined, options: options)
        } catch {
            throw MetalUnavailable(reason: "Corridor Key shader compile failed: \(error.localizedDescription)")
        }
    }
}

/// Minimal stand-in for XCTSkipError, mirroring the shape used by
/// `CorridorKeyToolboxMetalStagesTests`. Tests catch `MetalUnavailable`
/// and rethrow as this so the runner reports a skip instead of a failure
/// when the host has no GPU.
struct XCTSkip: Error, CustomStringConvertible {
    let underlying: any Error
    init(_ error: any Error) { self.underlying = error }
    var description: String { "Skipped: \(underlying)" }
}

/// Tiny shim that exposes `Bundle.module` from `CorridorKeyToolboxMetalStages`
/// so we can fish out the shared shader resources without making them
/// `public` on every consumer.
private enum MetalStagesResourceBundle {
    static var bundle: Bundle {
        // `Bundle.module` is module-scoped — referenced from a file that
        // belongs to `CorridorKeyToolboxMetalStages`, it returns that
        // module's resource bundle. We reach across via a public function
        // exposed there for exactly this purpose; if the symbol is
        // missing, we fall back to the test bundle and the shader-source
        // load will fail with a clean `MetalUnavailable`.
        MetalStagesResourceBundleAccessor.bundle()
    }
}
