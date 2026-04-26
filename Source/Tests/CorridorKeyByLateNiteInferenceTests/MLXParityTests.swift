//
//  MLXParityTests.swift
//  CorridorKeyToolboxInferenceTests
//
//  Validates that the two `MLXKeyingEngine.InputStrategy` paths produce
//  equivalent output for the same input. The `.zeroCopy` path is the
//  reference (it's what shipped in v1.0.0 build 2 and produces visually
//  correct mattes); `.cpuStaging` is the alternate we landed when
//  `.zeroCopy` was suspected to be slow on a particular clip. If they
//  disagree, the alternate has a parity bug and we cannot trust its
//  output.
//
//  Also asserts the alpha matte returned for a non-trivial input has
//  *some* structure — not all zeros, not all ones, more than one
//  distinct value. A wedged graph that emits a constant tensor would
//  produce an "unusable" matte that this catches without ground truth.
//

import Foundation
import Metal
import MLX
import Testing
@testable import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("MLXParity")
struct MLXParityTests {

    @Test("zeroCopy and cpuStaging strategies produce the same alpha output")
    func strategyParity() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()

        // Run the same input through each strategy. We override the
        // build-time `inputStrategy` via the test override entry
        // points (added below) so we don't have to rebuild the engine.
        let zeroCopy = try await runOneInference(
            strategy: .zeroCopy,
            bridgeURL: bridgeURL,
            entry: entry
        )
        let cpuStaging = try await runOneInference(
            strategy: .cpuStaging,
            bridgeURL: bridgeURL,
            entry: entry
        )

        #expect(zeroCopy.count == cpuStaging.count, "Different output counts: \(zeroCopy.count) vs \(cpuStaging.count)")

        // Floating-point tolerance: MLX may schedule the two runs on
        // slightly different code paths (rawPointer aliases vs owned
        // arrays), so allow rounding noise but not divergence.
        let tolerance: Float = 1e-3
        var maxAbsDelta: Float = 0
        var firstDifferIndex: Int = -1
        for index in 0..<min(zeroCopy.count, cpuStaging.count) {
            let delta = abs(zeroCopy[index] - cpuStaging[index])
            if delta > maxAbsDelta {
                maxAbsDelta = delta
                if delta > tolerance && firstDifferIndex < 0 {
                    firstDifferIndex = index
                }
            }
        }
        if maxAbsDelta > tolerance {
            print("First diverging pixel index: \(firstDifferIndex) — zeroCopy=\(zeroCopy[firstDifferIndex]), cpuStaging=\(cpuStaging[firstDifferIndex])")
            print("Max |Δ|: \(maxAbsDelta)")
        }
        #expect(maxAbsDelta <= tolerance, "Strategies diverge: max |Δ| = \(maxAbsDelta) (tolerance \(tolerance))")
    }

    @Test("MLX produces identical output for the same input across 30 sequential calls")
    func outputStableAcrossManyInferences() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)

        // Frame 1 reference.
        try engine.run(request: request, output: output)
        let frame1 = readAlpha(output: output)

        // Run 29 more, checking each against frame 1 — the input is
        // identical every iteration so output must be too. If the
        // cache-limit fix corrupts MLX's internal state under pressure,
        // this is what would catch it.
        for iteration in 2...30 {
            try engine.run(request: request, output: output)
            let frameN = readAlpha(output: output)

            #expect(frame1.count == frameN.count, "Output count diverged at iteration \(iteration).")

            var maxAbsDelta: Float = 0
            for index in 0..<min(frame1.count, frameN.count) {
                maxAbsDelta = max(maxAbsDelta, abs(frame1[index] - frameN[index]))
            }
            // Same input, deterministic graph — output should match
            // exactly (within tiny float-noise tolerance). Anything
            // larger means the cache cap is breaking computation.
            let tolerance: Float = 1e-4
            #expect(
                maxAbsDelta <= tolerance,
                "Output drifted at iteration \(iteration): max |Δ| = \(maxAbsDelta) (tolerance \(tolerance))"
            )
            if maxAbsDelta > tolerance {
                print("Drift detected at iteration \(iteration): \(maxAbsDelta)")
                break
            }
        }
    }

    @Test("MLX returns a non-degenerate alpha matte for a structured input")
    func outputHasStructure() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        let bridgeURL = try InferenceTestHarness.bridgeURL512()
        let alpha = try await runOneInference(
            strategy: .zeroCopy,
            bridgeURL: bridgeURL,
            entry: entry
        )

        var minValue: Float = .infinity
        var maxValue: Float = -.infinity
        var nonZero: Int = 0
        var nonOne: Int = 0
        for value in alpha {
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            if value > 0.001 { nonZero += 1 }
            if value < 0.999 { nonOne += 1 }
        }

        let pixelCount = alpha.count
        // Mid-range guard: a wedged model would emit all 0 or all 1.
        // Any non-degenerate model should have both kinds in a 512² output
        // for a structured RGB input.
        #expect(nonZero > pixelCount / 100, "Alpha is essentially all zeros (\(nonZero) of \(pixelCount) pixels > 0.001).")
        #expect(nonOne > pixelCount / 100, "Alpha is essentially all ones (\(nonOne) of \(pixelCount) pixels < 0.999).")
        #expect(maxValue > minValue + 0.1, "Alpha range too narrow: [\(minValue), \(maxValue)].")

        print("Alpha range: [\(minValue), \(maxValue)], non-zero \(nonZero)/\(pixelCount), non-one \(nonOne)/\(pixelCount)")
    }

    // MARK: - Helpers

    private func readAlpha(output: KeyingInferenceOutput) -> [Float] {
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

    /// Boots a fresh engine, applies the requested strategy, runs one
    /// inference on a deterministic gradient input, and returns the
    /// alpha texture pixels read back to CPU.
    private func runOneInference(
        strategy: MLXKeyingEngine.InputStrategy,
        bridgeURL: URL,
        entry: MetalDeviceCacheEntry
    ) async throws -> [Float] {
        let engine = MLXKeyingEngine(cacheEntry: entry)
        // Override the build-time strategy. The test entry point lives
        // on the engine for exactly this purpose; production paths
        // continue to use the default.
        engine.testOverrideInputStrategy(strategy)
        try await engine.prepare(bridgeURL: bridgeURL, rung: 512)

        let request = try makeRequest(rung: 512, entry: entry)
        let output = try makeOutput(rung: 512, entry: entry)
        try engine.run(request: request, output: output)

        return readAlpha(output: output)
    }

    /// Allocates a normalised input buffer filled with a smooth gradient
    /// — spatially structured so the model's response is non-trivial,
    /// deterministic so successive runs are bit-equal at the input.
    private func makeRequest(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceRequest {
        guard let buffer = entry.normalizedInputBuffer(forRung: rung) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate normalised input buffer.")
        }
        let elementCount = rung * rung * 4
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        // 4-channel structured pattern: greens for "screen", magentas for
        // "subject", with a soft circular boundary that the model should
        // segment cleanly. Same layout the normalise kernel produces:
        // [(R, G, B, hint)…] in row-major order.
        for row in 0..<rung {
            for column in 0..<rung {
                let dx = Float(column) - Float(rung) / 2
                let dy = Float(row) - Float(rung) / 2
                let radius = sqrt(dx * dx + dy * dy) / Float(rung) * 2
                let isSubject = radius < 0.4
                let baseIndex = (row * rung + column) * 4
                if isSubject {
                    pointer[baseIndex + 0] = 0.85   // R
                    pointer[baseIndex + 1] = 0.30   // G
                    pointer[baseIndex + 2] = 0.50   // B
                    pointer[baseIndex + 3] = 0.0    // hint: foreground
                } else {
                    pointer[baseIndex + 0] = 0.10   // R
                    pointer[baseIndex + 1] = 0.85   // G (green screen)
                    pointer[baseIndex + 2] = 0.10   // B
                    pointer[baseIndex + 3] = 1.0    // hint: background
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
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate raw source texture.")
        }
        return KeyingInferenceRequest(
            normalisedInputBuffer: buffer,
            rawSourceTexture: rawSource,
            inferenceResolution: rung
        )
    }

    private func makeOutput(rung: Int, entry: MetalDeviceCacheEntry) throws -> KeyingInferenceOutput {
        guard let alpha = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .r32Float,
            storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate alpha texture.")
        }
        guard let foreground = entry.makeIntermediateTexture(
            width: rung,
            height: rung,
            pixelFormat: .rgba32Float,
            storageMode: .shared
        ) else {
            throw InferenceTestHarness.MetalUnavailable(reason: "Could not allocate foreground texture.")
        }
        return KeyingInferenceOutput(alphaTexture: alpha, foregroundTexture: foreground)
    }
}

private struct XCTSkip: Error, CustomStringConvertible {
    let underlying: any Error
    init(_ error: any Error) { self.underlying = error }
    var description: String { "Skipped: \(underlying)" }
}
