//
//  MLXBridgeRungTests.swift
//  CorridorKeyToolboxInferenceTests
//
//  Exercises every production `.mlxfn` bridge that ships in the app
//  bundle. The crash reports from TestFlight all died in MLX eval, so
//  this suite validates both the direct MLX error path and the full
//  CorridorKey bridge load/eval/writeback path for every supported rung.
//

import Foundation
import MLX
import Testing
@testable import CorridorKeyToolboxMetalStages

@Suite("MLX bridge rungs", .serialized)
struct MLXBridgeRungTests {

    @Test("MLX scoped error handler converts runtime errors into Swift throws")
    func scopedErrorHandlerThrowsInsteadOfFatalError() throws {
        let left = MLXArray(0 ..< 10, [2, 5])
        let right = MLXArray(0 ..< 15, [3, 5])

        #expect(throws: (any Error).self) {
            try withError {
                let invalidBroadcast = left + right
                eval(invalidBroadcast)
            }
        }
    }

    @Test("every bundled MLX bridge loads, evaluates, and writes finite alpha")
    func everyBundledBridgeRunsOneInference() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try InferenceTestHarness.makeEntry() }
        catch { throw XCTSkip(error) }

        for rung in InferenceTestHarness.allBridgeRungs {
            try await runOneInference(rung: rung, entry: entry)
            MLX.Memory.clearCache()
        }
    }

    private func runOneInference(rung: Int, entry: MetalDeviceCacheEntry) async throws {
        let bridgeURL = try InferenceTestHarness.bridgeURL(forRung: rung)
        let engine = MLXKeyingEngine(cacheEntry: entry)
        try await engine.prepare(bridgeURL: bridgeURL, rung: rung)

        let request = try InferenceTestHarness.makeRequest(rung: rung, entry: entry)
        let output = try InferenceTestHarness.makeOutput(rung: rung, entry: entry)
        try engine.run(request: request, output: output)

        let alpha = InferenceTestHarness.readAlpha(output: output)
        var finiteCount = 0
        var minValue = Float.infinity
        var maxValue = -Float.infinity
        for value in alpha {
            guard value.isFinite else { continue }
            finiteCount += 1
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
        }

        #expect(finiteCount == alpha.count, "\(rung)px bridge produced non-finite alpha values.")
        #expect(maxValue > minValue + 0.001, "\(rung)px bridge produced a degenerate alpha range.")
        print("\(rung)px MLX bridge alpha range: [\(minValue), \(maxValue)]")
    }
}
