//
//  DeviceCapabilityCacheTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Verifies the rolling-median bookkeeping and ceiling-lift logic of
//  `DeviceCapabilityCache`. Each test builds an isolated `UserDefaults`
//  suite so concurrent test runs don't trample each other and so we
//  don't pollute the developer's actual preferences.
//

import Foundation
import Testing
@testable import CorridorKeyToolboxLogic

@Suite("DeviceCapabilityCache")
struct DeviceCapabilityCacheTests {

    /// Builds a cache backed by an ephemeral `UserDefaults` suite. The
    /// suite is removed in `Suite.tearDown` semantics — Swift Testing
    /// doesn't have an explicit tear-down hook, but `removeSuite` clears
    /// the persistent backing store.
    private func makeCache(suiteName: String = UUID().uuidString) -> DeviceCapabilityCache {
        let defaults = UserDefaults(suiteName: suiteName) ?? .standard
        defaults.removePersistentDomain(forName: suiteName)
        return DeviceCapabilityCache(userDefaultsForTesting: defaults)
    }

    @Test("Median follows the trailing window")
    func medianFollowsWindow() {
        let cache = makeCache()
        for value in [50.0, 60.0, 100.0] {
            cache.record(deviceRegistryID: 42, rung: 1024, milliseconds: value)
        }
        // Median of [50, 60, 100] = 60
        #expect(cache.medianMilliseconds(deviceRegistryID: 42, rung: 1024) == 60.0)

        // Add an outlier that pushes the median up.
        cache.record(deviceRegistryID: 42, rung: 1024, milliseconds: 200.0)
        // Median of [50, 60, 100, 200] = (60 + 100) / 2 = 80
        #expect(cache.medianMilliseconds(deviceRegistryID: 42, rung: 1024) == 80.0)
    }

    @Test("Window evicts oldest samples beyond capacity")
    func windowEvictsOldest() {
        let cache = makeCache()
        for i in 1...DeviceCapabilityCache.sampleHistory + 4 {
            cache.record(deviceRegistryID: 7, rung: 512, milliseconds: Double(i) * 10)
        }
        // After capacity-overflow, the median should reflect only the
        // last `sampleHistory` samples (5..12 for sampleHistory=8).
        guard let median = cache.medianMilliseconds(deviceRegistryID: 7, rung: 512) else {
            Issue.record("Expected a median after overflow.")
            return
        }
        // Median of [50, 60, 70, 80, 90, 100, 110, 120] = 85
        #expect(median == 85.0)
    }

    @Test("Recommended ceiling stays at static when no fast samples exist")
    func ceilingStaysWhenNoData() {
        let cache = makeCache()
        let ceiling = cache.recommendedCeiling(
            deviceRegistryID: 1,
            staticCeiling: 1024,
            ladder: [512, 768, 1024, 1536, 2048]
        )
        #expect(ceiling == 1024)
    }

    @Test("Recommended ceiling lifts when next rung is below budget")
    func ceilingLiftsWhenFast() {
        let cache = makeCache()
        // Fast inferences at the next rung (1536) → bump ceiling.
        for _ in 0..<5 {
            cache.record(deviceRegistryID: 99, rung: 1536, milliseconds: 80)
        }
        let ceiling = cache.recommendedCeiling(
            deviceRegistryID: 99,
            staticCeiling: 1024,
            ladder: [512, 768, 1024, 1536, 2048]
        )
        #expect(ceiling == 1536)
    }

    @Test("Ceiling stops walking at the first slow rung")
    func ceilingStopsAtSlowRung() {
        let cache = makeCache()
        // 1536 is fast, 2048 is slow → ceiling lifts to 1536 only.
        for _ in 0..<5 {
            cache.record(deviceRegistryID: 100, rung: 1536, milliseconds: 80)
        }
        for _ in 0..<5 {
            cache.record(deviceRegistryID: 100, rung: 2048, milliseconds: 500)
        }
        let ceiling = cache.recommendedCeiling(
            deviceRegistryID: 100,
            staticCeiling: 1024,
            ladder: [512, 768, 1024, 1536, 2048]
        )
        #expect(ceiling == 1536)
    }

    @Test("Ceiling never lowers below static")
    func ceilingNeverLowersBelowStatic() {
        let cache = makeCache()
        // Recorded 1024 as slow but the static ceiling already says 1024
        // is OK; the cache never lowers, only lifts.
        cache.record(deviceRegistryID: 200, rung: 1024, milliseconds: 999)
        let ceiling = cache.recommendedCeiling(
            deviceRegistryID: 200,
            staticCeiling: 1024,
            ladder: [512, 768, 1024, 1536, 2048]
        )
        #expect(ceiling == 1024)
    }

    @Test("QualityMode.automatic respects cache lift")
    func qualityModeAutomaticUsesCache() {
        let cache = makeCache()
        // Record fast 1536 inferences for device 555.
        for _ in 0..<5 {
            cache.record(deviceRegistryID: 555, rung: 1536, milliseconds: 100)
        }
        // 32 GB Mac normally caps automatic at 1024. With cache lift, it
        // should pick 1536 for a 4K clip.
        let lifted = QualityMode.automatic.resolvedInferenceResolution(
            forLongEdge: 3840,
            physicalMemoryBytes: 32 * (1 << 30),
            deviceRegistryID: 555,
            cache: cache
        )
        #expect(lifted == 1536)

        // A different device on the same Mac doesn't get the lift.
        let static1024 = QualityMode.automatic.resolvedInferenceResolution(
            forLongEdge: 3840,
            physicalMemoryBytes: 32 * (1 << 30),
            deviceRegistryID: 999,
            cache: cache
        )
        #expect(static1024 == 1024)
    }

    @Test("Garbage records are silently ignored")
    func ignoresInvalidRecords() {
        let cache = makeCache()
        cache.record(deviceRegistryID: 1, rung: 1024, milliseconds: -1)
        cache.record(deviceRegistryID: 1, rung: 1024, milliseconds: .infinity)
        cache.record(deviceRegistryID: 1, rung: 1024, milliseconds: .nan)
        cache.record(deviceRegistryID: 1, rung: 1024, milliseconds: 999_999)
        #expect(cache.medianMilliseconds(deviceRegistryID: 1, rung: 1024) == nil)
    }
}
