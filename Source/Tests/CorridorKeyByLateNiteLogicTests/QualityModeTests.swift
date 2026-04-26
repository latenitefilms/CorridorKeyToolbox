//
//  QualityModeTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Guards the automatic-quality rung ladder against accidental drift. The
//  plug-in's "Recommended" option is chosen to line up with CorridorKey's OFX
//  reference so editors see identical defaults across hosts — breaking the
//  mapping silently would quietly degrade output quality on real footage.
//

import Testing
@testable import CorridorKeyToolboxLogic

@Suite("QualityMode")
struct QualityModeTests {

    /// Reference ladder for `.automatic` on a host with 96 GB+ RAM where
    /// no rung gets clamped. Verifies the long-edge → rung mapping itself.
    @Test("Automatic mode scales with long edge (high-RAM host)", arguments: [
        (100, 512),
        (512, 512),
        (1000, 512),
        (1001, 1024),
        (1920, 1024),
        (2000, 1024),
        (2001, 1536),
        (3000, 1536),
        (3001, 2048),
        (4096, 2048),
        (8192, 2048)
    ])
    func automaticResolutionLadderHighMemory(longEdge: Int, expected: Int) {
        let highRAMBytes: UInt64 = 128 * 1024 * 1024 * 1024
        #expect(
            QualityMode.automatic.resolvedInferenceResolution(
                forLongEdge: longEdge,
                physicalMemoryBytes: highRAMBytes
            ) == expected
        )
    }

    /// On a 32 GB Mac, `.automatic` must cap at 1024. Hiera's working set
    /// at 2048 needs ~64 GB of unified memory; without this clamp a 4K
    /// clip on a 32 GB machine spends minutes per frame in swap.
    @Test("Automatic mode caps at 1024 on a 32 GB host", arguments: [1000, 1920, 4096])
    func automaticCapsAt1024OnSmallMemory(longEdge: Int) {
        let thirtyTwoGigabytes: UInt64 = 32 * 1024 * 1024 * 1024
        let resolved = QualityMode.automatic.resolvedInferenceResolution(
            forLongEdge: longEdge,
            physicalMemoryBytes: thirtyTwoGigabytes
        )
        #expect(resolved <= 1024, "Expected ≤ 1024 on 32 GB host; got \(resolved) for longEdge=\(longEdge).")
    }

    /// On a 64 GB Mac, `.automatic` caps at 1536 — large clips get the
    /// extra rung but stay below the 2048 cliff that overflows even 64 GB.
    @Test("Automatic mode caps at 1536 on a 64 GB host", arguments: [
        (1000, 512), (1920, 1024), (3000, 1536), (4096, 1536), (8192, 1536)
    ])
    func automaticCapsAt1536OnSixtyFourGigabytes(longEdge: Int, expected: Int) {
        let sixtyFourGigabytes: UInt64 = 64 * 1024 * 1024 * 1024
        let resolved = QualityMode.automatic.resolvedInferenceResolution(
            forLongEdge: longEdge,
            physicalMemoryBytes: sixtyFourGigabytes
        )
        #expect(resolved == expected, "longEdge=\(longEdge): expected \(expected), got \(resolved).")
    }

    /// Explicit rungs ignore the host-RAM ceiling — the user has opted in.
    /// Verifies they can still pick `.maximum2048` on a 32 GB Mac if they
    /// know what they're doing.
    @Test("Explicit rungs ignore host RAM ceiling")
    func explicitRungsBypassRAMCeiling() {
        let thirtyTwoGigabytes: UInt64 = 32 * 1024 * 1024 * 1024
        #expect(
            QualityMode.maximum2048.resolvedInferenceResolution(
                forLongEdge: 4096, physicalMemoryBytes: thirtyTwoGigabytes
            ) == 2048
        )
        #expect(
            QualityMode.ultra1536.resolvedInferenceResolution(
                forLongEdge: 4096, physicalMemoryBytes: thirtyTwoGigabytes
            ) == 1536
        )
    }

    @Test("Manual rungs ignore input size")
    func manualRungsAreFixed() {
        let longEdges = [100, 1920, 4096]
        for longEdge in longEdges {
            #expect(QualityMode.draft512.resolvedInferenceResolution(forLongEdge: longEdge) == 512)
            #expect(QualityMode.high1024.resolvedInferenceResolution(forLongEdge: longEdge) == 1024)
            #expect(QualityMode.ultra1536.resolvedInferenceResolution(forLongEdge: longEdge) == 1536)
            #expect(QualityMode.maximum2048.resolvedInferenceResolution(forLongEdge: longEdge) == 2048)
        }
    }

    @Test("Display names are stable for saved documents")
    func displayNamesAreStable() {
        #expect(QualityMode.automatic.displayName == "Recommended")
        #expect(QualityMode.draft512.displayName == "Draft (512)")
        #expect(QualityMode.high1024.displayName == "High (1024)")
        #expect(QualityMode.ultra1536.displayName == "Ultra (1536)")
        #expect(QualityMode.maximum2048.displayName == "Maximum (2048)")
    }
}
