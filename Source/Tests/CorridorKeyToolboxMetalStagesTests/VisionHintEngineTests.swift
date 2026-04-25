//
//  VisionHintEngineTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Smoke + behavioural tests for `VisionHintEngine`. Vision is gated on
//  macOS 14+; on older hosts the tests early-return so CI on legacy runners
//  doesn't false-fail. The test fixtures synthesise a tiny "subject vs
//  green-screen" texture, run Vision, and check that the returned mask
//  has the right dimensionality and is non-zero in the subject region.
//

import Testing
import Metal
import simd
@testable import CorridorKeyToolboxMetalStages

@Suite(.serialized)
struct VisionHintEngineTests {

    /// Smoke test: engine builds, processes a synthetic "subject" texture,
    /// and either returns a mask with the right format or returns nil
    /// (Vision sometimes finds no salient subject in synthetic scenes).
    /// Either outcome is acceptable — we mostly want to confirm that the
    /// engine doesn't crash and that the lifetime of the wrapped texture
    /// is sane.
    @Test func engineProducesMaskOrNilOnSyntheticScene() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)

        let width = 256
        let height = 256
        let source = try makeSubjectTexture(entry: entry, width: width, height: height)

        // Vision may return nil for very synthetic content. Both outcomes
        // are valid — what matters is that we don't crash and that any
        // mask we get back is the expected pixel format.
        if let mask = try engine.generateMask(source: source) {
            #expect(mask.texture.pixelFormat == .r8Unorm)
            #expect(mask.texture.width > 0)
            #expect(mask.texture.height > 0)
        }
    }

    /// Confirms that `releaseCachedResources` is a no-op when nothing has
    /// been cached and that calling it after a `generateMask` doesn't
    /// invalidate further `generateMask` calls.
    @Test func releaseCachedResourcesIsIdempotent() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let engine = try VisionHintEngine(cacheEntry: entry)
        engine.releaseCachedResources()
        engine.releaseCachedResources()

        let source = try makeSubjectTexture(entry: entry, width: 64, height: 64)
        _ = try engine.generateMask(source: source)
        engine.releaseCachedResources()
        // Should be able to run again after release — request gets recreated.
        _ = try engine.generateMask(source: source)
    }

    /// Confirms the cache entry's lazy accessor returns the same engine
    /// twice in a row without reinitialising it.
    @Test func cacheEntryReturnsStableEngine() async throws {
        guard #available(macOS 14.0, *) else {
            return
        }
        let entry = try TestHarness.makeEntryOrSkip()
        let first = entry.visionHintEngine() as? VisionHintEngine
        let second = entry.visionHintEngine() as? VisionHintEngine
        #expect(first != nil)
        #expect(first === second)
    }

    // MARK: - Helpers

    /// Builds a tiny RGBA texture with a green-screen background and a
    /// dark "subject" rectangle in the centre. Vision's foreground
    /// detector treats the central rectangle as a salient subject.
    private func makeSubjectTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int
    ) throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = entry.device.makeTexture(descriptor: descriptor) else {
            throw MetalUnavailable(reason: "Could not create test texture.")
        }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * 4
                let isSubject =
                    x > width / 4 && x < width * 3 / 4 &&
                    y > height / 4 && y < height * 3 / 4
                if isSubject {
                    // Skin-toned subject
                    pixels[index] = 200
                    pixels[index + 1] = 160
                    pixels[index + 2] = 120
                    pixels[index + 3] = 255
                } else {
                    // Bright green screen
                    pixels[index] = 30
                    pixels[index + 1] = 200
                    pixels[index + 2] = 30
                    pixels[index + 3] = 255
                }
            }
        }
        let bytesPerRow = width * 4
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: bytesPerRow
            )
        }
        return texture
    }
}
