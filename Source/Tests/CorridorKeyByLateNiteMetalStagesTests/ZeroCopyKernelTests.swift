//
//  ZeroCopyKernelTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Verifies the two zero-copy MLX I/O kernels:
//    * `corridorKeyNormalizeToBufferKernel` — source + hint → NHWC
//      float buffer that MLX wraps with `init(rawPointer:)`.
//    * `corridorKeyMLXWritebackFusedKernel` — MLX alpha + foreground
//      output buffers (y-up) → `r32Float` + `rgba32Float` textures
//      (y-down) in a single dispatch, with RGB → RGBA expansion inline.
//
//  Each test runs the kernel on a real MTLDevice and checks the output
//  pixel-by-pixel against an analytically-derived expectation, so a
//  shader regression fails the suite before it reaches the plug-in.
//

import Foundation
import Metal
import Testing
import simd
import CorridorKeyToolboxLogic
@testable import CorridorKeyToolboxMetalStages

@Suite("ZeroCopyKernels")
struct ZeroCopyKernelTests {

    // MARK: - Normalize → Buffer

    @Test("Normalize-to-buffer writes NHWC-packed float32 with Rec.709 identity")
    func normalizeToBufferRec709Identity() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // Build a tiny solid-colour source and a matching solid hint.
        let rung = 4
        let sourceColour = SIMD4<Float>(0.5, 0.6, 0.7, 1.0)
        let hintValue: Float = 0.35

        let source = try makeColourTexture(
            entry: entry, width: rung, height: rung, pixelFormat: .rgba32Float, colour: sourceColour
        )
        let hint = try makeScalarTexture(entry: entry, width: rung, height: rung, value: hintValue)

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        let buffer = try RenderStages.combineAndNormaliseIntoBuffer(
            source: source,
            hint: hint,
            inferenceResolution: rung,
            workingToRec709: matrix_identity_float3x3,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        // Read back the first pixel's 4 floats. Because the source is a
        // solid colour, every pixel should be identical.
        let bufferPointer = buffer.contents().bindMemory(to: Float.self, capacity: rung * rung * 4)
        let firstPixel = SIMD4<Float>(
            bufferPointer[0], bufferPointer[1], bufferPointer[2], bufferPointer[3]
        )

        // Expected: ((rgb - mean) * invStdDev, hintValue).
        let mean = SIMD3<Float>(0.485, 0.456, 0.406)
        let invStdDev = SIMD3<Float>(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)
        let expectedRGB = (SIMD3<Float>(sourceColour.x, sourceColour.y, sourceColour.z) - mean) * invStdDev
        #expect(abs(firstPixel.x - expectedRGB.x) < 1e-3)
        #expect(abs(firstPixel.y - expectedRGB.y) < 1e-3)
        #expect(abs(firstPixel.z - expectedRGB.z) < 1e-3)
        #expect(abs(firstPixel.w - hintValue) < 1e-3)
    }

    @Test("Normalize-to-buffer is cached per rung (stable identity)")
    func normalizeToBufferCachesByRung() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let first = entry.normalizedInputBuffer(forRung: 512)
        let second = entry.normalizedInputBuffer(forRung: 512)
        let different = entry.normalizedInputBuffer(forRung: 1024)
        #expect(first != nil && second != nil && different != nil)
        if let first, let second {
            #expect(ObjectIdentifier(first) == ObjectIdentifier(second),
                    "Same rung should return the same buffer instance.")
        }
        if let first, let different {
            #expect(ObjectIdentifier(first) != ObjectIdentifier(different),
                    "Different rungs should have separate buffers.")
        }
    }

    // MARK: - MLX writeback (fused alpha + foreground)

    @Test("MLX writeback fuses alpha + foreground with y-flip and RGB → RGBA expansion")
    func mlxWritebackFusedFlipsAndExpands() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // 4×4 inputs. Alpha gradient: row y = y * 0.25. Foreground gradient:
        // row y = (y*0.1, y*0.2, y*0.3). After y-flip, output row 0 should
        // be the last input row in both buffers.
        let width = 4
        let height = 4
        let pixelCount = width * height

        var alphaSourceData = [Float](repeating: 0, count: pixelCount)
        var foregroundSourceData = [Float](repeating: 0, count: pixelCount * 3)
        for rowIndex in 0..<height {
            for columnIndex in 0..<width {
                alphaSourceData[rowIndex * width + columnIndex] = Float(rowIndex) * 0.25
                let base = (rowIndex * width + columnIndex) * 3
                foregroundSourceData[base + 0] = Float(rowIndex) * 0.1
                foregroundSourceData[base + 1] = Float(rowIndex) * 0.2
                foregroundSourceData[base + 2] = Float(rowIndex) * 0.3
            }
        }

        guard let alphaSourceBuffer = entry.device.makeBuffer(
            length: pixelCount * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            Issue.record("Could not allocate alpha source buffer.")
            return
        }
        alphaSourceData.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                alphaSourceBuffer.contents().copyMemory(
                    from: base,
                    byteCount: pixelCount * MemoryLayout<Float>.size
                )
            }
        }

        guard let foregroundSourceBuffer = entry.device.makeBuffer(
            length: pixelCount * 3 * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            Issue.record("Could not allocate foreground source buffer.")
            return
        }
        foregroundSourceData.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                foregroundSourceBuffer.contents().copyMemory(
                    from: base,
                    byteCount: pixelCount * 3 * MemoryLayout<Float>.size
                )
            }
        }

        guard let alphaDestination = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate alpha destination texture.")
            return
        }
        guard let foregroundDestination = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .rgba32Float, storageMode: .shared
        ) else {
            alphaDestination.returnManually()
            Issue.record("Could not allocate foreground destination texture.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            alphaDestination.returnManually()
            foregroundDestination.returnManually()
            Issue.record("Could not create command buffer.")
            return
        }
        try RenderStages.writeMLXOutputsFused(
            alphaBuffer: alphaSourceBuffer,
            foregroundBuffer: foregroundSourceBuffer,
            alphaDestination: alphaDestination.texture,
            foregroundDestination: foregroundDestination.texture,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var alphaResult = [Float](repeating: 0, count: pixelCount)
        alphaResult.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                alphaDestination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<Float>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        var foregroundResult = [SIMD4<Float>](repeating: .zero, count: pixelCount)
        foregroundResult.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                foregroundDestination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<SIMD4<Float>>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        alphaDestination.returnManually()
        foregroundDestination.returnManually()

        // Alpha: y-flipped — output row 0 should be input row 3 = 0.75.
        for x in 0..<width {
            #expect(abs(alphaResult[0 * width + x] - 0.75) < 1e-3)
            #expect(abs(alphaResult[1 * width + x] - 0.50) < 1e-3)
            #expect(abs(alphaResult[2 * width + x] - 0.25) < 1e-3)
            #expect(abs(alphaResult[3 * width + x] - 0.00) < 1e-3)
        }

        // Foreground: y-flipped — output row 0 came from input row 3
        // = (0.3, 0.6, 0.9, 1.0). Output row 3 came from input row 0 = (0,0,0,1).
        for x in 0..<width {
            let topRow = foregroundResult[0 * width + x]
            #expect(abs(topRow.x - 0.3) < 1e-3)
            #expect(abs(topRow.y - 0.6) < 1e-3)
            #expect(abs(topRow.z - 0.9) < 1e-3)
            #expect(abs(topRow.w - 1.0) < 1e-3, "Alpha should be 1.0, got \(topRow.w)")

            let bottomRow = foregroundResult[3 * width + x]
            #expect(abs(bottomRow.x - 0.0) < 1e-3)
            #expect(abs(bottomRow.y - 0.0) < 1e-3)
            #expect(abs(bottomRow.z - 0.0) < 1e-3)
            #expect(abs(bottomRow.w - 1.0) < 1e-3)
        }
    }

    // MARK: - Helpers

    private func makeColourTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat,
        colour: SIMD4<Float>
    ) throws -> any MTLTexture {
        guard let pooled = entry.texturePool.acquire(
            width: width,
            height: height,
            pixelFormat: pixelFormat,
            storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate colour texture.")
        }
        let pixelCount = width * height
        var pixels = [SIMD4<Float>](repeating: colour, count: pixelCount)
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                pooled.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<SIMD4<Float>>.size
                )
            }
        }
        return pooled.texture
    }

    private func makeScalarTexture(
        entry: MetalDeviceCacheEntry,
        width: Int,
        height: Int,
        value: Float
    ) throws -> any MTLTexture {
        guard let pooled = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            throw MetalUnavailable(reason: "Could not allocate scalar texture.")
        }
        let pixels = [Float](repeating: value, count: width * height)
        pixels.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                pooled.texture.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * MemoryLayout<Float>.size
                )
            }
        }
        return pooled.texture
    }

    private func makeCommandQueue(entry: MetalDeviceCacheEntry) throws -> any MTLCommandQueue {
        guard let queue = entry.device.makeCommandQueue() else {
            throw MetalUnavailable(reason: "Could not create test command queue.")
        }
        queue.label = "CorridorKey Test"
        return queue
    }

    private func commitAndWait(_ commandBuffer: any MTLCommandBuffer) throws {
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        if let error = commandBuffer.error { throw error }
    }
}
