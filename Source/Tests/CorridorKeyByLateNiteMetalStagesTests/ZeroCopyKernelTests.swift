//
//  ZeroCopyKernelTests.swift
//  CorridorKeyToolboxMetalStagesTests
//
//  Verifies the three zero-copy MLX I/O kernels:
//    * `corridorKeyNormalizeToBufferKernel` — source + hint → NHWC
//      float buffer that MLX wraps with `init(rawPointer:)`.
//    * `corridorKeyAlphaBufferToTextureKernel` — MLX alpha output buffer
//      (y-up) → `r32Float` texture (y-down).
//    * `corridorKeyForegroundBufferToTextureKernel` — MLX foreground RGB
//      buffer (y-up) → `rgba32Float` texture (y-down) with alpha = 1.
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

    // MARK: - Alpha buffer → texture

    @Test("Alpha buffer → texture copies values with y-flip")
    func alphaBufferToTextureFlipsY() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        // 4×4 gradient: row 0 = 0.0, row 1 = 0.25, row 2 = 0.5, row 3 = 0.75.
        // After y-flip, row 0 in the output should be the last row of input (0.75),
        // row 3 should be 0.0.
        let width = 4
        let height = 4
        let pixelCount = width * height
        var sourceData = [Float](repeating: 0, count: pixelCount)
        for y in 0..<height {
            for x in 0..<width {
                sourceData[y * width + x] = Float(y) * 0.25
            }
        }

        guard let sourceBuffer = entry.device.makeBuffer(
            length: pixelCount * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            Issue.record("Could not allocate source buffer.")
            return
        }
        sourceData.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                sourceBuffer.contents().copyMemory(from: base, byteCount: pixelCount * MemoryLayout<Float>.size)
            }
        }

        guard let destination = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .r32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate destination texture.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        try RenderStages.writeAlphaBufferToTexture(
            buffer: sourceBuffer,
            destination: destination.texture,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var result = [Float](repeating: 0, count: pixelCount)
        result.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                destination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<Float>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        destination.returnManually()

        // y-flipped: output row 0 should be input row 3 = 0.75.
        for x in 0..<width {
            #expect(abs(result[0 * width + x] - 0.75) < 1e-3)
            #expect(abs(result[1 * width + x] - 0.50) < 1e-3)
            #expect(abs(result[2 * width + x] - 0.25) < 1e-3)
            #expect(abs(result[3 * width + x] - 0.00) < 1e-3)
        }
    }

    // MARK: - Foreground buffer → texture

    @Test("Foreground buffer → texture expands RGB to RGBA with y-flip")
    func foregroundBufferToTextureExpandsAndFlips() async throws {
        let entry: MetalDeviceCacheEntry
        do { entry = try TestHarness.makeEntry() } catch { throw XCTSkipError("\(error)") }

        let width = 4
        let height = 4
        let pixelCount = width * height
        // Pack RGB floats for each pixel. Row `y` gets colour `(y*0.1, y*0.2, y*0.3)`.
        var sourceData = [Float](repeating: 0, count: pixelCount * 3)
        for y in 0..<height {
            for x in 0..<width {
                let base = (y * width + x) * 3
                sourceData[base + 0] = Float(y) * 0.1
                sourceData[base + 1] = Float(y) * 0.2
                sourceData[base + 2] = Float(y) * 0.3
            }
        }

        guard let sourceBuffer = entry.device.makeBuffer(
            length: pixelCount * 3 * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            Issue.record("Could not allocate source buffer.")
            return
        }
        sourceData.withUnsafeBufferPointer { pointer in
            if let base = pointer.baseAddress {
                sourceBuffer.contents().copyMemory(
                    from: base,
                    byteCount: pixelCount * 3 * MemoryLayout<Float>.size
                )
            }
        }

        guard let destination = entry.texturePool.acquire(
            width: width, height: height, pixelFormat: .rgba32Float, storageMode: .shared
        ) else {
            Issue.record("Could not allocate destination texture.")
            return
        }

        let commandQueue = try makeCommandQueue(entry: entry)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Issue.record("Could not create command buffer.")
            return
        }
        try RenderStages.writeForegroundBufferToTexture(
            buffer: sourceBuffer,
            destination: destination.texture,
            entry: entry,
            commandBuffer: commandBuffer
        )
        try commitAndWait(commandBuffer)

        var result = [SIMD4<Float>](repeating: .zero, count: pixelCount)
        result.withUnsafeMutableBufferPointer { pointer in
            if let base = pointer.baseAddress {
                destination.texture.getBytes(
                    base,
                    bytesPerRow: width * MemoryLayout<SIMD4<Float>>.size,
                    from: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0
                )
            }
        }
        destination.returnManually()

        // y-flipped: output row 0 came from input row 3 = (0.3, 0.6, 0.9, 1.0).
        for x in 0..<width {
            let pixel = result[0 * width + x]
            #expect(abs(pixel.x - 0.3) < 1e-3)
            #expect(abs(pixel.y - 0.6) < 1e-3)
            #expect(abs(pixel.z - 0.9) < 1e-3)
            #expect(abs(pixel.w - 1.0) < 1e-3, "Alpha should be 1.0, got \(pixel.w)")
        }
        for x in 0..<width {
            let pixel = result[3 * width + x]
            #expect(abs(pixel.x - 0.0) < 1e-3)
            #expect(abs(pixel.y - 0.0) < 1e-3)
            #expect(abs(pixel.z - 0.0) < 1e-3)
            #expect(abs(pixel.w - 1.0) < 1e-3)
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
