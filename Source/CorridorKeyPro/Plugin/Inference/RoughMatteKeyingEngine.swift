//
//  RoughMatteKeyingEngine.swift
//  Corridor Key Pro
//
//  Fallback engine used when no neural model artefact is bundled (or when MLX
//  fails to load). Produces a simple `saturate((G - max(R, B)) * 2.5)` matte on
//  the GPU and copies the normalised input forward as the foreground texture
//  so the rest of the pipeline always has something to work with.
//

import Foundation
import Metal

final class RoughMatteKeyingEngine: KeyingInferenceEngine, @unchecked Sendable {
    let backendDisplayName: String
    var guideSourceDescription: String = "Green-channel fallback"

    private let cacheEntry: MetalDeviceCacheEntry

    init(cacheEntry: MetalDeviceCacheEntry) {
        self.cacheEntry = cacheEntry
        self.backendDisplayName = "Rough Matte (Metal) on \(cacheEntry.device.name)"
    }

    func supports(resolution: Int) -> Bool { true }

    func prepare(resolution: Int) async throws {
        // Nothing to prepare; all work happens per-frame on the GPU.
    }

    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws {
        guard let commandQueue = cacheEntry.borrowCommandQueue() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        defer { cacheEntry.returnCommandQueue(commandQueue) }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KeyingInferenceError.deviceUnavailable
        }
        commandBuffer.label = "Corridor Key Pro Rough Matte"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KeyingInferenceError.deviceUnavailable
        }

        // The normalised tensor texture holds (R, G, B, hint) where hint is in
        // the alpha channel. Re-reading with the rough-matte kernel produces a
        // crude alpha from `max(G - max(R, B), 0) * 2.5`.
        encoder.setComputePipelineState(cacheEntry.computePipelines.roughMatte)
        encoder.setTexture(request.normalisedInputTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.alphaTexture, index: Int(CKTextureIndexOutput.rawValue))
        dispatchThreads(encoder: encoder, pipeline: cacheEntry.computePipelines.roughMatte, width: output.alphaTexture.width, height: output.alphaTexture.height)

        encoder.setComputePipelineState(cacheEntry.computePipelines.resample)
        encoder.setTexture(request.normalisedInputTexture, index: Int(CKTextureIndexSource.rawValue))
        encoder.setTexture(output.foregroundTexture, index: Int(CKTextureIndexOutput.rawValue))
        dispatchThreads(encoder: encoder, pipeline: cacheEntry.computePipelines.resample, width: output.foregroundTexture.width, height: output.foregroundTexture.height)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
    }

    private func dispatchThreads(
        encoder: any MTLComputeCommandEncoder,
        pipeline: any MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        let threadgroupWidth = min(pipeline.threadExecutionWidth, max(width, 1))
        let threadgroupHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / max(threadgroupWidth, 1), max(height, 1)))
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        let threadsPerGrid = MTLSize(width: max(width, 1), height: max(height, 1), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}
