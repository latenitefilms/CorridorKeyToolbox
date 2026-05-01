//
//  KeyingInferenceEngine.swift
//  CorridorKey by LateNite
//
//  Abstraction over whichever neural backend is producing mattes. The render
//  pipeline is written against this protocol so we can swap CoreML, MLX, or a
//  pure-GPU rough-matte fallback without touching the FxPlug integration.
//

import Foundation
import Metal
import CoreMedia
#if CORRIDOR_KEY_SPM_MIRROR
import CorridorKeyToolboxLogic
#endif

/// Input bundle handed to an inference engine. The normalised tensor is
/// provided as a raw `MTLBuffer` rather than a texture so MLX can read it
/// via `MLXArray(rawPointer:)` without a CPU copy — this eliminates the
/// ~67 MB of per-frame CPU↔GPU round-tripping the pre-v1.0 texture-based
/// path paid at 2048² inference resolution.
///
/// GPU-only fallbacks (like the rough-matte engine) read from
/// `rawSourceTexture` instead, so the `normalisedInputBuffer` can stay
/// entirely in a format the neural bridge owns.
struct KeyingInferenceRequest {
    /// NHWC-packed tensor (`[1, rung, rung, 4]`) — the first three
    /// channels are ImageNet-normalised RGB, the fourth is the alpha hint.
    /// Element dtype matches `inputPrecision`: 4 bytes per scalar for
    /// fp32, 2 bytes for fp16. Storage mode is `.shared` so MLX can
    /// alias the bytes via `init(rawPointer:)`.
    let normalisedInputBuffer: any MTLBuffer
    /// Raw RGB source at inference resolution, in 0..1 linear/sRGB space.
    /// Any GPU engine that doesn't need the neural network should read
    /// from this texture instead of the buffer.
    let rawSourceTexture: any MTLTexture
    /// Square side length used for inference.
    let inferenceResolution: Int
    /// Precision the loaded `.mlxfn` bridge expects on its inputs and
    /// outputs. The pre-inference normalise kernel writes the buffer
    /// in this dtype; the writeback kernel reads MLX's outputs in the
    /// same dtype. Defaults to `.float32` for back-compat with callers
    /// that haven't been precision-plumbed yet.
    let inputPrecision: BridgePrecision

    init(
        normalisedInputBuffer: any MTLBuffer,
        rawSourceTexture: any MTLTexture,
        inferenceResolution: Int,
        inputPrecision: BridgePrecision = .float32
    ) {
        self.normalisedInputBuffer = normalisedInputBuffer
        self.rawSourceTexture = rawSourceTexture
        self.inferenceResolution = inferenceResolution
        self.inputPrecision = inputPrecision
    }
}

/// Textures the engine writes into. Both are created by the caller at the
/// requested inference resolution.
struct KeyingInferenceOutput {
    let alphaTexture: any MTLTexture
    let foregroundTexture: any MTLTexture
}

/// Error surface returned when inference cannot proceed.
enum KeyingInferenceError: Error, CustomStringConvertible {
    case modelUnavailable(String)
    case unsupportedResolution(Int)
    case deviceUnavailable

    var description: String {
        switch self {
        case .modelUnavailable(let detail):
            return "Keying model is unavailable: \(detail)"
        case .unsupportedResolution(let resolution):
            return "This build does not include a model tuned for \(resolution)px inference."
        case .deviceUnavailable:
            return "A Metal device capable of running the Corridor Key model was not found."
        }
    }
}

/// Minimal surface the orchestrator exercises. Concrete engines own their
/// session lifetime (model load, warm-up) but must be thread-safe because
/// Final Cut Pro may render multiple frames simultaneously.
///
/// `screenColor` is part of `supports` / `prepare` because each colour
/// ships its own MLX bridge file — Green and Blue are not interchangeable
/// at the engine level, even when the rung matches.
protocol KeyingInferenceEngine: AnyObject, Sendable {
    var backendDisplayName: String { get }
    var guideSourceDescription: String { get set }
    func supports(resolution: Int, screenColor: ScreenColor) -> Bool
    func prepare(resolution: Int, screenColor: ScreenColor) async throws
    func run(request: KeyingInferenceRequest, output: KeyingInferenceOutput) throws
}
