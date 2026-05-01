//
//  ScreenColorEstimator.swift
//  CorridorKey by LateNite
//
//  Historically produced a 3x3 matrix that rotated blue-screen footage into
//  the green domain so a single green-trained MLX model could process it,
//  with an inverse matrix to rotate the foreground back at the end. With
//  Corridor Digital's v1.0 dedicated blue model now bundled, both screen
//  colours have a native bridge — we ship `corridorkey_mlx_bridge_*.mlxfn`
//  for green and `corridorkeyblue_mlx_bridge_*.mlxfn` for blue, picked by
//  `MLXBridgeArtifact.filename(forResolution:screenColor:)`.
//
//  Consequence: this file now always returns `.identity`. Pre-inference no
//  longer has to rotate, the model receives footage in its native screen
//  domain (which is what it was trained on), and the despill / edge-decontam
//  kernels are passed the actual screen reference via
//  `ScreenColor.canonicalScreenReference` rather than always seeing
//  canonical green.
//
//  The struct is preserved (rather than ripped out) because every render
//  helper takes a `ScreenColorTransform` parameter and ignores it when
//  `isIdentity == true`. Keeping the surface stable means this change is
//  an isolated patch instead of a render-pipeline-wide rewrite.
//

import Foundation
import Metal
import simd

struct ScreenColorTransform: Sendable {
    let forwardMatrix: simd_float3x3
    let inverseMatrix: simd_float3x3
    let isIdentity: Bool
    let estimatedScreenReference: SIMD3<Float>

    static let identity = ScreenColorTransform(
        forwardMatrix: matrix_identity_float3x3,
        inverseMatrix: matrix_identity_float3x3,
        isIdentity: true,
        estimatedScreenReference: SIMD3<Float>(0.08, 0.84, 0.08)
    )
}

enum ScreenColorEstimator {
    /// Returns the transform appropriate for a screen colour without
    /// touching the GPU. Always identity now that both green and blue
    /// ship dedicated MLX bridges — the per-colour reference goes to
    /// despill / edge decontamination via
    /// `ScreenColor.canonicalScreenReference` instead.
    ///
    /// `estimatedScreenReference` carries the canonical reference for
    /// the chosen colour so kernels that previously read it from the
    /// transform (light wrap blends, edge decontamination) still see
    /// the right basis colour.
    static func defaultTransform(for screenColor: ScreenColor) -> ScreenColorTransform {
        ScreenColorTransform(
            forwardMatrix: matrix_identity_float3x3,
            inverseMatrix: matrix_identity_float3x3,
            isIdentity: true,
            estimatedScreenReference: screenColor.canonicalScreenReference
        )
    }
}
