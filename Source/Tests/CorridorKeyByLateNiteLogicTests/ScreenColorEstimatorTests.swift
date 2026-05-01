//
//  ScreenColorEstimatorTests.swift
//  CorridorKeyToolboxLogicTests
//
//  Both green and blue now ship dedicated MLX bridges, so the renderer
//  feeds source frames to inference in their native screen domain — no
//  rotation hop in either direction. These tests pin that contract: the
//  estimator returns identity for both colours, but exposes the right
//  `estimatedScreenReference` so the despill / edge decontaminate
//  kernels know which channel carries the spill.
//

import Testing
import simd
@testable import CorridorKeyToolboxLogic

@Suite("ScreenColorEstimator")
struct ScreenColorEstimatorTests {

    @Test("Green returns the identity transform")
    func greenIsIdentity() {
        let transform = ScreenColorEstimator.defaultTransform(for: .green)
        #expect(transform.isIdentity == true)
        expectMatricesEqual(transform.forwardMatrix, matrix_identity_float3x3)
        expectMatricesEqual(transform.inverseMatrix, matrix_identity_float3x3)
        expectVectorsEqual(transform.estimatedScreenReference, ScreenColor.green.canonicalScreenReference)
    }

    @Test("Blue returns the identity transform now that a native bridge ships")
    func blueIsIdentity() {
        let transform = ScreenColorEstimator.defaultTransform(for: .blue)
        #expect(transform.isIdentity == true)
        expectMatricesEqual(transform.forwardMatrix, matrix_identity_float3x3)
        expectMatricesEqual(transform.inverseMatrix, matrix_identity_float3x3)
    }

    @Test("Blue exposes the canonical blue screen reference for kernels")
    func blueExposesCanonicalReference() {
        let transform = ScreenColorEstimator.defaultTransform(for: .blue)
        expectVectorsEqual(transform.estimatedScreenReference, SIMD3<Float>(0.08, 0.16, 0.84))
    }

    // MARK: - Helpers

    private func expectMatricesEqual(
        _ lhs: simd_float3x3,
        _ rhs: simd_float3x3,
        tolerance: Float = 1e-5
    ) {
        for column in 0..<3 {
            for row in 0..<3 {
                let delta = abs(lhs[column][row] - rhs[column][row])
                #expect(delta < tolerance, "Mismatch at [\(column)][\(row)]: \(lhs[column][row]) vs \(rhs[column][row])")
            }
        }
    }

    private func expectVectorsEqual(
        _ lhs: SIMD3<Float>,
        _ rhs: SIMD3<Float>,
        tolerance: Float = 1e-5
    ) {
        for index in 0..<3 {
            let delta = abs(lhs[index] - rhs[index])
            #expect(delta < tolerance, "Mismatch at [\(index)]: \(lhs[index]) vs \(rhs[index])")
        }
    }
}
