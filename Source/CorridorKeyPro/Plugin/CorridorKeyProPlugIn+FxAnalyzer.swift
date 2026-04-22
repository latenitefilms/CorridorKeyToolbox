//
//  CorridorKeyProPlugIn+FxAnalyzer.swift
//  Corridor Key Pro
//
//  Implements the `FxAnalyzer` protocol so the plug-in can pre-compute
//  per-frame statistics – primarily the screen-colour reference – across the
//  whole clip before rendering. Mirrors the pattern shown in Apple's
//  `FxBrightnessAnalysis` sample but tailored to our needs: we store a
//  `SIMD3<Float>` reference plus a difficulty score per frame so the renderer
//  can pick a conservative quality rung when the signal is shaky.
//

import Foundation
import CoreMedia
import simd

extension CorridorKeyProPlugIn {

    // MARK: - Analysis time range

    @objc(desiredAnalysisTimeRange:forInputWithTimeRange:error:)
    func desiredAnalysisTimeRange(
        _ desiredRange: UnsafeMutablePointer<CMTimeRange>,
        forInputWithTimeRange inputTimeRange: CMTimeRange
    ) throws {
        // Analyse the entire clip the user applied the effect to.
        desiredRange.pointee = inputTimeRange
        PluginLog.debug("Analyser requested full input time range.")
    }

    @objc(setupAnalysisForTimeRange:frameDuration:error:)
    func setupAnalysis(
        for analysisRange: CMTimeRange,
        frameDuration: CMTime
    ) throws {
        analysisLock.lock()
        analyzedFrames.removeAll(keepingCapacity: true)
        analysisFrameDuration = frameDuration
        analysisLock.unlock()
        PluginLog.debug("Analyser set up for range of \(CMTimeGetSeconds(analysisRange.duration)) seconds.")
    }

    // MARK: - Per-frame analysis

    @objc(analyzeFrame:atTime:error:)
    func analyzeFrame(
        _ frame: FxImageTile,
        atTime frameTime: CMTime
    ) throws {
        // For the initial implementation we accumulate the screen reference
        // using the CorridorKey canonical value per screen colour. When the
        // GPU-side estimator is enabled we'll replace this with an async
        // readback of a downsampled patch.
        let reference = SIMD3<Float>(0.08, 0.84, 0.08)
        let difficulty = 0.0

        analysisLock.lock()
        analyzedFrames.append(
            AnalyzedFrame(
                frameTime: frameTime,
                screenReference: reference,
                estimatedDifficulty: difficulty
            )
        )
        analysisLock.unlock()
    }

    @objc(cleanupAnalysis:)
    func cleanupAnalysis() throws {
        analysisLock.lock()
        let snapshot = analyzedFrames
        analysisLock.unlock()

        PluginLog.debug("Analyser completed with \(snapshot.count) frames.")

        guard let setting = apiManager.api(for: FxParameterSettingAPI_v6.self) as? any FxParameterSettingAPI_v6 else {
            return
        }

        if let encoded = try? JSONEncoder().encode(snapshot),
           let string = String(data: encoded, encoding: .utf8),
           string.count < 1_000_000 {
            setting.setStringParameterValue(
                "Analysis complete (\(snapshot.count) frames)",
                toParameter: ParameterIdentifier.statusGuideSource
            )
            _ = string
        }
    }
}
