//
//  CorridorKeyProPlugIn+Render.swift
//  Corridor Key Pro
//
//  Hooks FxPlug's per-tile render callback into the Corridor Key render
//  pipeline. Final Cut Pro invokes this method on a background thread; the
//  render pipeline itself owns all Metal state internally.
//

import Foundation
import CoreMedia
import QuartzCore

extension CorridorKeyProPlugIn {

    @objc(renderDestinationImage:sourceImages:pluginState:atTime:error:)
    func renderDestinationImage(
        _ destinationImage: FxImageTile,
        sourceImages: [FxImageTile],
        pluginState: Data?,
        atTime renderTime: CMTime
    ) throws {
        guard let sourceImage = sourceImages.first else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_InvalidParameter,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Pro requires a source input."]
            )
        }

        var state = PluginStateData.decoded(from: pluginState.map { NSData(data: $0) })
        let width = Int(destinationImage.imagePixelBounds.right - destinationImage.imagePixelBounds.left)
        let height = Int(destinationImage.imagePixelBounds.top - destinationImage.imagePixelBounds.bottom)
        state.destinationLongEdgePixels = max(width, height)

        let alphaHint = sourceImages.count > 1 ? sourceImages[1] : nil
        let request = RenderRequest(
            destinationImage: destinationImage,
            sourceImage: sourceImage,
            alphaHintImage: alphaHint,
            state: state,
            renderTime: renderTime
        )

        let startTime = CACurrentMediaTime()
        PluginLog.debug("Render begin for tile \(destinationImage.tilePixelBounds.left),\(destinationImage.tilePixelBounds.bottom) — \(destinationImage.tilePixelBounds.right - destinationImage.tilePixelBounds.left)×\(destinationImage.tilePixelBounds.top - destinationImage.tilePixelBounds.bottom).")
        let report: RenderReport
        do {
            report = try renderPipeline.render(request)
        } catch {
            PluginLog.error("Render failed at \(CMTimeGetSeconds(renderTime))s: \(error.localizedDescription)")
            throw error
        }
        let elapsedMilliseconds = (CACurrentMediaTime() - startTime) * 1000
        lastFrameMilliseconds.set(elapsedMilliseconds)

        publishRuntimeStatus(for: state, report: report, elapsedMilliseconds: elapsedMilliseconds)
    }

    /// Updates the read-only status parameters so the user can see which
    /// backend is active and how fast the frame ran.
    private func publishRuntimeStatus(
        for state: PluginStateData,
        report: RenderReport,
        elapsedMilliseconds: Double
    ) {
        guard let setting = apiManager.api(for: FxParameterSettingAPI_v6.self) as? any FxParameterSettingAPI_v6 else {
            return
        }
        setting.setStringParameterValue(
            report.backendDescription,
            toParameter: ParameterIdentifier.statusBackend
        )
        setting.setStringParameterValue(
            "\(report.effectiveInferenceResolution)px",
            toParameter: ParameterIdentifier.statusEffectiveQuality
        )
        setting.setStringParameterValue(
            report.guideSourceDescription,
            toParameter: ParameterIdentifier.statusGuideSource
        )
        setting.setStringParameterValue(
            elapsedMilliseconds.formatted(.number.precision(.fractionLength(1))) + " ms",
            toParameter: ParameterIdentifier.statusLastFrameMs
        )
        setting.setStringParameterValue(
            report.deviceName,
            toParameter: ParameterIdentifier.statusDevice
        )
    }
}
