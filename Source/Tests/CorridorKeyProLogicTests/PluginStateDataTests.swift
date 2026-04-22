//
//  PluginStateDataTests.swift
//  CorridorKeyProLogicTests
//
//  Covers the JSON hand-off contract between pluginState and render. FxPlug
//  demands a fresh encoded blob per call, and saved FCP documents survive
//  across plug-in versions, so the decoder must tolerate missing keys that
//  existed in earlier releases (e.g. the removed temporalSmoothing field).
//

import Foundation
import Testing
@testable import CorridorKeyProLogic

@Suite("PluginStateData")
struct PluginStateDataTests {

    @Test("Round-trip preserves every field")
    func roundTripPreservesFields() throws {
        let original = PluginStateData(
            screenColor: .blue,
            qualityMode: .ultra1536,
            sourcePassthroughEnabled: false,
            passthroughErodeNormalized: 12,
            passthroughBlurNormalized: 18,
            alphaBlackPoint: 0.12,
            alphaWhitePoint: 0.88,
            alphaErodeNormalized: -2,
            alphaSoftnessNormalized: 1.5,
            alphaGamma: 1.4,
            autoDespeckleEnabled: true,
            despeckleSize: 850,
            despillStrength: 0.75,
            spillMethod: .doubleLimit,
            outputMode: .foregroundPlusMatte,
            upscaleMethod: .lanczos,
            renderQualityLevel: 3,
            longEdgeBaseline: 1920,
            destinationLongEdgePixels: 3840
        )
        let encoded = try original.encodedForHost()
        let decoded = PluginStateData.decoded(from: encoded)

        #expect(decoded.screenColor == .blue)
        #expect(decoded.qualityMode == .ultra1536)
        #expect(decoded.sourcePassthroughEnabled == false)
        #expect(decoded.passthroughErodeNormalized == 12)
        #expect(decoded.passthroughBlurNormalized == 18)
        #expect(decoded.alphaBlackPoint == 0.12)
        #expect(decoded.alphaWhitePoint == 0.88)
        #expect(decoded.alphaErodeNormalized == -2)
        #expect(decoded.alphaSoftnessNormalized == 1.5)
        #expect(decoded.alphaGamma == 1.4)
        #expect(decoded.autoDespeckleEnabled == true)
        #expect(decoded.despeckleSize == 850)
        #expect(decoded.despillStrength == 0.75)
        #expect(decoded.spillMethod == .doubleLimit)
        #expect(decoded.outputMode == .foregroundPlusMatte)
        #expect(decoded.upscaleMethod == .lanczos)
        #expect(decoded.renderQualityLevel == 3)
        #expect(decoded.longEdgeBaseline == 1920)
        #expect(decoded.destinationLongEdgePixels == 3840)
    }

    @Test("Empty blob falls back to defaults")
    func emptyBlobDecodesDefaults() {
        let decoded = PluginStateData.decoded(from: NSData())
        let defaults = PluginStateData()
        #expect(decoded.qualityMode == defaults.qualityMode)
        #expect(decoded.screenColor == defaults.screenColor)
        #expect(decoded.despillStrength == defaults.despillStrength)
    }

    @Test("Legacy documents with a temporal-smoothing key still decode")
    func legacyTemporalSmoothingKeyIgnored() {
        // Simulate a project file that was saved when the Temporal Smoothing
        // slider still existed. The decoder must skip unknown keys gracefully
        // rather than throwing and forcing the host back to all defaults.
        let legacyJSON = """
        {
          "screenColor": 0,
          "qualityMode": 0,
          "sourcePassthroughEnabled": true,
          "passthroughErodeNormalized": 3,
          "passthroughBlurNormalized": 7,
          "alphaBlackPoint": 0,
          "alphaWhitePoint": 1,
          "alphaErodeNormalized": 0,
          "alphaSoftnessNormalized": 0,
          "alphaGamma": 1,
          "autoDespeckleEnabled": false,
          "despeckleSize": 400,
          "despillStrength": 0.5,
          "spillMethod": 0,
          "outputMode": 0,
          "temporalSmoothing": 0.42,
          "upscaleMethod": 0,
          "renderQualityLevel": 2,
          "longEdgeBaseline": 1920,
          "destinationLongEdgePixels": 1920
        }
        """
        let data = NSData(data: Data(legacyJSON.utf8))
        let decoded = PluginStateData.decoded(from: data)
        #expect(decoded.despillStrength == 0.5)
        #expect(decoded.despeckleSize == 400)
    }

    @Test("destinationPixelRadius scales by clip size")
    func destinationPixelRadiusScales() {
        var state = PluginStateData()
        state.longEdgeBaseline = 1920
        state.destinationLongEdgePixels = 3840
        // 1920 baseline → 10-unit control yields 20px on a 4K clip.
        #expect(abs(state.destinationPixelRadius(fromNormalized: 10) - 20) < 0.001)

        state.destinationLongEdgePixels = 1920
        #expect(abs(state.destinationPixelRadius(fromNormalized: 10) - 10) < 0.001)
    }
}
