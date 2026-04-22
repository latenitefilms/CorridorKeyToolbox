//
//  CorridorKeyProPlugIn+Parameters.swift
//  Corridor Key Pro
//
//  Defines the entire inspector layout for the plug-in. The grouping and
//  defaults intentionally match the CorridorKey OFX panel so that an editor
//  moving between DaVinci Resolve and Final Cut Pro feels at home.
//

import Foundation
import AppKit
import CoreMedia

extension CorridorKeyProPlugIn {

    /// Registers every control FxPlug should draw in Final Cut Pro's inspector.
    /// Called once by FxPlug for each new instance.
    @objc(addParametersWithError:)
    func addParameters() throws {
        guard let create = apiManager.api(for: (any FxParameterCreationAPI_v5).self) as? any FxParameterCreationAPI_v5 else {
            throw NSError(
                domain: FxPlugErrorDomain,
                code: kFxError_APIUnavailable,
                userInfo: [NSLocalizedDescriptionKey: "Corridor Key Pro could not attach to the FxPlug parameter API."]
            )
        }

        try addKeySetupGroup(create: create)
        try addAnalysisGroup(create: create)
        try addInteriorDetailGroup(create: create)
        try addMatteGroup(create: create)
        try addEdgeAndSpillGroup(create: create)
        try addOutputGroup(create: create)
        try addPerformanceGroup(create: create)
        try addHelpGroup(create: create)
        PluginLog.notice("Parameters registered with Final Cut Pro.")
    }

    // MARK: - Custom parameter serialisation

    /// Tells FxPlug which Foundation classes may appear inside our hidden
    /// custom parameter so the host can safely decode it through
    /// `NSSecureCoding`. The analysis dictionary is a tree of NSDictionary /
    /// NSString / NSNumber / NSData nodes — listing any additional classes
    /// here would open the plug-in to decoding attacker-supplied payloads.
    func classes(forCustomParameterID parameterID: UInt32) -> Set<AnyHashable> {
        switch parameterID {
        case ParameterIdentifier.analysisData:
            let classes: [AnyClass] = [
                NSDictionary.self,
                NSString.self,
                NSNumber.self,
                NSData.self
            ]
            let set = NSSet(array: classes)
            return (set as? Set<AnyHashable>) ?? []
        default:
            return []
        }
    }

    // MARK: - Groups

    private func addKeySetupGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Key Setup",
            parameterID: ParameterIdentifier.keySetupGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Screen Colour",
            parameterID: ParameterIdentifier.screenColor,
            defaultValue: UInt32(ScreenColor.green.rawValue),
            menuEntries: ScreenColor.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Quality",
            parameterID: ParameterIdentifier.qualityMode,
            defaultValue: UInt32(QualityMode.automatic.rawValue),
            menuEntries: QualityMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addAnalysisGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Analyse",
            parameterID: ParameterIdentifier.analysisGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPushButton(
            withName: "Analyse Clip",
            parameterID: ParameterIdentifier.analyzeClip,
            selector: #selector(handleAnalyzeClip),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPushButton(
            withName: "Reset Analysis",
            parameterID: ParameterIdentifier.resetAnalysis,
            selector: #selector(handleResetAnalysis),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        let hiddenFlags = CorridorKeyParameterFlags(
            rawValue: CorridorKeyParameterFlags.hidden.rawValue |
                      CorridorKeyParameterFlags.notAnimatable.rawValue
        )
        create.addCustomParameter(
            withName: "Analysis Data",
            parameterID: ParameterIdentifier.analysisData,
            defaultValue: NSDictionary(),
            parameterFlags: hiddenFlags.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addInteriorDetailGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Interior Detail",
            parameterID: ParameterIdentifier.interiorGroup,
            parameterFlags: [CorridorKeyParameterFlags.default].reduce(into: CorridorKeyParameterFlags()) { $0.insert($1) }.fxFlags
        )

        create.addToggleButton(
            withName: "Source Passthrough",
            parameterID: ParameterIdentifier.sourcePassthrough,
            defaultValue: true,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Edge Erode",
            parameterID: ParameterIdentifier.passthroughErode,
            defaultValue: 3,
            parameterMin: 0,
            parameterMax: 100,
            sliderMin: 0,
            sliderMax: 40,
            delta: 1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Edge Blur",
            parameterID: ParameterIdentifier.passthroughBlur,
            defaultValue: 7,
            parameterMin: 0,
            parameterMax: 100,
            sliderMin: 0,
            sliderMax: 40,
            delta: 1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addMatteGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Matte",
            parameterID: ParameterIdentifier.matteGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Black Point",
            parameterID: ParameterIdentifier.alphaBlackPoint,
            defaultValue: 0,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "White Point",
            parameterID: ParameterIdentifier.alphaWhitePoint,
            defaultValue: 1,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Matte Erode",
            parameterID: ParameterIdentifier.alphaErode,
            defaultValue: 0,
            parameterMin: -10,
            parameterMax: 10,
            sliderMin: -10,
            sliderMax: 10,
            delta: 0.1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Softness",
            parameterID: ParameterIdentifier.alphaSoftness,
            defaultValue: 0,
            parameterMin: 0,
            parameterMax: 5,
            sliderMin: 0,
            sliderMax: 5,
            delta: 0.1,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Gamma",
            parameterID: ParameterIdentifier.alphaGamma,
            defaultValue: 1,
            parameterMin: 0.1,
            parameterMax: 10,
            sliderMin: 0.1,
            sliderMax: 4,
            delta: 0.05,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addToggleButton(
            withName: "Auto Despeckle",
            parameterID: ParameterIdentifier.autoDespeckle,
            defaultValue: false,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addIntSlider(
            withName: "Despeckle Size",
            parameterID: ParameterIdentifier.despeckleSize,
            defaultValue: 400,
            parameterMin: 50,
            parameterMax: 2000,
            sliderMin: 50,
            sliderMax: 2000,
            delta: 10,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addEdgeAndSpillGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Edge & Spill",
            parameterID: ParameterIdentifier.edgeSpillGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addFloatSlider(
            withName: "Despill Strength",
            parameterID: ParameterIdentifier.despillStrength,
            defaultValue: 0.5,
            parameterMin: 0,
            parameterMax: 1,
            sliderMin: 0,
            sliderMax: 1,
            delta: 0.01,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Spill Method",
            parameterID: ParameterIdentifier.spillMethod,
            defaultValue: UInt32(SpillMethod.average.rawValue),
            menuEntries: SpillMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addOutputGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Output",
            parameterID: ParameterIdentifier.outputGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPopupMenu(
            withName: "Output",
            parameterID: ParameterIdentifier.outputMode,
            defaultValue: UInt32(OutputMode.processed.rawValue),
            menuEntries: OutputMode.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addPerformanceGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Performance",
            parameterID: ParameterIdentifier.performanceGroup,
            parameterFlags: [CorridorKeyParameterFlags.default].reduce(into: CorridorKeyParameterFlags()) { $0.insert($1) }.fxFlags
        )

        create.addPopupMenu(
            withName: "Upscale Method",
            parameterID: ParameterIdentifier.upscaleMethod,
            defaultValue: UInt32(UpscaleMethod.bilinear.rawValue),
            menuEntries: UpscaleMethod.allCases.map(\.displayName),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    private func addHelpGroup(create: any FxParameterCreationAPI_v5) throws {
        create.startParameterSubGroup(
            "Help",
            parameterID: ParameterIdentifier.helpGroup,
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.addPushButton(
            withName: "Open User Guide",
            parameterID: ParameterIdentifier.openUserGuide,
            selector: #selector(handleOpenUserGuide),
            parameterFlags: CorridorKeyParameterFlags.default.fxFlags
        )

        create.endParameterSubGroup()
    }

    // MARK: - Callbacks

    @objc func handleOpenUserGuide() {
        guard let url = URL(string: "https://corridordigital.com/corridor-key-pro") else { return }
        NSWorkspace.shared.open(url)
    }

    /// Inspector button that kicks off a forward-analysis pass. The host
    /// drives the per-frame callbacks defined in `CorridorKeyProPlugIn+FxAnalyzer`.
    @objc func handleAnalyzeClip() {
        startForwardAnalysisPass()
    }

    /// Inspector button that clears the persisted analysis cache so the next
    /// "Analyse Clip" invocation starts from scratch.
    @objc func handleResetAnalysis() {
        clearAnalysisCache()
    }

    // MARK: - Parameter change notifications

    /// Final Cut Pro invokes this method when the user edits a control. We
    /// don't need to react per-parameter — the host will re-request the
    /// plug-in state and re-render automatically — but having it defined
    /// ensures FCP correctly invalidates any cached render when a control
    /// changes.
    @objc(parameterChanged:atTime:error:)
    func parameterChanged(_ paramID: UInt32, atTime time: CMTime) throws {
        // Intentionally empty. The host re-requests plug-in state and
        // triggers a re-render automatically when parameters change.
    }
}
