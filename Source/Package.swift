// swift-tools-version: 6.0
//
//  Package.swift
//  Corridor Key Toolbox
//
//  Headless regression suite for the plug-in's pure-Swift logic. The Xcode
//  project remains the build system of record for the FxPlug target itself;
//  this Swift Package lets us unit-test the pieces that don't depend on FxPlug
//  so CI and local developer loops can catch regressions without launching
//  Final Cut Pro.
//
//  Run with `swift test` from this directory.
//

import PackageDescription

let package = Package(
    name: "CorridorKeyToolboxLogic",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(
            name: "CorridorKeyToolboxLogic",
            targets: ["CorridorKeyToolboxLogic"]
        )
    ],
    targets: [
        .target(
            name: "CorridorKeyToolboxLogic",
            path: "CorridorKeyToolbox/Plugin",
            // Keep the FxPlug-dependent files, the Xcode-managed resources,
            // and the shaders out of the SPM build — they only compile inside
            // the Xcode target.
            exclude: [
                "Info.plist",
                "CorridorKeyToolbox.entitlements",
                "PluginLog.swift",
                "main.swift",
                "CorridorKeyToolboxPlugIn.swift",
                "CorridorKeyToolboxPlugIn+Properties.swift",
                "CorridorKeyToolboxPlugIn+Render.swift",
                "CorridorKeyToolboxPlugIn+RenderRects.swift",
                "CorridorKeyToolboxPlugIn+PluginState.swift",
                "CorridorKeyToolboxPlugIn+FxAnalyzer.swift",
                "Parameters/CorridorKeyToolboxPlugIn+Parameters.swift",
                "Parameters/ParameterIdentifiers.swift",
                "Inference/InferenceCoordinator.swift",
                "Inference/KeyingInferenceEngine.swift",
                "Inference/MLXKeyingEngine.swift",
                "Inference/RoughMatteKeyingEngine.swift",
                "Inspector/CorridorKeyToolboxPlugIn+CustomViews.swift",
                "Inspector/CorridorKeyInspectorBridge.swift",
                "Inspector/CorridorKeyHeaderView.swift",
                "Render",
                "Metal",
                "Resources",
                "Shared"
            ],
            sources: [
                "Parameters/ParameterEnumerations.swift",
                "Parameters/PluginStateData.swift",
                "PostProcess/ScreenColorEstimator.swift",
                "Inference/AnalysisData.swift",
                "Inference/MatteCodec.swift",
                "Inspector/CorridorKeyAnalysisSnapshot.swift"
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
        .testTarget(
            name: "CorridorKeyToolboxLogicTests",
            dependencies: ["CorridorKeyToolboxLogic"],
            path: "Tests/CorridorKeyToolboxLogicTests",
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        )
    ]
)
