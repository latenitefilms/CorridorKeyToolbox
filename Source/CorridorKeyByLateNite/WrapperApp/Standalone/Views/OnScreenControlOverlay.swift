//
//  OnScreenControlOverlay.swift
//  CorridorKey by LateNite — Standalone Editor
//
//  Mirrors Final Cut Pro's on-screen control for the Corridor Key
//  effect: a click target that lets the user drop foreground or
//  background hint dots onto the preview, plus a render layer that
//  draws the dots over the keyed image. The hint set is shared with
//  the FxPlug renderer via `PluginStateData.hintPointSet`, so the
//  same dots flow into MLX's 4th input channel during analysis.
//
//  The overlay sits inside the same `ZStack` as the `MetalPreviewView`
//  and matches the renderer's aspect-fit rectangle so click positions
//  resolve correctly regardless of inspector / window resizing.
//
//  The overlay's click target accepts a `contextMenu` view builder
//  so right-click works even while a hint tool is active. Without
//  that the overlay's hit testing swallowed right-clicks before the
//  underlying `MetalPreviewView`'s context menu could see them, and
//  the user couldn't change their preview backdrop without first
//  switching the tool back to Off.
//

import SwiftUI

struct OnScreenControlOverlay<MenuContent: View>: View {
    @Bindable var viewModel: EditorViewModel
    /// Logical render size of the loaded clip — used so the click
    /// target sits over the same letterboxed quad the preview shader
    /// draws.
    let renderSize: CGSize
    /// Right-click context menu attached to the click target. Sharing
    /// the editor's preview-backdrop menu here keeps the right-click
    /// behaviour consistent whether the user is dropping hints or
    /// just scrubbing.
    @ViewBuilder let contextMenu: () -> MenuContent

    var body: some View {
        GeometryReader { proxy in
            let fittedRect = aspectFittedRect(for: renderSize, in: proxy.size)
            ZStack(alignment: .topLeading) {
                // Click target — only intercepts hits when an OSC
                // tool is active so the rest of the time the user can
                // still drag the parent window from the preview area.
                if viewModel.oscTool != .disabled {
                    Rectangle()
                        .fill(Color.white.opacity(0.0001))
                        .frame(width: fittedRect.width, height: fittedRect.height)
                        .position(x: fittedRect.midX, y: fittedRect.midY)
                        .contentShape(.rect)
                        .onTapGesture { location in
                            let normalisedX = (location.x - fittedRect.minX) / fittedRect.width
                            let normalisedY = (location.y - fittedRect.minY) / fittedRect.height
                            viewModel.handleOSCClick(
                                atNormalizedPoint: CGPoint(x: normalisedX, y: normalisedY)
                            )
                        }
                        .contextMenu { contextMenu() }
                        .help(toolHint)
                }

                ForEach(Array(viewModel.state.hintPointSet.points.enumerated()), id: \.offset) { item in
                    HintPointMarker(
                        point: item.element,
                        screenColor: viewModel.state.screenColor
                    )
                    .position(
                        x: fittedRect.minX + CGFloat(item.element.x) * fittedRect.width,
                        y: fittedRect.minY + CGFloat(item.element.y) * fittedRect.height
                    )
                }
            }
            .allowsHitTesting(viewModel.oscTool != .disabled)
        }
    }

    private var toolHint: String {
        switch viewModel.oscTool {
        case .disabled: return ""
        case .foregroundHint: return "Click to drop a foreground hint."
        case .backgroundHint: return "Click to drop a background hint."
        case .eraseHint: return "Click near a hint to remove it."
        }
    }

    private func aspectFittedRect(for content: CGSize, in container: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0,
              container.width > 0, container.height > 0
        else { return CGRect(origin: .zero, size: container) }
        let scale = min(container.width / content.width, container.height / content.height)
        let fittedSize = CGSize(width: content.width * scale, height: content.height * scale)
        let originX = (container.width - fittedSize.width) * 0.5
        let originY = (container.height - fittedSize.height) * 0.5
        return CGRect(origin: CGPoint(x: originX, y: originY), size: fittedSize)
    }
}

/// Visual marker for a single hint point. Foreground points show a
/// person silhouette so the marker reads as "this is the subject" no
/// matter what the screen colour is — earlier builds used a green
/// dot for foreground, which was confusing on a green-screen comp.
/// Background points show a coloured rectangle that picks up the
/// screen colour (green / blue / magenta) so the user can see at a
/// glance which area of the frame they're telling the keyer to drop.
struct HintPointMarker: View {
    let point: HintPoint
    let screenColor: ScreenColor

    var body: some View {
        Group {
            switch point.kind {
            case .foreground:
                foregroundMarker
            case .background:
                backgroundMarker
            }
        }
        .shadow(color: .black.opacity(0.45), radius: 1.8, y: 1)
    }

    /// Foreground marker: a circular badge with a `person.fill`
    /// silhouette inside. The badge fill is a near-black tone so the
    /// white silhouette stays readable against bright matte regions.
    private var foregroundMarker: some View {
        ZStack {
            Circle()
                .fill(Color(red: 0.10, green: 0.12, blue: 0.18))
                .frame(width: 18, height: 18)
            Circle()
                .strokeBorder(Color.white, lineWidth: 1.5)
                .frame(width: 18, height: 18)
            Image(systemName: "person.fill")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .foregroundStyle(.white)
                .frame(width: 10, height: 10)
        }
    }

    /// Background marker: a small rounded square coloured to match
    /// the active screen colour. Wrapped in a black border so the
    /// shape stays legible on a frame whose dominant tone matches
    /// the screen colour itself.
    private var backgroundMarker: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 3)
                .fill(screenColorSwatch)
                .frame(width: 16, height: 14)
            RoundedRectangle(cornerRadius: 3)
                .strokeBorder(Color.black, lineWidth: 1.5)
                .frame(width: 16, height: 14)
        }
    }

    private var screenColorSwatch: Color {
        switch screenColor {
        case .green: return Color(red: 0.20, green: 0.78, blue: 0.30)
        case .blue: return Color(red: 0.20, green: 0.42, blue: 0.95)
        }
    }
}

/// Compact `Menu` that the transport bar embeds next to the Loop
/// toggle. Replaces the earlier floating `OSCToolbar` overlay — the
/// floating overlay was visually competing with the keyed preview
/// and (worse) intercepting right-clicks meant for the backdrop
/// picker. Living in the transport bar gives the controls a stable
/// home next to the rest of the playback chrome.
///
/// Each tool gets a `Cmd-Shift-Letter` shortcut so a power user can
/// switch hint tools without leaving the keyboard. Conventional
/// modifiers were chosen to stay clear of single-letter conflicts
/// with the inspector's text fields and the standard playback
/// shortcuts (Space, Cmd-Z, etc.).
struct SubjectHintsMenu: View {
    @Bindable var viewModel: EditorViewModel

    var body: some View {
        Menu {
            Button("Off", systemImage: "circle.slash") {
                viewModel.oscTool = .disabled
            }
            .keyboardShortcut("o", modifiers: [.command, .shift])

            Button("Mark Foreground", systemImage: "person.fill.badge.plus") {
                viewModel.oscTool = .foregroundHint
            }
            .keyboardShortcut("f", modifiers: [.command, .shift])

            Button("Mark Background", systemImage: "rectangle.fill.badge.minus") {
                viewModel.oscTool = .backgroundHint
            }
            .keyboardShortcut("b", modifiers: [.command, .shift])

            Button("Erase Nearest Hint", systemImage: "eraser") {
                viewModel.oscTool = .eraseHint
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])

            Divider()

            Button("Clear All Hints", systemImage: "trash") {
                viewModel.clearAllHints()
            }
            .disabled(viewModel.state.hintPointSet.isEmpty)
            .keyboardShortcut(.delete, modifiers: [.command, .shift])
        } label: {
            Label(menuLabelTitle, systemImage: menuLabelIcon)
        }
        .menuStyle(.borderlessButton)
        .controlSize(.regular)
        .help("Mark foreground / background regions to guide the keyer.")
    }

    private var menuLabelTitle: String {
        switch viewModel.oscTool {
        case .disabled: return "Subject"
        case .foregroundHint: return "Marking FG"
        case .backgroundHint: return "Marking BG"
        case .eraseHint: return "Erasing"
        }
    }

    private var menuLabelIcon: String {
        switch viewModel.oscTool {
        case .disabled: return "person.crop.rectangle"
        case .foregroundHint: return "person.fill.badge.plus"
        case .backgroundHint: return "rectangle.fill.badge.minus"
        case .eraseHint: return "eraser"
        }
    }
}

extension OnScreenControlTool {
    /// Menu-row title that disambiguates the four states beyond just
    /// the enum's displayName ("Off" / "Foreground" / "Background" /
    /// "Erase"). The transport-bar menu reads more naturally as
    /// "Mark Foreground" than the bare "Foreground".
    var menuDisplayName: String {
        switch self {
        case .disabled: return "Off"
        case .foregroundHint: return "Mark Foreground"
        case .backgroundHint: return "Mark Background"
        case .eraseHint: return "Erase Nearest Hint"
        }
    }
}
