//
//  TransportBar.swift
//  Corridor Key Toolbox — Standalone Editor
//
//  Mac-native transport bar that sits beneath the preview surface.
//  Modelled after QuickTime Player's controls: a centred play / pause
//  button with frame-step buttons either side, a loop toggle on the
//  right, a current-time / duration label, and a scrubber spanning
//  the bar. The leading column hosts the import / close / export
//  workflow buttons; the trailing column hosts the loop and OSC
//  affordances. The slider drives `EditorViewModel.scrub`, which
//  snaps to the nearest source frame.
//

import SwiftUI
import CoreMedia

struct TransportBar: View {
    @Bindable var viewModel: EditorViewModel
    let onImport: () -> Void
    let onClose: () -> Void
    let onExport: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button("Import…", systemImage: "square.and.arrow.down", action: onImport)
                    .buttonStyle(.bordered)

                Button("Close Clip", systemImage: "xmark.circle", action: onClose)
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.phase.isReady)

                Spacer()

                playbackCluster

                Spacer()

                Toggle(isOn: $viewModel.loopEnabled) {
                    Label("Loop", systemImage: "repeat")
                }
                .toggleStyle(.button)
                .controlSize(.regular)
                .help("Restart playback when the clip ends.")

                Button("Export…", systemImage: "square.and.arrow.up", action: onExport)
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.phase.isReady || viewModel.exportStatus.inProgress)
            }

            scrubberRow
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(.regularMaterial)
    }

    private var playbackCluster: some View {
        HStack(spacing: 14) {
            Button("Step Back", systemImage: "backward.frame.fill") {
                viewModel.step(byFrames: -1)
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.large)

            Button(
                viewModel.isPlaying ? "Pause" : "Play",
                systemImage: viewModel.isPlaying ? "pause.fill" : "play.fill"
            ) {
                viewModel.togglePlayback()
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.extraLarge)
            .keyboardShortcut(.space, modifiers: [])
            .font(.title2)

            Button("Step Forward", systemImage: "forward.frame.fill") {
                viewModel.step(byFrames: 1)
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .disabled(!viewModel.phase.isReady)
            .controlSize(.large)
        }
    }

    private var scrubberRow: some View {
        HStack(spacing: 10) {
            Text(timeLabel(for: viewModel.playheadTime))
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 90, alignment: .leading)

            Slider(
                value: Binding(
                    get: { normalizedPlayhead },
                    set: { viewModel.scrub(toNormalized: $0) }
                ),
                in: 0...1
            )
            .disabled(!viewModel.phase.isReady)

            Text("\(currentFrameLabel) / \(totalFrameLabel)")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .trailing)
        }
    }

    private var normalizedPlayhead: Double {
        guard let info = viewModel.clipInfo, info.duration.seconds > 0 else { return 0 }
        return min(max(viewModel.playheadTime.seconds / info.duration.seconds, 0), 1)
    }

    private var currentFrameLabel: String {
        guard let info = viewModel.clipInfo else { return "0" }
        let frameRate = max(Double(info.nominalFrameRate), 0.001)
        let frame = Int((viewModel.playheadTime.seconds * frameRate).rounded())
        return String(frame + 1)
    }

    private var totalFrameLabel: String {
        viewModel.totalFrames > 0 ? "\(viewModel.totalFrames) f" : "—"
    }

    private func timeLabel(for time: CMTime) -> String {
        guard time.isValid && time.isNumeric else { return "00:00:00" }
        let total = max(time.seconds, 0)
        let totalSeconds = Int(total)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let seconds = totalSeconds % 60
        let centiseconds = Int((total - Double(totalSeconds)) * 100)
        return String(format: "%02d:%02d:%02d.%02d", hours, minutes, seconds, centiseconds)
    }
}
