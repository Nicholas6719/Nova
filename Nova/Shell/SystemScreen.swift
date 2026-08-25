//
//  SystemScreen.swift
//  Nova
//
//  What Nova is made of, and what she is allowed to do.
//
//  Four panels: the ports she is listening on, the models running on this Mac,
//  what she remembers, and which permissions she actually holds.
//
//  The last of those is the reason this screen earns its place. Nova's most
//  confusing failures have all been permission failures wearing a different
//  costume — the calendar refusing a bare interpreter, screen capture quietly
//  returning wallpaper, NSWorkspace answering from a stale snapshot. Every one
//  of those looked like a bug in a feature. A screen that says plainly which
//  grants exist turns a morning of debugging into a glance.
//
//  Nothing here is optimistic. A permission whose state was never checked reads
//  UNKNOWN, not GRANTED: a hopeful default on this screen would be worse than
//  showing nothing at all.
//

import SwiftUI

/// What the shell knows about the machine underneath it. Populated by the
/// backend over the existing status feed; every field optional, because
/// "we have not been told yet" is a real state and deserves to be shown.
struct SystemInfo: Equatable {
    var httpPort: Int = 5001
    var wsPort: Int = 8766
    var connected: Bool = false

    var llm: String? = nil
    var stt: String? = nil
    var tts: String? = nil
    var wake: String? = nil

    var facts: Int? = nil
    var conversations: Int? = nil
    var ragDocs: Int? = nil

    var microphone: Bool? = nil
    var accessibility: Bool? = nil
    var screenRecording: Bool? = nil
    var location: Bool? = nil
}

struct SystemScreen: View {
    let info: SystemInfo
    let tint: Color

    private let columns = [GridItem(.adaptive(minimum: 280), spacing: 18)]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                LazyVGrid(columns: columns, alignment: .leading, spacing: 18) {
                    connections
                    models
                    memory
                    permissions
                }
            }
            .padding(.horizontal, NovaDesign.contentPaddingH)
            .padding(.vertical, NovaDesign.contentPaddingV)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("System")
                .font(.system(size: 21, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))
            Text("EVERYTHING RUNS ON THIS MAC")
                .font(NovaDesign.label(9))
                .tracking(1.8)
                .foregroundStyle(tint.opacity(0.7))
        }
    }

    private var connections: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "Connections")
            StatRow(label: "HTTP", value: ":\(info.httpPort)")
            StatRow(label: "WebSocket", value: ":\(info.wsPort)")
            HStack(spacing: 6) {
                Circle()
                    .fill(info.connected ? NovaDesign.positive.opacity(0.85)
                          : NovaDesign.negative.opacity(0.8))
                    .frame(width: 5, height: 5)
                Text(info.connected ? "ONLINE" : "OFFLINE")
                    .font(NovaDesign.label(9)).tracking(1.4)
                    .foregroundStyle(info.connected ? NovaDesign.positive.opacity(0.8)
                                     : NovaDesign.negative.opacity(0.75))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private var models: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "On-device models")
            StatRow(label: "Language", value: info.llm ?? "—")
            StatRow(label: "Speech to text", value: info.stt ?? "—")
            StatRow(label: "Text to speech", value: info.tts ?? "—")
            StatRow(label: "Wake word", value: info.wake ?? "—")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private var memory: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "Memory")
            StatRow(label: "Facts", value: count(info.facts))
            StatRow(label: "Conversations", value: count(info.conversations))
            StatRow(label: "Indexed documents", value: count(info.ragDocs))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private var permissions: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "Permissions")
            PermissionRow(name: "Microphone", granted: info.microphone)
            PermissionRow(name: "Accessibility", granted: info.accessibility)
            PermissionRow(name: "Screen Recording", granted: info.screenRecording)
            PermissionRow(name: "Location", granted: info.location)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private func count(_ n: Int?) -> String {
        guard let n else { return "—" }
        return n.formatted()
    }
}

/// Health, and the honesty pattern for a screen that does not exist.
///
/// Dimmed skeletons rather than a blank page or a spinner: a spinner implies
/// something is coming, and a blank page implies something broke. Striped
/// placeholders at low opacity read immediately as "this is the shape of a
/// thing that is not here yet", and the note underneath says why.
struct HealthScreen: View {
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Health")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.92))
                Text("NOT BUILT YET")
                    .font(NovaDesign.label(9)).tracking(1.8)
                    .foregroundStyle(.white.opacity(0.3))
            }

            HStack(spacing: 18) {
                ForEach(["Steps", "Sleep", "Heart Rate"], id: \.self) { name in
                    VStack(alignment: .leading, spacing: 14) {
                        CardLabel(text: name)
                        DiagonalStripes()
                            .stroke(Color.white.opacity(0.06), lineWidth: 1)
                            .frame(height: 54)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glass()
                    .opacity(0.45)
                }
            }

            Text("Health data lives on your phone. Nova can't see it until "
                 + "there's an iOS app to send it — HealthKit doesn't exist on macOS.")
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.4))
                .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
    }
}

struct DiagonalStripes: Shape {
    var spacing: CGFloat = 9

    func path(in rect: CGRect) -> Path {
        var p = Path()
        var x = rect.minX - rect.height
        while x <= rect.maxX {
            p.move(to: CGPoint(x: x, y: rect.maxY))
            p.addLine(to: CGPoint(x: x + rect.height, y: rect.minY))
            x += spacing
        }
        return p
    }
}
