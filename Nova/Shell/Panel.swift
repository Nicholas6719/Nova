//
//  Panel.swift
//  Nova
//
//  The structured half of Nova's answers, rendered.
//
//  Every block here was built by a deterministic handler in Python — the same
//  templated numbers the voice speaks. Nothing on a panel is phrased by the
//  model, which is what keeps invented figures off the screen.
//
//  The block vocabulary is deliberately small (stat / rows / items / text /
//  note / sections) so a new panel is a backend change only and never needs a
//  new SwiftUI view.
//

import SwiftUI

// MARK: - Model

struct Panel {
    var title: String = ""
    var subtitle: String = ""
    var blocks: [PanelBlock] = []
    var sections: [PanelSection] = []

    var isEmpty: Bool { blocks.isEmpty && sections.isEmpty }

    /// One block on its own, for home's side cards.
    nonisolated init(single block: PanelBlock) {
        blocks = [block]
    }

    /// The card in a named home slot, if anything is there.
    nonisolated func block(inSlot slot: String) -> PanelBlock? {
        blocks.first { $0.slot == slot }
    }

    /// The bottom-left instrumentation row, if Nova is showing it.
    nonisolated var statusReadings: [PanelMetric] {
        for b in blocks {
            if case let .metrics(_, readings) = b.content, b.slot == "status" {
                return readings
            }
        }
        return []
    }

    /// Decoded defensively: a malformed payload yields an empty panel and the
    /// orb keeps the screen, rather than throwing away Nova's answer.
    ///
    /// `nonisolated` throughout this file: the target builds with
    /// SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor, which makes even these pure
    /// data types actor-isolated, and decoding a payload is not main-actor
    /// work. Fixed on the types rather than by relaxing the build setting —
    /// that default is load-bearing everywhere else in the app.
    nonisolated init(_ data: [String: Any]) {
        title = data["title"] as? String ?? ""
        subtitle = data["subtitle"] as? String ?? ""
        blocks = (data["blocks"] as? [[String: Any]] ?? []).compactMap(PanelBlock.init)
        sections = (data["sections"] as? [[String: Any]] ?? []).map(PanelSection.init)
    }
}

struct PanelSection: Identifiable {
    let id = UUID()
    let title: String
    let items: [PanelItem]

    nonisolated init(_ d: [String: Any]) {
        title = d["title"] as? String ?? ""
        items = (d["items"] as? [[String: Any]] ?? []).map(PanelItem.init)
    }
}

struct PanelItem: Identifiable {
    let id = UUID()
    var title = ""
    var detail = ""
    var meta = ""
    /// Menu entries for screens that do not exist yet are shown dimmed with
    /// the reason, never silently omitted.
    var available = true
    var note = ""
    /// A hint from the engine about what KIND of row this is — "reminder"
    /// marks the reminders mixed into Upcoming, so calendar and reminders stay
    /// tellable apart without needing two cards.
    var accent = ""

    nonisolated init(_ d: [String: Any]) {
        title = d["title"] as? String ?? d["name"] as? String ?? ""
        detail = d["detail"] as? String ?? d["say"] as? String ?? ""
        meta = d["meta"] as? String ?? ""
        available = d["available"] as? Bool ?? true
        note = d["note"] as? String ?? ""
        accent = d["accent"] as? String ?? ""
    }
}

/// One reading on the status row. Value is templated by the engine; `pct`
/// drives the hairline so a level is readable without reading the number.
struct PanelMetric: Identifiable {
    let id = UUID()
    var label = ""
    var value = ""
    var pct: Double?
    var flag = ""
    var alert = false

    nonisolated init(_ d: [String: Any]) {
        label = d["label"] as? String ?? ""
        value = d["value"] as? String ?? ""
        pct = d["pct"] as? Double
        flag = d["flag"] as? String ?? ""
        alert = d["alert"] as? Bool ?? false
    }
}

/// One line of what Nova is doing, while she is doing it.
struct PanelStep: Identifiable {
    let id = UUID()
    var label = ""
    var state = "pending"      // done | running | pending | failed
    var meta = ""

    nonisolated init(_ d: [String: Any]) {
        label = d["label"] as? String ?? ""
        state = d["state"] as? String ?? "pending"
        meta = d["meta"] as? String ?? ""
    }
}

/// A block, plus where on the home grid it belongs.
///
/// Slot and card live on the WRAPPER rather than inside every content case:
/// they are placement, not content, and threading them through five enum
/// payloads would mean every render site had to ignore them by hand.
struct PanelBlock: Identifiable {
    let id = UUID()
    let content: PanelContent
    /// "L1".."R3" for a home card, "status" for the bottom row, "" elsewhere.
    let slot: String
    /// Stable identity for a card across renders — this is what lets a card
    /// FLY to its new slot when he moves it, instead of blinking out of one
    /// place and into another.
    let card: String

    nonisolated init?(_ d: [String: Any]) {
        guard let content = PanelContent(d) else { return nil }
        self.content = content
        self.slot = d["slot"] as? String ?? ""
        self.card = d["card"] as? String ?? ""
    }
}

enum PanelContent {
    case stat(value: String, label: String, detail: String)
    case rows(title: String, pairs: [(String, String)])
    case items(title: String, items: [PanelItem])
    case text(title: String, body: String)
    case note(body: String)
    case metrics(title: String, readings: [PanelMetric])
    case steps(title: String, detail: String, entries: [PanelStep])

    nonisolated init?(_ d: [String: Any]) {
        switch d["kind"] as? String {
        case "stat":
            self = .stat(value: d["value"] as? String ?? "",
                         label: d["label"] as? String ?? "",
                         detail: d["detail"] as? String ?? "")
        case "rows":
            let pairs = (d["rows"] as? [[String: Any]] ?? []).map {
                ($0["label"] as? String ?? "", $0["value"] as? String ?? "")
            }
            self = .rows(title: d["title"] as? String ?? "", pairs: pairs)
        case "items":
            self = .items(title: d["title"] as? String ?? "",
                          items: (d["items"] as? [[String: Any]] ?? []).map(PanelItem.init))
        case "text":
            self = .text(title: d["title"] as? String ?? "",
                         body: d["text"] as? String ?? "")
        case "note":
            self = .note(body: d["text"] as? String ?? "")
        case "metrics":
            self = .metrics(title: d["title"] as? String ?? "",
                            readings: (d["metrics"] as? [[String: Any]] ?? []).map(PanelMetric.init))
        case "steps":
            self = .steps(title: d["title"] as? String ?? "",
                          detail: d["detail"] as? String ?? "",
                          entries: (d["steps"] as? [[String: Any]] ?? []).map(PanelStep.init))
        default:
            return nil
        }
    }
}

// MARK: - View

struct PanelView: View {
    let panel: Panel
    let tint: Color
    /// A home CARD hugs its content; a full answer panel scrolls and fills.
    /// Without this a one-line card stretched the whole window height.
    var compact: Bool = false

    var body: some View {
        Group {
            if compact {
                content.fixedSize(horizontal: false, vertical: true)
            } else {
                ScrollView { content }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.white.opacity(0.028))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Color.white.opacity(0.07), lineWidth: 1)
                )
        )
    }

    private var content: some View {
        VStack(alignment: .leading, spacing: compact ? 14 : 22) {
            header
            ForEach(panel.blocks) { block(for: $0) }
            ForEach(panel.sections) { section($0) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(compact ? 18 : 26)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 3) {
            if !panel.title.isEmpty {
                Text(panel.title)
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.92))
            }
            if !panel.subtitle.isEmpty {
                Text(panel.subtitle.uppercased())
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .tracking(1.8)
                    .foregroundStyle(tint.opacity(0.7))
            }
        }
    }

    @ViewBuilder
    private func block(for b: PanelBlock) -> some View {
        switch b.content {
        case let .stat(value, label, detail):
            VStack(alignment: .leading, spacing: 2) {
                if !label.isEmpty { caption(label) }
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(value)
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(.white.opacity(0.95))
                    if !detail.isEmpty {
                        Text(detail)
                            .font(.system(size: 14))
                            .foregroundStyle(.white.opacity(0.55))
                    }
                }
            }

        case let .rows(title, pairs):
            VStack(alignment: .leading, spacing: 8) {
                if !title.isEmpty { caption(title) }
                ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                    HStack {
                        Text(pair.0)
                            .foregroundStyle(.white.opacity(0.5))
                        Spacer(minLength: 20)
                        Text(pair.1)
                            .foregroundStyle(.white.opacity(0.88))
                            .monospacedDigit()
                    }
                    .font(.system(size: 13))
                }
            }

        case let .items(title, list):
            VStack(alignment: .leading, spacing: 10) {
                if !title.isEmpty { caption(title) }
                ForEach(list) { row(for: $0) }
            }

        case let .text(title, body):
            VStack(alignment: .leading, spacing: 6) {
                if !title.isEmpty { caption(title) }
                Text(body)
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.78))
                    .fixedSize(horizontal: false, vertical: true)
            }

        case let .note(body):
            Text(body)
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.4))

        case let .metrics(title, readings):
            // Rendered inline here only for completeness — on home the status
            // row is pulled out and pinned along the bottom by ShellView,
            // because it is a line of instrumentation and not a card.
            VStack(alignment: .leading, spacing: 8) {
                if !title.isEmpty { caption(title) }
                StatusRow(readings: readings, tint: tint)
            }

        case let .steps(title, detail, entries):
            VStack(alignment: .leading, spacing: 8) {
                if !title.isEmpty { caption(title) }
                if !detail.isEmpty {
                    Text(detail)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.white.opacity(0.9))
                }
                StepList(entries: entries, tint: tint)
            }
        }
    }

    private func section(_ s: PanelSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if !s.title.isEmpty { caption(s.title) }
            ForEach(s.items) { row(for: $0) }
        }
    }

    private func row(for item: PanelItem) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Rectangle()
                .fill(item.available
                      // A reminder mixed into Upcoming reads as a different
                      // kind of thing from a meeting, and one colour is
                      // cheaper than a second card.
                      ? (item.accent == "reminder" ? Color.green.opacity(0.65)
                                                   : tint.opacity(0.55))
                      : Color.white.opacity(0.12))
                .frame(width: 2)
                .frame(maxHeight: .infinity)

            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(item.available ? 0.9 : 0.4))
                if !item.detail.isEmpty {
                    Text(item.detail)
                        .font(.system(size: 12))
                        .foregroundStyle(.white.opacity(0.5))
                }
                // Why a destination is unavailable, rather than hiding it.
                if !item.note.isEmpty {
                    Text(item.note)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.32))
                }
            }

            Spacer(minLength: 8)

            if !item.meta.isEmpty {
                Text(item.meta)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.55))
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func caption(_ s: String) -> some View {
        Text(s.uppercased())
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .tracking(1.6)
            .foregroundStyle(.white.opacity(0.35))
    }
}


// MARK: - Status row

/// CPU, memory and battery as a LINE, not a card.
///
/// His call and the right one: this is glanceable furniture, and a box would
/// give it the same visual weight as his calendar. Each reading is a label, a
/// templated value, and a hairline whose fill is the level — so he can read
/// the state without reading a single number.
struct StatusRow: View {
    let readings: [PanelMetric]
    let tint: Color

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(readings.enumerated()), id: \.element.id) { index, r in
                if index > 0 {
                    Text("—")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.14))
                        .padding(.horizontal, 11)
                }
                reading(r)
            }
        }
        .animation(.easeInOut(duration: 0.45), value: readings.map(\.value))
    }

    private func reading(_ r: PanelMetric) -> some View {
        HStack(spacing: 6) {
            Text(r.label.uppercased())
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(.white.opacity(0.30))
            Text(r.value)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .tracking(0.8)
                .foregroundStyle(.white.opacity(0.55))
                .monospacedDigit()
            if let pct = r.pct {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.10))
                        Capsule()
                            .fill(r.alert ? Color.orange.opacity(0.9) : tint.opacity(0.8))
                            .frame(width: max(1, geo.size.width * pct))
                    }
                }
                .frame(width: 22, height: 2)
            }
            if r.flag == "charging" {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 7.5))
                    .foregroundStyle(Color.green.opacity(0.85))
            }
        }
    }
}

// MARK: - Step list

/// What Nova is doing, while she is doing it.
struct StepList: View {
    let entries: [PanelStep]
    let tint: Color
    /// Drives the pulse on the running step. One clock for the whole list, so
    /// several steps could never breathe out of phase with each other.
    @State private var pulse = false

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(entries) { step in
                HStack(alignment: .firstTextBaseline, spacing: 9) {
                    marker(step)
                        .frame(width: 12, alignment: .leading)
                    Text(step.label)
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(opacity(step)))
                    Spacer(minLength: 8)
                    if !step.meta.isEmpty {
                        Text(step.meta)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.3))
                    }
                }
                .animation(.easeOut(duration: 0.3), value: step.state)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }

    @ViewBuilder
    private func marker(_ step: PanelStep) -> some View {
        switch step.state {
        case "done":
            Image(systemName: "checkmark")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Color.green.opacity(0.8))
        case "running":
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)
                .opacity(pulse ? 0.25 : 1.0)
        case "failed":
            Image(systemName: "xmark")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Color.red.opacity(0.75))
        default:
            Circle()
                .stroke(Color.white.opacity(0.22), lineWidth: 1)
                .frame(width: 6, height: 6)
        }
    }

    private func opacity(_ step: PanelStep) -> Double {
        switch step.state {
        case "running": return 0.95
        case "done":    return 0.6
        case "failed":  return 0.6
        default:        return 0.3
        }
    }
}
