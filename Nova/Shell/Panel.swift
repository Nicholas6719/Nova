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

    /// Decoded defensively: a malformed payload yields an empty panel and the
    /// orb keeps the screen, rather than throwing away Nova's answer.
    init(_ data: [String: Any]) {
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

    init(_ d: [String: Any]) {
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

    init(_ d: [String: Any]) {
        title = d["title"] as? String ?? d["name"] as? String ?? ""
        detail = d["detail"] as? String ?? d["say"] as? String ?? ""
        meta = d["meta"] as? String ?? ""
        available = d["available"] as? Bool ?? true
        note = d["note"] as? String ?? ""
    }
}

enum PanelBlock: Identifiable {
    case stat(id: UUID, value: String, label: String, detail: String)
    case rows(id: UUID, title: String, pairs: [(String, String)])
    case items(id: UUID, title: String, items: [PanelItem])
    case text(id: UUID, title: String, body: String)
    case note(id: UUID, body: String)

    var id: UUID {
        switch self {
        case let .stat(id, _, _, _), let .rows(id, _, _),
             let .items(id, _, _), let .text(id, _, _), let .note(id, _):
            return id
        }
    }

    init?(_ d: [String: Any]) {
        let id = UUID()
        switch d["kind"] as? String {
        case "stat":
            self = .stat(id: id,
                         value: d["value"] as? String ?? "",
                         label: d["label"] as? String ?? "",
                         detail: d["detail"] as? String ?? "")
        case "rows":
            let pairs = (d["rows"] as? [[String: Any]] ?? []).map {
                ($0["label"] as? String ?? "", $0["value"] as? String ?? "")
            }
            self = .rows(id: id, title: d["title"] as? String ?? "", pairs: pairs)
        case "items":
            self = .items(id: id, title: d["title"] as? String ?? "",
                          items: (d["items"] as? [[String: Any]] ?? []).map(PanelItem.init))
        case "text":
            self = .text(id: id, title: d["title"] as? String ?? "",
                         body: d["text"] as? String ?? "")
        case "note":
            self = .note(id: id, body: d["text"] as? String ?? "")
        default:
            return nil
        }
    }
}

// MARK: - View

struct PanelView: View {
    let panel: Panel
    let tint: Color

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                ForEach(panel.blocks) { block(for: $0) }
                ForEach(panel.sections) { section($0) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(26)
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
        switch b {
        case let .stat(_, value, label, detail):
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

        case let .rows(_, title, pairs):
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

        case let .items(_, title, list):
            VStack(alignment: .leading, spacing: 10) {
                if !title.isEmpty { caption(title) }
                ForEach(list) { row(for: $0) }
            }

        case let .text(_, title, body):
            VStack(alignment: .leading, spacing: 6) {
                if !title.isEmpty { caption(title) }
                Text(body)
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.78))
                    .fixedSize(horizontal: false, vertical: true)
            }

        case let .note(_, body):
            Text(body)
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.4))
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
                .fill(item.available ? tint.opacity(0.55) : Color.white.opacity(0.12))
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
