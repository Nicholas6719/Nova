//
//  NovaChrome.swift
//  Nova
//
//  The frame every screen sits in: a rail down the left, a strip across the top.
//
//  Both were indicative in the shipped shell — speech was the only navigation
//  and there was nothing to click anywhere in the app. The redesign makes the
//  rail a real target, which is a deliberate change and worth naming: voice
//  stays primary and still drives every destination, but reaching for one
//  screen should not require saying a sentence. Chrome that shows you where you
//  are and refuses to take you there is a worse deal than it looks.
//

import SwiftUI

struct NovaRail: View {
    let active: String
    let tint: Color
    let onSelect: (NovaDestination) -> Void

    var body: some View {
        VStack(spacing: 10) {
            ForEach(NovaDestination.rail) { dest in
                RailButton(dest: dest, active: dest.id == active, tint: tint) {
                    onSelect(dest)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 16)
        .frame(width: NovaDesign.railWidth)
        .background(
            ZStack(alignment: .trailing) {
                Color.white.opacity(0.02)
                Rectangle().fill(Color.white.opacity(0.06)).frame(width: 1)
            }
        )
    }
}

private struct RailButton: View {
    let dest: NovaDestination
    let active: Bool
    let tint: Color
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: dest.symbol)
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(foreground)
                .frame(width: 40, height: 40)
                .background(
                    RoundedRectangle(cornerRadius: NovaDesign.railRadius,
                                     style: .continuous)
                        .fill(active ? tint.opacity(0.14)
                              : Color.white.opacity(hovering ? 0.06 : 0.03))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: NovaDesign.railRadius,
                                     style: .continuous)
                        .stroke(active ? tint.opacity(0.50)
                                : Color.white.opacity(0.08), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        // Unbuilt destinations are reachable and honest about themselves —
        // they are not hidden and not disabled, they say so when you arrive.
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.16), value: active)
        .animation(.easeOut(duration: 0.16), value: hovering)
        .help(dest.built ? dest.title : "\(dest.title) — not built yet")
        .accessibilityLabel(dest.title)
    }

    private var foreground: Color {
        if active { return tint }
        return .white.opacity(dest.built ? 0.28 : 0.13)
    }
}

/// One line across the top: that Nova is awake, and what day it is.
struct NovaStrip: View {
    let state: NovaState
    let subtitle: String

    var body: some View {
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(state == .sleeping ? Color.white.opacity(0.2) : state.tint)
                    .frame(width: 5, height: 5)
                Text(state == .sleeping ? "NOVA IS ASLEEP" : "NOVA IS ACTIVE")
                    .foregroundStyle(state == .sleeping ? .white.opacity(0.22)
                                     : state.tint.opacity(0.75))
            }
            Spacer()
            Text(subtitle.uppercased())
                .foregroundStyle(.white.opacity(0.28))
        }
        .font(NovaDesign.label(9))
        .tracking(1.6)
        .padding(.horizontal, 16)
        .frame(height: NovaDesign.stripHeight)
        .background(
            ZStack(alignment: .bottom) {
                Color.white.opacity(0.02)
                Rectangle().fill(Color.white.opacity(0.06)).frame(height: 1)
            }
        )
        .animation(.easeInOut(duration: 0.35), value: state)
    }
}

// MARK: - Shared pieces

/// The mono, uppercase, wide-tracked caption above everything.
struct CardLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(NovaDesign.label(9))
            .tracking(1.6)
            .foregroundStyle(.white.opacity(0.35))
    }
}

/// A label/value line, the shape most of System is made of.
struct StatRow: View {
    let label: String
    let value: String
    var accent: Color? = nil

    var body: some View {
        HStack(spacing: 12) {
            Text(label)
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.5))
            Spacer(minLength: 8)
            Text(value)
                .font(NovaDesign.data(12))
                .foregroundStyle(accent ?? .white.opacity(0.88))
        }
    }
}

/// Granted / not granted, and never a guess.
///
/// A permission row that says "granted" because nobody checked is worse than
/// no row at all — it is the one place in this app where an optimistic default
/// would be actively misleading.
struct PermissionRow: View {
    let name: String
    let granted: Bool?

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(dotColor)
                .frame(width: 5, height: 5)
            Text(name)
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.5))
            Spacer(minLength: 8)
            Text(caption)
                .font(NovaDesign.label(9))
                .tracking(1.4)
                .foregroundStyle(captionColor)
        }
    }

    private var dotColor: Color {
        guard let granted else { return .white.opacity(0.2) }
        return granted ? NovaDesign.positive.opacity(0.85) : NovaDesign.negative.opacity(0.8)
    }
    private var caption: String {
        guard let granted else { return "UNKNOWN" }
        return granted ? "GRANTED" : "NOT GRANTED"
    }
    private var captionColor: Color {
        guard let granted else { return .white.opacity(0.25) }
        return granted ? NovaDesign.positive.opacity(0.8) : NovaDesign.negative.opacity(0.75)
    }
}
