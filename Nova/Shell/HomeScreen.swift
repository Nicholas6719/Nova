//
//  HomeScreen.swift
//  Nova
//
//  The resting screen: orb in the middle, glass either side, instrumentation
//  along the bottom.
//
//  The orb is centred on the CONTENT, and the columns are laid over it rather
//  than beside it. That is deliberate and it is the same lesson the shipped
//  home learned the hard way: put the orb in an HStack between two columns and
//  an empty column shoves it off centre — and home spends most of its life
//  uneven, because Now Playing and the proactive notice both come and go on
//  their own. Overlaying keeps it nailed to the middle whatever appears.
//
//  The centre column takes no hits. Everything there is a readout, and a
//  readout that swallows a click is a readout that feels broken.
//

import SwiftUI

/// What Nova can see, and what she is offering to do about it.
///
/// The suggestion is violet rather than cyan on purpose: a thing Nova is
/// PROPOSING is a different kind of object from a thing she is reporting, and
/// the two should never be mistaken for one another at a glance.
enum AwarenessStage: Equatable {
    case idle
    case suggesting(String)
    case checking
    case done(String)
}

struct HomeScreen: View {
    let state: NovaState
    let greeting: String
    let name: String
    let weather: (value: String, detail: String)?
    let nowPlaying: (title: String, artist: String)?
    let notice: String?
    let markets: [Quote]
    let upcoming: [CalendarEntry]
    let awarenessApp: String?
    let awarenessContext: String?
    let awareness: AwarenessStage
    let metrics: [PanelMetric]
    let onAwarenessYes: () -> Void
    let onAwarenessNo: () -> Void

    var body: some View {
        ZStack {
            HStack(alignment: .top, spacing: 24) {
                VStack(spacing: 14) { leftColumn }
                    .frame(width: NovaDesign.sideColumn)
                Spacer(minLength: 40)
                VStack(spacing: 14) { rightColumn }
                    .frame(width: NovaDesign.sideColumn)
            }

            centre.allowsHitTesting(false)

            VStack {
                Spacer(minLength: 0)
                HStack {
                    StatusRow(readings: metrics, tint: state.tint)
                    Spacer(minLength: 0)
                }
            }
            .allowsHitTesting(false)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
    }

    // MARK: Centre

    private var centre: some View {
        VStack(spacing: 10) {
            OrbView(state: state, density: .reactor)
                .frame(maxWidth: 300, maxHeight: 300)
                .aspectRatio(1, contentMode: .fit)
            VStack(spacing: 2) {
                Text(greeting.uppercased())
                    .font(NovaDesign.label(11)).tracking(2.6)
                    .foregroundStyle(.white.opacity(0.45))
                Text(name.uppercased())
                    .font(.system(size: 30, weight: .light))
                    .tracking(6)
                    .foregroundStyle(state.tint.opacity(0.95))
            }
            Text(state.readout.uppercased())
                .font(NovaDesign.label(10)).tracking(2.4)
                .foregroundStyle(state.tint.opacity(state == .sleeping ? 0.35 : 0.6))
        }
        .animation(.easeInOut(duration: 0.35), value: state)
    }

    // MARK: Columns

    @ViewBuilder private var leftColumn: some View {
        if let w = weather {
            VStack(alignment: .leading, spacing: 12) {
                CardLabel(text: "Weather")
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(w.value)
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(.white.opacity(0.95))
                    Text(w.detail)
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.55))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
        }
        if let np = nowPlaying {
            VStack(alignment: .leading, spacing: 10) {
                CardLabel(text: "Now playing")
                Text(np.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                Text(np.artist)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
            .transition(.opacity.combined(with: .scale(scale: 0.96)))
        }
        if let notice {
            // Amber, because it is the one card that arrived without being
            // asked for. It should look like an interruption, briefly.
            VStack(alignment: .leading, spacing: 8) {
                CardLabel(text: "Heads up")
                Text(notice)
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.85))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
            .overlay(
                RoundedRectangle(cornerRadius: NovaDesign.cardRadius, style: .continuous)
                    .stroke(NovaState.working.tint.opacity(0.45), lineWidth: 1))
            .transition(.opacity.combined(with: .move(edge: .top)))
        }
        Spacer(minLength: 0)
    }

    @ViewBuilder private var rightColumn: some View {
        if !markets.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                CardLabel(text: "Markets")
                ForEach(markets) { q in
                    HStack(spacing: 8) {
                        Text(q.symbol)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(.white.opacity(0.9))
                        Spacer(minLength: 8)
                        Text(q.priceText)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(.white.opacity(0.6))
                        Text(q.changeText)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(q.tint.opacity(0.9))
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
        }
        if !upcoming.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                CardLabel(text: "Upcoming")
                ForEach(upcoming.prefix(4)) { e in
                    HStack(alignment: .top, spacing: 10) {
                        Rectangle()
                            .fill(e.isReminder ? NovaDesign.positive.opacity(0.65)
                                  : state.tint.opacity(0.55))
                            .frame(width: 2).frame(maxHeight: .infinity)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(e.title)
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(.white.opacity(0.9))
                            if !e.time.isEmpty {
                                Text(e.time)
                                    .font(.system(size: 11))
                                    .foregroundStyle(.white.opacity(0.45))
                            }
                        }
                        Spacer(minLength: 0)
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
        }
        awarenessCard
        Spacer(minLength: 0)
    }

    // MARK: Screen awareness

    @ViewBuilder private var awarenessCard: some View {
        if let app = awarenessApp {
            VStack(alignment: .leading, spacing: 10) {
                CardLabel(text: "On your screen")
                VStack(alignment: .leading, spacing: 2) {
                    Text(app)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.white.opacity(0.88))
                    if let ctx = awarenessContext {
                        Text(ctx)
                            .font(.system(size: 12))
                            .foregroundStyle(.white.opacity(0.45))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                awarenessBody
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass()
            .animation(.easeOut(duration: 0.22), value: awareness)
        }
    }

    @ViewBuilder private var awarenessBody: some View {
        switch awareness {
        case .idle:
            EmptyView()
        case .suggesting(let text):
            VStack(alignment: .leading, spacing: 10) {
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(NovaDesign.suggestion.opacity(0.95))
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    Button("Yes", action: onAwarenessYes)
                        .buttonStyle(PillButton(tint: NovaDesign.suggestion, filled: true))
                    Button("Not now", action: onAwarenessNo)
                        .buttonStyle(PillButton(tint: .white, filled: false))
                }
            }
        case .checking:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Checking…")
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.55))
            }
        case .done(let result):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "checkmark")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(NovaDesign.positive.opacity(0.85))
                Text(result)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.7))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Puck

/// The floating companion: a glass square, not a bare circle.
///
/// Two dots carry everything the full window would have told him — work mode
/// on the right, a next event on the left. In the mockup a legend explained
/// them; a legend on a 180pt puck would be most of the puck, so they are
/// glanceable indicators and nothing else. If he cannot learn them in a day
/// they are the wrong indicators, not under-explained ones.
struct PuckView: View {
    let state: NovaState
    let workMode: Bool
    let hasUpcoming: Bool

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .fill(.ultraThinMaterial)
                .opacity(0.5)
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .fill(Color.white.opacity(0.035))
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(Color(red: 120/255, green: 200/255, blue: 255/255).opacity(0.16),
                        lineWidth: 1)

            VStack(spacing: 4) {
                OrbView(state: state, density: .array)
                    .frame(width: 116, height: 116)
                Text(state.readout.uppercased())
                    .font(NovaDesign.label(8)).tracking(1.8)
                    .foregroundStyle(state.tint.opacity(state == .sleeping ? 0.3 : 0.6))
                    .lineLimit(1)
            }
            .padding(.top, 6)

            VStack {
                HStack {
                    dot(NovaState.working.tint, on: hasUpcoming)
                    Spacer()
                    dot(NovaState.working.tint, on: workMode)
                }
                Spacer()
            }
            .padding(12)
        }
        .frame(width: 180, height: 180)
    }

    private func dot(_ color: Color, on: Bool) -> some View {
        Circle()
            .fill(on ? color.opacity(0.9) : .clear)
            .frame(width: 6, height: 6)
            .animation(.easeOut(duration: 0.25), value: on)
    }
}
