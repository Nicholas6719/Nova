//
//  ShellView.swift
//  Nova
//
//  The whole interface. An orb, a word under it, and nothing else.
//
//  There is no transcript and no chat: Nova speaks, and the orb says what she
//  is doing. The typing field is deliberately hidden until asked for — speech
//  is the primary channel, and typing exists for the times he cannot talk.
//
//  A panel sits beside the orb when Nova has structure to show. Everything on
//  it was built by a deterministic handler in Python — the model never touches
//  a panel, which is what keeps invented numbers off the screen.
//

import SwiftUI

struct ShellView: View {
    @StateObject private var vm: ShellViewModel
    @State private var typing = false
    @State private var draft = ""
    @FocusState private var draftFocused: Bool
    /// Cards keep their identity across slots, so moving one by voice makes it
    /// FLY there rather than blink out of one place and into another.
    @Namespace private var cardSpace

    init() {
        _vm = StateObject(wrappedValue: ShellViewModel())
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if vm.isPuck {
                // The puck is the orb and nothing else. ignoresSafeArea is
                // load-bearing: .hiddenTitleBar still reserves the titlebar
                // inset, and without this the orb sits measurably below centre
                // (13pt, in a 130pt window).
                OrbView(state: vm.state, density: .array)
                    .padding(2)
                    .ignoresSafeArea()
            } else {
                fullShell
            }
        }
        .onAppear {
            WindowChrome.makeFrameless()
            // Again on the next runloop: window restoration applies its saved
            // frame after onAppear, so a single pass loses to it and Nova comes
            // back puck-sized after being quit while parked.
            DispatchQueue.main.async { WindowChrome.makeFrameless() }
        }
        // Cmd-T reaches the typing field without a visible control cluttering
        // an interface whose whole point is not having one.
        .background(
            ZStack {
                Button("") { toggleTyping() }
                    .keyboardShortcut("t", modifiers: .command)
                // Phase 3 puts Nova in the puck by voice, and automatically
                // when she starts doing things on his Mac. This is the manual
                // way in and out until then — and worth keeping regardless.
                Button("") { vm.setPuck(!vm.isPuck) }
                    .keyboardShortcut("m", modifiers: [.command, .shift])
                // SPACE stops her talking. Deterministic where acoustics are
                // not: no threshold, no cancellation, nothing to mishear. It
                // is unmodified on purpose — reaching for a chord while she is
                // mid-sentence is not interrupting, it is admin. Harmless when
                // she is silent, and the composer takes the key back the
                // moment he is typing.
                Button("") { vm.interrupt() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(typing)
            }
            .opacity(0)
        )
    }

    /// The window chrome: a rail down the left, a strip across the top, and
    /// whichever layout the current view calls for.
    ///
    /// Both pieces are deliberately NOT controls. Nova is navigated by voice —
    /// there is nothing to click anywhere in this app — so the rail's job is to
    /// say what exists and where you are, and the strip's is to say she is
    /// listening and what day it is. Without them the date in every payload had
    /// nowhere to render at all.
    private var fullShell: some View {
        HStack(spacing: 0) {
            iconRail
            VStack(spacing: 0) {
                statusStrip
                VStack(spacing: 0) {
                    if isWorking {
                        workLayout
                    } else if isHome {
                        homeLayout
                    } else {
                        answerLayout
                    }
                    if typing { composer.padding(.top, 20) }
                }
                .padding(24)
            }
        }
        .animation(.easeOut(duration: 0.24), value: vm.view)
        .animation(.easeOut(duration: 0.30), value: panel.title)
        .animation(.easeOut(duration: 0.22), value: panel.isEmpty)
        .animation(.spring(response: 0.5, dampingFraction: 0.82), value: isWorking)
        // Cards rearrange on a spring rather than a fade: a card that MOVES
        // reads as the same card in a new place, which is the whole point of
        // being able to move one.
        .animation(.spring(response: 0.46, dampingFraction: 0.84),
                   value: slotSignature)
    }

    /// The five destinations, and which one you are looking at.
    ///
    /// Indicative, not clickable: speech is the navigation. "health" is dimmed
    /// further because its screen does not exist yet — the same honesty the
    /// menu applies, carried into the chrome.
    private var iconRail: some View {
        VStack(spacing: 13) {
            ForEach(Self.railItems, id: \.view) { item in
                railIcon(item)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 14)
        .frame(width: 52)
        .background(
            ZStack(alignment: .trailing) {
                Color.white.opacity(0.02)
                Rectangle().fill(Color.white.opacity(0.06)).frame(width: 1)
            }
        )
    }

    private static let railItems: [(view: String, glyph: String, built: Bool)] = [
        ("home",     "house",              true),
        ("menu",     "square.grid.2x2",    true),
        ("calendar", "calendar",           true),
        ("finance",  "chart.line.uptrend.xyaxis", true),
        ("health",   "heart",              false),
    ]

    private func railIcon(_ item: (view: String, glyph: String, built: Bool)) -> some View {
        let active = vm.view == item.view
        return Image(systemName: item.glyph)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(active ? vm.state.tint
                             : .white.opacity(item.built ? 0.28 : 0.13))
            .frame(width: 24, height: 24)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(active ? vm.state.tint.opacity(0.12) : .clear)
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .stroke(active ? vm.state.tint.opacity(0.55)
                                    : Color.white.opacity(0.07), lineWidth: 1)
                    )
            )
            .animation(.easeOut(duration: 0.25), value: active)
            .accessibilityLabel(item.view)
    }

    /// One line across the top: that Nova is awake, and what day it is.
    private var statusStrip: some View {
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(vm.state == .sleeping ? Color.white.opacity(0.2)
                                                : vm.state.tint)
                    .frame(width: 5, height: 5)
                Text(vm.state == .sleeping ? "NOVA IS ASLEEP" : "NOVA IS ACTIVE")
                    .foregroundStyle(vm.state == .sleeping ? .white.opacity(0.22)
                                                           : vm.state.tint.opacity(0.75))
            }
            Spacer()
            // The date the backend already put in every payload, which until
            // now had nowhere on screen to go.
            Text(panel.subtitle.isEmpty ? "" : panel.subtitle.uppercased())
                .foregroundStyle(.white.opacity(0.28))
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
        .tracking(1.6)
        .padding(.horizontal, 16)
        .frame(height: 32)
        .background(
            ZStack(alignment: .bottom) {
                Color.white.opacity(0.02)
                Rectangle().fill(Color.white.opacity(0.06)).frame(height: 1)
            }
        )
        .animation(.easeInOut(duration: 0.35), value: vm.state)
    }

    private var isHome: Bool { vm.view == "home" }

    /// True while Nova is showing her working — a live step list on screen.
    private var isWorking: Bool {
        panel.blocks.contains { if case .steps = $0.content { return true }
                                return false }
    }

    /// Which card is in which slot, as one comparable value. This is what a
    /// move animates against.
    private var slotSignature: String {
        panel.blocks.map { "\($0.card):\($0.slot)" }.sorted().joined(separator: ",")
    }

    /// HOME: the orb is the centre of the SCREEN — not the centre of whatever
    /// space the cards leave over.
    ///
    /// That distinction is the whole layout. Flanking the orb with two columns
    /// in an HStack means an empty column on one side shoves it off centre,
    /// and home spends most of its life with an uneven number of cards (Now
    /// Playing comes and goes on its own). Stacking instead keeps the orb
    /// nailed to the middle no matter what appears beside it.
    private var homeLayout: some View {
        ZStack {
            HStack(alignment: .top, spacing: 24) {
                slotColumn(["L1", "L2", "L3"])
                Spacer(minLength: 40)
                slotColumn(["R1", "R2", "R3"])
            }

            VStack(spacing: 10) {
                OrbView(state: vm.state, density: .reactor)
                    .frame(maxWidth: 380, maxHeight: 380)
                    .aspectRatio(1, contentMode: .fit)
                greeting
                readout
            }
            // The orb owns the middle and must never be pushed by a card, so
            // it does not participate in the columns' layout at all.
            .allowsHitTesting(false)

            VStack {
                Spacer(minLength: 0)
                HStack {
                    statusRow
                    Spacer(minLength: 0)
                }
            }
        }
    }

    /// One side of the home grid. Empty slots take no room, so a column with
    /// one card sits at the top rather than floating in the middle of a gap.
    private func slotColumn(_ slots: [String]) -> some View {
        VStack(spacing: 14) {
            ForEach(slots, id: \.self) { slot in
                if let block = panel.block(inSlot: slot) {
                    PanelView(panel: Panel(single: block), tint: vm.state.tint,
                              compact: true)
                        .matchedGeometryEffect(id: block.card.isEmpty ? slot : block.card,
                                               in: cardSpace)
                        .transition(.asymmetric(
                            insertion: .opacity.combined(with: .scale(scale: 0.94))
                                .combined(with: .move(edge: .bottom)),
                            removal: .opacity.combined(with: .scale(scale: 0.96))))
                }
            }
            Spacer(minLength: 0)
        }
        .frame(width: 250)
    }

    /// CPU, memory and battery along the very bottom left. Absent entirely
    /// unless Nova is showing it — at launch, or because he just asked.
    @ViewBuilder
    private var statusRow: some View {
        let readings = panel.statusReadings
        if !readings.isEmpty {
            StatusRow(readings: readings, tint: vm.state.tint)
                .transition(.opacity.combined(with: .move(edge: .leading)))
                .animation(.easeOut(duration: 0.4), value: readings.count)
        }
    }

    /// WORKING: the orb steps aside and shrinks, and the room it gives up
    /// becomes what she is doing right now.
    ///
    /// Same orb, same panel renderer — only the proportions change, so the
    /// transition is a movement rather than a screen swap. That is what makes
    /// it read as Nova turning to a task instead of the app changing pages.
    private var workLayout: some View {
        HStack(spacing: 26) {
            VStack(spacing: 8) {
                Spacer(minLength: 0)
                OrbView(state: vm.state, density: .reactor)
                    .frame(maxWidth: 230, maxHeight: 230)
                    .aspectRatio(1, contentMode: .fit)
                readout
                Spacer(minLength: 0)
            }
            .frame(width: 260)

            PanelView(panel: panel, tint: vm.state.tint)
                .frame(maxWidth: .infinity)
                .transition(.opacity.combined(with: .move(edge: .trailing)))
        }
    }

    /// An ANSWER: the orb steps aside and the panel gets the room.
    private var answerLayout: some View {
        HStack(spacing: 26) {
            orbColumn
                .frame(maxWidth: panel.isEmpty ? .infinity : 340)
            if !panel.isEmpty {
                PanelView(panel: panel, tint: vm.state.tint)
                    .frame(maxWidth: .infinity)
                    .transition(.opacity)
            }
        }
    }

    /// The greeting, under the orb, as in his concept — and it goes away the
    /// moment he speaks. The backend signals that by clearing the panel title
    /// (views.py `spoken_yet`), so BOTH halves hang off it. Conditioning only
    /// the "good evening" line left NICHOLAS marooned under the orb for the
    /// rest of the session.
    @ViewBuilder
    private var greeting: some View {
        if !panel.title.isEmpty {
            VStack(spacing: 2) {
                Text(panel.title.uppercased())
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .tracking(2.6)
                    .foregroundStyle(.white.opacity(0.45))
                Text(userName.uppercased())
                    .font(.system(size: 30, weight: .light))
                    .tracking(6)
                    .foregroundStyle(vm.state.tint.opacity(0.95))
            }
            .transition(.opacity)
        }
    }

    private var userName: String { "Nicholas" }

    private var orbColumn: some View {
        VStack(spacing: 6) {
            Spacer(minLength: 0)
            OrbView(state: vm.state, density: .reactor)
                .frame(maxWidth: 420, maxHeight: 420)
                .aspectRatio(1, contentMode: .fit)
            readout
            Spacer(minLength: 0)
        }
    }

    /// The screen Nova last put up. Empty means orb only.
    private var panel: Panel { Panel(vm.viewData) }

    /// Tiny, dim, and easy to ignore — but it is the only thing that
    /// distinguishes idle from sleeping at a glance.
    private var readout: some View {
        Text(vm.state.readout.uppercased())
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .tracking(2.4)
            .foregroundStyle(vm.state.tint.opacity(vm.state == .sleeping ? 0.35 : 0.6))
            .animation(.easeInOut(duration: 0.35), value: vm.state)
            .accessibilityHidden(true)   // the orb already carries this
    }

    private var composer: some View {
        HStack(spacing: 10) {
            TextField("Type to Nova", text: $draft)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .focused($draftFocused)
                .onSubmit(send)

            Text("REPLIES IN TEXT")
                .font(.system(size: 8, weight: .medium, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(.white.opacity(0.28))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .stroke(Color.white.opacity(0.09), lineWidth: 1)
                )
        )
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    private func toggleTyping() {
        withAnimation(.easeOut(duration: 0.18)) { typing.toggle() }
        draftFocused = typing
    }

    private func send() {
        // Typed in, typed back: he is typing because he cannot talk.
        vm.send(draft, silent: true)
        draft = ""
        withAnimation(.easeOut(duration: 0.18)) { typing = false }
    }
}
