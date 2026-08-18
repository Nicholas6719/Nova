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
            }
            .opacity(0)
        )
    }

    private var fullShell: some View {
        VStack(spacing: 0) {
            if isHome {
                homeLayout
            } else {
                answerLayout
            }
            if typing { composer.padding(.top, 20) }
        }
        .padding(28)
        .animation(.easeOut(duration: 0.24), value: vm.view)
        .animation(.easeOut(duration: 0.22), value: panel.isEmpty)
    }

    private var isHome: Bool { vm.view == "home" }

    /// HOME: the orb is the centre of the screen, the greeting sits under it,
    /// and what he wants at a glance flanks it. This is his concept, and it is
    /// a different shape from an answer — an answer steps the orb aside, home
    /// puts it in the middle.
    private var homeLayout: some View {
        HStack(alignment: .center, spacing: 24) {
            homeCards(Array(panel.blocks.prefix(1)))
                .frame(width: 250)

            VStack(spacing: 10) {
                Spacer(minLength: 0)
                OrbView(state: vm.state, density: .reactor)
                    .frame(maxWidth: 380, maxHeight: 380)
                    .aspectRatio(1, contentMode: .fit)
                greeting
                readout
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity)

            homeCards(Array(panel.blocks.dropFirst(1)))
                .frame(width: 250)
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

    /// The greeting, under the orb, as in his concept.
    private var greeting: some View {
        VStack(spacing: 2) {
            if !panel.title.isEmpty {
                Text(panel.title.uppercased())
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .tracking(2.6)
                    .foregroundStyle(.white.opacity(0.45))
            }
            Text(userName.uppercased())
                .font(.system(size: 30, weight: .light))
                .tracking(6)
                .foregroundStyle(vm.state.tint.opacity(0.95))
        }
    }

    private var userName: String { "Nicholas" }

    /// Home's blocks as cards down a side column, rather than one long panel.
    @ViewBuilder
    private func homeCards(_ blocks: [PanelBlock]) -> some View {
        if blocks.isEmpty {
            Color.clear
        } else {
            VStack(spacing: 14) {
                ForEach(blocks) { b in
                    PanelView(panel: Panel(single: b), tint: vm.state.tint,
                              compact: true)
                }
                Spacer(minLength: 0)
            }
        }
    }

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
