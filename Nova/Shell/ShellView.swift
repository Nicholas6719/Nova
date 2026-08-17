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
            HStack(spacing: 26) {
                orbColumn
                    // With a panel up the orb steps aside rather than dominating;
                    // on its own it takes the whole window.
                    .frame(maxWidth: panel.isEmpty ? .infinity : 340)

                if !panel.isEmpty {
                    PanelView(panel: panel, tint: vm.state.tint)
                        .frame(maxWidth: .infinity)
                        .transition(.opacity)
                }
            }
            if typing { composer.padding(.top, 20) }
        }
        .padding(28)
        .animation(.easeOut(duration: 0.22), value: panel.isEmpty)
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
