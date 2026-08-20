//
//  ShellViewModel.swift
//  Nova
//
//  What the window needs to know: which state Nova is in, which screen she is
//  showing, and whether she is parked in the corner.
//
//  This REPLACES ChatViewModel rather than untangling it. That file was 1,022
//  lines of the old on-device pipeline — Swift-side wake word, recording
//  sessions, sentence-by-sentence TTS, auto-listen — all of which the Python
//  backend has owned since the migration. It was dead weight wrapped around a
//  live WebSocket client, and picking it apart would have been surgery on a
//  state machine we are retiring anyway.
//

import Combine
import SwiftUI

@MainActor
final class ShellViewModel: ObservableObject {

    /// What Nova is doing. Drives the orb and the readout under it.
    @Published private(set) var state: NovaState = .idle
    /// Which screen is showing, and its panel data, as sent by the backend.
    @Published private(set) var view: String = "home"
    @Published private(set) var viewData: [String: Any] = [:]
    /// Parked in the corner, above everything, while working alongside him.
    @Published private(set) var isPuck: Bool = false
    @Published private(set) var isConnected: Bool = false

    private let api: NovaAPIClient
    private var cancellables = Set<AnyCancellable>()

    init() {
        self.api = NovaAPIClient()

        api.$currentState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] wire in self?.state = NovaState(wire: wire) }
            .store(in: &cancellables)

        api.$isConnected
            .receive(on: DispatchQueue.main)
            .assign(to: \.isConnected, on: self)
            .store(in: &cancellables)

        api.$currentView
            .receive(on: DispatchQueue.main)
            .sink { [weak self] payload in
                guard let payload else { return }
                self?.view = payload.name
                self?.viewData = payload.data
            }
            .store(in: &cancellables)

        // Work mode is decided by the backend — it knows when Nova acted on his
        // Mac and when he said "go home" — so the window follows it rather than
        // the two keeping separate opinions about where Nova should be.
        api.$isPuck
            .receive(on: DispatchQueue.main)
            .sink { [weak self] puck in self?.setPuck(puck) }
            .store(in: &cancellables)

        api.connect()
    }

    /// Typed input. Deliberately secondary — speech is the primary channel —
    /// but there has to be a way to reach Nova when he cannot talk.
    func send(_ text: String, silent: Bool) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        Task { await api.sendMessage(trimmed, silent: silent) }
    }

    /// Space bar: stop her talking, now.
    ///
    /// Fire and forget — the reply is irrelevant and waiting on it would make
    /// the one interaction that must feel instant depend on a round trip.
    func interrupt() {
        api.interrupt()
    }

    func setPuck(_ on: Bool) {
        guard isPuck != on else { return }
        isPuck = on
        WindowChrome.setPuck(on)
    }
}
