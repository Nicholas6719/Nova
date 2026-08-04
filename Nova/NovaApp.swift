//
//  NovaApp.swift
//  Nova
//
//  Nova — Neural Omniscient Voice Assistant. Minimal entry point.
//

import SwiftUI

@main
struct NovaApp: App {
    /// Owns and supervises the Python backend process for the app's lifetime.
    /// Created here (not in ChatViewModel) so it survives view re-creation and
    /// there is a single process owner.
    @StateObject private var backendManager = BackendManager()

    var body: some Scene {
        WindowGroup {
            ContentView(backendManager: backendManager)
                .onAppear { backendManager.start() }
        }
    }
}
