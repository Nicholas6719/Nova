//
//  NovaApp.swift
//  Nova
//
//  Nova — Neural Omniscient Voice Assistant. Minimal entry point.
//

import SwiftUI
import AppKit

@main
struct NovaApp: App {
    /// Owns and supervises the Python backend process for the app's lifetime.
    /// Created here (not in ChatViewModel) so it survives view re-creation and
    /// there is a single process owner.
    @StateObject private var backendManager = BackendManager()

    /// The AppDelegate exists solely to terminate the backend on app quit —
    /// SwiftUI's App has no reliable "will terminate" hook, but AppKit does.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView(backendManager: backendManager)
                .onAppear {
                    appDelegate.backendManager = backendManager
                    backendManager.start()
                }
        }
    }
}

/// Ensures the Python backend is stopped when the app quits, so it never lingers
/// headless (still holding the microphone) after the window closes.
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var backendManager: BackendManager?

    func applicationWillTerminate(_ notification: Notification) {
        backendManager?.stop()
    }

    // Quitting the app when its last window closes keeps the lifecycle simple:
    // no windowless app left running the backend.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
