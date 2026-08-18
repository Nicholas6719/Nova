//
//  NovaApp.swift
//  Nova
//
//  Nova — Neural Omniscient Voice Assistant. Minimal entry point.
//

import Combine
import SwiftUI
import AppKit

@main
struct NovaApp: App {
    /// Owns and supervises the Python backend process for the app's lifetime.
    /// Created here (not in the view model) so it survives view re-creation and
    /// there is a single process owner.
    @StateObject private var backendManager = BackendManager()

    /// Owns CoreLocation. It lives in Swift because only the APP bundle carries
    /// NSLocationWhenInUseUsageDescription — a python subprocess asking for
    /// location is silently ignored and never prompts. See LocationProvider.
    @StateObject private var locationProvider = LocationProvider()

    /// Captures the system audio mix and streams it to the backend, so the
    /// echo canceller finally has a reference for everything the speakers play
    /// rather than only for Nova's own voice. Lives here for the same reason
    /// LocationProvider does: ScreenCaptureKit is delegate-driven and needs the
    /// app's run loop and its Screen Recording grant, neither of which a
    /// headless python child has.
    @StateObject private var systemAudio = SystemAudioTapBox()

    /// The AppDelegate exists solely to terminate the backend on app quit —
    /// SwiftUI's App has no reliable "will terminate" hook, but AppKit does.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ShellView()
                .onAppear {
                    appDelegate.backendManager = backendManager
                    backendManager.start()
                    // Ask once at launch so the permission dialog appears while
                    // Nicholas is actually at the keyboard, and so NovaOS shows
                    // up under Privacy & Security > Location Services.
                    locationProvider.requestLocation()
                    // After the backend, so its socket is listening before
                    // the first packet arrives.
                    systemAudio.start()
                }
        }
        // The SwiftUI-native way, and the one that actually works. Measured:
        // titlebarAppearsTransparent, fullSizeContentView, a black window
        // background and hiding NSTitlebarContainerView all applied cleanly and
        // STILL left a 32pt #1D1F20 strip across the top, because the scene
        // reserves and paints that area itself. AppKit was the wrong layer to
        // fight this at.
        .windowStyle(.hiddenTitleBar)
        // Part of "Nova always opens full size". macOS restores the previous
        // frame AFTER onAppear, so setting the size from the view alone loses
        // the race; this establishes the size before restoration can apply.
        .defaultSize(width: 1120, height: 760)
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


/// Owns the tap and keeps it alive for the life of the app.
///
/// A tiny ObservableObject wrapper because SCStream capture is only available
/// on macOS 13+, and `@StateObject` cannot itself be conditioned on
/// availability — the check has to live inside.
final class SystemAudioTapBox: ObservableObject {
    private var tap: AnyObject?

    func start() {
        guard #available(macOS 13.0, *) else {
            NSLog("[Nova] system audio tap needs macOS 13 or newer")
            return
        }
        let tap = SystemAudioTap()
        self.tap = tap
        tap.start()
    }

    func stop() {
        if #available(macOS 13.0, *), let tap = tap as? SystemAudioTap {
            tap.stop()
        }
    }
}
