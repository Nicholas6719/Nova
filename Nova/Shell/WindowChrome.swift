//
//  WindowChrome.swift
//  Nova
//
//  The window itself: frameless at full size, and a small puck parked in the
//  corner while Nova works alongside him.
//
//  The puck is the state Nova spends most of her life in. It has to float above
//  everything — including fullscreen apps and other Spaces — because the whole
//  point is working together in whatever app he is already in.
//
//  Nova ALWAYS launches full size, even if she was quit while parked. Coming
//  back as a dot in the corner with no memory of why is a bad first
//  second of the day.
//

import AppKit
import SwiftUI

enum WindowChrome {

    /// RileyJarvis used 190. Too big for something that sits in the corner all
    /// day while he works in another app — this is deliberately smaller.
    static let puckSize: CGFloat = 130
    private static let margin: CGFloat = 18

    /// Bounds to restore when leaving puck mode. Captured on the way in.
    private static var restoreFrame: NSRect?

    private static var window: NSWindow? {
        NSApp.windows.first { $0.isVisible && !($0 is NSPanel) }
    }

    /// Black background, draggable anywhere, no traffic lights.
    ///
    /// The titlebar itself is removed by `.windowStyle(.hiddenTitleBar)` on the
    /// scene, NOT here. That matters, because AppKit is the wrong layer for it:
    /// measured, `titlebarAppearsTransparent` + `fullSizeContentView` + a black
    /// window background + hiding `NSTitlebarContainerView` all applied cleanly
    /// and still left a 32pt #1D1F20 strip, because the scene reserves and
    /// paints that area itself.
    static func makeFrameless() {
        guard let w = window else { return }
        w.isMovableByWindowBackground = true
        w.isOpaque = true
        w.backgroundColor = .black
        w.hasShadow = true

        // Nova ALWAYS opens full size. macOS otherwise restores the last frame,
        // so quitting while parked brought her back as a 130pt dot in the
        // corner with the full-size layout crammed into it — orb the size of a
        // coin, readout wrapping. Restoration is turned off, and any restored
        // puck-sized frame is corrected on the way in.
        w.isRestorable = false
        if !isPuckMode && (w.frame.width < 420 || w.frame.height < 520) {
            w.setFrame(NSRect(x: 0, y: 0, width: 1120, height: 760), display: false)
            w.center()
        }

        hideButtons(w)

        if !isPuckMode { w.minSize = NSSize(width: 420, height: 520) }
    }

    /// The orb IS the window; the traffic lights would be the only thing on it
    /// that isn't the orb. Cmd-W and Cmd-Q still work, and the window drags from
    /// anywhere.
    ///
    /// Called after EVERY transition, not once: changing the style mask on the
    /// way into puck mode rebuilds the frame view and brings the buttons back,
    /// so they reappeared on top of the puck.
    private static func hideButtons(_ w: NSWindow) {
        for button in [NSWindow.ButtonType.closeButton,
                       .miniaturizeButton, .zoomButton] {
            w.standardWindowButton(button)?.isHidden = true
        }
    }

    /// Tracked so re-applying the frameless chrome does not undo the puck's
    /// smaller minimum size and knock it back to 420pt wide.
    private static var isPuckMode = false

    static func setPuck(_ on: Bool) {
        guard let w = window else { return }
        on ? enterPuck(w) : leavePuck(w)
    }

    private static func enterPuck(_ w: NSWindow) {
        isPuckMode = true
        let current = w.frame
        // Guard against saving an already-puck frame as the restore target,
        // which would strand him at puck size forever.
        if current.width > 400 && current.height > 400 {
            restoreFrame = current
        }

        // Park on whichever screen the cursor is on, not whichever screen the
        // window happened to be on — he is working over there.
        let mouse = NSEvent.mouseLocation
        let screen = NSScreen.screens.first { $0.frame.contains(mouse) }
            ?? w.screen ?? NSScreen.main
        let area = screen?.visibleFrame ?? .zero

        w.minSize = NSSize(width: 110, height: 110)
        w.isRestorable = false
        w.level = .floating
        w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary,
                                .stationary]
        w.setFrame(NSRect(x: area.minX + margin,
                          y: area.minY + margin,
                          width: puckSize, height: puckSize),
                   display: true, animate: true)
        w.styleMask.remove(.resizable)
        hideButtons(w)
    }

    private static func leavePuck(_ w: NSWindow) {
        isPuckMode = false
        w.level = .normal
        w.collectionBehavior = [.fullScreenPrimary]
        w.styleMask.insert(.resizable)
        w.minSize = NSSize(width: 420, height: 520)
        if let frame = restoreFrame {
            w.setFrame(frame, display: true, animate: true)
        } else {
            w.setFrame(NSRect(x: 0, y: 0, width: 1120, height: 760),
                       display: true, animate: true)
            w.center()
        }
        hideButtons(w)
    }
}
