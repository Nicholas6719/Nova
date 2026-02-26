//
//  DebugLog.swift
//  Nova
//
//  Minimal debug logging; only prints in DEBUG when enabled.
//  MUST be non-blocking: no DispatchQueue, no locks, no MainActor hops.
//

import Foundation

enum DebugLog {
    /// Enable/disable debug logs at compile-time
    static let enabled: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()

    /// MUST be non-blocking: no DispatchQueue.sync, no locks, no MainActor hops.
    @inline(__always)
    static func d(_ message: String) {
        guard enabled else { return }
        print(message)
    }
}
