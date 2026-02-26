//
//  Log.swift
//  Nova
//
//  Flow logging for debugging.
//

import Foundation

/// Thread-safe logging; callable from any context (nonisolated).
nonisolated func log(_ message: String) {
    #if DEBUG
    print("[Nova] \(message)")
    #endif
}
