//
//  MemoryContext.swift
//  Nova
//
//  Lightweight conversation context for memory follow-up corrections.
//  Tracks last memory key/value discussed for "actually it's X" updates.
//

import Foundation

/// Session-scoped context for memory follow-up. Thread-safe.
enum MemoryContext: Sendable {
    private static let lock = NSLock()
    private static var _lastKey: String?
    private static var _lastValue: String?

    /// Last memory key we saved, recalled, or updated. Used for "actually it's X" follow-up.
    static func lastKey() -> String? {
        lock.lock()
        defer { lock.unlock() }
        return _lastKey
    }

    /// Last memory value we returned or stored. Used for "not X" confirmation.
    static func lastValue() -> String? {
        lock.lock()
        defer { lock.unlock() }
        return _lastValue
    }

    /// Update after save, recall, or correction. Call when memory interaction completes.
    static func updateLastDiscussed(key: String, value: String) {
        lock.lock()
        defer { lock.unlock() }
        _lastKey = key
        _lastValue = value
    }
}
