//
//  LLMResponseCache.swift
//  Nova
//
//  In-memory cache for OpenAI fallback responses. Safe, low-cost, reversible.
//  Fully nonisolated with serial DispatchQueue to avoid actor deadlocks.
//

import Foundation

/// Thread-safe in-memory cache for LLM responses. Evicts by TTL and LRU when full.
/// Synchronous get/set; accessible from any isolation context.
final class LLMResponseCache: @unchecked Sendable {

    nonisolated static let shared = LLMResponseCache()

    private let queue = DispatchQueue(label: "nova.llmResponseCache")
    nonisolated(unsafe) private var entries: [String: (value: String, timestamp: Date)] = [:]
    private let maxEntries: Int
    private let ttlSeconds: TimeInterval

    init(maxEntries: Int = 50, ttlSeconds: TimeInterval = 1800) {
        self.maxEntries = maxEntries
        self.ttlSeconds = ttlSeconds
    }

    nonisolated func get(key: String) -> String? {
        queue.sync {
            evictIfNeededUnsafe()
            guard let entry = entries[key] else { return nil }
            guard Date().timeIntervalSince(entry.timestamp) < ttlSeconds else {
                entries.removeValue(forKey: key)
                return nil
            }
            return entry.value
        }
    }

    nonisolated func set(key: String, value: String) {
        queue.sync {
            evictIfNeededUnsafe()
            entries[key] = (value: value, timestamp: Date())
            while entries.count > maxEntries, let oldest = entries.min(by: { $0.value.timestamp < $1.value.timestamp }) {
                entries.removeValue(forKey: oldest.key)
            }
        }
    }

    private nonisolated func evictIfNeededUnsafe() {
        let now = Date()
        let expiredKeys = entries.filter { now.timeIntervalSince($0.value.timestamp) >= ttlSeconds }.map(\.key)
        for k in expiredKeys {
            entries.removeValue(forKey: k)
        }
    }
}
