//
//  MemoryStore.swift
//  Nova
//
//  Local-only memory persistence. v1: [String: String] store with JSON on disk.
//  Crash-safe: no force unwraps, try!, or fatalError. All failures return nil/false.
//

import Foundation

/// Local memory store. JSON-backed, app-local. Thread-safe for single-process use.
struct MemoryStore: Sendable {

    private static let fileName = "memories.json"
    private static let lock = NSLock()

    /// Supported memory keys for v1.
    static let supportedKeys: Set<String> = [
        "name", "nickname", "favorite_ide", "favorite_game", "favorite_color",
        "favorite_food", "hometown", "company", "job"
    ]

    /// Resolve storage URL and ensure directory exists. Returns nil on any failure.
    private static func resolveStorageURL() -> URL? {
        guard let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        let novaDir = dir.appendingPathComponent("Nova", isDirectory: true)
        if !FileManager.default.fileExists(atPath: novaDir.path) {
            do {
                try FileManager.default.createDirectory(at: novaDir, withIntermediateDirectories: true)
            } catch {
                DebugLog.d("[Memory] directory creation failed: \(error.localizedDescription)")
                return nil
            }
        }
        return novaDir.appendingPathComponent(fileName)
    }

    /// Load from disk. Returns empty dict on any failure; never throws or crashes.
    static func load() -> [String: String] {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else {
            return [:]
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            return [:]
        }
        guard let data = try? Data(contentsOf: url) else {
            DebugLog.d("[Memory] load failed (read), using empty store")
            return [:]
        }
        guard let dict = try? JSONDecoder().decode([String: String].self, from: data) else {
            DebugLog.d("[Memory] load failed (decode), using empty store")
            return [:]
        }
        return dict.filter { supportedKeys.contains($0.key) }
    }

    /// Save to disk. Returns false on any failure.
    static func save(_ dict: [String: String]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else {
            return false
        }
        let filtered = dict.filter { supportedKeys.contains($0.key) }
        guard let data = try? JSONEncoder().encode(filtered) else {
            DebugLog.d("[Memory] save failed (encode)")
            return false
        }
        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            DebugLog.d("[Memory] save failed (write): \(error.localizedDescription)")
            return false
        }
    }

    /// Get value for key. Returns nil if missing or any error.
    static func get(_ key: String) -> String? {
        let dict = load()
        return dict[key]
    }

    /// Set value for key and persist. Returns false if save fails.
    static func set(_ key: String, value: String) -> Bool {
        guard supportedKeys.contains(key) else { return false }
        var dict = load()
        dict[key] = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let ok = save(dict)
        if ok {
            DebugLog.d("[Memory] save success key=\(key)")
        } else {
            DebugLog.d("[Memory] save failed key=\(key)")
        }
        return ok
    }
}
