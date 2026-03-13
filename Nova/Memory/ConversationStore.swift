//
//  ConversationStore.swift
//  Nova
//
//  Conversation history persistence. v1: versioned JSON on disk.
//  Crash-safe: no force unwraps, try!, or fatalError. All failures return nil/false/[].
//

import Foundation

/// Persists conversation history to disk. JSON-backed, app-local. Thread-safe for single-process use.
struct ConversationStore: Sendable {

    private static let fileName = "conversations.json"
    private static let lock = NSLock()
    private static let currentVersion = 1

    // MARK: - Persistence-only types (decoupled from UI Message model)

    private struct PersistedMessage: Codable {
        let id: String
        let role: String
        let content: String
        let timestamp: TimeInterval

        static func from(_ message: Message) -> PersistedMessage {
            PersistedMessage(
                id: message.id.uuidString,
                role: message.role == .user ? "user" : "assistant",
                content: message.content,
                timestamp: message.timestamp.timeIntervalSince1970
            )
        }

        func toMessage() -> Message? {
            guard let uuid = UUID(uuidString: id) else { return nil }
            let messageRole: MessageRole
            switch role {
            case "user": messageRole = .user
            case "assistant": messageRole = .assistant
            default: return nil
            }
            return Message(
                id: uuid,
                role: messageRole,
                content: content,
                timestamp: Date(timeIntervalSince1970: timestamp)
            )
        }
    }

    private struct ConversationFile: Codable {
        let version: Int
        let messages: [PersistedMessage]
    }

    // MARK: - Storage

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
                DebugLog.d("[Conversation] directory creation failed: \(error.localizedDescription)")
                return nil
            }
        }
        return novaDir.appendingPathComponent(fileName)
    }

    // MARK: - Public API

    /// Load conversation from disk. Returns empty array on any failure; never throws or crashes.
    static func load() -> [Message] {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else { return [] }
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        guard let data = try? Data(contentsOf: url) else {
            DebugLog.d("[Conversation] load failed (read), using empty history")
            return []
        }
        guard let file = try? JSONDecoder().decode(ConversationFile.self, from: data) else {
            DebugLog.d("[Conversation] load failed (decode), using empty history")
            return []
        }
        guard file.version <= currentVersion else {
            DebugLog.d("[Conversation] unknown version \(file.version), using empty history")
            return []
        }
        let messages = file.messages.compactMap { $0.toMessage() }
        return messages
    }

    /// Save conversation to disk. Returns false on any failure.
    @discardableResult
    static func save(_ messages: [Message]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else { return false }
        let persisted = messages.map { PersistedMessage.from($0) }
        let file = ConversationFile(version: currentVersion, messages: persisted)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(file) else {
            DebugLog.d("[Conversation] save failed (encode)")
            return false
        }
        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            DebugLog.d("[Conversation] save failed (write): \(error.localizedDescription)")
            return false
        }
    }

    /// Delete conversation file. Returns false on any failure.
    @discardableResult
    static func clear() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else { return false }
        guard FileManager.default.fileExists(atPath: url.path) else { return true }
        do {
            try FileManager.default.removeItem(at: url)
            DebugLog.d("[Conversation] cleared")
            return true
        } catch {
            DebugLog.d("[Conversation] clear failed: \(error.localizedDescription)")
            return false
        }
    }
}
