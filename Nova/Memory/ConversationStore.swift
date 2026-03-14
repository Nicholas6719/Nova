//
//  ConversationStore.swift
//  Nova
//
//  Conversation history persistence. v2: active conversation with metadata + day-based archives.
//  Crash-safe: no force unwraps, try!, or fatalError. All failures return nil/false/[].
//

import Foundation

/// Persists conversation history to disk. JSON-backed, app-local. Thread-safe for single-process use.
struct ConversationStore: Sendable {

    private static let fileName = "conversations.json"
    private static let archiveDirName = "archives"
    private static let lock = NSLock()
    private static let currentVersion = 2

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

    /// Wraps a single conversation with metadata.
    private struct ConversationEnvelope: Codable {
        let id: String              // UUID string
        let startDate: TimeInterval // seconds since 1970
        let messages: [PersistedMessage]
    }

    // MARK: - File formats

    /// v1 format (legacy): flat message array.
    private struct ConversationFileV1: Codable {
        let version: Int
        let messages: [PersistedMessage]
    }

    /// v2 format: active conversation with metadata.
    private struct ActiveFileV2: Codable {
        let version: Int
        let conversation: ConversationEnvelope
    }

    /// Archive format: array of conversations for a single calendar day.
    private struct ArchiveFile: Codable {
        let version: Int
        let conversations: [ConversationEnvelope]
    }

    /// Generic version detector.
    private struct VersionOnly: Codable {
        let version: Int
    }

    // MARK: - Storage helpers

    /// Resolve storage URL for conversations.json. Creates Nova dir if needed.
    private static func resolveStorageURL() -> URL? {
        guard let novaDir = resolveNovaDir() else { return nil }
        return novaDir.appendingPathComponent(fileName)
    }

    /// Resolve the Nova application support directory. Creates it if needed.
    private static func resolveNovaDir() -> URL? {
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
        return novaDir
    }

    /// Resolve the archives subdirectory. Creates it if needed.
    private static func resolveArchiveDir() -> URL? {
        guard let novaDir = resolveNovaDir() else { return nil }
        let archiveDir = novaDir.appendingPathComponent(archiveDirName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: archiveDir.path) {
            do {
                try FileManager.default.createDirectory(at: archiveDir, withIntermediateDirectories: true)
            } catch {
                DebugLog.d("[Conversation] archive directory creation failed: \(error.localizedDescription)")
                return nil
            }
        }
        return archiveDir
    }

    /// Date formatter for archive filenames (YYYY-MM-DD.json).
    private static func archiveFileName(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return "\(formatter.string(from: date)).json"
    }

    // MARK: - Rollover check

    /// Returns true when the active conversation belongs to a different calendar day than today.
    /// Rollover should happen only when this returns true.
    static func needsRollover(startDate: Date) -> Bool {
        return !Calendar.current.isDate(startDate, inSameDayAs: Date())
    }

    // MARK: - v2 Active conversation API

    /// Load the active conversation. Returns nil if no file, empty, or any failure.
    /// Handles v1 → v2 migration transparently.
    static func loadActive() -> (id: UUID, startDate: Date, messages: [Message])? {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else { return nil }
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let data = try? Data(contentsOf: url) else {
            DebugLog.d("[Conversation] loadActive failed (read)")
            return nil
        }

        // Detect version
        guard let versionInfo = try? JSONDecoder().decode(VersionOnly.self, from: data) else {
            DebugLog.d("[Conversation] loadActive failed (version decode)")
            return nil
        }

        if versionInfo.version == 1 {
            // v1 migration: read flat messages, wrap in envelope
            return migrateV1(data: data)
        }

        guard versionInfo.version <= currentVersion else {
            DebugLog.d("[Conversation] unknown version \(versionInfo.version)")
            return nil
        }

        // v2 decode
        guard let file = try? JSONDecoder().decode(ActiveFileV2.self, from: data) else {
            DebugLog.d("[Conversation] loadActive failed (v2 decode)")
            return nil
        }

        guard let uuid = UUID(uuidString: file.conversation.id) else { return nil }
        let messages = file.conversation.messages.compactMap { $0.toMessage() }
        let startDate = Date(timeIntervalSince1970: file.conversation.startDate)
        return (id: uuid, startDate: startDate, messages: messages)
    }

    /// Save the active conversation in v2 format. Returns false on any failure.
    @discardableResult
    static func saveActive(id: UUID, startDate: Date, messages: [Message]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let url = resolveStorageURL() else { return false }
        let envelope = ConversationEnvelope(
            id: id.uuidString,
            startDate: startDate.timeIntervalSince1970,
            messages: messages.map { PersistedMessage.from($0) }
        )
        let file = ActiveFileV2(version: currentVersion, conversation: envelope)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(file) else {
            DebugLog.d("[Conversation] saveActive failed (encode)")
            return false
        }
        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            DebugLog.d("[Conversation] saveActive failed (write): \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - Archive API

    /// Archive a conversation to the day-based archive file. Appends to existing day file.
    /// Returns false on any failure.
    @discardableResult
    static func archive(id: UUID, startDate: Date, messages: [Message]) -> Bool {
        guard !messages.isEmpty else { return true } // nothing to archive
        lock.lock()
        defer { lock.unlock() }
        guard let archiveDir = resolveArchiveDir() else { return false }

        let fileName = archiveFileName(for: startDate)
        let fileURL = archiveDir.appendingPathComponent(fileName)

        let envelope = ConversationEnvelope(
            id: id.uuidString,
            startDate: startDate.timeIntervalSince1970,
            messages: messages.map { PersistedMessage.from($0) }
        )

        // Load existing day archive if present, append new conversation
        var conversations: [ConversationEnvelope] = []
        if FileManager.default.fileExists(atPath: fileURL.path),
           let data = try? Data(contentsOf: fileURL),
           let existing = try? JSONDecoder().decode(ArchiveFile.self, from: data) {
            conversations = existing.conversations
        }
        conversations.append(envelope)

        let archiveFile = ArchiveFile(version: currentVersion, conversations: conversations)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(archiveFile) else {
            DebugLog.d("[Conversation] archive failed (encode)")
            return false
        }
        do {
            try data.write(to: fileURL, options: .atomic)
            DebugLog.d("[Conversation] archived \(messages.count) messages to \(fileName)")
            return true
        } catch {
            DebugLog.d("[Conversation] archive failed (write): \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - v1 Migration

    /// Migrate v1 flat message array to v2 format. Returns the conversation tuple or nil.
    private static func migrateV1(data: Data) -> (id: UUID, startDate: Date, messages: [Message])? {
        guard let v1File = try? JSONDecoder().decode(ConversationFileV1.self, from: data) else {
            DebugLog.d("[Conversation] v1 migration failed (decode)")
            return nil
        }
        let messages = v1File.messages.compactMap { $0.toMessage() }
        guard !messages.isEmpty else { return nil }

        let newId = UUID()
        let startDate = messages.first?.timestamp ?? Date()

        DebugLog.d("[Conversation] migrating v1 → v2 (\(messages.count) messages)")

        // Write v2 format immediately so future loads use v2
        let envelope = ConversationEnvelope(
            id: newId.uuidString,
            startDate: startDate.timeIntervalSince1970,
            messages: v1File.messages
        )
        let v2File = ActiveFileV2(version: currentVersion, conversation: envelope)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        if let v2Data = try? encoder.encode(v2File),
           let url = resolveStorageURL() {
            // Note: lock is already held by caller (loadActive)
            try? v2Data.write(to: url, options: .atomic)
            DebugLog.d("[Conversation] v1 → v2 migration complete")
        }

        return (id: newId, startDate: startDate, messages: messages)
    }

    // MARK: - Legacy v1 API (temporary backward compat)

    /// Load conversation from disk (v1 compat). Returns empty array on any failure.
    static func load() -> [Message] {
        guard let result = loadActive() else { return [] }
        return result.messages
    }

    /// Save conversation to disk (v1 compat). Returns false on any failure.
    @discardableResult
    static func save(_ messages: [Message]) -> Bool {
        // Delegate to saveActive with a default id/date. This keeps old call sites working
        // until ChatViewModel is updated in Milestone 2.
        return saveActive(id: UUID(), startDate: Date(), messages: messages)
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
