//
//  Message.swift
//  Nova
//
//  Conversation message model for Nova (memory and context).
//

import Foundation

/// Role of the speaker in the conversation.
/// Must remain non-@MainActor for use from NovaEngineCore's nonisolated context.
enum MessageRole: Sendable, Equatable {
    case user
    case assistant

    /// Explicit nonisolated Equatable so comparisons work from nonisolated context.
    nonisolated static func == (lhs: MessageRole, rhs: MessageRole) -> Bool {
        switch (lhs, rhs) {
        case (.user, .user), (.assistant, .assistant): return true
        default: return false
        }
    }
}

/// A single message in the conversation history.
struct Message: Identifiable, Sendable {
    let id: UUID
    let role: MessageRole
    let content: String
    let timestamp: Date

    init(id: UUID = UUID(), role: MessageRole, content: String, timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}
