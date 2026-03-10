//
//  NovaToolResult.swift
//  Nova
//
//  Shared result type for tool execution.
//

import Foundation

/// Result of a tool execution. Sendable for use from async contexts.
struct NovaToolResult: Sendable {
    let success: Bool
    let spokenResponse: String
    let didPerformAction: Bool

    nonisolated static func success(spoken: String, didPerform: Bool = true) -> NovaToolResult {
        NovaToolResult(success: true, spokenResponse: spoken, didPerformAction: didPerform)
    }

    nonisolated static func failure(spoken: String) -> NovaToolResult {
        NovaToolResult(success: false, spokenResponse: spoken, didPerformAction: false)
    }
}
