//
//  ToolRouter.swift
//  Nova
//
//  Routes user text to one of the first 3 tools. Simple, explicit matching.
//

import Foundation

enum ToolIntent: Sendable, Equatable {
    case openApp(name: String)
    case batteryStatus(chargingIntent: Bool)
    case webSearch(query: String)
    case none
}

struct ToolRouter: Sendable {

    /// Match user text to a tool intent. Returns .none if no match.
    static func match(from input: String) -> ToolIntent {
        let normalized = input
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "'", with: "")
        let words = normalized.split(separator: " ").map { String($0) }
        if words.isEmpty { return .none }

        // Strip leading wake/greeting words
        var rest = normalized
        for _ in 0..<6 {
            if rest.hasPrefix("nova ") {
                rest = String(rest.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            } else if rest.hasPrefix("hey nova ") {
                rest = String(rest.dropFirst(9)).trimmingCharacters(in: .whitespaces)
            } else if rest.hasPrefix("hi ") || rest.hasPrefix("hello ") {
                rest = String(rest.dropFirst(rest.prefix(5).count)).trimmingCharacters(in: .whitespaces)
            } else { break }
        }

        // --- TOOL 1: Open App / Settings ---
        if rest.hasPrefix("open ") || rest.hasPrefix("launch ") {
            let after = rest.hasPrefix("open ") ? String(rest.dropFirst(5)) : String(rest.dropFirst(7))
            let target = after.trimmingCharacters(in: .whitespacesAndNewlines)
            if !target.isEmpty {
                return .openApp(name: target)
            }
        }

        // --- TOOL 2: Battery ---
        let chargingPhrases = ["am i charging", "is my battery charging", "are we charging"]
        let isChargingIntent = chargingPhrases.contains { rest == $0 || rest.hasPrefix($0 + " ") }
        if isChargingIntent {
            return .batteryStatus(chargingIntent: true)
        }
        if rest.contains("battery") || rest.contains("charging") {
            return .batteryStatus(chargingIntent: false)
        }

        // --- TOOL 3: Web Search ---
        // "search the web for x", "search for x", "look up x", "google x"
        let searchPrefixes = ["search the web for ", "search for ", "look up ", "lookup ", "google ", "search web for "]
        for prefix in searchPrefixes {
            if rest.hasPrefix(prefix) {
                let query = String(rest.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
                if !query.isEmpty {
                    return .webSearch(query: query)
                }
                break
            }
        }
        if rest.hasPrefix("search ") {
            let after = String(rest.dropFirst(7))
            if !after.isEmpty && !after.hasPrefix("the ") {
                return .webSearch(query: after.trimmingCharacters(in: .whitespaces))
            }
        }

        return .none
    }
}
