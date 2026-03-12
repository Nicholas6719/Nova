//
//  MemoryRouter.swift
//  Nova
//
//  Parses memory save and recall phrases. Deterministic, explicit matching only.
//

import Foundation

enum MemoryIntent: Sendable, Equatable {
    case save(key: String, value: String)
    case recall(key: String)
    case update(key: String, value: String, previousValue: String?)
}

struct MemoryRouter: Sendable {

    /// Map phrases to memory keys. Order matters for prefix matching.
    private static let recallPhraseToKey: [(String, String)] = [
        ("what company do i work at", "company"),
        ("what do i do for work", "job"),
        ("where am i from", "hometown"),
        ("whats my name", "name"),
        ("what is my name", "name"),
        ("whats my nickname", "nickname"),
        ("what is my nickname", "nickname"),
        ("whats my favorite ide", "favorite_ide"),
        ("what is my favorite ide", "favorite_ide"),
        ("whats my favorite editor", "favorite_ide"),
        ("what is my favorite editor", "favorite_ide"),
        ("do you remember my favorite ide", "favorite_ide"),
        ("whats my favorite game", "favorite_game"),
        ("what is my favorite game", "favorite_game"),
        ("do you remember my favorite game", "favorite_game"),
        ("do you remember my name", "name"),
        ("do you remember my nickname", "nickname"),
        ("do you remember my favorite color", "favorite_color"),
        ("do you remember my favorite food", "favorite_food"),
        ("do you remember my hometown", "hometown"),
        ("do you remember my company", "company"),
        ("do you remember my job", "job"),
        ("whats my favorite color", "favorite_color"),
        ("what is my favorite color", "favorite_color"),
        ("whats my favorite food", "favorite_food"),
        ("what is my favorite food", "favorite_food"),
        ("whats my hometown", "hometown"),
        ("what is my hometown", "hometown"),
        ("whats my company", "company"),
        ("what is my company", "company"),
        ("whats my job", "job"),
        ("what is my job", "job"),
    ]

    /// Normalize input for matching: lowercase, strip apostrophes.
    private static func normalize(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "'", with: "")
    }

    /// Strip leading "remember ", "remember that ", etc. for save parsing.
    private static func stripRememberPrefix(_ s: String) -> String? {
        let n = normalize(s)
        if n.hasPrefix("remember that ") {
            return String(n.dropFirst(13)).trimmingCharacters(in: .whitespaces)
        }
        if n.hasPrefix("remember ") {
            return String(n.dropFirst(9)).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    /// Try to match a save intent. Returns (key, value) or nil.
    static func matchSave(from input: String) -> (key: String, value: String)? {
        guard let afterRemember = stripRememberPrefix(input), !afterRemember.isEmpty else {
            return nil
        }
        let n = normalize(afterRemember)

        // "i work at <value>" -> company
        if n.hasPrefix("i work at ") {
            let value = String(n.dropFirst(10)).trimmingCharacters(in: .whitespaces)
            if !value.isEmpty { return ("company", value) }
        }

        // "my <field> is <value>"
        if n.hasPrefix("my ") {
            let rest = String(n.dropFirst(3))
            if let range = rest.range(of: " is ") {
                let fieldPart = String(rest[..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
                let valuePart = String(rest[range.upperBound...]).trimmingCharacters(in: .whitespaces)
                if !valuePart.isEmpty, let key = fieldPhraseToKey(fieldPart) {
                    return (key, valuePart)
                }
            }
        }

        return nil
    }

    /// Map "name", "favorite ide", "favorite editor", etc. to key.
    private static func fieldPhraseToKey(_ field: String) -> String? {
        switch field {
        case "name": return "name"
        case "nickname": return "nickname"
        case "favorite ide", "favorite editor": return "favorite_ide"
        case "favorite game": return "favorite_game"
        case "favorite color": return "favorite_color"
        case "favorite food": return "favorite_food"
        case "hometown": return "hometown"
        case "company": return "company"
        case "job": return "job"
        default: return nil
        }
    }

    /// Correction prefixes: "actually", "no", "correction"
    private static let correctionPrefixes = ["actually ", "no ", "correction "]

    /// Strip correction prefix. Returns remainder or nil if no match.
    private static func stripCorrectionPrefix(_ s: String) -> String? {
        let n = normalize(s)
        for prefix in correctionPrefixes {
            if n.hasPrefix(prefix) {
                return String(n.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    /// Try to match an update/correction intent.
    /// - Explicit: "actually my name is X" -> (key, value, nil)
    /// - Follow-up: "actually it's X" with lastKey -> (lastKey, value, previousValue)
    /// Returns nil if no match or follow-up has no context.
    static func matchUpdate(from input: String) -> (key: String, value: String, previousValue: String?)? {
        guard let afterPrefix = stripCorrectionPrefix(input), !afterPrefix.isEmpty else {
            return nil
        }
        let n = normalize(afterPrefix)

        // Explicit: "i work at <value>" -> company
        if n.hasPrefix("i work at ") {
            let valuePart = String(n.dropFirst(10)).trimmingCharacters(in: .whitespaces)
            if !valuePart.isEmpty {
                let prev = MemoryStore.get("company")
                return ("company", valuePart, prev)
            }
        }

        // Explicit: "my <field> is <value>"
        if n.hasPrefix("my ") {
            let rest = String(n.dropFirst(3))
            if let range = rest.range(of: " is ") {
                let fieldPart = String(rest[..<range.lowerBound]).trimmingCharacters(in: .whitespaces)
                let valuePart = String(rest[range.upperBound...]).trimmingCharacters(in: .whitespaces)
                if !valuePart.isEmpty, let key = fieldPhraseToKey(fieldPart) {
                    let prev = MemoryStore.get(key)
                    return (key, valuePart, prev)
                }
            }
        }

        // Follow-up: "it's <value>" or "it is <value>" — requires lastMemoryKeyDiscussed
        if n.hasPrefix("its ") || n.hasPrefix("it is ") {
            let valuePart: String
            if n.hasPrefix("its ") {
                valuePart = String(n.dropFirst(4)).trimmingCharacters(in: .whitespaces)
            } else {
                valuePart = String(n.dropFirst(6)).trimmingCharacters(in: .whitespaces)
            }
            guard !valuePart.isEmpty, let lastKey = MemoryContext.lastKey() else {
                return nil
            }
            let prev = MemoryContext.lastValue()
            return (lastKey, valuePart, prev)
        }

        return nil
    }

    /// Try to match a recall intent. Returns key or nil.
    static func matchRecall(from input: String) -> String? {
        let n = normalize(input)
        for (phrase, key) in recallPhraseToKey {
            if n == phrase || n.hasPrefix(phrase + " ") {
                return key
            }
        }
        return nil
    }
}
