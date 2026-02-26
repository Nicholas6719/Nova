//
//  IntentDetector.swift
//  Nova
//
//  Lightweight intent detection for Nova.
//  Normalizes text and maps common phrases to high-level intents.
//  This sits in front of the main NovaEngine logic and can be expanded over time.
//

import Foundation

/// High-level intents Nova can handle locally before falling back to general logic/AI.
enum IntentType: Sendable, Equatable {
    case getDate
    case getTime
    case getDayOfWeek
    case unknown

    /// Explicit nonisolated Equatable so comparisons work from nonisolated context.
    nonisolated static func == (lhs: IntentType, rhs: IntentType) -> Bool {
        switch (lhs, rhs) {
        case (.getDate, .getDate), (.getTime, .getTime), (.getDayOfWeek, .getDayOfWeek), (.unknown, .unknown): return true
        default: return false
        }
    }
}

/// Detects intents from user utterances using simple keyword/phrase matching.
struct IntentDetector: Sendable {

    /// Detect intent from raw user input.
    /// - Parameter input: Raw text (e.g. from speech recognition).
    /// - Returns: Detected intent or `.unknown` if no rule matches.
    nonisolated static func detect(from input: String) -> IntentType {
        let normalized = normalize(input)
        if normalized.isEmpty { return .unknown }

        // Day of week (more specific) first
        if normalized.contains("day of the week") {
            return .getDayOfWeek
        }

        // Date-related phrases (flexible matching)
        //
        // Matches:
        // - phrases containing \"date\"
        // - phrases containing \"what day\" or \"day is it\"
        // - phrases that mention today with a wh-word:
        //     \"what's today\", \"whats today\", \"what is today\"
        let containsDate = normalized.contains("date")
        let containsWhatDay = normalized.contains("what day")
        let containsDayIsIt = normalized.contains("day is it")
        let containsToday = normalized.contains("today")
        let containsWhatOrWhats = normalized.contains("what") || normalized.contains("whats")

        if containsDate
            || containsWhatDay
            || containsDayIsIt
            || (containsToday && containsWhatOrWhats) {
            return .getDate
        }

        // Time-related phrases
        if normalized.contains("what time is it")
            || normalized.contains("current time")
            || normalized.contains("time") {
            return .getTime
        }

        return .unknown
    }

    /// Normalize text for intent detection:
    /// - remove apostrophes
    /// - lowercased
    /// - punctuation removed
    /// - trimmed of surrounding whitespace.
    private nonisolated static func normalize(_ input: String) -> String {
        // Remove apostrophes explicitly so \"what's\" → \"whats\", \"today's\" → \"todays\".
        let withoutApostrophes = input.replacingOccurrences(of: "'", with: "")
        let lower = withoutApostrophes.lowercased()
        let scalars = lower.unicodeScalars.filter { !CharacterSet.punctuationCharacters.contains($0) }
        let stripped = String(String.UnicodeScalarView(scalars))
        return stripped.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

