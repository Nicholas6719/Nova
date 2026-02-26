//
//  NovaEngineCore.swift
//  Nova
//
//  Pure non-actor core for response generation. Safe to call from any context.
//
//  Compound intent verification (DEBUG):
//  1. "Hi Nova, what time is it right now?" -> compound, time (local)
//  2. "Hello, what's the date tomorrow?" -> compound, date (local)
//  3. "Hey Nova, explain quantum computing in one sentence" -> OpenAI (unknown, no local intent)
//

import Foundation

/// Pure struct for response generation.
struct NovaEngineCore: Sendable {

    func generateResponse(
        messages: [Message],
        newInput: String,
        systemPrompt: String,
        llmConfig: LLMConfig?,
        now: Date = Date(),
        onStreamStart: (@Sendable () async -> Void)? = nil,
        onStreamDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        defer { NovaLogger.info("[Engine] DEFER fired — exiting generateResponse") }
        DebugLog.d("[Flow] entered generateResponse")
        let trimmedInput = newInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let input = trimmedInput.lowercased()
        if input.isEmpty {
            return "I didn't catch that. Say something and I'll respond."
        }

        // Compound: greeting + local intent → greet first, then answer
        if hasGreetingWord(input) && hasLocalIntent(input: input, trimmedInput: trimmedInput, messages: messages) {
            let intentResponse = generateLocalIntentResponse(messages: messages, newInput: newInput, input: input, trimmedInput: trimmedInput, now: now)
            if let response = intentResponse {
                let briefGreeting = briefGreetingFromInput(input)
                #if DEBUG
                DebugLog.d("[Nova] compound intent: greeting+local -> \(response.prefix(40))...")
                #endif
                return "\(briefGreeting) \(response)"
            }
        }

        // Pure greeting: greeting only, no substantive question (otherwise let OpenAI handle)
        if isGreetingPhrase(input) && !hasSubstantiveQuestion(input: input) {
            let priorGreetings = messages.dropLast().filter { $0.role == .user }.filter { isGreetingPhrase($0.content) }
            return greetingResponse(priorGreetings: priorGreetings, now: now)
        }

        let intent = IntentDetector.detect(from: newInput)
        if intent != .unknown {
            switch intent {
            case .getDate:
                return responseForDateIntent(input: input, now: now)

            case .getTime:
                let formatter = DateFormatter()
                formatter.locale = Locale.current
                formatter.timeStyle = .short
                let timeString = formatter.string(from: now)
                return "The current time is \(timeString)."

            case .getDayOfWeek:
                let formatter = DateFormatter()
                formatter.locale = Locale.current
                formatter.dateFormat = "EEEE"
                return formatter.string(from: now)

            case .unknown:
                break
            }
        }

        if input.contains("what did i just say") || input.contains("what did i say") {
            let userMessages = messages.filter { $0.role == .user }
            let previousUserMessages = userMessages.dropLast()
            if let previous = previousUserMessages.last {
                let text = previous.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty {
                    return "You spoke just before this, but it was very short."
                }
                let snippet = text.prefix(120)
                let suffix = text.count > 120 ? "…" : ""
                return "You just said something like: \"\(snippet)\(suffix)\"."
            }
            return "You haven't said anything prior to that."
        }

        if input.contains("what was your last response") || input.contains("what did you last say") || input.contains("what did you say") {
            let assistantMessages = messages.filter { $0.role == .assistant }
            if let last = assistantMessages.last {
                let text = last.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty {
                    return "My last response was extremely short."
                }
                let snippet = text.prefix(120)
                let suffix = text.count > 120 ? "…" : ""
                return "My last response was roughly: \"\(snippet)\(suffix)\"."
            }
            return "I haven't responded yet in this conversation."
        }

        if input.contains("remind me what we discussed") || input.contains("what we discussed") || (input.contains("remind me") && input.contains("discussed")) {
            return summarizeConversation(messages: messages)
        }
        if input.contains("set a reminder") || (input.contains("remind me") && !input.contains("discussed")) {
            return "I don't set timed reminders yet, but I can summarize what we've discussed if you'd like."
        }

        if let mathResult = evaluateSimpleMath(input: trimmedInput) {
            return mathResult
        }

        // OpenAI fallback: config passed in (no ProcessInfo access here).
        guard let cfg = llmConfig else {
            return "I'm missing my API key setup."
        }
        NovaLogger.info("[Engine] E0 openai path entered")
        NovaLogger.info("[Engine] E0.1 apiKey.len=\(cfg.apiKey.count)")
        NovaLogger.info("[Engine] E1 before LLMClient.generateResponse")
        let resp = try await LLMClient.generateResponse(config: cfg, messages: messages, systemPrompt: systemPrompt)
        NovaLogger.info("[Engine] E2 after LLMClient.generateResponse len=\(resp.count)")
        return resp
    }

    // MARK: - Cache key (model + canonical query; greetings/wake word stripped at start only)

    /// Safe, deterministic O(n) cache key builder. No regex, no while loops, no String.Index.
    /// Strips leading wake word + greetings so "Hi Nova, explain X" and "Explain X" map to same key.
    private nonisolated func makeOpenAICacheKey(rawQuery: String) -> String {
        let capped = rawQuery.unicodeScalars.prefix(500)
        // Single pass over scalars: A-Z -> a-z, keep a-z/0-9, collapse whitespace
        var normalized = ""
        var lastWasSpace = true
        for scalar in capped {
            let v = scalar.value
            if v >= 65 && v <= 90 {
                normalized.unicodeScalars.append(Unicode.Scalar(v + 32)!)
                lastWasSpace = false
            } else if (v >= 97 && v <= 122) || (v >= 48 && v <= 57) {
                normalized.unicodeScalars.append(scalar)
                lastWasSpace = false
            } else if v == 32 || v == 9 || v == 10 || v == 13 {
                if !lastWasSpace {
                    normalized.unicodeScalars.append(Unicode.Scalar(32)!)
                    lastWasSpace = true
                }
            }
        }
        let trimmed = normalized.trimmingCharacters(in: .whitespaces)
        let words = trimmed.split(separator: " ").map { String($0) }

        // Drop leading greeting/wake words: max 6 iterations, no while
        var remaining = words
        for _ in 0..<6 {
            if remaining.isEmpty { break }
            let first = remaining[0]
            if first == "nova" {
                remaining.removeFirst()
            } else if first == "hi" || first == "hello" || first == "hey" {
                remaining.removeFirst()
            } else if first == "good" && remaining.count >= 2 {
                let second = remaining[1]
                if second == "morning" || second == "afternoon" || second == "evening" {
                    remaining.removeFirst()
                    remaining.removeFirst()
                } else { break }
            } else if first == "morning" || first == "afternoon" || first == "evening" {
                remaining.removeFirst()
            } else {
                break
            }
        }

        let s: String
        if remaining.count > 50 {
            s = remaining.prefix(50).joined(separator: " ")
        } else {
            s = remaining.joined(separator: " ")
        }
        let finalNorm = s.count > 200 ? String(s.prefix(200)) : s
        return "gpt-4o-mini|\(finalNorm)"
    }

    // MARK: - Helpers

    private nonisolated func getDate(for keyword: String, now: Date) -> Date? {
        let calendar = Calendar.current
        let normalized = keyword.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.contains("day after tomorrow") {
            return calendar.date(byAdding: .day, value: 2, to: now)
        }
        if normalized.contains("tomorrow") {
            return calendar.date(byAdding: .day, value: 1, to: now)
        }
        if normalized.contains("today") || normalized.isEmpty {
            return now
        }
        return nil
    }

    private nonisolated func responseForDateIntent(input: String, now: Date) -> String {
        let normalized = input.replacingOccurrences(of: "'", with: "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let keyword: String
        if normalized.contains("day after tomorrow") {
            keyword = "day after tomorrow"
        } else if normalized.contains("tomorrow") {
            keyword = "tomorrow"
        } else {
            keyword = "today"
        }

        guard let date = getDate(for: keyword, now: now) else {
            let formatter = DateFormatter()
            formatter.locale = Locale.current
            formatter.dateStyle = .long
            return "Today is \(formatter.string(from: now)). If you need information about events or tasks for that day, please let me know."
        }

        let formatter = DateFormatter()
        formatter.locale = Locale.current
        formatter.dateStyle = .long
        let dateString = formatter.string(from: date)
        let label: String
        switch keyword {
        case "day after tomorrow": label = "The day after tomorrow"
        case "tomorrow": label = "Tomorrow"
        default: label = "Today"
        }

        if label == "Today" {
            return "Today is \(dateString). If you need information about events or tasks for that day, please let me know."
        }
        return "\(label) is \(dateString). If you need information about events or tasks for that day, please let me know."
    }

    /// Word-boundary safe: avoids "hi" in "high", "hey" in "they".
    private nonisolated func hasGreetingWord(_ text: String) -> Bool {
        let c = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !c.isEmpty else { return false }
        let padded = " \(c) "
        let phrases = [" hi ", " hello ", " hey ", " good morning ", " good afternoon ", " good evening "]
        for p in phrases {
            if padded.contains(p) { return true }
        }
        return c == "hi" || c == "hello" || c == "hey"
    }

    private nonisolated func isGreetingPhrase(_ text: String) -> Bool {
        let c = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !c.isEmpty else { return false }
        return hasGreetingWord(c)
    }

    private nonisolated func hasLocalIntent(input: String, trimmedInput: String, messages: [Message]) -> Bool {
        if IntentDetector.detect(from: trimmedInput) != .unknown { return true }
        if input.contains("what did i just say") || input.contains("what did i say") { return true }
        if input.contains("what was your last response") || input.contains("what did you last say") || input.contains("what did you say") { return true }
        if input.contains("remind me what we discussed") || input.contains("what we discussed") || (input.contains("remind me") && input.contains("discussed")) { return true }
        if input.contains("set a reminder") || (input.contains("remind me") && !input.contains("discussed")) { return true }
        if evaluateSimpleMath(input: trimmedInput) != nil { return true }
        return false
    }

    private nonisolated func generateLocalIntentResponse(messages: [Message], newInput: String, input: String, trimmedInput: String, now: Date) -> String? {
        let intent = IntentDetector.detect(from: newInput)
        if intent != .unknown {
            switch intent {
            case .getDate: return responseForDateIntent(input: input, now: now)
            case .getTime:
                let formatter = DateFormatter()
                formatter.locale = Locale.current
                formatter.timeStyle = .short
                return "The current time is \(formatter.string(from: now))."
            case .getDayOfWeek:
                let formatter = DateFormatter()
                formatter.locale = Locale.current
                formatter.dateFormat = "EEEE, MMMM d, yyyy"
                return "Today is \(formatter.string(from: now))."
            case .unknown: break
            }
        }
        if input.contains("what did i just say") || input.contains("what did i say") {
            let userMessages = messages.filter { $0.role == .user }
            let prev = userMessages.dropLast().last
            if let p = prev {
                let text = p.content.trimmingCharacters(in: .whitespacesAndNewlines)
                let snippet = text.prefix(120)
                return text.isEmpty ? "You spoke just before this, but it was very short." : "You just said: \"\(snippet)\(text.count > 120 ? "…" : "")\"."
            }
            return "You haven't said anything prior to that."
        }
        if input.contains("what was your last response") || input.contains("what did you last say") || input.contains("what did you say") {
            let assistantMessages = messages.filter { $0.role == .assistant }
            if let last = assistantMessages.last {
                let text = last.content.trimmingCharacters(in: .whitespacesAndNewlines)
                let snippet = text.prefix(120)
                return text.isEmpty ? "My last response was extremely short." : "My last response was roughly: \"\(snippet)\(text.count > 120 ? "…" : "")\"."
            }
            return "I haven't responded yet in this conversation."
        }
        if input.contains("remind me what we discussed") || input.contains("what we discussed") || (input.contains("remind me") && input.contains("discussed")) {
            return summarizeConversation(messages: messages)
        }
        if input.contains("set a reminder") || (input.contains("remind me") && !input.contains("discussed")) {
            return "I don't set timed reminders yet, but I can summarize what we've discussed if you'd like."
        }
        if let mathResult = evaluateSimpleMath(input: trimmedInput) {
            return mathResult
        }
        return nil
    }

    /// Matches the user's greeting phrase (no double-greet). Order matters: longer phrases first.
    private nonisolated func briefGreetingFromInput(_ input: String) -> String {
        let padded = " \(input) "
        if padded.contains(" good morning ") { return "Good morning!" }
        if padded.contains(" good afternoon ") { return "Good afternoon!" }
        if padded.contains(" good evening ") { return "Good evening!" }
        return "Hi!"
    }

    /// True if input contains question-like phrases (explain, what is, how, tell me, etc.).
    private nonisolated func hasSubstantiveQuestion(input: String) -> Bool {
        let phrases = ["explain", "what is", "whats", "what's", "how ", "how do", "why ", "tell me", "define", "describe"]
        return phrases.contains { input.contains($0) }
    }

    private nonisolated func greetingResponse(priorGreetings: [Message], now: Date) -> String {
        let hour = Calendar.current.component(.hour, from: now)
        let greeting: String
        if hour >= 5 && hour < 12 {
            greeting = "Good morning! How can I help you today?"
        } else if hour >= 12 && hour < 17 {
            greeting = "Good afternoon! What can I do for you?"
        } else if hour >= 17 && hour < 22 {
            greeting = "Good evening! How can I assist?"
        } else {
            greeting = "Hello! Burning the midnight oil? How can I help?"
        }
        if priorGreetings.isEmpty {
            return greeting
        }
        return "Hello again. " + greeting
    }

    private nonisolated func evaluateSimpleMath(input: String) -> String? {
        let lower = input.lowercased()
        let clean = lower
            .replacingOccurrences(of: "what is", with: "")
            .replacingOccurrences(of: "whats", with: "")
            .replacingOccurrences(of: "what's", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        typealias Op = (String, (Double, Double) -> Double)
        let ops: [Op] = [
            ("plus", +),
            ("minus", { $0 - $1 }),
            ("times", *),
            ("multiplied by", *),
            ("divided by", { $1 != 0 ? $0 / $1 : 0 }),
        ]

        for (name, operation) in ops {
            guard let idx = lower.range(of: name) else { continue }
            let before = String(lower[..<idx.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let after = String(lower[idx.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            let aNum = Double(before.filter { $0.isNumber || $0 == "." })
            let bNum = Double(after.filter { $0.isNumber || $0 == "." })
            guard let a = aNum, let b = bNum else { continue }
            if name == "divided by" && b == 0 { continue }
            let res = operation(a, b)
            let opWord = name == "plus" ? "plus" : name == "minus" ? "minus" : name == "times" || name == "multiplied by" ? "times" : "divided by"
            let intResult = res.rounded()
            if res == intResult && abs(res) <= 1_000_000 {
                return "\(Int(a)) \(opWord) \(Int(b)) equals \(Int(intResult))."
            }
            return "\(a) \(opWord) \(b) equals \(res)."
        }

        let symbolOps: [(Character, (Double, Double) -> Double)] = [
            ("+", +),
            ("-", { $0 - $1 }),
            ("*", *),
            ("/", { $1 != 0 ? $0 / $1 : 0 }),
        ]
        for (char, operation) in symbolOps {
            guard let idx = clean.firstIndex(of: char) else { continue }
            let before = String(clean[..<idx]).trimmingCharacters(in: .whitespacesAndNewlines)
            let after = String(clean[clean.index(after: idx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let a = Double(before.filter { $0.isNumber || $0 == "." }),
                  let b = Double(after.filter { $0.isNumber || $0 == "." }),
                  b != 0 || char != "/" else { continue }
            let result = operation(a, b)
            let opWord = char == "+" ? "plus" : char == "-" ? "minus" : char == "*" ? "times" : "divided by"
            let intResult = result.rounded()
            if result == intResult && abs(result) <= 1_000_000 {
                return "\(Int(a)) \(opWord) \(Int(b)) equals \(Int(intResult))."
            }
            return "\(a) \(opWord) \(b) equals \(result)."
        }

        return nil
    }

    private nonisolated func summarizeConversation(messages: [Message]) -> String {
        let recent = Array(messages.dropLast().suffix(6))
        if recent.isEmpty {
            return "We haven't discussed anything yet in this conversation."
        }
        var parts: [String] = []
        for msg in recent {
            let content = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if content.isEmpty { continue }
            let snippet = content.prefix(60)
            let suffix = content.count > 60 ? "…" : ""
            if msg.role == .user {
                parts.append("you mentioned \"\(snippet)\(suffix)\"")
            } else {
                parts.append("I talked about \"\(snippet)\(suffix)\"")
            }
        }
        if parts.isEmpty {
            return "We've only exchanged very short messages so far."
        }
        return "Here's what we've covered so far: " + parts.joined(separator: "; ") + "."
    }
}
