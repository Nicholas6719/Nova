import Foundation

/// Pure struct for response generation. Safe to call from any context.
struct NovaEngineCore: Sendable {

    func generateResponse(
        messages: [Message],
        newInput: String,
        systemPrompt: String,
        llmConfig: LLMConfig?,
        now: Date = Date(),
        onStreamStart: (@Sendable () -> Void)? = nil,
        onStreamDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        let trimmedInput = newInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let input = trimmedInput.lowercased()
        let strippedQuery = stripWakeWords(from: input)

        if input.isEmpty {
            return "I didn't catch that. Say something and I'll respond."
        }

        // MARK: Local math (normalize → binary → chain → mathy guard)
        let mathNormalized = Self.normalizeMathOperators(input)

        if let mathReply = Self.tryEvaluateSimpleBinaryMath(mathNormalized) {
            return mathReply
        }

        if let chainReply = Self.tryEvaluateChainedMath(mathNormalized) {
            return chainReply
        }

        // If input looks like a math expression we can't handle, skip intent detection → OpenAI
        localChecks: do {
            if Self.looksLikeMathExpression(mathNormalized) {
                break localChecks
            }

            // Compound: greeting + local intent → greet first, then answer
            if hasGreetingWord(input) && hasLocalIntent(input: input, trimmedInput: strippedQuery, messages: messages) {
                let intentResponse = generateLocalIntentResponse(messages: messages, newInput: newInput, input: input, trimmedInput: strippedQuery.isEmpty ? trimmedInput : strippedQuery, now: now)
                if let response = intentResponse {
                    let briefGreeting = briefGreetingFromInput(input)
                    return "\(briefGreeting) \(response)"
                }
            }

            // Pure greeting (no substantive question)
            if isGreetingPhrase(input) && !hasSubstantiveQuestion(input: input) {
                let priorGreetings = messages.dropLast().filter { $0.role == .user }.filter { isGreetingPhrase($0.content) }
                return greetingResponse(priorGreetings: priorGreetings, now: now)
            }

            // Intent detection (time, date, day of week)
            let intent = IntentDetector.detect(from: strippedQuery.isEmpty ? newInput : strippedQuery)
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

            // Recall: what did I say / what did you say
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
        }

        // MARK: Tool routing (before OpenAI)
        let toolIntent = ToolRouter.match(from: trimmedInput)
        switch toolIntent {
        case .openApp(let name):
            return (await PlatformTools.executeOpenApp(name: name)).spokenResponse
        case .quitApp:
            return "Closing apps by name isn't available yet."
        case .batteryStatus(let chargingIntent):
            return (await PlatformTools.executeBatteryStatus(chargingIntent: chargingIntent)).spokenResponse
        case .webSearch(let query):
            return (await PlatformTools.executeWebSearch(query: query)).spokenResponse
        case .none:
            break
        }

        // MARK: OpenAI fallback
        guard let cfg = llmConfig else {
            return "I'm missing my API key setup."
        }
        guard !cfg.apiKey.isEmpty else {
            return "I'm missing my OpenAI API key configuration."
        }

        let fallbackMsg = "Sorry — my online brain is taking too long right now. Please try again."

        // Streaming path: if caller provided streaming callbacks, use SSE streaming.
        if let streamStart = onStreamStart, let streamDelta = onStreamDelta {
            let resp: String
            do {
                resp = try await withThrowingTaskGroup(of: String.self) { group in
                    group.addTask {
                        try await Task.detached(priority: .userInitiated) {
                            try await LLMClient.streamResponse(
                                config: cfg,
                                systemPrompt: systemPrompt,
                                messages: messages,
                                onStreamStart: streamStart,
                                onDelta: streamDelta
                            )
                        }.value
                    }
                    group.addTask {
                        try await Task.sleep(nanoseconds: 60_000_000_000)
                        return fallbackMsg
                    }
                    let first = try await group.next()!
                    group.cancelAll()
                    return first
                }
            } catch {
                throw error
            }
            return resp
        }

        // Non-streaming path: original behavior.
        let resp: String
        do {
            resp = try await withThrowingTaskGroup(of: String.self) { group in
                group.addTask {
                    try await Task.detached(priority: .userInitiated) {
                        try await LLMClient.generateResponse(config: cfg, messages: messages, systemPrompt: systemPrompt)
                    }.value
                }
                group.addTask {
                    try await Task.sleep(nanoseconds: 10_000_000_000)
                    return fallbackMsg
                }
                let first = try await group.next()!
                group.cancelAll()
                return first
            }
        } catch {
            throw error
        }
        return resp
    }

    // MARK: - Math Helpers (static, pure — no self, no MainActor)

    /// Normalize Unicode math symbols from speech recognition (×→*, ÷→/, •→*, ·→*).
    private static func normalizeMathOperators(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{00D7}", with: "*")
         .replacingOccurrences(of: "\u{00F7}", with: "/")
         .replacingOccurrences(of: "\u{2022}", with: "*")
         .replacingOccurrences(of: "\u{00B7}", with: "*")
    }

    /// Evaluate a single binary operation: "8-2", "300*2", "12/3".
    /// Returns nil for multi-op expressions, non-numeric input, or divide-by-zero.
    private static func tryEvaluateSimpleBinaryMath(_ normalized: String) -> String? {
        let s = normalized.replacingOccurrences(of: " ", with: "")
        guard s.rangeOfCharacter(from: .decimalDigits) != nil else { return nil }

        let ops: [Character] = ["+", "-", "*", "/"]
        var found: (Character, Int)? = nil
        for (i, ch) in s.enumerated() {
            if ops.contains(ch) {
                if i == 0 { return nil }
                if found != nil { return nil }
                found = (ch, i)
            }
        }
        guard let (op, idx) = found else { return nil }

        let lhs = String(s.prefix(idx))
        let rhs = String(s.suffix(s.count - idx - 1))
        guard let a = Double(lhs), let b = Double(rhs) else { return nil }

        let result: Double
        switch op {
        case "+": result = a + b
        case "-": result = a - b
        case "*": result = a * b
        case "/":
            if b == 0 { return nil }
            result = a / b
        default: return nil
        }

        func fmt(_ x: Double) -> String {
            x.rounded() == x && abs(x) < 1e15 ? String(Int(x)) : String(x)
        }

        let spokenOp: String
        switch op {
        case "+": spokenOp = "plus"
        case "-": spokenOp = "minus"
        case "*": spokenOp = "times"
        case "/": spokenOp = "divided by"
        default: spokenOp = "?"
        }
        return "\(fmt(a)) \(spokenOp) \(fmt(b)) equals \(fmt(result))."
    }

    /// Evaluate chained +/− expressions: "300-200+2", "-5+2", "calculate 10+5-3".
    /// Strips common prefixes. Rejects input containing *, /, parens, or letters.
    private static func tryEvaluateChainedMath(_ raw: String) -> String? {
        var s = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        for prefix in ["nova ", "what is ", "what's ", "whats ", "calculate ", "solve "] {
            if s.hasPrefix(prefix) {
                s = String(s.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }

        s = s.replacingOccurrences(of: " ", with: "")

        for ch in s {
            guard ch.isNumber || ch == "." || ch == "+" || ch == "-" else { return nil }
        }
        guard !s.isEmpty else { return nil }

        let inner = s.first == "-" ? String(s.dropFirst()) : s
        guard inner.contains("+") || inner.contains("-") else { return nil }

        var result = 0.0
        var currentNum = ""
        var pendingOp: Character = "+"
        var isFirst = true

        for ch in s {
            if (ch == "+" || ch == "-") && !currentNum.isEmpty {
                guard let val = Double(currentNum) else { return nil }
                result = pendingOp == "+" ? result + val : result - val
                currentNum = ""
                pendingOp = ch
                isFirst = false
            } else if ch == "-" && currentNum.isEmpty && isFirst {
                currentNum.append(ch)
            } else if ch.isNumber || ch == "." {
                currentNum.append(ch)
            } else {
                return nil
            }
        }

        guard !currentNum.isEmpty, let val = Double(currentNum) else { return nil }
        result = pendingOp == "+" ? result + val : result - val

        let formatted: String
        if result.truncatingRemainder(dividingBy: 1) == 0 && abs(result) < 1e15 {
            formatted = String(Int(result))
        } else {
            formatted = String(result)
        }
        return "That equals \(formatted)."
    }

    /// True if input (after stripping spaces) contains digits + math operators but NO letters.
    /// Used to skip compound/intent detection for expressions we can't evaluate locally.
    private static func looksLikeMathExpression(_ normalized: String) -> Bool {
        let s = normalized.replacingOccurrences(of: " ", with: "")
        var hasDigit = false
        var hasOp = false
        let mathChars: Set<Character> = ["+", "-", "*", "/", "(", ")"]
        for ch in s {
            if ch.isNumber || ch == "." { hasDigit = true }
            else if mathChars.contains(ch) { hasOp = true }
            else if ch.isLetter { return false }
        }
        return hasDigit && hasOp
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

    /// Strip leading wake words and greetings for intent detection.
    private nonisolated func stripWakeWords(from input: String) -> String {
        let words = input.split(separator: " ").map { String($0) }
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
        return remaining.joined(separator: " ").trimmingCharacters(in: .whitespaces)
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
        let intent = IntentDetector.detect(from: trimmedInput.isEmpty ? newInput : trimmedInput)
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

    private nonisolated func briefGreetingFromInput(_ input: String) -> String {
        let padded = " \(input) "
        if padded.contains(" good morning ") { return "Good morning!" }
        if padded.contains(" good afternoon ") { return "Good afternoon!" }
        if padded.contains(" good evening ") { return "Good evening!" }
        return "Hi!"
    }

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

    /// Instance math evaluator used by compound greeting+math detection (hasLocalIntent/generateLocalIntentResponse).
    /// Strips common prefixes ("what's", "calculate", etc.), parses single binary op.
    private nonisolated func evaluateSimpleMath(input: String) -> String? {
        let candidate = mathCandidate(from: input)
        guard let candidate, !candidate.isEmpty else { return nil }

        let ops: [(Character, String, (Double, Double) -> Double)] = [
            ("+", "plus", +),
            ("-", "minus", { $0 - $1 }),
            ("*", "times", *),
            ("/", "divided by", { $1 != 0 ? $0 / $1 : .nan }),
        ]
        for (sym, word, fn) in ops {
            guard let idx = candidate.firstIndex(of: sym) else { continue }
            let lhs = String(candidate[..<idx]).trimmingCharacters(in: .whitespaces)
            let rhs = String(candidate[candidate.index(after: idx)...]).trimmingCharacters(in: .whitespaces)
            guard let a = Double(lhs), let b = Double(rhs) else { continue }
            if sym == "/" && b == 0 { return nil }
            let res = fn(a, b)
            if res.isNaN || res.isInfinite { return nil }
            let aStr = a == a.rounded() && abs(a) < 1e7 ? "\(Int(a))" : "\(a)"
            let bStr = b == b.rounded() && abs(b) < 1e7 ? "\(Int(b))" : "\(b)"
            let rStr = res == res.rounded() && abs(res) < 1e7 ? "\(Int(res))" : String(format: "%.2g", res)
            return "\(aStr) \(word) \(bStr) equals \(rStr)."
        }
        return nil
    }

    private nonisolated func mathCandidate(from input: String) -> String? {
        var s = input
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        for prefix in ["what's ", "whats ", "what is ", "calculate ", "solve ", "nova "] {
            if s.hasPrefix(prefix) {
                s = String(s.dropFirst(prefix.count))
                break
            }
        }

        var out = ""
        var lastSpace = false
        for c in s {
            let v = c.unicodeScalars.first.map { $0.value } ?? 0
            if (v >= 48 && v <= 57) || c == "." || c == "+" || c == "-" || c == "*" || c == "/" {
                out.append(c)
                lastSpace = false
            } else if v == 32 || v == 9 {
                if !lastSpace && !out.isEmpty { out.append(" "); lastSpace = true }
            }
        }
        let result = out.trimmingCharacters(in: .whitespaces)
        guard !result.isEmpty else { return nil }

        var numCount = 0; var opCount = 0; var inNum = false
        for c in result {
            if c.isNumber || c == "." {
                if !inNum { numCount += 1; inNum = true }
            } else if c == "+" || c == "-" || c == "*" || c == "/" {
                opCount += 1; inNum = false
            } else { inNum = false }
        }
        guard numCount >= 2 && opCount >= 1 else { return nil }
        return result
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
