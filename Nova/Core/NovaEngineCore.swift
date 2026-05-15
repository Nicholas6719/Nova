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

        // MARK: Local math
        if let mathReply = MathRouter.localMathResponse(for: input) {
            return mathReply
        }

        // If input looks like a math expression we can't handle, skip intent detection → OpenAI
        localChecks: do {
            if MathRouter.looksLikeMath(input) {
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
            if hasGreetingWord(input) && !hasSubstantiveQuestion(input: input) {
                let priorGreetings = messages.dropLast().filter { $0.role == .user }.filter { hasGreetingWord($0.content) }
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

        // MARK: Memory update / save / recall (before tools)
        let memoryInput = strippedQuery.isEmpty ? input : strippedQuery

        // Update/correction: "actually my name is X" or "actually it's X" (with last key)
        if let update = MemoryRouter.matchUpdate(from: memoryInput) {
            guard MemoryStore.set(update.key, value: update.value) else {
                DebugLog.d("[Memory] returning update failure")
                return "I couldn't update that memory."
            }
            MemoryContext.updateLastDiscussed(key: update.key, value: update.value)
            let key = update.key
            let val = update.value
            let displayVal = val.prefix(1).uppercased() + val.dropFirst()
            let label = MemoryStore.displayLabel(for:key)
            let response: String
            if let p = update.previousValue, !p.isEmpty {
                let pDisplay = p.prefix(1).uppercased() + p.dropFirst()
                response = "Okay, I'll remember that your \(label) is \(displayVal), not \(pDisplay)."
            } else {
                response = "Okay, I'll remember that your \(label) is \(displayVal)."
            }
            DebugLog.d("[Memory] returning update response")
            return response
        }

        if let save = MemoryRouter.matchSave(from: memoryInput) {
            guard MemoryStore.set(save.key, value: save.value) else {
                DebugLog.d("[Memory] returning save failure")
                return "I couldn't save that memory."
            }
            MemoryContext.updateLastDiscussed(key: save.key, value: save.value)
            let displayValue = save.value.prefix(1).uppercased() + save.value.dropFirst()
            let fieldLabel = MemoryStore.displayLabel(for:save.key)
            let response = "Got it. I'll remember that your \(fieldLabel) is \(displayValue)."
            DebugLog.d("[Memory] returning save response")
            return response
        }
        if let recallKey = MemoryRouter.matchRecall(from: memoryInput) {
            if let value = MemoryStore.get(recallKey) {
                MemoryContext.updateLastDiscussed(key: recallKey, value: value)
                let displayValue = value.prefix(1).uppercased() + value.dropFirst()
                let fieldLabel = MemoryStore.displayLabel(for:recallKey)
                let response = "Your \(fieldLabel) is \(displayValue)."
                DebugLog.d("[Memory] returning recall response")
                return response
            }
            DebugLog.d("[Memory] returning recall miss")
            return "I don't know that yet."
        }

        // MARK: Tool routing (before OpenAI)
        let toolIntent = ToolRouter.match(from: trimmedInput)
        switch toolIntent {
        case .openApp(let name):
            return (await PlatformTools.executeOpenApp(name: name)).spokenResponse
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
                    guard let first = try await group.next() else {
                        throw CancellationError()
                    }
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
                guard let first = try await group.next() else {
                    throw CancellationError()
                }
                group.cancelAll()
                return first
            }
        } catch {
            throw error
        }
        return resp
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
        if MathRouter.localMathResponse(for: trimmedInput) != nil { return true }
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
        if let mathResult = MathRouter.localMathResponse(for: trimmedInput) {
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
