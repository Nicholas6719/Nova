//
//  LLMClient.swift
//  Nova
//
//  OpenAI API client for GPT-4o-mini. Reads API key from environment.
//  Used by NovaEngine when intent is unknown (personality + memory preserved).
//  Config passed in to avoid MainActor isolation in detached request path.
//

import Foundation

/// Actor-neutral config for OpenAI requests. Built outside detached tasks.
struct LLMConfig: Sendable {
    let apiKey: String
    let endpoint: URL
    let model: String
    let temperature: Double
}

/// Errors thrown by the LLM client.
enum LLMClientError: LocalizedError {
    case missingAPIKey
    case emptyContent
    case invalidResponse
    case networkError(underlying: Error)
    case serverError(statusCode: Int, body: String?)
    case timeout

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenAI API key is not set. Add OPENAI_API_KEY to your environment."
        case .emptyContent:
            return "The model returned an empty response."
        case .invalidResponse:
            return "The API response could not be parsed."
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .serverError(let code, let body):
            if let body = body, !body.isEmpty {
                return "API error (\(code)): \(body)"
            }
            return "API error: HTTP \(code)"
        case .timeout:
            return "Network request timed out. Please try again."
        }
    }
}

/// OpenAI chat completions client. Config passed in to avoid MainActor isolation.
struct LLMClient {

    /// Non-streaming chat completions. Simple direct call (no Task.detached).
    static func generateResponse(config: LLMConfig, messages: [Message], systemPrompt: String?) async throws -> String {
        NovaLogger.info("[LLMClient] request start")
        guard !config.apiKey.isEmpty else {
            throw LLMClientError.missingAPIKey
        }

        var apiMessages: [[String: String]] = [["role": "system", "content": systemPrompt ?? ""]]
        for msg in messages {
            let role = msg.role == .user ? "user" : "assistant"
            apiMessages.append(["role": role, "content": msg.content])
        }
        let payload: [String: Any] = [
            "model": config.model,
            "temperature": config.temperature,
            "messages": apiMessages
        ]
        guard let bodyData = try? JSONSerialization.data(withJSONObject: payload) else {
            throw LLMClientError.invalidResponse
        }

        var request = URLRequest(url: config.endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 15
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = bodyData

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw LLMClientError.invalidResponse
        }
        NovaLogger.info("[LLMClient] status=\(http.statusCode)")
        guard (200...299).contains(http.statusCode) else {
            let bodyString = String(data: data, encoding: .utf8)
            throw LLMClientError.serverError(statusCode: http.statusCode, body: bodyString)
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) else {
            throw LLMClientError.invalidResponse
        }
        let root = json as? [String: Any]
        let choices = root?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        guard let content = message?["content"] as? String else {
            throw LLMClientError.emptyContent
        }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LLMClientError.emptyContent
        }
        NovaLogger.info("[LLMClient] decode success")
        NovaLogger.info("[LLMClient] returning len=\(trimmed.count)")
        return trimmed
    }

    // MARK: - Streaming (not used in current OpenAI path; kept for future use)

    /// Stream OpenAI chat completions; call onDelta as chunks arrive; return full text.
    static func streamResponse(
        config: LLMConfig,
        systemPrompt: String,
        messages: [Message],
        onDelta: @escaping @Sendable (String) -> Void
    ) async throws -> String {
        guard !config.apiKey.isEmpty else {
            throw LLMClientError.missingAPIKey
        }

        var apiMessages: [[String: String]] = [["role": "system", "content": systemPrompt]]
        for msg in messages {
            let role = msg.role == .user ? "user" : "assistant"
            apiMessages.append(["role": role, "content": msg.content])
        }
        let payload: [String: Any] = [
            "model": config.model,
            "temperature": config.temperature,
            "messages": apiMessages,
            "stream": true
        ]
        guard let bodyData = try? JSONSerialization.data(withJSONObject: payload) else {
            throw LLMClientError.invalidResponse
        }

        var request = URLRequest(url: config.endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = bodyData

        DebugLog.d("[LLMClient] stream start")

        let (bytes, response) = try await URLSession.shared.bytes(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw LLMClientError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            var bodyData = Data()
            for try await chunk in bytes { bodyData.append(chunk) }
            let bodyString = String(data: bodyData, encoding: .utf8)
            throw LLMClientError.serverError(statusCode: http.statusCode, body: bodyString)
        }

        var buffer = ""
        var fullText = ""

        for try await byte in bytes {
            let c = Character(Unicode.Scalar(byte))
            if c == "\n" {
                let trimmed = buffer.trimmingCharacters(in: .whitespaces)
                buffer = ""
                guard trimmed.hasPrefix("data: "), trimmed != "data: [DONE]" else { continue }
                let jsonStr = String(trimmed.dropFirst(6))
                guard let data = jsonStr.data(using: .utf8),
                      let delta = parseStreamDelta(data), !delta.isEmpty else { continue }
                fullText += delta
                onDelta(delta)
            } else {
                buffer.append(c)
            }
        }

        DebugLog.d("[LLMClient] stream end")
        let trimmed = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LLMClientError.emptyContent
        }
        return trimmed
    }

    private static func parseStreamDelta(_ data: Data) -> String? {
        struct StreamDeltaChunk: Decodable {
            let choices: [StreamChoice]?
            struct StreamChoice: Decodable {
                let delta: StreamDelta?
                struct StreamDelta: Decodable {
                    let content: String?
                }
            }
        }
        guard let decoded = try? JSONDecoder().decode(StreamDeltaChunk.self, from: data),
              let content = decoded.choices?.first?.delta?.content else { return nil }
        return content
    }
}
