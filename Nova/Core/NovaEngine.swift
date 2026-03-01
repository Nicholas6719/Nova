import Foundation

/// Generates assistant responses from conversation history and new input.
/// Delegates to NovaEngineCore (pure non-actor) for execution.
struct NovaEngine {

    private let core = NovaEngineCore()

    /// Produce a response given full message history and the latest user input.
    /// If onStreamStart/onStreamDelta are provided and OpenAI fallback is used, streams to UI.
    func generateResponse(
        messages: [Message],
        newInput: String,
        llmConfig: LLMConfig?,
        onStreamStart: (@Sendable () -> Void)? = nil,
        onStreamDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        return try await core.generateResponse(
            messages: messages,
            newInput: newInput,
            systemPrompt: NovaPersonality.systemPrompt(),
            llmConfig: llmConfig,
            onStreamStart: onStreamStart,
            onStreamDelta: onStreamDelta
        )
    }
}
