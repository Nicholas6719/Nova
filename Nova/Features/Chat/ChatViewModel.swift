import Foundation
import Combine

/// UI status for the header indicator.
enum ChatStatus: String {
    case idle = "Idle"
    case listening = "Listening…"
    case processing = "Processing…"
}

@MainActor
final class ChatViewModel: ObservableObject {

    // MARK: - Published

    @Published private(set) var messages: [Message] = []
    @Published private(set) var isRecording: Bool = false
    @Published private(set) var liveTranscript: String = ""
    @Published private(set) var errorMessage: String?
    @Published private(set) var isProcessing: Bool = false

    var status: ChatStatus {
        if isRecording { return .listening }
        if isProcessing { return .processing }
        return .idle
    }

    /// Mic is enabled only when not processing and not speaking (single-flight).
    var isMicEnabled: Bool { !isProcessing && !speechManager.isSpeaking }

    // MARK: - Dependencies

    private let speechRecognizer = SpeechRecognizer()
    private let speechManager = SpeechManager()
    private let engine = NovaEngine()

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Streaming state (MainActor-owned)

    /// Full text accumulated from all deltas so far.
    private var streamingFullText = ""
    /// Number of characters currently revealed in the UI bubble.
    private var streamingShownCount = 0
    /// The message ID of the placeholder assistant bubble.
    private var streamingMessageId: UUID?
    /// The ticker task that reveals characters at a steady pace.
    private var streamingTickerTask: Task<Void, Never>?
    /// True while the SSE stream is still receiving deltas.
    private var streamingIsActive = false
    /// The final complete text from the engine (set when engine returns).
    private var streamingFinalText: String?

    /// Debounce: prevent double-commit of the same transcript within 0.5s.
    private var lastCommittedTranscript = ""
    private var lastCommittedTime: CFAbsoluteTime = 0

    init() {
        speechRecognizer.$transcript
            .assign(to: &$liveTranscript)

        speechRecognizer.$isRecording
            .assign(to: &$isRecording)

        speechRecognizer.$errorMessage
            .compactMap { $0 }
            .sink { [weak self] message in
                self?.errorMessage = message
            }
            .store(in: &cancellables)

        speechRecognizer.onRecordingDidStop = { [weak self] text in
            Task { @MainActor in
                self?.handleRecordingStopped(with: text)
            }
        }

        speechManager.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    // MARK: - Permissions

    func requestPermissions() {
        speechRecognizer.requestPermissions()
    }

    // MARK: - Microphone

    func toggleRecording() {
        guard !isProcessing else {
            errorMessage = "One sec — I'm finishing my response."
            return
        }
        if !isMicEnabled { return }
        if isRecording {
            speechRecognizer.stopListening()
        } else {
            errorMessage = nil
            speechRecognizer.clearTranscript()
            speechRecognizer.startListening()
        }
    }

    /// Called when speech stops. Clear live bubble synchronously, then commit user message.
    private func handleRecordingStopped(with text: String) {
        let finalText = text.trimmingCharacters(in: .whitespacesAndNewlines)

        liveTranscript = ""
        speechRecognizer.clearTranscript()

        guard !finalText.isEmpty else {
            errorMessage = "No speech recognized. Try again."
            return
        }

        let now = CFAbsoluteTimeGetCurrent()
        if finalText == lastCommittedTranscript && (now - lastCommittedTime) < 0.5 {
            return
        }
        lastCommittedTranscript = finalText
        lastCommittedTime = now

        processUserInput(text: finalText)
    }

    // MARK: - Process user input

    private func processUserInput(text: String) {
        guard !text.isEmpty else { return }
        guard !isProcessing else { return }
        isProcessing = true
        errorMessage = nil

        cancelStreamingState()

        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)
        DebugLog.d("[Chat] append user: \(text.prefix(60))\(text.count > 60 ? "…" : "")")

        let messageSnapshot = messages

        let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] ?? ""
        let cfg = LLMConfig(
            apiKey: apiKey,
            endpoint: URL(string: "https://api.openai.com/v1/chat/completions")!,
            model: "gpt-4o-mini",
            temperature: 0.7
        )

        let onStreamStart: @Sendable () -> Void = { [weak self] in
            Task { @MainActor [weak self] in
                self?.beginStreaming()
            }
        }

        let onStreamDelta: @Sendable (String) -> Void = { [weak self] chunk in
            Task { @MainActor [weak self] in
                self?.receiveStreamDelta(chunk)
            }
        }

        let engineRef = engine
        Task.detached(priority: .userInitiated) { [cfg, messageSnapshot, text, onStreamStart, onStreamDelta] in
            do {
                let resp = try await Task.detached(priority: .userInitiated) {
                    try await engineRef.generateResponse(
                        messages: messageSnapshot,
                        newInput: text,
                        llmConfig: cfg,
                        onStreamStart: onStreamStart,
                        onStreamDelta: onStreamDelta
                    )
                }.value
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.onEngineComplete(fullText: resp)
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    DebugLog.d("[Chat] engine error: \(error)")
                    self.onEngineComplete(fullText: "Sorry — I couldn't reach my online brain. Please try again.")
                }
            }
        }
    }

    // MARK: - Streaming: start / delta / ticker

    @MainActor
    private func beginStreaming() {
        let pid = UUID()
        streamingMessageId = pid
        streamingFullText = ""
        streamingShownCount = 0
        streamingIsActive = true
        streamingFinalText = nil
        messages.append(Message(id: pid, role: .assistant, content: ""))
        startTicker()
    }

    @MainActor
    private func receiveStreamDelta(_ delta: String) {
        streamingFullText += delta
    }

    /// Continuous ticker: reveals 5 characters every 120ms (~41 chars/sec).
    @MainActor
    private func startTicker() {
        streamingTickerTask?.cancel()
        streamingTickerTask = Task { @MainActor [weak self] in
            let step = 5
            let intervalNs: UInt64 = 120_000_000 // 120ms

            while !Task.isCancelled {
                guard let self else { return }

                let available = self.streamingFullText.count
                if self.streamingShownCount < available {
                    self.streamingShownCount = min(self.streamingShownCount + step, available)
                    self.updateBubble(charCount: self.streamingShownCount)
                }

                // If stream is done AND we've revealed everything, finish.
                if !self.streamingIsActive && self.streamingShownCount >= self.streamingFullText.count {
                    self.commitFinalStreamedResponse()
                    return
                }

                try? await Task.sleep(nanoseconds: intervalNs)
            }
        }
    }

    @MainActor
    private func updateBubble(charCount: Int) {
        guard let pid = streamingMessageId,
              let idx = messages.lastIndex(where: { $0.id == pid }) else { return }
        let text = String(streamingFullText.prefix(charCount))
        messages[idx] = Message(id: pid, role: .assistant, content: text)
    }

    /// Called by the ticker when it has fully caught up after the stream ended.
    @MainActor
    private func commitFinalStreamedResponse() {
        streamingTickerTask?.cancel()
        streamingTickerTask = nil

        let fullText = streamingFinalText ?? streamingFullText

        if let pid = streamingMessageId,
           let idx = messages.lastIndex(where: { $0.id == pid }) {
            messages[idx] = Message(id: pid, role: .assistant, content: fullText)
        }

        DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")

        clearStreamingState()
        speechManager.speak(fullText)
        isProcessing = false
    }

    // MARK: - Engine completion

    /// Called when the engine returns the full response text.
    /// For streaming: marks the stream as finished and lets the ticker drain.
    /// For non-streaming (local): appends the bubble directly.
    @MainActor
    private func onEngineComplete(fullText: String) {
        if streamingMessageId != nil {
            streamingFinalText = fullText
            streamingFullText = fullText
            streamingIsActive = false
            // Ticker will keep running and call commitFinalStreamedResponse when caught up.
        } else {
            messages.append(Message(role: .assistant, content: fullText))
            DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")
            speechManager.speak(fullText)
            isProcessing = false
        }
    }

    // MARK: - Streaming cleanup

    @MainActor
    private func cancelStreamingState() {
        streamingTickerTask?.cancel()
        streamingTickerTask = nil
        clearStreamingState()
    }

    @MainActor
    private func clearStreamingState() {
        streamingMessageId = nil
        streamingFullText = ""
        streamingShownCount = 0
        streamingIsActive = false
        streamingFinalText = nil
    }

    func clearError() {
        errorMessage = nil
    }
}
