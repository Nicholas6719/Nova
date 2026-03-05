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

    /// Mic enabled when not processing, or when Nova is speaking (barge-in).
    /// Disabled only while waiting on OpenAI/engine (and not yet speaking).
    var isMicEnabled: Bool { !isProcessing || speechManager.isSpeaking }

    // MARK: - Dependencies

    private let speechRecognizer = SpeechRecognizer()
    private let speechManager = SpeechManager()
    private let engine = NovaEngine()

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Streaming UI state (MainActor-owned)

    /// Invalidated on barge-in so late deltas/completion become no-ops.
    private var activeStreamToken: UUID?
    private var streamingFullText = ""
    private var streamingShownCount = 0
    private var streamingMessageId: UUID?
    private var streamingTickerTask: Task<Void, Never>?
    private var streamingIsActive = false
    private var streamingFinalText: String?

    // MARK: - Streaming speech state (MainActor-owned)

    /// Accumulates unspoken text. Sentences are extracted and spoken one at a time.
    private var speechBuffer = ""
    /// True while SpeechManager is speaking a sentence we extracted from the buffer.
    private var isSpeakingStreamChunk = false
    /// True once at least one sentence has been spoken during this streaming session.
    private var hasSpokenAnyStreamChunk = false

    // MARK: - Debounce

    private var lastCommittedTranscript = ""
    private var lastCommittedTime: CFAbsoluteTime = 0

    /// Hands-free: auto-start listening after TTS completes; cancel after timeout.
    private var autoListenTask: Task<Void, Never>?
    private let autoListenTimeout: TimeInterval = 6

    /// End-of-speech: stop listening ~0.9s after last transcript update.
    private var endOfSpeechTask: Task<Void, Never>?
    private var lastTranscriptUpdate = Date()
    private let endOfSpeechDelay: TimeInterval = 0.9

    /// Gate: auto-listen only after Nova has spoken (not on launch).
    private var allowAutoListen = false

    // MARK: - Init

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

        // Drive sentence-by-sentence TTS: when speech finishes, try the next queued sentence.
        speechManager.$isSpeaking
            .dropFirst()
            .removeDuplicates()
            .filter { !$0 }
            .sink { [weak self] _ in self?.onSpeechFinished() }
            .store(in: &cancellables)

        speechManager.onSpeechFinished = { [weak self] in
            Task { @MainActor in
                self?.handleSpeechFinished()
            }
        }

        speechRecognizer.onPartialTranscript = { [weak self] _ in
            Task { @MainActor in
                self?.lastTranscriptUpdate = Date()
            }
        }

        speechRecognizer.onFinalTranscript = { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isRecording else { return }
                self.endOfSpeechTask?.cancel()
                self.endOfSpeechTask = nil
                self.speechRecognizer.stopListening()
            }
        }
    }

    // MARK: - Permissions

    func requestPermissions() {
        speechRecognizer.requestPermissions()
    }

    // MARK: - Microphone

    func toggleRecording() {
        // Barge-in: tap during speech, streaming, or processing → stop + start listening (1 tap).
        if speechManager.isSpeaking || activeStreamToken != nil || isProcessing {
            autoListenTask?.cancel()
            autoListenTask = nil
            endOfSpeechTask?.cancel()
            endOfSpeechTask = nil
            speechManager.prepareForBargeIn()
            cancelStreamingState()
            isProcessing = false
            errorMessage = nil
            speechRecognizer.clearTranscript()
            speechRecognizer.startListening()
            return
        }
        if !isMicEnabled { return }
        autoListenTask?.cancel()
        autoListenTask = nil
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil
        if isRecording {
            speechRecognizer.stopListening()
        } else {
            errorMessage = nil
            speechRecognizer.clearTranscript()
            speechRecognizer.startListening()
        }
    }

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
        autoListenTask?.cancel()
        autoListenTask = nil
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil

        cancelStreamingState()

        let token = UUID()
        activeStreamToken = token

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
                self?.beginStreaming(token: token)
            }
        }

        let onStreamDelta: @Sendable (String) -> Void = { [weak self] chunk in
            Task { @MainActor [weak self] in
                self?.receiveStreamDelta(chunk, token: token)
            }
        }

        let engineRef = engine
        Task.detached(priority: .userInitiated) { [cfg, messageSnapshot, text, onStreamStart, onStreamDelta, token] in
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
                    self.onEngineComplete(fullText: resp, token: token)
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    DebugLog.d("[Chat] engine error: \(error)")
                    self.onEngineComplete(fullText: "Sorry — I couldn't reach my online brain. Please try again.", token: token)
                }
            }
        }
    }

    // MARK: - Streaming: start / delta / ticker

    @MainActor
    private func beginStreaming(token: UUID) {
        guard activeStreamToken == token else { return }
        let pid = UUID()
        streamingMessageId = pid
        streamingFullText = ""
        streamingShownCount = 0
        streamingIsActive = true
        streamingFinalText = nil
        speechBuffer = ""
        isSpeakingStreamChunk = false
        hasSpokenAnyStreamChunk = false
        messages.append(Message(id: pid, role: .assistant, content: ""))
        startTicker(token: token)
    }

    @MainActor
    private func receiveStreamDelta(_ delta: String, token: UUID) {
        guard activeStreamToken == token else { return }
        streamingFullText += delta
        speechBuffer += delta
        trySpeakNextSentence()
    }

    /// Continuous ticker: reveals 5 characters every 120ms (~41 chars/sec).
    @MainActor
    private func startTicker(token: UUID) {
        streamingTickerTask?.cancel()
        streamingTickerTask = Task { @MainActor [weak self] in
            let step = 5
            let intervalNs: UInt64 = 120_000_000

            while !Task.isCancelled {
                guard let self else { return }
                guard self.activeStreamToken == token else { return }

                let available = self.streamingFullText.count
                if self.streamingShownCount < available {
                    self.streamingShownCount = min(self.streamingShownCount + step, available)
                    self.updateBubble(charCount: self.streamingShownCount, token: token)
                }

                if !self.streamingIsActive && self.streamingShownCount >= self.streamingFullText.count {
                    self.commitFinalStreamedResponse(token: token)
                    return
                }

                try? await Task.sleep(nanoseconds: intervalNs)
            }
        }
    }

    @MainActor
    private func updateBubble(charCount: Int, token: UUID) {
        guard activeStreamToken == token else { return }
        guard let pid = streamingMessageId,
              let idx = messages.lastIndex(where: { $0.id == pid }) else { return }
        let text = String(streamingFullText.prefix(charCount))
        messages[idx] = Message(id: pid, role: .assistant, content: text)
    }

    // MARK: - Sentence-by-sentence TTS

    /// Extract the first complete sentence from speechBuffer and speak it.
    /// A sentence boundary is a terminator (. ! ?) followed by whitespace, or at
    /// the end of the buffer when the stream has finished.
    @MainActor
    private func trySpeakNextSentence() {
        guard activeStreamToken != nil else { return }
        guard !isSpeakingStreamChunk else { return }
        guard let sentence = extractFirstSentence() else { return }
        isSpeakingStreamChunk = true
        hasSpokenAnyStreamChunk = true
        allowAutoListen = true
        speechManager.speak(sentence)
    }

    @MainActor
    private func extractFirstSentence() -> String? {
        let terminators: Set<Character> = [".", "!", "?"]
        var i = speechBuffer.startIndex
        while i < speechBuffer.endIndex {
            let ch = speechBuffer[i]
            let nextIdx = speechBuffer.index(after: i)
            if terminators.contains(ch) {
                if nextIdx < speechBuffer.endIndex {
                    let next = speechBuffer[nextIdx]
                    if next == " " || next.isNewline {
                        let sentence = String(speechBuffer[speechBuffer.startIndex...i])
                            .trimmingCharacters(in: .whitespaces)
                        speechBuffer = String(speechBuffer[nextIdx...])
                            .trimmingCharacters(in: .whitespaces)
                        return sentence.isEmpty ? nil : sentence
                    }
                } else if !streamingIsActive {
                    let sentence = String(speechBuffer[speechBuffer.startIndex...i])
                        .trimmingCharacters(in: .whitespaces)
                    speechBuffer = ""
                    return sentence.isEmpty ? nil : sentence
                }
            }
            i = nextIdx
        }
        return nil
    }

    /// Called via Combine when speechManager.isSpeaking transitions to false.
    @MainActor
    private func onSpeechFinished() {
        guard isSpeakingStreamChunk else { return }
        guard activeStreamToken != nil else { return }
        isSpeakingStreamChunk = false

        if streamingMessageId == nil {
            // Ticker already finished — speak any remaining buffer text.
            speakRemainingBuffer()
        } else {
            trySpeakNextSentence()
        }
    }

    @MainActor
    private func speakRemainingBuffer() {
        let remaining = speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        clearSpeechState()
        if !remaining.isEmpty {
            allowAutoListen = true
            speechManager.speak(remaining)
        }
    }

    // MARK: - Finalize streaming response

    /// Called by the ticker when all text has been revealed in the UI.
    @MainActor
    private func commitFinalStreamedResponse(token: UUID) {
        guard activeStreamToken == token else { return }
        streamingTickerTask?.cancel()
        streamingTickerTask = nil

        let fullText = streamingFinalText ?? streamingFullText

        if let pid = streamingMessageId,
           let idx = messages.lastIndex(where: { $0.id == pid }) {
            messages[idx] = Message(id: pid, role: .assistant, content: fullText)
        }

        DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")

        // Clear UI streaming state; speech state may survive briefly for onSpeechFinished.
        activeStreamToken = nil
        streamingMessageId = nil
        streamingFullText = ""
        streamingShownCount = 0
        streamingFinalText = nil
        streamingIsActive = false

        if !hasSpokenAnyStreamChunk {
            clearSpeechState()
            allowAutoListen = true
            speechManager.speak(fullText)
        } else if !isSpeakingStreamChunk {
            speakRemainingBuffer()
        }
        // else: speech still playing → onSpeechFinished will call speakRemainingBuffer

        isProcessing = false
    }

    // MARK: - Engine completion

    @MainActor
    private func onEngineComplete(fullText: String, token: UUID) {
        if streamingMessageId != nil {
            guard activeStreamToken == token else { return }
            streamingFinalText = fullText
            streamingFullText = fullText
            streamingIsActive = false
            // Ticker keeps running to finish reveal. Also try to speak if idle.
            if !isSpeakingStreamChunk {
                trySpeakNextSentence()
            }
        } else {
            // Non-streaming (local) OR invalidated streaming (barge-in before placeholder).
            guard activeStreamToken == token else { return }
            messages.append(Message(role: .assistant, content: fullText))
            DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")
            allowAutoListen = true
            speechManager.speak(fullText)
            isProcessing = false
        }
    }

    // MARK: - Streaming cleanup

    @MainActor
    private func cancelStreamingState() {
        activeStreamToken = nil
        streamingTickerTask?.cancel()
        streamingTickerTask = nil
        speechManager.stop()
        streamingMessageId = nil
        streamingFullText = ""
        streamingShownCount = 0
        streamingIsActive = false
        streamingFinalText = nil
        clearSpeechState()
    }

    @MainActor
    private func clearSpeechState() {
        speechBuffer = ""
        isSpeakingStreamChunk = false
        hasSpokenAnyStreamChunk = false
    }

    func clearError() {
        errorMessage = nil
    }

    // MARK: - Hands-free auto-listen

    @MainActor
    private func handleSpeechFinished() {
        guard allowAutoListen else { return }
        guard !speechManager.isSpeaking else { return }
        guard !isProcessing else { return }
        guard !isRecording else { return }

        allowAutoListen = false
        autoListenTask?.cancel()
        autoListenTask = nil
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil
        errorMessage = nil
        speechRecognizer.clearTranscript()
        lastTranscriptUpdate = Date()
        speechRecognizer.startListening()

        DebugLog.d("[AutoListen] auto-listen start")

        startEndOfSpeechMonitor()

        autoListenTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64((self?.autoListenTimeout ?? 6) * 1_000_000_000))
            guard let self else { return }
            guard !Task.isCancelled else { return }
            self.endOfSpeechTask?.cancel()
            self.endOfSpeechTask = nil
            if self.isRecording {
                DebugLog.d("[AutoListen] hard timeout triggered")
                self.speechRecognizer.stopListening()
            }
            self.autoListenTask = nil
        }
    }

    @MainActor
    private func startEndOfSpeechMonitor() {
        endOfSpeechTask?.cancel()
        endOfSpeechTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                try? await Task.sleep(nanoseconds: 150_000_000)
                guard !Task.isCancelled else { return }
                guard self.isRecording else { return }
                let transcript = self.liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !transcript.isEmpty else { continue }

                if Date().timeIntervalSince(self.lastTranscriptUpdate) >= self.endOfSpeechDelay {
                    DebugLog.d("[AutoListen] end-of-speech triggered (silence >= \(self.endOfSpeechDelay)s)")
                    self.endOfSpeechTask?.cancel()
                    self.endOfSpeechTask = nil
                    self.speechRecognizer.stopListening()
                    return
                }
            }
        }
    }
}
