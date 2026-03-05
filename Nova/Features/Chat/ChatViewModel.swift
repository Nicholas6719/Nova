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
    private let hardTimeoutManual: TimeInterval = 15
    private let hardTimeoutAuto: TimeInterval = 6

    /// End-of-speech: stop listening after last transcript update (manual: transcript-based only).
    private var endOfSpeechTask: Task<Void, Never>?
    private var lastTranscriptUpdate = Date()
    private let endOfSpeechDelay: TimeInterval = 1.2

    /// Skip next handleRecordingStopped when we committed from partial (stopListeningAndCommitNow).
    private var skipNextRecordingStopped = false

    /// Gate: auto-listen only after Nova has spoken (not on launch).
    private var allowAutoListen = false

    private enum RecordingMode { case manual, auto }
    private var recordingMode: RecordingMode = .manual
    private var recordingSessionId: UUID = UUID()

    /// VAD: unified for both manual and auto. Silence-based EOS; hard cap is safety only.
    private var recordingStartedAt: Date = .init()
    private var vadHeardSpeech = false
    private var firstSpeechAt: Date?
    private var lastSpeechOrTranscriptAt: Date?
    private var vadDidTriggerStop = false
    private let vadSpeechThreshold: Float = 0.012
    private let vadSilenceDuration: TimeInterval = 1.0
    private let vadMinSpeechBeforeStop: TimeInterval = 0.6

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
            Task { @MainActor [weak self] in
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

        speechManager.onSpeechStarted = { [weak self] in
            Task { @MainActor [weak self] in
                self?.handleSpeechStarted()
            }
        }
        speechManager.onSpeechFinished = { [weak self] in
            Task { @MainActor [weak self] in
                self?.handleSpeechFinished()
            }
        }

        speechRecognizer.onPartialTranscript = { [weak self] text in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                self.lastTranscriptUpdate = Date()
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let now = Date()
                    self.vadHeardSpeech = true
                    if self.firstSpeechAt == nil { self.firstSpeechAt = now }
                    self.lastSpeechOrTranscriptAt = now
                }
            }
        }

        speechRecognizer.onFinalTranscript = { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                self.lastTranscriptUpdate = Date()
                self.stopListeningAndCommitNow(reason: "final", sessionId: self.recordingSessionId)
            }
        }

        speechRecognizer.onAudioEnergy = { [weak self] rms in
            Task { @MainActor [weak self] in
                self?.handleAudioEnergy(rms)
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
            speechManager.prepareForBargeIn()
            cancelStreamingState()
            isProcessing = false
            errorMessage = nil
            beginRecordingSession(mode: .manual)
            return
        }
        if !isMicEnabled { return }
        if isRecording {
            stopListeningAndCommitNow(reason: "manual tap", sessionId: recordingSessionId)
        } else {
            beginRecordingSession(mode: .manual)
        }
    }

    private func handleRecordingStopped(with text: String) {
        if skipNextRecordingStopped {
            skipNextRecordingStopped = false
            liveTranscript = ""
            speechRecognizer.clearTranscript()
            return
        }
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

    /// Single entry point for TTS: arms auto-listen, logs, speaks, and starts 400ms fallback if TTS never starts.
    @MainActor
    private func speakWithAutoListen(_ text: String, source: String) {
        allowAutoListen = true
        speechManager.speak(text)
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard let self else { return }
            if self.allowAutoListen, !self.speechManager.isSpeaking, !self.isProcessing, !self.isRecording {
                self.allowAutoListen = false
                self.beginRecordingSession(mode: .auto)
            }
        }
    }

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
        speakWithAutoListen(sentence, source: "openai")
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
        if isSpeakingStreamChunk, activeStreamToken != nil {
            isSpeakingStreamChunk = false
            if streamingMessageId == nil {
                speakRemainingBuffer()
            } else {
                trySpeakNextSentence()
            }
        }
        if allowAutoListen, !speechManager.isSpeaking, !isProcessing, !isRecording {
            allowAutoListen = false
            beginRecordingSession(mode: .auto)
        }
    }

    @MainActor
    private func speakRemainingBuffer() {
        let remaining = speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        clearSpeechState()
        if !remaining.isEmpty {
            speakWithAutoListen(remaining, source: "openai")
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
            speakWithAutoListen(fullText, source: "openai")
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
            speakWithAutoListen(fullText, source: "local")
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

    // MARK: - Recording session (session-scoped cancellation safety)

    @MainActor
    private func beginRecordingSession(mode: RecordingMode) {
        recordingSessionId = UUID()
        recordingMode = mode
        recordingStartedAt = Date()
        lastTranscriptUpdate = Date()
        vadHeardSpeech = false
        firstSpeechAt = nil
        lastSpeechOrTranscriptAt = nil
        vadDidTriggerStop = false
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil
        autoListenTask?.cancel()
        autoListenTask = nil
        errorMessage = nil
        speechRecognizer.clearTranscript()
        speechRecognizer.startListening()
        DebugLog.d("[EOS] begin mode=\(recordingMode) sid=\(recordingSessionId)")
        startEndOfSpeechMonitor(sessionId: recordingSessionId)
        startHardTimeout(sessionId: recordingSessionId, mode: mode)
    }

    @MainActor
    private func handleAudioEnergy(_ rms: Float) {
        guard isRecording else { return }
        guard !vadDidTriggerStop else { return }
        let sessionId = recordingSessionId
        let now = Date()
        if rms > vadSpeechThreshold {
            if !vadHeardSpeech {
                vadHeardSpeech = true
                firstSpeechAt = now
                DebugLog.d("[VAD] speech start sid=\(sessionId)")
            }
            lastSpeechOrTranscriptAt = now
        } else if vadHeardSpeech, let lastAt = lastSpeechOrTranscriptAt, let firstAt = firstSpeechAt {
            let silenceElapsed = now.timeIntervalSince(lastAt)
            let elapsedSinceFirstSpeech = now.timeIntervalSince(firstAt)
            guard elapsedSinceFirstSpeech >= vadMinSpeechBeforeStop else { return }
            if silenceElapsed >= vadSilenceDuration {
                vadDidTriggerStop = true
                DebugLog.d("[VAD] stop sid=\(sessionId) mode=\(recordingMode) silence=\(String(format: "%.2f", silenceElapsed))s")
                stopListeningAndCommitNow(reason: "vad-silence", sessionId: sessionId)
            }
        }
    }

    @MainActor
    private func stopListeningAndCommitNow(reason: String, sessionId: UUID) {
        guard sessionId == recordingSessionId else {
            DebugLog.d("[EOS] ignore stale stop reason=\(reason) staleSid=\(sessionId) currentSid=\(recordingSessionId)")
            return
        }
        guard isRecording else { return }
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil
        autoListenTask?.cancel()
        autoListenTask = nil

        let text = liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        speechRecognizer.stopListening()
        liveTranscript = ""
        speechRecognizer.clearTranscript()
        skipNextRecordingStopped = true

        guard !text.isEmpty else {
            if reason == "vad-silence" {
                errorMessage = "No speech recognized. Try again."
            }
            return
        }
        DebugLog.d("[EOS] stop reason=\(reason) mode=\(recordingMode) sid=\(sessionId) text.len=\(text.count)")
        processUserInput(text: text)
    }

    @MainActor
    private func startEndOfSpeechMonitor(sessionId: UUID) {
        endOfSpeechTask?.cancel()
        endOfSpeechTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard !Task.isCancelled else { return }
                guard sessionId == self.recordingSessionId else { return }
                guard self.isRecording else { return }
                let transcript = self.liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !transcript.isEmpty else { continue }

                if Date().timeIntervalSince(self.lastTranscriptUpdate) >= self.endOfSpeechDelay {
                    self.endOfSpeechTask?.cancel()
                    self.endOfSpeechTask = nil
                    let reason = self.recordingMode == .manual ? "partial-silence" : "partial-silence"
                    self.stopListeningAndCommitNow(reason: reason, sessionId: sessionId)
                    return
                }
            }
        }
    }

    @MainActor
    private func startHardTimeout(sessionId: UUID, mode: RecordingMode) {
        let timeout: TimeInterval = mode == .manual ? hardTimeoutManual : hardTimeoutAuto
        autoListenTask?.cancel()
        autoListenTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            guard let self else { return }
            guard !Task.isCancelled else { return }
            guard sessionId == self.recordingSessionId else { return }
            if self.isRecording {
                self.stopListeningAndCommitNow(reason: "hard-timeout", sessionId: sessionId)
            }
            self.autoListenTask = nil
        }
    }

    // MARK: - Hands-free auto-listen

    @MainActor
    private func handleSpeechStarted() {
        // TTS actually started; fallback will no-op (isSpeaking true or we'll get onSpeechFinished).
    }

    /// Called by SpeechManager when TTS fully completes (all utterances done).
    @MainActor
    private func handleSpeechFinished() {
        guard allowAutoListen, !speechManager.isSpeaking, !isProcessing, !isRecording else { return }
        allowAutoListen = false
        beginRecordingSession(mode: .auto)
    }
}
