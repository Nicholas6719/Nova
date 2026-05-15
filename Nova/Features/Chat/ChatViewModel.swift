import Foundation
import Combine
import Speech
import AVFoundation

/// UI status for the header indicator.
enum ChatStatus: String {
    case idle = "Idle"
    case standby = "Standby"
    case listening = "Listening"
    case processing = "Processing"
    case speaking = "Speaking"
}

@MainActor
final class ChatViewModel: ObservableObject {

    // MARK: - StreamingSession

    /// Holds all mutable state for one streaming response session.
    /// Created at the start of streaming; nilled out on completion or cancel.
    private final class StreamingSession {
        let messageId: UUID
        var fullText: String = ""
        var shownCount: Int = 0
        var tickerTask: Task<Void, Never>?
        var isActive: Bool = true
        var finalText: String?

        init(messageId: UUID) {
            self.messageId = messageId
        }
    }

    // MARK: - Published

    @Published private(set) var messages: [Message] = []
    @Published private(set) var isRecording: Bool = false
    @Published private(set) var liveTranscript: String = ""
    @Published private(set) var errorMessage: String?
    @Published private(set) var isProcessing: Bool = false

    var status: ChatStatus {
        if speechManager.isSpeaking { return .speaking }
        if isProcessing { return .processing }
        if isRecording {
            return recordingMode == .wake ? .standby : .listening
        }
        if isWakeTriggered { return .listening }
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

    // MARK: - Streaming state (MainActor-owned)

    /// Invalidated on barge-in so late deltas/completion become no-ops.
    private var activeStreamToken: UUID?

    /// Non-nil while a streaming response session is in progress.
    private var stream: StreamingSession?

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
    /// Scheduled auto-listen start (250ms after TTS finished). Cancelled when a new one is scheduled.
    private var scheduledAutoListenTask: Task<Void, Never>?
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

    // MARK: - Wake word (Slice A: detection only, no command capture)

    private var wakeWordEnabled = true
    @Published private(set) var isWakeTriggered: Bool = false
    @Published private(set) var wakeTriggerPhrase: String?
    private enum PendingTransition: Equatable { case none; case toWakeTriggered(phrase: String); case toReturnToWake; case toFreshCommandAfterBareWake }
    private var pendingTransition: PendingTransition = .none
    private let hardTimeoutWake: TimeInterval = 300

    private var isWakeListening: Bool { recordingMode == .wake }

    private enum RecordingMode { case manual, auto, wake, command }
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

        speechRecognizer.$authorizationStatus
            .dropFirst()
            .sink { [weak self] status in
                Task { @MainActor in
                    guard let self else { return }
                    if status == .authorized {
                        self.errorMessage = nil
                        DebugLog.d("[Wake] cleared stale permission error")
                        self.startWakeListeningIfIdle()
                    }
                }
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

        speechManager.onSpeechFinished = { [weak self] in
            Task { @MainActor [weak self] in
                self?.handleSpeechFinished()
            }
        }

        speechRecognizer.onPartialTranscript = { [weak self] text in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                if self.recordingMode == .wake {
                    self.handleWakeTranscript(text)
                    return
                }
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
                if self.recordingMode == .wake { return }
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
        #if os(iOS)
        // iOS: explicitly request mic permission at startup and retry wake start
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { _ in
                Task { @MainActor [weak self] in
                    self?.startWakeListeningIfIdle()
                }
            }
        } else {
            AVAudioSession.sharedInstance().requestRecordPermission { _ in
                Task { @MainActor [weak self] in
                    self?.startWakeListeningIfIdle()
                }
            }
        }
        // Fallback: retry after delay in case both permissions were already granted
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            self?.startWakeListeningIfIdle()
        }
        #endif
    }

    /// Start wake only when auth and mic ready, app idle. Called from auth sink or after manual response.
    func startWakeListeningIfIdle() {
        DebugLog.d("[Chat] startWakeListeningIfIdle called")
        guard wakeWordEnabled else { return }
        guard speechRecognizer.authorizationStatus == .authorized else {
            DebugLog.d("[Wake] auth not ready")
            return
        }
        #if os(iOS)
        let micGranted: Bool = if #available(iOS 17.0, *) {
            AVAudioApplication.shared.recordPermission == .granted
        } else {
            AVAudioSession.sharedInstance().recordPermission == .granted
        }
        guard micGranted else {
            DebugLog.d("[Wake] auth not ready")
            return
        }
        #endif
        if errorMessage != nil {
            errorMessage = nil
            DebugLog.d("[Wake] cleared stale permission error")
        }
        guard !isRecording else { return }
        guard recordingMode != .wake else { return }
        guard case .none = pendingTransition else { return }
        guard !isProcessing else { return }
        guard !speechManager.isSpeaking else { return }
        DebugLog.d("[Wake] auth ready -> start wake")
        startWakeListening()
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
        isWakeTriggered = false
        wakeTriggerPhrase = nil
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
            if pendingTransition == .toFreshCommandAfterBareWake {
                pendingTransition = .none
                vadHeardSpeech = false
                firstSpeechAt = nil
                lastSpeechOrTranscriptAt = nil
                vadDidTriggerStop = false
                DebugLog.d("[Wake] fresh command session started after bare wake")
                beginRecordingSession(mode: .command)
            }
            return
        }

        let transition = pendingTransition
        pendingTransition = .none
        liveTranscript = ""
        speechRecognizer.clearTranscript()

        switch transition {
        case .toWakeTriggered:
            return
        case .toFreshCommandAfterBareWake:
            return
        case .toReturnToWake:
            returnToWakeListeningIfIdle()
            return
        case .none:
            break
        }

        let finalText = text.trimmingCharacters(in: .whitespacesAndNewlines)

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

    // MARK: - Wake phrase stripping

    /// Strip first "hey nova" or "nova" occurrence (word boundary); return trimmed remainder.
    private func extractWakeRemainder(from text: String) -> String {
        let s = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let wordBoundary: (Character) -> Bool = { $0.isWhitespace || $0.isPunctuation }
        if let range = s.range(of: "hey nova") {
            let rest = String(s[range.upperBound...])
            if rest.isEmpty || rest.first.map(wordBoundary) ?? true {
                return rest.trimmingCharacters(in: CharacterSet.whitespaces.union(CharacterSet(charactersIn: ",.")))
            }
        }
        if let range = s.range(of: "nova") {
            let rest = String(s[range.upperBound...])
            if rest.isEmpty || rest.first.map(wordBoundary) ?? true {
                return rest.trimmingCharacters(in: CharacterSet.whitespaces.union(CharacterSet(charactersIn: ",.")))
            }
        }
        return s.trimmingCharacters(in: CharacterSet.whitespaces.union(CharacterSet(charactersIn: ",.")))
    }

    // MARK: - Process user input

    private func processUserInput(text: String) {
        guard !text.isEmpty else { return }
        guard !isProcessing else { return }
        isProcessing = true
        errorMessage = nil
        autoListenTask?.cancel()
        autoListenTask = nil
        scheduledAutoListenTask?.cancel()
        scheduledAutoListenTask = nil
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil

        cancelStreamingState()

        let token = UUID()
        activeStreamToken = token

        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)
        DebugLog.d("[Chat] append user: \(text.prefix(60))\(text.count > 60 ? "…" : "")")

        let messageSnapshot = messages

        let apiKey = APIKeyProvider.openAIKey
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
                let resp = try await engineRef.generateResponse(
                    messages: messageSnapshot,
                    newInput: text,
                    llmConfig: cfg,
                    onStreamStart: onStreamStart,
                    onStreamDelta: onStreamDelta
                )
                Task { @MainActor [weak self] in
                    self?.onEngineComplete(fullText: resp, token: token)
                }
            } catch {
                Task { @MainActor [weak self] in
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
        stream = StreamingSession(messageId: pid)
        speechBuffer = ""
        isSpeakingStreamChunk = false
        hasSpokenAnyStreamChunk = false
        messages.append(Message(id: pid, role: .assistant, content: ""))
        startTicker(token: token)
    }

    @MainActor
    private func receiveStreamDelta(_ delta: String, token: UUID) {
        guard activeStreamToken == token else { return }
        stream?.fullText += delta
        speechBuffer += delta
        trySpeakNextSentence()
    }

    /// Continuous ticker: reveals 5 characters every 120ms (~41 chars/sec).
    @MainActor
    private func startTicker(token: UUID) {
        stream?.tickerTask?.cancel()
        stream?.tickerTask = Task { @MainActor [weak self] in
            let step = 5
            let intervalNs: UInt64 = 120_000_000

            while !Task.isCancelled {
                guard let self else { return }
                guard self.activeStreamToken == token else { return }
                guard let s = self.stream else { return }

                let available = s.fullText.count
                if s.shownCount < available {
                    s.shownCount = min(s.shownCount + step, available)
                    self.updateBubble(charCount: s.shownCount, token: token)
                }

                if !s.isActive && s.shownCount >= s.fullText.count {
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
        guard let s = stream,
              let idx = messages.lastIndex(where: { $0.id == s.messageId }) else { return }
        let text = String(s.fullText.prefix(charCount))
        messages[idx] = Message(id: s.messageId, role: .assistant, content: text)
    }

    // MARK: - Sentence-by-sentence TTS

    /// Single entry point for TTS: arms auto-listen and speaks. No fallback — auto-listen starts only from TTS-finished signal.
    @MainActor
    private func speakWithAutoListen(_ text: String) {
        allowAutoListen = true
        speechManager.speak(text)
    }

    /// Extract the first complete sentence from speechBuffer and speak it.
    /// A sentence boundary is a terminator (. ! ?) followed by whitespace, or at
    /// the end of the buffer when the stream has finished.
    @MainActor
    private func trySpeakNextSentence() {
        guard stream != nil else { return }
        guard !isSpeakingStreamChunk else { return }
        guard let sentence = extractFirstSentence() else { return }
        isSpeakingStreamChunk = true
        hasSpokenAnyStreamChunk = true
        speakWithAutoListen(sentence)
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
                } else if !(stream?.isActive ?? true) {
                    let sentence = String(speechBuffer[speechBuffer.startIndex...i])
                        .trimmingCharacters(in: .whitespaces)
                    speechBuffer = ""
                    return sentence.isEmpty ? nil : sentence
                }
            }
            i = nextIdx
        }
        if !(stream?.isActive ?? true) {
            let remainder = speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
            speechBuffer = ""
            return remainder.isEmpty ? nil : remainder
        }
        return nil
    }

    /// Called via Combine when speechManager.isSpeaking transitions to false.
    @MainActor
    private func onSpeechFinished() {
        if isSpeakingStreamChunk {
            isSpeakingStreamChunk = false
            if stream != nil {
                trySpeakNextSentence()
            } else {
                speakRemainingBuffer()
            }
        }
        // Schedule only from handleSpeechFinished (engine callback) to avoid duplicate restart
    }

    @MainActor
    private func speakRemainingBuffer() {
        let remaining = speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        clearSpeechState()
        if !remaining.isEmpty {
            speakWithAutoListen(remaining)
        }
    }

    // MARK: - Finalize streaming response

    /// Called by the ticker when all text has been revealed in the UI.
    @MainActor
    private func commitFinalStreamedResponse(token: UUID) {
        guard activeStreamToken == token else { return }
        stream?.tickerTask?.cancel()

        let fullText = stream?.finalText ?? stream?.fullText ?? ""

        if let s = stream,
           let idx = messages.lastIndex(where: { $0.id == s.messageId }) {
            messages[idx] = Message(id: s.messageId, role: .assistant, content: fullText)
        }

        DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")

        // Clear UI streaming state; speech state may survive briefly for onSpeechFinished.
        stream = nil
        activeStreamToken = nil

        if !hasSpokenAnyStreamChunk {
            clearSpeechState()
            speakWithAutoListen(fullText)
        } else if !isSpeakingStreamChunk {
            speakRemainingBuffer()
        }
        // else: speech still playing → onSpeechFinished will call speakRemainingBuffer

        isProcessing = false
    }

    // MARK: - Engine completion

    @MainActor
    private func onEngineComplete(fullText: String, token: UUID) {
        if stream != nil {
            guard activeStreamToken == token else { return }
            stream?.finalText = fullText
            stream?.fullText = fullText
            stream?.isActive = false
            // Ticker keeps running to finish reveal. Also try to speak if idle.
            if !isSpeakingStreamChunk {
                trySpeakNextSentence()
            }
        } else {
            // Non-streaming (local) OR invalidated streaming (barge-in before placeholder).
            guard activeStreamToken == token else { return }
            messages.append(Message(role: .assistant, content: fullText))
            DebugLog.d("[Chat] append assistant: \(fullText.prefix(60))\(fullText.count > 60 ? "…" : "")")
            speakWithAutoListen(fullText)
            isProcessing = false
        }
    }

    // MARK: - Streaming cleanup

    @MainActor
    private func cancelStreamingState() {
        activeStreamToken = nil
        stream?.tickerTask?.cancel()
        stream = nil
        speechManager.stop()
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
        if mode == .auto {
            let hasUnspoken = !speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            if speechManager.isSpeaking || speechManager.hasPendingSpeech || hasUnspoken {
                return
            }
        }
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
        scheduledAutoListenTask?.cancel()
        scheduledAutoListenTask = nil
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
        guard recordingMode != .wake else { return }
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
        guard sessionId == recordingSessionId else { return }
        guard isRecording else { return }
        endOfSpeechTask?.cancel()
        endOfSpeechTask = nil
        autoListenTask?.cancel()
        autoListenTask = nil
        scheduledAutoListenTask?.cancel()
        scheduledAutoListenTask = nil

        let text = liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        speechRecognizer.stopListening()
        liveTranscript = ""
        speechRecognizer.clearTranscript()
        skipNextRecordingStopped = true

        if recordingMode == .command {
            let remainder = extractWakeRemainder(from: text)
            if remainder.isEmpty {
                if reason == "hard-timeout" {
                    Task { @MainActor [weak self] in
                        try? await Task.sleep(nanoseconds: 100_000_000)
                        self?.returnToWakeListeningIfIdle()
                    }
                } else {
                    DebugLog.d("[Wake] bare wake phrase -> restart fresh command session")
                    pendingTransition = .toFreshCommandAfterBareWake
                }
                return
            }
            DebugLog.d("[Wake] extracted remainder=\"\(remainder.prefix(80))\(remainder.count > 80 ? "…" : "")\"")
            processUserInput(text: remainder)
            return
        }

        guard !text.isEmpty else {
            if recordingMode == .auto && wakeWordEnabled {
                Task { @MainActor [weak self] in
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    self?.returnToWakeListeningIfIdle()
                }
            } else if reason == "vad-silence" {
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
                guard self.recordingMode != .wake else { continue }
                let transcript = self.liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !transcript.isEmpty else { continue }

                if Date().timeIntervalSince(self.lastTranscriptUpdate) >= self.endOfSpeechDelay {
                    self.endOfSpeechTask?.cancel()
                    self.endOfSpeechTask = nil
                    self.stopListeningAndCommitNow(reason: "partial-silence", sessionId: sessionId)
                    return
                }
            }
        }
    }

    @MainActor
    private func startHardTimeout(sessionId: UUID, mode: RecordingMode) {
        let timeout: TimeInterval = switch mode {
        case .manual: hardTimeoutManual
        case .wake: hardTimeoutWake
        default: hardTimeoutAuto
        }
        autoListenTask?.cancel()
        autoListenTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            guard let self else { return }
            guard !Task.isCancelled else { return }
            guard sessionId == self.recordingSessionId else { return }
            if self.isRecording {
                if self.recordingMode == .wake {
                    self.pendingTransition = .toReturnToWake
                }
                self.stopListeningAndCommitNow(reason: "hard-timeout", sessionId: sessionId)
            }
            self.autoListenTask = nil
        }
    }

    // MARK: - Hands-free auto-listen

    /// Called by SpeechManager when TTS fully completes (all utterances done).
    @MainActor
    private func handleSpeechFinished() {
        scheduleAutoListenStart(source: "handleSpeechFinished")
    }

    /// Schedules auto-listen start after stabilization delay. Single path from engine callback.
    @MainActor
    private func scheduleAutoListenStart(source: String) {
        guard allowAutoListen else { return }
        scheduledAutoListenTask?.cancel()
        scheduledAutoListenTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 250_000_000)  // 250ms stabilization
            guard let self else { return }
            guard !Task.isCancelled else { return }
            self.scheduledAutoListenTask = nil
            let hasUnspoken = !self.speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let ok = self.allowAutoListen && !self.isProcessing && !self.isRecording
                && !self.speechManager.isSpeaking && !self.speechManager.hasPendingSpeech
                && !hasUnspoken && self.stream == nil && !self.isSpeakingStreamChunk
            if !ok { return }
            self.allowAutoListen = false
            self.beginRecordingSession(mode: .auto)
        }
    }

    // MARK: - Wake word helpers (Slice A)

    @MainActor
    private func startWakeListening() {
        guard !isRecording else { return }
        guard recordingMode != .wake else { return }
        guard case .none = pendingTransition else { return }
        isWakeTriggered = false
        wakeTriggerPhrase = nil
        beginRecordingSession(mode: .wake)
    }

    @MainActor
    private func handleWakeTranscript(_ text: String) {
        let lower = text.lowercased()
        let trigger: String? = lower.contains("hey nova") ? "hey nova" : (lower.contains("nova") ? "nova" : nil)
        guard let trigger else { return }
        DebugLog.d("[Wake] detected trigger=\(trigger)")
        DebugLog.d("[Wake] stay in same session for command capture")
        isWakeTriggered = true
        wakeTriggerPhrase = trigger
        recordingMode = .command
        lastTranscriptUpdate = Date()
        vadHeardSpeech = true
        firstSpeechAt = firstSpeechAt ?? Date()
        lastSpeechOrTranscriptAt = Date()
        autoListenTask?.cancel()
        autoListenTask = nil
        startHardTimeout(sessionId: recordingSessionId, mode: .command)
    }

    @MainActor
    private func returnToWakeListeningIfIdle() {
        guard !isRecording else { return }
        guard !isProcessing else { return }
        guard !speechManager.isSpeaking else { return }
        guard !speechManager.hasPendingSpeech else { return }
        let hasUnspoken = !speechBuffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard !hasUnspoken else { return }
        guard case .none = pendingTransition else { return }
        DebugLog.d("[Wake] return to wake")
        startWakeListening()
    }
}
