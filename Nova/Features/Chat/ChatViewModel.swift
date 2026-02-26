//
//  ChatViewModel.swift
//  Nova
//
//  Manages conversation history (Message[]) and voice I/O.
//  Passes full message history to NovaEngine for context-aware responses.
//

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

    /// Throttled streaming: buffer accumulates deltas; UI updates at most every ~60ms.
    private var streamingBuffer = ""
    private var streamingLastFlushTime: CFAbsoluteTime = 0
    private var streamingFlushTask: Task<Void, Never>?
    private var streamingPlaceholderId: UUID?

    init() {
        // SpeechRecognizer updates @Published on MainActor; no receive(on:) needed.
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
        if !isMicEnabled {
            errorMessage = "One sec — I'm finishing my response."
            return
        }
        if isRecording {
            speechRecognizer.stopListening()
        } else {
            errorMessage = nil
            speechRecognizer.clearTranscript()
            speechRecognizer.startListening()
        }
    }

    /// Called when speech stops. Capture transcript, clear immediately, then process in one place.
    private func handleRecordingStopped(with text: String) {
        let finalText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        speechRecognizer.clearTranscript()

        guard !finalText.isEmpty else {
            errorMessage = "No speech recognized. Try again."
            return
        }

        processUserInput(text: finalText)
    }

    /// EMERGENCY: Bypass engine — call LLMClient directly. No streaming, no cache, no throttle.
    private func processUserInput(text: String) {
        guard !text.isEmpty else { return }
        guard !isProcessing else { return }
        isProcessing = true
        errorMessage = nil

        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)
        #if DEBUG
        DebugLog.d("[Chat] append user: \(text.prefix(60))\(text.count > 60 ? "…" : "")")
        #endif

        let messageSnapshot = messages
        let systemPrompt = NovaPersonality.systemPrompt()

        // Build cfg on MainActor (ProcessInfo blocks from background).
        let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] ?? ""
        let cfg = LLMConfig(
            apiKey: apiKey,
            endpoint: URL(string: "https://api.openai.com/v1/chat/completions")!,
            model: "gpt-4o-mini",
            temperature: 0.7
        )

        // Run OpenAI call OFF main actor — no engine, no streaming, no TaskGroup.
        Task.detached(priority: .userInitiated) { [cfg, messageSnapshot, systemPrompt] in
            do {
                print("[OPENAI] O1 starting detached request")
                let resp = try await LLMClient.generateResponse(
                    config: cfg,
                    messages: messageSnapshot,
                    systemPrompt: systemPrompt
                )
                print("[OPENAI] O2 got response len=\(resp.count)")

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.appendAssistant(resp)
                    self.speechManager.speak(resp)
                    self.isProcessing = false
                }
            } catch {
                print("[OPENAI] OX error: \(error)")
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.appendAssistant("Sorry — I couldn't reach my online brain. Please try again.")
                    self.isProcessing = false
                }
            }
        }
    }

    @MainActor
    private func appendAssistant(_ text: String) {
        messages.append(Message(role: .assistant, content: text))
        print("[Chat] append assistant: \(text.prefix(60))\(text.count > 60 ? "…" : "")")
    }

    func clearError() {
        errorMessage = nil
    }

    private func flushStreamingToUI(placeholderId: UUID) {
        guard let idx = messages.lastIndex(where: { $0.id == placeholderId }),
              messages[idx].role == .assistant else { return }
        messages[idx] = Message(id: placeholderId, role: .assistant, content: streamingBuffer)
    }

    /// Called from Sendable streaming callbacks via static ref; no self capture in concurrent closures.
    private static weak var activeStreamingHandler: ChatViewModel?

    @MainActor private func prepareForStreaming(placeholderId: UUID) {
        streamingBuffer = ""
        streamingLastFlushTime = 0
        streamingFlushTask?.cancel()
        streamingFlushTask = nil
        messages.append(Message(id: placeholderId, role: .assistant, content: ""))
    }

    @MainActor private func handleStreamDelta(_ delta: String) {
        guard let placeholderId = streamingPlaceholderId,
              messages.contains(where: { $0.id == placeholderId }) else { return }
        streamingBuffer += delta
        let now = CFAbsoluteTimeGetCurrent()
        let interval: CFAbsoluteTime = 0.06
        let elapsed = streamingLastFlushTime == 0 ? interval + 1 : now - streamingLastFlushTime
        if elapsed >= interval {
            flushStreamingToUI(placeholderId: placeholderId)
            streamingLastFlushTime = now
        } else if streamingFlushTask == nil {
            let delay = interval - elapsed
            let pid = placeholderId
            streamingFlushTask = Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                guard !Task.isCancelled else { return }
                guard let self else { return }
                self.flushStreamingToUI(placeholderId: pid)
                self.streamingLastFlushTime = CFAbsoluteTimeGetCurrent()
                self.streamingFlushTask = nil
            }
        }
    }
}
