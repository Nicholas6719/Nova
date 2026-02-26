//
//  SpeechRecognizer.swift
//  Nova
//
//  Minimal, stable voice input using SFSpeechRecognizer + AVAudioEngine.
//  All engine work runs off the main thread in SpeechRecognitionEngine actor
//  to avoid sync dispatch deadlocks when stopping/restarting.
//

import Foundation
import Speech
import AVFoundation
import Combine

// MARK: - Engine actor (off-main audio and recognition work)

/// Runs all AVAudioEngine and SFSpeechRecognitionTask work off the main thread
/// so that engine start/stop and recognition callbacks never block or sync-dispatch to main.
private actor SpeechRecognitionEngine {

    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    /// Start listening. Callbacks invoked via Task { @MainActor in } (no MainActor.run / unsafeForcedSync).
    func start(
        recognizer: SFSpeechRecognizer,
        onTranscript: @Sendable @escaping (String) -> Void,
        onError: @Sendable @escaping (String?) -> Void,
        onStarted: @Sendable @escaping () -> Void,
        onStartFailed: @Sendable @escaping (String) -> Void
    ) async throws {
        // Clean previous session (all on actor executor, not main)
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        recognitionRequest = request

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [request] buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            audioEngine.inputNode.removeTap(onBus: 0)
            recognitionRequest = nil
            onStartFailed(error.localizedDescription)
            throw error
        }

        onStarted()

        recognitionTask = recognizer.recognitionTask(with: request) { result, error in
            if let error = error {
                let ns = error as NSError
                let cancelled = ns.code == 216
                if !cancelled {
                    onError(error.localizedDescription)
                }
            }
            if let result = result {
                onTranscript(result.bestTranscription.formattedString)
            }
        }
    }

    /// Stop engine and recognition. Safe to call from any context; does not touch main.
    func stop() async {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
    }
}

// MARK: - State guard actor (replaces NSLock for Swift 6)

private actor SpeechStateGuard {
    var isStopping = false
    var isStarting = false

    func tryStart(isRecording: Bool) -> Bool {
        if isStarting || isRecording { return false }
        isStarting = true
        return true
    }

    func didFinishStart() {
        isStarting = false
    }

    func tryStop() -> Bool {
        if isStopping { return false }
        isStopping = true
        return true
    }

    func didFinishStop() {
        isStopping = false
    }
}

// MARK: - SpeechRecognizer (engine work off main; UI updates on MainActor only)

final class SpeechRecognizer: ObservableObject {

    // MARK: - Published (updated on MainActor only)

    @Published private(set) var transcript: String = ""
    @Published private(set) var isRecording: Bool = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var authorizationStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined

    var onRecordingDidStop: ((String) -> Void)?

    // MARK: - Private

    private let engine = SpeechRecognitionEngine()
    private var speechRecognizer: SFSpeechRecognizer?
    private let stateGuard = SpeechStateGuard()

    init() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale.current)
        if speechRecognizer == nil {
            speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        }
    }

    // MARK: - Authorization

    func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            Task { @MainActor in
                self?.authorizationStatus = status
                switch status {
                case .authorized:
                    self?.errorMessage = nil
                case .denied:
                    self?.errorMessage = "Speech recognition was denied."
                case .restricted:
                    self?.errorMessage = "Speech recognition is restricted."
                case .notDetermined:
                    self?.errorMessage = "Speech recognition not determined."
                @unknown default:
                    self?.errorMessage = "Unknown authorization status."
                }
            }
        }
    }

    // MARK: - Listen

    func startListening() {
        let recording = isRecording
        let authStatus = authorizationStatus
        let recognizer = speechRecognizer

        Task { @MainActor [weak self] in
            guard let self else { return }
            let canStart = await self.stateGuard.tryStart(isRecording: recording)
            guard canStart else { return }
            defer { Task { await self.stateGuard.didFinishStart() } }

            guard authStatus == .authorized else {
                self.errorMessage = "Please allow speech recognition in Settings."
                self.requestPermissions()
                return
            }

            guard let recognizer, recognizer.isAvailable else {
                self.errorMessage = "Speech recognition is not available."
                return
            }
            #if os(iOS)
            let granted = await requestMicrophonePermission()
            guard granted else {
                self.errorMessage = "Microphone access was denied."
                return
            }
            do {
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.record, mode: .measurement, options: .duckOthers)
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            } catch {
                self.errorMessage = "Could not configure audio: \(error.localizedDescription)"
                return
            }
            #endif

            self.transcript = ""
            self.errorMessage = nil

            do {
                try await engine.start(
                    recognizer: recognizer,
                    onTranscript: { [weak self] text in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.transcript = text
                        }
                    },
                    onError: { [weak self] message in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.errorMessage = message
                        }
                    },
                    onStarted: { [weak self] in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.isRecording = true
                            print("[SpeechRecognizer] speech start")
                        }
                    },
                    onStartFailed: { [weak self] message in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.errorMessage = message
                        }
                    }
                )
            } catch {
                // Already reported via onStartFailed
            }
        }
    }

    /// Non-blocking: returns immediately; teardown runs off main, then callback on MainActor.
    /// Engine stop must NOT run on MainActor (Speech framework can reenter during cancel).
    func stopListening() {
        let finalText = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let engineRef = engine
        let stateGuardRef = stateGuard

        Task.detached { [weak self] in
            guard let self else { return }
            let canStop = await stateGuardRef.tryStop()
            guard canStop else { return }
            await engineRef.stop()
            #if os(iOS)
            do {
                try AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
                print("[AudioSession] deactivated after stopListening")
            } catch {
                print("[AudioSession] deactivate failed: \(error.localizedDescription)")
            }
            #endif
            await stateGuardRef.didFinishStop()
            await MainActor.run {
                self.isRecording = false
                print("[SpeechRecognizer] speech stop")
                self.onRecordingDidStop?(finalText)
            }
        }
    }

    func clearTranscript() {
        Task { @MainActor [weak self] in
            self?.transcript = ""
            self?.errorMessage = nil
        }
    }

    // MARK: - Helpers

    #if os(iOS)
    private func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { cont in
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { granted in
                    cont.resume(returning: granted)
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    cont.resume(returning: granted)
                }
            }
        }
    }
    #endif
}
