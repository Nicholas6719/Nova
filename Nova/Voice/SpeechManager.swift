//
//  SpeechManager.swift
//  Nova
//
//  Pluggable TTS: SpeechManager delegates to a TTSEngine (default AVSpeechSynthesizer).
//  Future: swap in a neural TTS engine (on-device or cloud) by providing a different
//  TTSEngine implementation without changing the rest of the app.
//

import Foundation
import AVFoundation
import Combine

// MARK: - TTSEngine Protocol (pluggable TTS)

/// Protocol for any TTS backend. Implement this to swap in a different engine
/// (e.g. on-device neural TTS or cloud TTS) without changing SpeechManager or callers.
///
/// Future neural TTS integration: add a type like `NeuralTTSEngine: TTSEngine` that
/// wraps your neural model (on-device or streaming from cloud), then use
/// `SpeechManager(engine: NeuralTTSEngine())` so the rest of the app stays unchanged.
protocol TTSEngine: AnyObject {
    func speak(_ text: String)
    func stop()
    var isSpeaking: Bool { get }
    /// True if synthesizer is speaking or queue has pending utterances.
    var hasPendingSpeech: Bool { get }
}

/// Optional: engines can expose a display name for debug logging.
extension TTSEngine {
    var engineTypeName: String { "TTS" }
    var hasPendingSpeech: Bool { isSpeaking }
}

// MARK: - SpeechManager (orchestrator, engine-agnostic)

/// Manages text-to-speech via a pluggable TTSEngine. Default is AVSpeechSynthesizer;
/// replace `engine` with a neural TTS implementation when ready.
@MainActor
final class SpeechManager: ObservableObject {

    // MARK: - Published

    @Published private(set) var isSpeaking: Bool = false

    /// True if TTS is speaking or has utterances queued. Use to avoid starting mic during TTS.
    var hasPendingSpeech: Bool { engine.hasPendingSpeech }

    // MARK: - Engine (swap here for future neural TTS)

    /// Active TTS backend. Default: AVSpeech. Future: inject a neural TTS engine, e.g.:
    /// `SpeechManager(engine: OnDeviceNeuralTTSEngine())` or `SpeechManager(engine: CloudTTSEngine())`.
    private let engine: TTSEngine

    /// Called when TTS actually starts (didStart utterance). Used for fallback auto-listen.
    var onSpeechStarted: (() -> Void)?
    /// Called when TTS fully completes (all utterances done). Used for hands-free auto-listen.
    var onSpeechFinished: (() -> Void)?

    // MARK: - Init

    init(engine providedEngine: TTSEngine? = nil) {
        let engine: TTSEngine = providedEngine ?? AVSpeechTTSEngine()
        self.engine = engine
        if let avEngine = engine as? AVSpeechTTSEngine {
            avEngine.onSpeakingStateChanged = { [weak self] speaking in
                Task { @MainActor in
                    self?.isSpeaking = speaking
                }
            }
            avEngine.onSpeechStarted = { [weak self] in
                self?.onSpeechStarted?()
            }
            avEngine.onSpeechFinished = { [weak self] in
                self?.onSpeechFinished?()
            }
        }
    }

    // MARK: - Public API (unchanged for app code)

    /// Speak the given text. Does nothing if text is empty.
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        Task { @MainActor [weak self] in
            guard let self else { return }
            #if os(iOS)
            await AudioSessionQueue.configureForPlayback()
            #endif
            (self.engine as? AVSpeechTTSEngine)?.ensureReadyForPlayback()
            self.isSpeaking = true
            self.engine.speak(trimmed)
        }
    }

    /// Stop current speech.
    func stop() {
        engine.stop()
    }

    /// Stop current speech immediately (barge-in). Idempotent.
    func stopSpeaking() {
        guard isSpeaking else { return }
        #if DEBUG
        DebugLog.d("[TTS] stopSpeaking (barge-in)")
        #endif
        isSpeaking = false
        engine.stop()
    }

    /// Call before mic starts (barge-in). Stops TTS and configures session for recording.
    /// Session config runs on audioSessionQueue; subsequent startListening will serialize after this.
    func prepareForBargeIn() {
        stopSpeaking()
        engine.stop()
        #if os(iOS)
        AudioSessionQueue.prepareForBargeIn()
        #endif
    }
}

// MARK: - AVSpeechTTSEngine (default implementation of TTSEngine)

/// Default TTS using AVSpeechSynthesizer. Fine-tuned for natural pacing and warmth.
///
/// Future neural TTS integration point: add another class conforming to TTSEngine
/// (e.g. `OnDeviceNeuralTTSEngine` or `CloudTTSEngine`) with the same speak/stop/isSpeaking
/// API, then construct SpeechManager with that engine instead of AVSpeechTTSEngine.
@MainActor
final class AVSpeechTTSEngine: NSObject, TTSEngine {

    var engineTypeName: String { "AVSpeechSynthesizer" }

    /// Called when speech starts or stops. Invoked from delegate callbacks.
    var onSpeakingStateChanged: ((Bool) -> Void)?
    /// Called when first utterance starts (didStart). Used for fallback auto-listen.
    var onSpeechStarted: (() -> Void)?
    /// Called when all speech is done (didFinish with empty queue). Used for hands-free auto-listen.
    var onSpeechFinished: (() -> Void)?

    // MARK: - Configuration (Ava Premium, natural pacing and warmth)

    /// Ava Premium: light, natural, friendly, conversational (iOS and macOS).
    private static let avaPremiumIdentifier = "com.apple.voice.premium.ava"

    /// Rate for natural pacing (slightly faster, lighter cadence).
    private static var rate: Float { AVSpeechUtteranceDefaultSpeechRate * 1.05 }
    /// Slight pitch increase for warmth.
    private static let pitchMultiplier: Float = 1.05

    // MARK: - State

    private var synthesizer: AVSpeechSynthesizer
    private let selectedVoice: AVSpeechSynthesisVoice?
    /// Pending sentences to speak sequentially after the current utterance finishes.
    private var sentenceQueue: [String] = []
    /// Recent stop timestamps for barge-in recovery (rebuild after 3 stops in 5s).
    private var stopTimestamps: [CFAbsoluteTime] = []

    // MARK: - Init & preload

    override init() {
        self.selectedVoice = AVSpeechSynthesisVoice(identifier: Self.avaPremiumIdentifier)
        self.synthesizer = AVSpeechSynthesizer()
        super.init()
        synthesizer.delegate = self
        preloadVoice()
    }

    /// Rebuild synthesizer after repeated barge-ins. Called before playback.
    func ensureReadyForPlayback() {
        let now = CFAbsoluteTimeGetCurrent()
        let cutoff = now - 5.0
        stopTimestamps.removeAll { $0 < cutoff }
        if stopTimestamps.count >= 3 {
            #if DEBUG
            DebugLog.d("[TTS] rebuild synthesizer (3+ stops in 5s)")
            #endif
            synthesizer = AVSpeechSynthesizer()
            synthesizer.delegate = self
            stopTimestamps.removeAll()
        }
    }

    /// Warm the synthesizer to reduce first-speech delay (caching).
    private func preloadVoice() {
        let utterance = AVSpeechUtterance(string: " ")
        configure(utterance)
        utterance.volume = 0.01
        synthesizer.speak(utterance)
        synthesizer.stopSpeaking(at: .immediate)
    }

    // MARK: - Utterance configuration

    private func configure(_ utterance: AVSpeechUtterance) {
        utterance.rate = Self.rate
        utterance.pitchMultiplier = Self.pitchMultiplier
        utterance.volume = 1.0
        utterance.postUtteranceDelay = 0.08
        if let voice = selectedVoice {
            utterance.voice = voice
        }
    }

    // MARK: - Sentence splitting (smoother speech for long responses)

    /// Split text into sentence-sized chunks by ". ", "? ", "! " and newlines.
    private static func sentences(from text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let lines = trimmed.components(separatedBy: .newlines)
        var result: [String] = []
        for line in lines {
            let t = line.trimmingCharacters(in: .whitespaces)
            guard !t.isEmpty else { continue }
            var parts = t.components(separatedBy: ". ")
            parts = parts.flatMap { $0.components(separatedBy: "? ") }
            parts = parts.flatMap { $0.components(separatedBy: "! ") }
            result.append(contentsOf: parts
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty })
        }
        return result
    }

    /// Speak one sentence (same config as full utterance).
    private func speakOne(_ sentence: String) {
        let utterance = AVSpeechUtterance(string: sentence)
        configure(utterance)
        synthesizer.speak(utterance)
    }

    // MARK: - TTSEngine

    func speak(_ text: String) {
        let list = Self.sentences(from: text)
        guard !list.isEmpty else { return }

        sentenceQueue = []
        onSpeakingStateChanged?(true)
        if list.count == 1 {
            speakOne(list[0])
            return
        }

        sentenceQueue = Array(list.dropFirst())
        speakOne(list[0])
    }

    func stop() {
        sentenceQueue = []
        synthesizer.stopSpeaking(at: .immediate)
        stopTimestamps.append(CFAbsoluteTimeGetCurrent())
        #if os(iOS)
        AudioSessionQueue.deactivate()
        #endif
        onSpeakingStateChanged?(false)
    }

    var isSpeaking: Bool {
        synthesizer.isSpeaking
    }

    var hasPendingSpeech: Bool {
        synthesizer.isSpeaking || !sentenceQueue.isEmpty
    }
}

// MARK: - AVSpeechSynthesizerDelegate (sequential sentence queue)
// Delegate runs on synthesizer thread; bounce to MainActor to avoid data races.

extension AVSpeechTTSEngine: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.onSpeakingStateChanged?(true)
            self?.onSpeechStarted?()
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            let next = self.sentenceQueue.isEmpty ? nil : self.sentenceQueue.removeFirst()

            if let sentence = next {
                self.speakOne(sentence)
            } else {
                #if DEBUG
                DebugLog.d("[TTS] finished queueEmpty=true sentenceQueueRemaining=0")
                #endif
                #if os(iOS)
                AudioSessionQueue.deactivate()
                #endif
                self.onSpeakingStateChanged?(false)
                self.onSpeechFinished?()
            }
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        #if os(iOS)
        AudioSessionQueue.deactivate()
        #endif
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.onSpeakingStateChanged?(false)
        }
    }
}
