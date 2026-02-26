//
//  SpeechSynthesizer.swift
//  Nova
//
//  Voice output using AVSpeechSynthesizer (AVFoundation).
//  Works on both iOS and macOS.
//

import Foundation
import AVFoundation

/// Speaks text using the system voice. Reusable and simple.
final class SpeechSynthesizer: NSObject {

    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Speak the given text. Does nothing if text is empty.
    func speak(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        // Use default voice for current locale.
        synthesizer.speak(utterance)
    }

    /// Stop current speech.
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    /// True when speech is in progress.
    var isSpeaking: Bool {
        synthesizer.isSpeaking
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension SpeechSynthesizer: AVSpeechSynthesizerDelegate {
    // Optional: add callbacks here later (e.g. when speech finishes) if needed.
}
