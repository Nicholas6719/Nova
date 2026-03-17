//
//  AudioSessionQueue.swift
//  Nova
//
//  Dedicated queue for AVAudioSession operations to avoid QoS priority inversion.
//  All setCategory, setMode, setActive calls are routed through this queue.
//

import Foundation
import AVFoundation

enum AudioSessionQueue {

    private static let queue = DispatchQueue(label: "nova.audio.session", qos: .userInteractive)

    /// Run AVAudioSession configuration on the dedicated queue. Non-blocking; use async version for callers.
    static func async(work: @escaping () -> Void) {
        queue.async(execute: work)
    }

    #if os(iOS)
    /// Configure session for playback (TTS). Category .playAndRecord, mode .voiceChat, active=true.
    static func configureForPlayback() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                do {
                    let session = AVAudioSession.sharedInstance()
                    try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
                    try session.setActive(true)
                } catch { }
                cont.resume()
            }
        }
    }

    /// Configure session for recording. Category .playAndRecord, mode .voiceChat, active=true.
    /// Throws if configuration fails.
    static func configureForRecording() async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    let session = AVAudioSession.sharedInstance()
                    try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
                    try session.setActive(true, options: .notifyOthersOnDeactivation)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Deactivate session (notify others). Fire-and-forget.
    static func deactivate() {
        queue.async {
            do {
                try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            } catch { }
        }
    }

    /// Deactivate session and run continuation when done.
    static func deactivateAsync() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                do {
                    try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
                } catch { }
                cont.resume()
            }
        }
    }

    /// Prepare for barge-in: category + deactivate. Fire-and-forget (caller does not wait).
    static func prepareForBargeIn() {
        queue.async {
            do {
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
                try session.setActive(false, options: .notifyOthersOnDeactivation)
            } catch { }
        }
    }
    #endif
}
