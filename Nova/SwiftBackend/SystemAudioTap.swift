//
//  SystemAudioTap.swift
//  Nova
//
//  What the speakers are playing, handed to the backend so it can be
//  subtracted from the microphone.
//
//  This is the half Nova was missing. `echo_canceller.py` has always been able
//  to remove a sound from the mic — measured 27-36 dB — but it could only ever
//  do it for Nova's OWN voice, because Kokoro is the one signal she synthesises
//  and therefore the one she has a clean reference for. Everything else coming
//  out of the speakers (his music, a video, a call) was unknowable, so the only
//  defence was pausing the player.
//
//  ScreenCaptureKit closes that gap: it delivers the system mix itself. Give
//  the canceller that instead, and "do not hear anything from my speakers"
//  becomes one rule rather than a list of special cases — and barge-in falls
//  out of it, because Nova's own voice is in the system mix too.
//
//  WHY IT LIVES IN SWIFT. ScreenCaptureKit is a delegate-driven AVFoundation
//  API and the backend is a headless Python child with no run loop to service
//  it — the same reason NSWorkspace's app list is stale over there. The app has
//  the run loop and the Screen Recording grant, so the capture belongs here and
//  the samples travel.
//
//  WHY UDP. The reference only has value while it is current; a frame that
//  arrives late describes something the microphone already heard. UDP cannot
//  block the audio thread and cannot build a backlog — if a packet is lost the
//  canceller simply has a hole, which it handles, whereas a stalled socket
//  would wedge capture. On loopback, loss is close to theoretical anyway.
//

import AVFoundation
import Foundation
import Network
import ScreenCaptureKit

@available(macOS 13.0, *)
final class SystemAudioTap: NSObject, SCStreamOutput, SCStreamDelegate {

    /// Nova's third port. 5001 is HTTP and 8766 is the WebSocket (invariant 1);
    /// this is deliberately a separate one so a flood of audio can never
    /// interleave with the control channel.
    static let port: UInt16 = 8767

    private var stream: SCStream?
    private var converter: AVAudioConverter?
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "nova.systemaudio", qos: .userInitiated)
    private(set) var running = false

    /// 16 kHz mono Int16 — the microphone's own format, so Python can hand the
    /// samples straight to the canceller without touching them.
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                          sampleRate: 16_000,
                                          channels: 1,
                                          interleaved: true)!

    // MARK: - Lifecycle

    func start() {
        guard !running else { return }
        Task { await self.startCapture() }
    }

    func stop() {
        running = false
        stream?.stopCapture { _ in }
        stream = nil
        connection?.cancel()
        connection = nil
    }

    private func startCapture() async {
        do {
            // Asking for shareable content is also the permission check: it
            // throws when Screen Recording has not been granted, which is the
            // honest place to find out rather than silently capturing nothing.
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                NSLog("[Nova] system audio: no display to attach to")
                return
            }

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.sampleRate = 48_000
            config.channelCount = 2
            // Nova's own voice is IN the system mix, and that is wanted: it is
            // what makes barge-in work. The backend stops feeding Kokoro
            // separately while this tap is live, so nothing is counted twice.
            config.excludesCurrentProcessAudio = false
            // A video stream is mandatory, so make it the smallest and slowest
            // one the API will accept. We never look at a single frame of it.
            // Small, but not absurd: SCStreamConfiguration VALIDATES these
            // and raises an Objective-C exception on nonsense values, which
            // Swift cannot catch — it takes the whole app down. 2x2 did
            // exactly that. We still never look at a video frame.
            config.width = 128
            config.height = 128
            config.minimumFrameInterval = CMTime(value: 1, timescale: 2)
            config.queueDepth = 3

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let stream = SCStream(filter: filter, configuration: config, delegate: self)
            try stream.addStreamOutput(self, type: .audio,
                                       sampleHandlerQueue: queue)
            try await stream.startCapture()

            self.stream = stream
            openSocket()
            running = true
            NSLog("[Nova] system audio tap running -> udp 127.0.0.1:\(Self.port)")
        } catch {
            // Never fatal. Without the tap Nova behaves exactly as she did
            // before: half duplex, and the music paused while she listens.
            NSLog("[Nova] system audio tap unavailable: \(error.localizedDescription)")
        }
    }

    private func openSocket() {
        let conn = NWConnection(
            host: .ipv4(.loopback),
            port: NWEndpoint.Port(rawValue: Self.port)!,
            using: .udp)
        conn.start(queue: queue)
        connection = conn
    }

    // MARK: - Sample delivery

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, running, sampleBuffer.isValid else { return }
        guard let input = pcmBuffer(from: sampleBuffer) else { return }
        guard let out = convert(input) else { return }
        send(out)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        NSLog("[Nova] system audio tap stopped: \(error.localizedDescription)")
        running = false
    }

    /// CMSampleBuffer -> AVAudioPCMBuffer, without copying more than necessary.
    private func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        // THE CRASH LIVED HERE. The previous version built an
        // AudioStreamBasicDescription, wrapped it in an array, and took
        // `withUnsafeBufferPointer { $0.baseAddress! }` — a pointer into a
        // temporary that is already invalid by the time AVAudioFormat reads it.
        // The resulting format was garbage, AVAudioPCMBuffer's initialiser
        // raised an Objective-C exception on it, and Swift cannot catch those:
        // the app aborted instead of degrading (EXC_CRASH, _objc_terminate).
        //
        // `AVAudioFormat(cmAudioFormatDescription:)` is the supported path and
        // takes the CMFormatDescription directly, so there is no pointer to
        // outlive.
        guard let fmtDesc = sampleBuffer.formatDescription else { return nil }
        let format = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
        guard format.channelCount > 0, format.sampleRate > 0 else { return nil }

        let frames = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frames > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
        else { return nil }
        buffer.frameLength = frames

        // copyPCMData is the supported path and validates shapes itself; the
        // hand-rolled memcpy version had to assume interleaving and channel
        // count and would corrupt or crash when SCK changed either.
        do {
            try sampleBuffer.copyPCMData(fromRange: 0..<Int(frames),
                                         into: buffer.mutableAudioBufferList)
        } catch {
            return nil
        }
        return buffer
    }

    /// Down to 16 kHz mono. AVAudioConverter rather than hand-rolled decimation:
    /// a reference with aliasing in it is a reference the canceller cannot match.
    private func convert(_ input: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        if converter == nil || converter?.inputFormat != input.format {
            converter = AVAudioConverter(from: input.format, to: outFormat)
        }
        guard let converter else { return nil }

        let ratio = outFormat.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat,
                                         frameCapacity: capacity) else { return nil }

        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return input
        }
        if error != nil || out.frameLength == 0 { return nil }
        return out
    }

    private func send(_ buffer: AVAudioPCMBuffer) {
        guard let connection,
              let channel = buffer.int16ChannelData else { return }
        let byteCount = Int(buffer.frameLength) * MemoryLayout<Int16>.size
        let data = Data(bytes: channel[0], count: byteCount)
        // Unreliable on purpose — see the note at the top of the file.
        connection.send(content: data, completion: .idempotent)
    }
}
