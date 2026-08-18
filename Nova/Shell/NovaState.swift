//
//  NovaState.swift
//  Nova
//
//  The seven things Nova can be doing, and how each one looks.
//
//  Nova's UI has no transcript. The orb is the entire interface, so these
//  states are not decoration — they are the only channel telling Nicholas
//  whether he was heard, whether she is thinking, and whether she is awake.
//  `unsure` exists because removing the transcript removed the one place a
//  mis-hear used to be visible.
//

import SwiftUI

enum NovaState: String, CaseIterable {
    case idle
    case listening
    case thinking
    case speaking
    case working
    case sleeping
    case unsure

    /// Parsed from the backend's `{"type":"state"}` message.
    ///
    /// The backend's vocabulary is not identical to this enum, and the gap was
    /// invisible: the main path broadcasts "processing", which had no case and
    /// silently fell back to .idle — so the orb sat at IDLE for the whole time
    /// Nova was thinking, and the thinking state effectively never appeared.
    /// Aliases are mapped explicitly; anything genuinely unknown still falls
    /// back to idle rather than freezing on a stale state.
    init(wire: String) {
        let raw = wire.lowercased()
        switch raw {
        case "processing", "generating":
            self = .thinking
        case "busy":
            self = .working
        case "asleep":
            self = .sleeping
        default:
            self = NovaState(rawValue: raw) ?? .idle
        }
    }

    /// The word under the orb. Tiny and dim by design — it is the only thing
    /// that distinguishes idle from sleeping at a glance.
    var readout: String {
        switch self {
        case .idle:      return "Idle"
        case .listening: return "Listening"
        case .thinking:  return "Thinking"
        case .speaking:  return "Speaking"
        case .working:   return "Working"
        case .sleeping:  return "Sleeping"
        case .unsure:    return "Didn't catch that"
        }
    }

    /// One hue family, walked by temperature rather than jumped between.
    ///
    /// He said the colours did not match: the old set mixed a cyan, a
    /// blue-violet and a near-white that shared no common ground, so every
    /// state change looked like a different palette. These stay on a
    /// teal-to-blue axis and differ by DEPTH — idle is deep and desaturated,
    /// speaking is the same hue brought forward. Only `working` and `unsure`
    /// leave the family, because those two must be unmistakable at a glance.
    var tint: Color {
        switch self {
        case .idle:      return Color(red: 0.157, green: 0.404, blue: 0.478)
        case .listening: return Color(red: 0.208, green: 0.706, blue: 0.831)
        case .thinking:  return Color(red: 0.286, green: 0.518, blue: 0.808)
        case .speaking:  return Color(red: 0.322, green: 0.808, blue: 0.898)
        case .working:   return Color(red: 0.902, green: 0.643, blue: 0.310)
        case .sleeping:  return Color(red: 0.086, green: 0.176, blue: 0.227)
        case .unsure:    return Color(red: 0.878, green: 0.494, blue: 0.337)
        }
    }

    /// Everything dims together when asleep, rather than each element
    /// deciding for itself.
    var dim: Double { self == .sleeping ? 0.45 : 1.0 }

    /// How fast the arc segments turn.
    var spin: Double {
        switch self {
        case .thinking: return 1.9
        case .working:  return 1.1
        case .sleeping: return 0.05
        default:        return 0.30
        }
    }

    /// Arcs stutter instead of turning smoothly when Nova isn't sure she
    /// understood. Deliberately visible without being alarming.
    var jitters: Bool { self == .unsure }

    /// Only `working` draws the determinate sweep, so "busy" never looks
    /// like "stuck".
    var showsProgress: Bool { self == .working }

    /// A synthetic voice envelope, used until the backend streams real audio
    /// levels. Speaking gets a syllabic rhythm; listening gets a quieter, less
    /// regular signal standing in for a human voice.
    func envelope(_ t: Double) -> Double {
        switch self {
        case .speaking:
            let syl = abs(sin(t * 5.1)) * abs(sin(t * 1.7 + 0.6))
            return 0.30 + syl * 0.70
        case .listening:
            return 0.18 + (abs(sin(t * 3.3 + 1.2)) * 0.6 + abs(sin(t * 7.9)) * 0.25) * 0.5
        case .thinking:  return 0.36 + sin(t * 2.4) * 0.12
        case .working:   return 0.30 + sin(t * 1.8) * 0.09
        case .unsure:    return 0.24 + abs(sin(t * 7.2)) * 0.32
        case .sleeping:  return 0.06 + sin(t * 0.42) * 0.03
        case .idle:      return 0.15 + sin(t * 0.75) * 0.06
        }
    }
}
