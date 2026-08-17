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

    /// Parsed from the backend's `{"type":"state"}` message. Anything
    /// unrecognised falls back to idle rather than freezing on a stale state.
    init(wire: String) {
        self = NovaState(rawValue: wire.lowercased()) ?? .idle
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

    /// Cyan throughout, as in the mockup. Only `working` and `unsure` break the
    /// palette, because those two have to be unmistakable from across a room.
    var tint: Color {
        switch self {
        case .idle:      return Color(red: 0.243, green: 0.561, blue: 0.659)
        case .listening: return Color(red: 0.251, green: 0.839, blue: 1.000)
        case .thinking:  return Color(red: 0.494, green: 0.612, blue: 1.000)
        case .speaking:  return Color(red: 0.659, green: 0.925, blue: 1.000)
        case .working:   return Color(red: 1.000, green: 0.745, blue: 0.353)
        case .sleeping:  return Color(red: 0.106, green: 0.212, blue: 0.282)
        case .unsure:    return Color(red: 1.000, green: 0.588, blue: 0.376)
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
