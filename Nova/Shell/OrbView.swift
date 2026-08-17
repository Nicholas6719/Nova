//
//  OrbView.swift
//  Nova
//
//  The orb. Reactor at full size, Array in the puck — the same design language
//  at two densities, so Nova sheds detail as she shrinks rather than becoming a
//  different shape.
//
//  The signature element is the core TORUS: not a filled ball but a ring of
//  light that goes white-hot at its crest, drawn as a radial gradient whose
//  stops peak at the ring radius. Everything else — the concentric rings, the
//  arc segments, the dashed tracks, the drifting particles — is arranged
//  around it.
//
//  Drawn with Canvas + TimelineView so it is one retained-mode redraw per
//  frame rather than a pile of animated SwiftUI shapes.
//

import SwiftUI

// MARK: - Density

/// One rotating arc segment. A named type rather than a 5-tuple: the literal
/// arrays below are long enough that inference on tuples makes SourceKit give
/// up on the whole expression.
struct ArcSpec {
    let radius: Double      // × R
    let phase: Double       // radians at t = 0
    let length: Double      // radians
    let direction: Double   // +1 / -1
    let weight: Double      // stroke width
}

enum OrbDensity {
    /// Full window. Five rings, eight arcs, two dashed tracks, particles.
    case reactor
    /// 190pt puck. One torus, two rings, four long arcs, nothing else — the
    /// version that has to stay legible at the edge of his vision all day.
    case array
}

// MARK: - Orb

struct OrbView: View {
    let state: NovaState
    var density: OrbDensity = .reactor
    /// 0…1 voice level. Nil uses the state's synthetic envelope.
    var level: Double? = nil

    var body: some View {
        TimelineView(.animation) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            Canvas { context, size in
                draw(context: &context, size: size, t: t)
            }
            .drawingGroup()
        }
        .accessibilityLabel("Nova is \(state.readout.lowercased())")
    }

    private func draw(context: inout GraphicsContext, size: CGSize, t: Double) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        // Array fills more of its frame: the puck is small, and dead margin
        // inside it is wasted pixels he has to squint past.
        let R = min(size.width, size.height) * (density == .reactor ? 0.44 : 0.47)
        let lvl = level ?? state.envelope(t)
        let tint = state.tint
        let dim = state.dim
        let open = 1 + lvl * (density == .reactor ? 0.10 : 0.08)

        context.addFilter(.blur(radius: R * 0.012))

        ambientField(&context, center, R, tint, 0.13 * dim)

        switch density {
        case .reactor: drawReactor(&context, center, R, t, lvl, open, tint, dim)
        case .array:   drawArray(&context, center, R, t, lvl, open, tint, dim)
        }

        if state.showsProgress {
            progressSweep(&context, center, R * (density == .reactor ? 1.14 : 1.14) * open,
                          t, tint)
        }

        // The hero, drawn last so it sits on top of everything.
        let coreR = R * (density == .reactor ? 0.30 : 0.32) * open
        torus(&context, center, coreR, R * (0.045 + lvl * 0.05), tint,
              (0.55 + lvl * 0.45) * dim)
    }

    // MARK: Reactor

    private func drawReactor(_ c: inout GraphicsContext, _ o: CGPoint, _ R: Double,
                             _ t: Double, _ lvl: Double, _ open: Double,
                             _ tint: Color, _ dim: Double) {
        for (i, f) in [0.46, 0.60, 0.78, 0.90, 1.02].enumerated() {
            ring(&c, o, R * f * open, tint, (0.20 - Double(i) * 0.026) * dim, 1)
        }
        dashedRing(&c, o, R * 0.68 * open, tint, 0.30 * dim, [2, 6], t * state.spin * 0.5)
        dashedRing(&c, o, R * 1.10 * open, tint, 0.18 * dim, [1, 9], -t * state.spin * 0.3)

        for (i, a) in Self.reactorArcs.enumerated() {
            var phase = t * state.spin * a.direction + a.phase
            if state.jitters { phase += sin(t * 24 + Double(i)) * 0.14 }
            arc(&c, o, R * a.radius * open, phase, a.length, tint,
                0.72 * dim, a.weight)
        }

        particles(&c, o, R * 1.25 * open, t, tint, 0.5 * dim, count: 46)
    }

    // MARK: Array

    private func drawArray(_ c: inout GraphicsContext, _ o: CGPoint, _ R: Double,
                           _ t: Double, _ lvl: Double, _ open: Double,
                           _ tint: Color, _ dim: Double) {
        ring(&c, o, R * 0.64 * open, tint, 0.16 * dim, 1)
        ring(&c, o, R * 1.02 * open, tint, 0.11 * dim, 1)

        for (i, a) in Self.arrayArcs.enumerated() {
            var phase = t * state.spin * a.direction + a.phase
            if state.jitters { phase += sin(t * 20 + Double(i)) * 0.18 }
            arc(&c, o, R * a.radius * open, phase, a.length, tint,
                0.60 * dim, a.weight)
        }
    }

    // MARK: Arc layouts

    private static let reactorArcs: [ArcSpec] = [
        ArcSpec(radius: 0.53, phase: 0.10, length: 1.60, direction:  1, weight: 2.6),
        ArcSpec(radius: 0.53, phase: 3.40, length: 0.70, direction:  1, weight: 2.6),
        ArcSpec(radius: 0.70, phase: 2.20, length: 1.15, direction: -1, weight: 1.8),
        ArcSpec(radius: 0.70, phase: 5.00, length: 0.55, direction: -1, weight: 1.8),
        ArcSpec(radius: 0.86, phase: 1.10, length: 2.10, direction:  1, weight: 1.4),
        ArcSpec(radius: 0.86, phase: 4.20, length: 0.80, direction:  1, weight: 1.4),
        ArcSpec(radius: 1.02, phase: 0.60, length: 1.30, direction: -1, weight: 1.1),
        ArcSpec(radius: 1.02, phase: 3.90, length: 1.90, direction: -1, weight: 1.1),
    ]

    private static let arrayArcs: [ArcSpec] = [
        ArcSpec(radius: 0.82, phase: 0.00, length: 2.30, direction:  1, weight: 1.5),
        ArcSpec(radius: 0.82, phase: 3.40, length: 1.60, direction:  1, weight: 1.5),
        ArcSpec(radius: 1.02, phase: 1.60, length: 2.60, direction: -1, weight: 1.0),
        ArcSpec(radius: 1.02, phase: 4.90, length: 1.10, direction: -1, weight: 1.0),
    ]

    // MARK: Primitives

    /// A ring of light, white-hot at its crest. Built from gradient stops that
    /// peak at `r` — a stroked circle cannot bloom like this.
    private func torus(_ c: inout GraphicsContext, _ o: CGPoint, _ r: Double,
                       _ band: Double, _ tint: Color, _ alpha: Double) {
        let outer = r + band * 3.2
        guard outer > 0 else { return }
        let p = r / outer, b = band / outer
        let stops: [Gradient.Stop] = [
            .init(color: tint.opacity(alpha * 0.12), location: 0),
            .init(color: tint.opacity(alpha * 0.10), location: max(0.001, p - b * 2.4)),
            .init(color: tint.opacity(alpha * 0.62), location: max(0.002, p - b * 0.9)),
            .init(color: Color.white.opacity(alpha),  location: p),
            .init(color: tint.opacity(alpha * 0.55), location: min(0.998, p + b * 0.9)),
            .init(color: tint.opacity(alpha * 0.08), location: min(0.999, p + b * 2.4)),
            .init(color: tint.opacity(0),             location: 1),
        ]
        c.fill(Path(ellipseIn: rect(o, outer)),
               with: .radialGradient(Gradient(stops: stops), center: o,
                                     startRadius: 0, endRadius: outer))
    }

    private func ambientField(_ c: inout GraphicsContext, _ o: CGPoint, _ R: Double,
                              _ tint: Color, _ alpha: Double) {
        let outer = R * 1.5
        c.fill(Path(ellipseIn: rect(o, outer)),
               with: .radialGradient(
                Gradient(stops: [
                    .init(color: tint.opacity(alpha), location: 0),
                    .init(color: tint.opacity(0), location: 1),
                ]), center: o, startRadius: 0, endRadius: outer))
    }

    private func ring(_ c: inout GraphicsContext, _ o: CGPoint, _ r: Double,
                      _ tint: Color, _ alpha: Double, _ weight: Double) {
        c.stroke(Path(ellipseIn: rect(o, r)), with: .color(tint.opacity(alpha)),
                 lineWidth: weight)
    }

    private func dashedRing(_ c: inout GraphicsContext, _ o: CGPoint, _ r: Double,
                            _ tint: Color, _ alpha: Double, _ dash: [CGFloat],
                            _ rotation: Double) {
        var style = StrokeStyle(lineWidth: 1, dash: dash)
        style.dashPhase = CGFloat(-rotation * r)
        c.stroke(Path(ellipseIn: rect(o, r)), with: .color(tint.opacity(alpha)),
                 style: style)
    }

    private func arc(_ c: inout GraphicsContext, _ o: CGPoint, _ r: Double,
                     _ from: Double, _ length: Double, _ tint: Color,
                     _ alpha: Double, _ weight: Double) {
        var path = Path()
        path.addArc(center: o, radius: r,
                    startAngle: .radians(from), endAngle: .radians(from + length),
                    clockwise: false)
        c.stroke(path, with: .color(tint.opacity(alpha)),
                 style: StrokeStyle(lineWidth: weight, lineCap: .round))
    }

    private func progressSweep(_ c: inout GraphicsContext, _ o: CGPoint, _ r: Double,
                               _ t: Double, _ tint: Color) {
        let p = t.truncatingRemainder(dividingBy: 2.5) / 2.5
        var path = Path()
        path.addArc(center: o, radius: r,
                    startAngle: .radians(-.pi / 2),
                    endAngle: .radians(-.pi / 2 + p * 2 * .pi), clockwise: false)
        c.stroke(path, with: .color(tint.opacity(0.9)),
                 style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
    }

    /// Deterministic dust — seeded from the index so the field is stable
    /// between frames instead of boiling.
    private func particles(_ c: inout GraphicsContext, _ o: CGPoint, _ R: Double,
                           _ t: Double, _ tint: Color, _ alpha: Double, count: Int) {
        for i in 0..<count {
            let seed = Double(i) * 0.6180339887
            let a0 = seed.truncatingRemainder(dividingBy: 1) * 2 * .pi
            let v = 0.3 + (seed * 7).truncatingRemainder(dividingBy: 1) * 0.9
            let radial = 0.35 + (seed * 13).truncatingRemainder(dividingBy: 1) * 0.85
            let angle = a0 + t * 0.05 * v
            let p = CGPoint(x: o.x + cos(angle) * R * radial,
                            y: o.y + sin(angle) * R * radial)
            let twinkle = 0.45 + abs(sin(t * 1.6 * v + a0)) * 0.55
            let s = 1.0 + (seed * 3).truncatingRemainder(dividingBy: 1) * 1.2
            c.fill(Path(ellipseIn: CGRect(x: p.x - s / 2, y: p.y - s / 2,
                                          width: s, height: s)),
                   with: .color(tint.opacity(alpha * twinkle)))
        }
    }

    private func rect(_ o: CGPoint, _ r: Double) -> CGRect {
        CGRect(x: o.x - r, y: o.y - r, width: r * 2, height: r * 2)
    }
}

#Preview("Reactor") {
    VStack(spacing: 24) {
        ForEach(NovaState.allCases, id: \.self) { s in
            HStack(spacing: 20) {
                OrbView(state: s, density: .reactor).frame(width: 130, height: 130)
                OrbView(state: s, density: .array).frame(width: 90, height: 90)
                Text(s.readout).font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
    }
    .padding(40)
    .background(Color.black)
}
