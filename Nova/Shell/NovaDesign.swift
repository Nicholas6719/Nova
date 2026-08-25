//
//  NovaDesign.swift
//  Nova
//
//  The visual language, in one place.
//
//  Eight screens share a base colour, a vignette, a grid, and a glass card. If
//  each of them owns its own copy of those, they drift — one screen ends up a
//  shade darker than the others and nobody can say when it happened. So the
//  tokens live here and the screens compose them.
//
//  Values come from the design handoff and are deliberately literal: the
//  background is #050810, a near-black navy rather than the pure black the
//  shipped shell uses, and the difference is visible the moment they sit side
//  by side. The state tints are NOT redefined here — they already exist in
//  NovaState.swift and are unchanged by the redesign, so this reads them
//  rather than restating them, and there is no second copy to fall out of date.
//

import SwiftUI

enum NovaDesign {

    // MARK: - Base

    /// Near-black navy. Not #000: the glass cards need something to sit ON,
    /// and pure black gives their translucency nothing to pick up.
    static let background = Color(red: 0x05 / 255, green: 0x08 / 255, blue: 0x10 / 255)

    /// New data accents. Same lightness family as the state tints, hue-shifted,
    /// so a rising market and a listening orb look like they belong to one app.
    static let positive = Color(red: 0x4A / 255, green: 0xDE / 255, blue: 0x80 / 255)
    static let negative = Color(red: 0xF8 / 255, green: 0x50 / 255, blue: 0x50 / 255)
    /// Screen awareness and anything Nova is SUGGESTING rather than reporting.
    /// Deliberately outside the cyan family: a suggestion is a different kind
    /// of thing from a reading, and should not be mistaken for one.
    static let suggestion = Color(red: 180 / 255, green: 140 / 255, blue: 255 / 255)

    // MARK: - Metrics

    static let railWidth: CGFloat = 72
    static let stripHeight: CGFloat = 44
    static let contentPaddingH: CGFloat = 32
    static let contentPaddingV: CGFloat = 28
    static let sideColumn: CGFloat = 270
    static let financeColumn: CGFloat = 320

    static let cardRadius: CGFloat = 18
    static let heroRadius: CGFloat = 20
    static let railRadius: CGFloat = 12

    // MARK: - Type

    /// Mono, uppercase, wide tracking — every label, every unit, every piece of
    /// metadata. The rule that keeps data legible next to prose.
    static func label(_ size: CGFloat = 9, tracking: CGFloat = 1.6) -> Font {
        .system(size: size, weight: .medium, design: .monospaced)
    }

    static func data(_ size: CGFloat = 12) -> Font {
        .system(size: size, design: .monospaced)
    }
}

// MARK: - Glass

/// The card everything sits in.
///
/// `.ultraThinMaterial` rather than a flat fill: the handoff asks for
/// `backdrop-filter: blur(20px) saturate(140%)`, and a material is how AppKit
/// does that natively — it picks up the vignette behind it, which a solid
/// colour cannot. The inset highlight along the top edge is what reads as
/// "glass" rather than "grey box"; without it these look like flat panels.
struct GlassCard: ViewModifier {
    var radius: CGFloat = NovaDesign.cardRadius
    var padding: CGFloat = 18

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                ZStack {
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(.ultraThinMaterial)
                        .opacity(0.45)
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(Color.white.opacity(0.035))
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(Color(red: 120 / 255, green: 200 / 255, blue: 255 / 255)
                        .opacity(0.14), lineWidth: 1)
            )
            .overlay(alignment: .top) {
                // The inset top highlight. One hairline, and it is the whole
                // difference between glass and a grey rectangle.
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .fill(LinearGradient(
                        colors: [Color.white.opacity(0.06), .clear],
                        startPoint: .top, endPoint: .bottom))
                    .frame(height: radius)
                    .allowsHitTesting(false)
            }
    }
}

extension View {
    func glass(radius: CGFloat = NovaDesign.cardRadius,
               padding: CGFloat = 18) -> some View {
        modifier(GlassCard(radius: radius, padding: padding))
    }
}

// MARK: - Ambience

/// The tinted vignette and the grid, behind everything.
///
/// The vignette is driven by the CURRENT STATE tint, so the whole room changes
/// colour with Nova rather than only the orb. It is the reason the redesign
/// reads as one surface instead of cards floating on black.
struct NovaBackdrop: View {
    let tint: Color

    var body: some View {
        ZStack {
            NovaDesign.background
            RadialGradient(
                colors: [tint.opacity(0.09), .clear],
                center: .top, startRadius: 0, endRadius: 720)
            GridTexture()
                .stroke(Color.white.opacity(0.025), lineWidth: 1)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

/// 34px repeating lines on both axes, at 2.5%.
///
/// Drawn as a Path rather than tiled with an image: there are no bitmap assets
/// in this app and there is no reason for this to be the first.
struct GridTexture: Shape {
    var spacing: CGFloat = 34

    func path(in rect: CGRect) -> Path {
        var p = Path()
        var x = rect.minX
        while x <= rect.maxX {
            p.move(to: CGPoint(x: x, y: rect.minY))
            p.addLine(to: CGPoint(x: x, y: rect.maxY))
            x += spacing
        }
        var y = rect.minY
        while y <= rect.maxY {
            p.move(to: CGPoint(x: rect.minX, y: y))
            p.addLine(to: CGPoint(x: rect.maxX, y: y))
            y += spacing
        }
        return p
    }
}

// MARK: - Rail

/// The eight destinations, in rail order.
///
/// `built` is honest rather than cosmetic: a destination whose screen does not
/// exist is dimmed and says so when you reach it, the same rule the menu has
/// always followed. Health ships unbuilt on purpose.
struct NovaDestination: Identifiable, Hashable {
    let id: String
    let symbol: String
    let title: String
    var built: Bool = true

    static let rail: [NovaDestination] = [
        .init(id: "home",       symbol: "house",                     title: "Home"),
        .init(id: "calendar",   symbol: "calendar",                  title: "Calendar"),
        .init(id: "finance",    symbol: "chart.line.uptrend.xyaxis", title: "Finance"),
        .init(id: "automation", symbol: "bolt",                      title: "Automation"),
        .init(id: "browser",    symbol: "globe",                     title: "Browser"),
        .init(id: "health",     symbol: "heart",                     title: "Health",
              built: false),
        .init(id: "system",     symbol: "cpu",                       title: "System"),
    ]
}
