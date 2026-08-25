//
//  FinanceScreen.swift
//  Nova
//
//  The deep dive: indices, a hero chart, a watchlist, and the news behind it.
//
//  Every number on this screen came from market_engine and was templated in
//  Python. That is the same rule the spoken answers follow and it matters more
//  here than anywhere else in the app — a wrong price looks exactly like a
//  right one, and this screen shows dozens at a glance. Nothing here is
//  phrased, inferred, or interpolated by the UI; the charts draw the points
//  they were given and nothing between them.
//
//  Nova is a RESEARCHER, not an advisor. There is deliberately no
//  buy/sell/hold verdict anywhere on this screen beyond reporting what analysts
//  themselves say, attributed as such.
//

import SwiftUI

struct Quote: Identifiable, Equatable {
    var id: String { symbol }
    var symbol: String
    var name: String
    var price: Double
    var changePct: Double
    var series: [Double] = []

    var up: Bool { changePct >= 0 }
    var tint: Color { up ? NovaDesign.positive : NovaDesign.negative }
    var changeText: String {
        String(format: "%@%.2f%%", up ? "+" : "", changePct)
    }
    var priceText: String {
        price.formatted(.number.precision(.fractionLength(2)))
    }
}

struct NewsItem: Identifiable, Equatable {
    let id = UUID()
    var headline: String
    var source: String
}

struct Analysts: Equatable {
    var buy: Int = 0
    var hold: Int = 0
    var sell: Int = 0
    var total: Int { max(1, buy + hold + sell) }
}

struct FinanceScreen: View {
    let indices: [Quote]
    let watchlist: [Quote]
    let selected: Quote?
    let news: [NewsItem]
    let analysts: Analysts?
    let fundamentals: [(String, String)]
    let range: String
    let ranges: [String]
    let tint: Color
    let onSelect: (Quote) -> Void
    let onRange: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            HStack(spacing: 14) {
                ForEach(indices) { indexCard($0) }
            }
            HStack(alignment: .top, spacing: 18) {
                hero
                VStack(spacing: 18) {
                    watchlistCard
                    newsCard
                }
                .frame(width: NovaDesign.financeColumn)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Markets")
                .font(.system(size: 21, weight: .semibold))
                .foregroundStyle(.white.opacity(0.92))
            LivePill(tint: tint)
            Spacer()
            // Attribution, always. The data is not Nova's and the screen should
            // never imply otherwise.
            Text("YAHOO FINANCE · FINNHUB")
                .font(NovaDesign.label(9)).tracking(1.6)
                .foregroundStyle(.white.opacity(0.28))
        }
    }

    private func indexCard(_ q: Quote) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            CardLabel(text: q.name)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(q.priceText)
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(.white.opacity(0.95))
                Text(q.changeText)
                    .font(NovaDesign.data(11))
                    .foregroundStyle(q.tint.opacity(0.9))
            }
            Sparkline(points: q.series)
                .stroke(q.tint.opacity(0.8), style: .init(lineWidth: 1.4, lineJoin: .round))
                .frame(height: 22)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass(padding: 14)
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 18) {
            if let q = selected {
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(q.symbol)
                            .font(.system(size: 21, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.92))
                        Text(q.name)
                            .font(.system(size: 12))
                            .foregroundStyle(.white.opacity(0.45))
                    }
                    Spacer()
                    HStack(spacing: 4) {
                        ForEach(ranges, id: \.self) { r in
                            Button { onRange(r) } label: {
                                Text(r)
                                    .font(NovaDesign.label(9)).tracking(1.2)
                                    .foregroundStyle(r == range ? tint : .white.opacity(0.35))
                                    .padding(.horizontal, 9).padding(.vertical, 5)
                                    .background(
                                        Capsule().fill(r == range ? tint.opacity(0.12) : .clear))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(q.priceText)
                        .font(.system(size: 44, weight: .light))
                        .foregroundStyle(.white.opacity(0.95))
                    Text(q.changeText)
                        .font(NovaDesign.data(12))
                        .foregroundStyle(q.tint)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Capsule().fill(q.tint.opacity(0.14)))
                }

                PriceChart(points: q.series, tint: q.tint)
                    .frame(height: 190)

                if !fundamentals.isEmpty {
                    HStack(spacing: 26) {
                        ForEach(fundamentals.indices, id: \.self) { i in
                            VStack(alignment: .leading, spacing: 3) {
                                CardLabel(text: fundamentals[i].0)
                                Text(fundamentals[i].1)
                                    .font(NovaDesign.data(13))
                                    .foregroundStyle(.white.opacity(0.85))
                            }
                        }
                        Spacer(minLength: 0)
                    }
                }

                if let a = analysts { analystBar(a) }
            } else {
                Text("No market data just now.")
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.4))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass(radius: NovaDesign.heroRadius, padding: 22)
    }

    /// What analysts say, reported and attributed — never Nova's own view.
    private func analystBar(_ a: Analysts) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            CardLabel(text: "Analyst coverage")
            GeometryReader { geo in
                HStack(spacing: 2) {
                    segment(a.buy, a.total, NovaDesign.positive, geo.size.width)
                    segment(a.hold, a.total, Color.white.opacity(0.35), geo.size.width)
                    segment(a.sell, a.total, NovaDesign.negative, geo.size.width)
                }
            }
            .frame(height: 6)
            HStack(spacing: 16) {
                legend("Buy", a.buy, NovaDesign.positive)
                legend("Hold", a.hold, .white.opacity(0.45))
                legend("Sell", a.sell, NovaDesign.negative)
                Spacer(minLength: 0)
            }
        }
    }

    private func segment(_ n: Int, _ total: Int, _ color: Color, _ width: CGFloat) -> some View {
        Capsule().fill(color.opacity(0.8))
            .frame(width: max(0, width * CGFloat(n) / CGFloat(total)))
    }

    private func legend(_ label: String, _ n: Int, _ color: Color) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text("\(label) \(n)")
                .font(NovaDesign.label(9)).tracking(1.0)
                .foregroundStyle(.white.opacity(0.45))
        }
    }

    private var watchlistCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            CardLabel(text: "Watchlist")
            ForEach(watchlist) { q in
                Button { onSelect(q) } label: {
                    HStack(spacing: 10) {
                        Text(q.symbol)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(.white.opacity(0.9))
                            .frame(width: 46, alignment: .leading)
                        Sparkline(points: q.series)
                            .stroke(q.tint.opacity(0.7), lineWidth: 1.2)
                            .frame(height: 16)
                        Text(q.priceText)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(.white.opacity(0.7))
                        Text(q.changeText)
                            .font(NovaDesign.data(11))
                            .foregroundStyle(q.tint.opacity(0.9))
                            .frame(width: 58, alignment: .trailing)
                    }
                    .padding(.vertical, 5).padding(.horizontal, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(q.symbol == selected?.symbol ? tint.opacity(0.10) : .clear))
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private var newsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            CardLabel(text: "News")
            if news.isEmpty {
                Text("No coverage available.")
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.3))
            } else {
                ForEach(news) { n in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(n.headline)
                            .font(.system(size: 13))
                            .foregroundStyle(.white.opacity(0.82))
                            .fixedSize(horizontal: false, vertical: true)
                        Text(n.source)
                            .font(NovaDesign.data(10))
                            .foregroundStyle(.white.opacity(0.35))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }
}

// MARK: - Charts

/// A bare line. No axes, no labels — it is a shape, not a chart, and its only
/// job is the direction of travel.
struct Sparkline: Shape {
    let points: [Double]

    func path(in rect: CGRect) -> Path {
        var p = Path()
        guard points.count > 1 else { return p }
        let lo = points.min() ?? 0, hi = points.max() ?? 1
        let span = max(1e-9, hi - lo)
        for (i, v) in points.enumerated() {
            let x = rect.minX + rect.width * CGFloat(i) / CGFloat(points.count - 1)
            let y = rect.maxY - rect.height * CGFloat((v - lo) / span)
            i == 0 ? p.move(to: .init(x: x, y: y)) : p.addLine(to: .init(x: x, y: y))
        }
        return p
    }
}

/// The hero chart: three grid lines, a gradient fill, and the line itself.
struct PriceChart: View {
    let points: [Double]
    let tint: Color

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                ForEach(0..<3, id: \.self) { _ in
                    Rectangle().fill(Color.white.opacity(0.05)).frame(height: 1)
                    Spacer(minLength: 0)
                }
            }
            Sparkline(points: points)
                .fill(LinearGradient(colors: [tint.opacity(0.22), .clear],
                                     startPoint: .top, endPoint: .bottom))
            Sparkline(points: points)
                .stroke(tint.opacity(0.9), style: .init(lineWidth: 1.8, lineJoin: .round))
        }
    }
}

struct LivePill: View {
    let tint: Color
    @State private var on = false

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(tint).frame(width: 5, height: 5).opacity(on ? 0.25 : 1)
            Text("LIVE").font(NovaDesign.label(8)).tracking(1.4)
                .foregroundStyle(tint.opacity(0.8))
        }
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(Capsule().fill(tint.opacity(0.10)))
        .onAppear {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                on = true
            }
        }
    }
}
