//
//  CalendarScreen.swift
//  Nova
//
//  Today down the left, the month down the right.
//
//  Events and reminders live in ONE list rather than two, because that is how
//  he thinks about a day — the question is "what is coming", not "what kind of
//  record is it". They stay tellable apart by the accent bar alone: state tint
//  for an event, green for a reminder. One colour instead of a second column,
//  which is the same trade the home screen's Upcoming card already makes.
//

import SwiftUI

struct CalendarEntry: Identifiable, Equatable {
    let id = UUID()
    var time: String
    var title: String
    var detail: String = ""
    var isReminder: Bool = false
}

struct CalendarScreen: View {
    let entries: [CalendarEntry]
    let monthDays: [Int]
    let today: Int
    let subtitle: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Calendar")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.92))
                Text(subtitle.uppercased())
                    .font(NovaDesign.label(9)).tracking(1.8)
                    .foregroundStyle(tint.opacity(0.7))
            }

            HStack(alignment: .top, spacing: 18) {
                todayList
                VStack(spacing: 18) {
                    monthGrid
                    summary
                }
                .frame(width: NovaDesign.sideColumn)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
    }

    private var todayList: some View {
        VStack(alignment: .leading, spacing: 14) {
            CardLabel(text: "Today")
            if entries.isEmpty {
                Text("Nothing on your calendar today.")
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.4))
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(entries) { row(for: $0) }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private func row(for e: CalendarEntry) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(e.time)
                .font(NovaDesign.label(10)).tracking(0.6)
                .foregroundStyle(.white.opacity(0.45))
                .frame(width: 58, alignment: .leading)
            Rectangle()
                .fill(e.isReminder ? NovaDesign.positive.opacity(0.65) : tint.opacity(0.55))
                .frame(width: 2)
                .frame(maxHeight: .infinity)
            VStack(alignment: .leading, spacing: 2) {
                Text(e.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                if !e.detail.isEmpty {
                    Text(e.detail)
                        .font(.system(size: 12))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            Spacer(minLength: 0)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var monthGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            CardLabel(text: "This month")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4),
                                     count: 7), spacing: 6) {
                ForEach(["S", "M", "T", "W", "T", "F", "S"].indices, id: \.self) { i in
                    Text(["S", "M", "T", "W", "T", "F", "S"][i])
                        .font(NovaDesign.label(8)).tracking(0.8)
                        .foregroundStyle(.white.opacity(0.25))
                }
                ForEach(monthDays.indices, id: \.self) { i in
                    let day = monthDays[i]
                    Text(day == 0 ? "" : "\(day)")
                        .font(NovaDesign.data(11))
                        .foregroundStyle(day == today ? Color.black.opacity(0.85)
                                         : .white.opacity(0.55))
                        .frame(width: 24, height: 24)
                        .background(
                            Circle().fill(day == today ? tint : .clear)
                        )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    private var summary: some View {
        let events = entries.filter { !$0.isReminder }.count
        let reminders = entries.count - events
        return VStack(alignment: .leading, spacing: 8) {
            CardLabel(text: "Summary")
            Text(line(events: events, reminders: reminders))
                .font(.system(size: 13))
                .foregroundStyle(.white.opacity(0.6))
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }

    /// Phrased in Python everywhere else; phrased here because it is a count of
    /// what is already on screen, not a fact fetched from anywhere.
    private func line(events: Int, reminders: Int) -> String {
        if events == 0 && reminders == 0 { return "Nothing scheduled." }
        var parts: [String] = []
        if events > 0 { parts.append("\(events) event\(events == 1 ? "" : "s")") }
        if reminders > 0 { parts.append("\(reminders) reminder\(reminders == 1 ? "" : "s")") }
        return parts.joined(separator: " and ") + " today."
    }
}
