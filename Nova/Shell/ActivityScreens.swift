//
//  ActivityScreens.swift
//  Nova
//
//  Browser Activity and Automation — the two screens that watch Nova work.
//
//  Both are step lists, and both reuse `StepList` from Panel.swift rather than
//  growing their own. That component already renders the backend's `steps`
//  block, already has the done/running/pending marker language, and is already
//  under test; a second implementation would only be a second thing to drift.
//
//  What each screen adds on top is a HISTORY. A live step list answers "what is
//  she doing"; the log answers "what did she do", which is the question you
//  actually have afterwards — and in Automation's case it is arguably the whole
//  point, since work mode mostly happens with Nova parked as the puck and
//  nobody watching this screen at the time.
//

import SwiftUI

struct ActivityEvent: Identifiable, Equatable {
    let id = UUID()
    var time: String
    var text: String
    var failed: Bool = false
}

// MARK: - Browser

struct BrowserScreen: View {
    let steps: [PanelStep]
    let query: String
    let result: (title: String, source: String)?
    let history: [ActivityEvent]
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Browser Activity")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.92))
                Text("WHAT SHE READ, AND WHERE")
                    .font(NovaDesign.label(9)).tracking(1.8)
                    .foregroundStyle(tint.opacity(0.7))
            }

            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading, spacing: 16) {
                    if !query.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            CardLabel(text: "Searching for")
                            Text(query)
                                .font(.system(size: 15, weight: .medium))
                                .foregroundStyle(.white.opacity(0.9))
                        }
                    }
                    if steps.isEmpty {
                        Text("Nothing running. Ask her to search for something.")
                            .font(.system(size: 13))
                            .foregroundStyle(.white.opacity(0.4))
                    } else {
                        StepList(entries: steps, tint: tint)
                    }
                    if let result {
                        Divider().overlay(Color.white.opacity(0.06))
                        VStack(alignment: .leading, spacing: 4) {
                            CardLabel(text: "On screen now")
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(result.title)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundStyle(.white.opacity(0.9))
                                Spacer(minLength: 8)
                                Text(result.source)
                                    .font(NovaDesign.data(10))
                                    .foregroundStyle(.white.opacity(0.4))
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .glass(padding: 22)

                HistoryLog(title: "Recent", events: history)
                    .frame(width: NovaDesign.sideColumn)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
    }
}

// MARK: - Automation

struct AutomationScreen: View {
    let workMode: Bool
    let steps: [PanelStep]
    let pendingConfirmation: String?
    let history: [ActivityEvent]
    let state: NovaState
    let onToggleWorkMode: (Bool) -> Void
    let onConfirm: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            if workMode {
                live
            } else {
                off
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, NovaDesign.contentPaddingH)
        .padding(.vertical, NovaDesign.contentPaddingV)
        .animation(.easeOut(duration: 0.24), value: workMode)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Automation")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.92))
                Text("TYPING AND CLICKING, ON YOUR MAC")
                    .font(NovaDesign.label(9)).tracking(1.8)
                    .foregroundStyle(state.tint.opacity(0.7))
            }
            Spacer()
            Toggle("", isOn: Binding(get: { workMode }, set: onToggleWorkMode))
                .labelsHidden()
                .toggleStyle(.switch)
                .tint(state.tint)
        }
    }

    /// The refusal, in the backend's own words. Copying it keeps the screen and
    /// the voice saying the same thing — two phrasings of one rule is how they
    /// end up meaning two different rules.
    private var off: some View {
        Text("Nova only types and clicks while you're working together. "
             + "Say \"work with me\" to start.")
            .font(.system(size: 13))
            .foregroundStyle(.white.opacity(0.45))
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass(padding: 22)
    }

    private var live: some View {
        HStack(alignment: .top, spacing: 18) {
            VStack(alignment: .leading, spacing: 16) {
                if steps.isEmpty && pendingConfirmation == nil {
                    Text("Ready. She'll show each step here as she takes it.")
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.4))
                } else {
                    StepList(entries: steps, tint: state.tint)
                }
                if let pending = pendingConfirmation {
                    confirmation(pending)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .glass(padding: 22)

            HistoryLog(title: "History", events: history)
                .frame(width: NovaDesign.sideColumn)
        }
    }

    /// The gate. Deliberately the loudest thing on the screen, because it is
    /// the one moment where a wrong click is not recoverable.
    private func confirmation(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Divider().overlay(Color.white.opacity(0.06))
            Text(text)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.white.opacity(0.92))
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) {
                Button("Confirm", action: onConfirm)
                    .buttonStyle(PillButton(tint: NovaState.working.tint, filled: true))
                Button("Cancel", action: onCancel)
                    .buttonStyle(PillButton(tint: .white, filled: false))
            }
        }
    }
}

// MARK: - Shared

struct HistoryLog: View {
    let title: String
    let events: [ActivityEvent]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            CardLabel(text: title)
            if events.isEmpty {
                Text("Nothing yet.")
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.3))
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(events) { e in
                            HStack(alignment: .firstTextBaseline, spacing: 10) {
                                Text(e.time)
                                    .font(NovaDesign.data(10))
                                    .foregroundStyle(.white.opacity(0.3))
                                Text(e.text)
                                    .font(.system(size: 12))
                                    .foregroundStyle(e.failed ? NovaDesign.negative.opacity(0.75)
                                                     : .white.opacity(0.6))
                                Spacer(minLength: 0)
                            }
                            .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .glass()
    }
}

struct PillButton: ButtonStyle {
    let tint: Color
    let filled: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(filled ? Color.black.opacity(0.85) : .white.opacity(0.7))
            .padding(.horizontal, 16)
            .padding(.vertical, 7)
            .background(
                Capsule().fill(filled ? tint.opacity(configuration.isPressed ? 0.75 : 1)
                               : Color.white.opacity(configuration.isPressed ? 0.10 : 0.05))
            )
            .overlay(
                Capsule().stroke(filled ? .clear : Color.white.opacity(0.12), lineWidth: 1)
            )
    }
}
