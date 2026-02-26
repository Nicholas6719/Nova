//
//  ContentView.swift
//  Nova
//
//  Chat UI: header, status, message bubbles, microphone button.
//  Displays Message (user/assistant) from conversation history.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    private static let liveTranscriptID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    var body: some View {
        VStack(spacing: 0) {
            headerView
            statusView

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        if !viewModel.liveTranscript.isEmpty {
                            MessageBubble(message: Message(id: Self.liveTranscriptID, role: .user, content: viewModel.liveTranscript))
                                .opacity(0.85)
                                .id(Self.liveTranscriptID)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let last = viewModel.messages.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    } else if !viewModel.liveTranscript.isEmpty {
                        proxy.scrollTo(Self.liveTranscriptID, anchor: .bottom)
                    }
                }
            }

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .padding(.vertical, 6)
            }

            micButton
                .padding(.vertical, 20)
        }
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onAppear {
            viewModel.requestPermissions()
        }
    }

    private var headerView: some View {
        VStack(spacing: 4) {
            Text("Nova")
                .font(.title.weight(.bold))
            Text("Neural Omniscient Voice Assistant")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
    }

    private var statusView: some View {
        Text(viewModel.status.rawValue)
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .padding(.vertical, 6)
    }

    private var micButton: some View {
        Button {
            viewModel.toggleRecording()
        } label: {
            ZStack {
                Circle()
                    .fill(viewModel.isRecording ? Color.red.opacity(0.2) : Color.gray.opacity(0.15))
                    .frame(width: 88, height: 88)
                    .scaleEffect(viewModel.isRecording ? 1.08 : 1.0)
                    .animation(.easeInOut(duration: 0.25), value: viewModel.isRecording)

                Circle()
                    .stroke(viewModel.isRecording ? Color.red : Color.gray.opacity(0.4), lineWidth: 3)
                    .frame(width: 88, height: 88)
                    .scaleEffect(viewModel.isRecording ? 1.05 : 1.0)
                    .animation(.easeInOut(duration: 0.25), value: viewModel.isRecording)

                Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(viewModel.isRecording ? .red : .gray)
            }
            .frame(width: 100, height: 100)
            .contentShape(Circle())
            .opacity(viewModel.isMicEnabled ? 1.0 : 0.5)
        }
        .buttonStyle(MicButtonStyle())
        .disabled(!viewModel.isMicEnabled)
    }
}

// MARK: - Message bubble (user right, assistant left)

private struct MessageBubble: View {
    let message: Message

    private var isFromUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if isFromUser { Spacer(minLength: 40) }
            Text(message.content)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(isFromUser ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.2))
                )
                .foregroundStyle(isFromUser ? Color.accentColor : .primary)
            if !isFromUser { Spacer(minLength: 40) }
        }
        .frame(maxWidth: .infinity, alignment: isFromUser ? .trailing : .leading)
    }
}

// MARK: - Mic button style

private struct MicButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .contentShape(Circle())
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

#Preview {
    NavigationStack {
        ContentView()
    }
}
