import Foundation
import Combine

/// Talks to the Python backend over its two channels:
/// - HTTP REST on port 5001 (`/api/status`, `/api/messages`, `/api/message`, `/api/mute`).
/// - WebSocket on port 8766 for live `state`, `message`, and `token` events.
///
/// Observable state (`messages`, `currentState`, `isConnected`) is published so
/// `ChatViewModel` can drive the UI directly.
///
/// The project uses `SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor`; this type is
/// MainActor-isolated. The WebSocket receive loop runs off the main actor and
/// hops back only to mutate published state.
@MainActor
final class NovaAPIClient: ObservableObject {

    // MARK: - Observable state

    @Published private(set) var messages: [Message] = []
    @Published private(set) var currentState: String = "idle"
    @Published private(set) var isConnected: Bool = false
    /// Which screen the UI should be showing, and its panel data. The backend
    /// sends this on connect as well as on every navigation, so a relaunched
    /// app comes up on the right screen instead of blank.
    @Published private(set) var currentView: ViewPayload?

    struct ViewPayload {
        let name: String
        let data: [String: Any]
    }

    // MARK: - Configuration

    /// Ports are fixed by Invariant 1 — do not change.
    private let httpPort: Int = 5001
    private let wsPort: Int = 8766

    private var httpBase: URL { URL(string: "http://localhost:\(httpPort)")! }
    private var wsURL: URL { URL(string: "ws://localhost:\(wsPort)")! }

    // MARK: - Private state

    private let session = URLSession(configuration: .default)
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?

    /// True between `connect()` and `disconnect()`. Drives the reconnect loop so
    /// a WebSocket failure retries instead of giving up — the backend often is
    /// not listening yet when the app first launches (it loads models for a few
    /// seconds), and it may also be restarted by BackendManager.
    private var shouldStayConnected = false

    /// ID of the assistant message currently being assembled from `token` events,
    /// if any.
    private var streamingMessageID: UUID?

    // MARK: - Connection lifecycle

    /// Begins maintaining a WebSocket connection to the backend, retrying until
    /// it is reachable and reconnecting if it later drops. Idempotent.
    func connect() {
        guard receiveTask == nil else { return }
        shouldStayConnected = true
        receiveTask = Task { [weak self] in
            await self?.maintainConnection()
        }
    }

    /// Stops connecting and closes any open WebSocket.
    func disconnect() {
        shouldStayConnected = false
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
    }

    /// Reconnect supervisor: (re)opens the socket and pumps messages until it
    /// fails, then waits briefly and tries again for as long as we should stay
    /// connected. On each successful open it reloads history so the UI catches
    /// up on anything sent while it was down.
    private func maintainConnection() async {
        while shouldStayConnected && !Task.isCancelled {
            let task = session.webSocketTask(with: wsURL)
            webSocketTask = task
            task.resume()

            // Confirm the backend is actually reachable before declaring
            // connected — webSocketTask.resume() succeeds optimistically even
            // when nothing is listening.
            await loadHistory()

            await receiveLoop()   // returns when the connection drops

            isConnected = false
            webSocketTask?.cancel(with: .goingAway, reason: nil)
            webSocketTask = nil

            if !shouldStayConnected || Task.isCancelled { break }
            try? await Task.sleep(nanoseconds: 1_000_000_000)  // 1s backoff
        }
    }

    // MARK: - REST: outbound

    /// Sends user text to the pipeline via `POST /api/message`. The backend
    /// echoes the user's turn back over the WebSocket as a `message` frame, so we
    /// do NOT append it locally here — doing both showed every message twice.
    /// The backend echo is the single source of truth for the message list.
    /// `silent` asks the backend to answer in text only, without speaking. When
    /// Nicholas types instead of talking he is usually somewhere he cannot talk,
    /// so a spoken reply is the last thing he wants.
    func sendMessage(_ text: String, silent: Bool = false) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let url = httpBase.appendingPathComponent("api/message")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: ["content": trimmed, "silent": silent])
        _ = try? await session.data(for: request)
    }

    /// Toggles mute via `POST /api/mute`.
    func toggleMute() async {
        let url = httpBase.appendingPathComponent("api/mute")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        _ = try? await session.data(for: request)
    }

    // MARK: - REST: inbound

    /// Loads the recent message history via `GET /api/messages` and replaces the
    /// local list. Useful on launch to restore the last 50 turns.
    func loadHistory() async {
        let url = httpBase.appendingPathComponent("api/messages")
        guard let (data, _) = try? await session.data(from: url) else { return }
        guard let raw = try? JSONSerialization.jsonObject(with: data) else { return }

        // Accept either a bare array or `{ "messages": [...] }`.
        let array: [[String: Any]]
        if let dict = raw as? [String: Any], let list = dict["messages"] as? [[String: Any]] {
            array = list
        } else if let list = raw as? [[String: Any]] {
            array = list
        } else {
            return
        }

        let parsed: [Message] = array.compactMap { entry in
            guard
                let roleString = entry["role"] as? String,
                let role = Self.messageRole(from: roleString),
                let content = entry["content"] as? String
            else { return nil }
            return Message(role: role, content: content)
        }
        messages = parsed
        streamingMessageID = nil
    }

    /// Maps a backend role string ("user" | "assistant") to `MessageRole`.
    private static func messageRole(from raw: String) -> MessageRole? {
        switch raw {
        case "user": return .user
        case "assistant": return .assistant
        default: return nil
        }
    }

    // MARK: - WebSocket receive loop

    private func receiveLoop() async {
        while !Task.isCancelled, let task = webSocketTask {
            do {
                let message = try await task.receive()
                // A successful receive proves the backend is up and listening.
                isConnected = true
                switch message {
                case .string(let text):
                    handle(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) { handle(text) }
                @unknown default:
                    break
                }
            } catch {
                // Connection failed or dropped (often: backend not up yet at
                // launch). Return so the supervisor waits and reconnects.
                return
            }
        }
    }

    /// Parses one inbound JSON frame and applies it to published state.
    private func handle(_ jsonText: String) {
        guard
            let data = jsonText.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = object["type"] as? String
        else { return }

        switch type {
        case "state":
            if let state = object["state"] as? String {
                currentState = state
                // Leaving an active state finalizes any in-flight stream.
                if state == "idle" || state == "listening" {
                    streamingMessageID = nil
                }
            }

        case "message":
            guard
                let roleString = object["role"] as? String,
                let role = Self.messageRole(from: roleString),
                let content = object["content"] as? String
            else { return }
            apply(role: role, content: content)

        case "token":
            if let token = object["token"] as? String {
                appendToken(token)
            }

        case "view":
            if let name = object["view"] as? String {
                currentView = ViewPayload(name: name,
                                          data: object["data"] as? [String: Any] ?? [:])
            }

        case "need_location":
            // The backend answers a location question and has no usable fix.
            // Only the app bundle can obtain one, so it has to originate here.
            // Broadcast rather than call LocationProvider directly: this client
            // is owned by ChatViewModel, and reaching across for a reference
            // would mean threading a dependency through a 994-line view model
            // that is already scheduled for its own refactor.
            NotificationCenter.default.post(name: .novaNeedsLocation, object: nil)

        default:
            break
        }
    }

    /// Applies a complete message frame. If an assistant message was being
    /// streamed, the final frame replaces its accumulated content.
    ///
    /// `Message.content` is immutable, so an in-place update rebuilds the struct
    /// preserving its `id` (and therefore its SwiftUI identity).
    private func apply(role: MessageRole, content: String) {
        if role == .assistant, let id = streamingMessageID,
           let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index] = Message(id: id, role: .assistant, content: content)
            streamingMessageID = nil
            return
        }
        messages.append(Message(role: role, content: content))
    }

    /// Appends a streaming token to the in-flight assistant message, creating one
    /// if this is the first token of a response.
    private func appendToken(_ token: String) {
        if let id = streamingMessageID,
           let index = messages.firstIndex(where: { $0.id == id }) {
            let updated = messages[index].content + token
            messages[index] = Message(id: id, role: .assistant, content: updated)
        } else {
            let message = Message(role: .assistant, content: token)
            streamingMessageID = message.id
            messages.append(message)
        }
    }
}
