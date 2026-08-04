import Foundation
import Combine

/// Launches and supervises the Python backend (`nova_backend/nova.py`).
///
/// Responsibilities:
/// - Locate a Python interpreter and `nova_backend/nova.py`.
/// - Launch `python nova.py` as a child process with `NOVA_DATA_DIR` set so
///   data survives app updates (Invariant 8).
/// - Poll `GET /api/status` on port 5001 until the backend reports ready.
/// - Restart the process if it exits unexpectedly.
///
/// The Xcode project uses `SWIFT_DEFAULT_ACTOR_ISOLATION=MainActor`, so this
/// type is MainActor-isolated. All blocking subprocess and polling work is
/// pushed off the main actor with `Task.detached`; observable state is mutated
/// back on the main actor.
@MainActor
final class BackendManager: ObservableObject {

    // MARK: - Observable state

    @Published private(set) var isRunning: Bool = false
    @Published private(set) var isReady: Bool = false

    // MARK: - Configuration

    /// HTTP port the backend serves on (Invariant 1 — do not change).
    private let httpPort: Int = 5001

    /// Where Nova's persistent data lives. Set into the child environment as
    /// `NOVA_DATA_DIR` (Invariant 8).
    private let dataDirectory: URL = {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
        return base.appendingPathComponent("Nova", isDirectory: true)
    }()

    // MARK: - Private state

    private var process: Process?
    private var supervisor: Task<Void, Never>?
    private var stoppedIntentionally = false

    // MARK: - Lifecycle

    /// Start the backend and begin supervising it. Idempotent.
    func start() {
        guard supervisor == nil else { return }
        stoppedIntentionally = false
        supervisor = Task { [weak self] in
            await self?.superviseLoop()
        }
    }

    /// Stop the backend and cancel supervision. Safe to call when not running.
    func stop() {
        stoppedIntentionally = true
        supervisor?.cancel()
        supervisor = nil
        terminateProcess()
        isRunning = false
        isReady = false
    }

    // MARK: - Supervision

    /// Launches the process, waits for readiness, then waits for exit. Restarts
    /// on unexpected exit with a short backoff until cancelled or stopped.
    private func superviseLoop() async {
        while !Task.isCancelled && !stoppedIntentionally {
            do {
                let proc = try makeProcess()
                self.process = proc

                try proc.run()
                isRunning = true
                isReady = false

                // Wait for the HTTP server to come up.
                let ready = await waitUntilReady()
                isReady = ready

                // Block (off main actor) until the process exits.
                await waitForExit(proc)

                isRunning = false
                isReady = false
                self.process = nil

                if stoppedIntentionally || Task.isCancelled { break }
            } catch {
                isRunning = false
                isReady = false
                self.process = nil
                if stoppedIntentionally || Task.isCancelled { break }
            }

            // Backoff before restarting after a crash.
            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
    }

    /// Builds the child `Process` configured to run `python nova.py`.
    private func makeProcess() throws -> Process {
        let pythonPath = try locatePython()
        let scriptURL = try locateNovaScript()

        try FileManager.default.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true
        )

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = [scriptURL.path]
        proc.currentDirectoryURL = scriptURL.deletingLastPathComponent()

        var env = ProcessInfo.processInfo.environment
        env["NOVA_DATA_DIR"] = dataDirectory.path
        // Ensure stdout/stderr from Python is unbuffered for live logs.
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        return proc
    }

    /// Waits for the child process to terminate without blocking the main actor.
    private nonisolated func waitForExit(_ proc: Process) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            proc.terminationHandler = { _ in
                continuation.resume()
            }
            // Guard against a race where the process already exited before the
            // handler was installed.
            if !proc.isRunning {
                proc.terminationHandler = nil
                continuation.resume()
            }
        }
    }

    private func terminateProcess() {
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()
        process = nil
    }

    // MARK: - Readiness polling

    /// Polls `GET /api/status` until it returns a 200, the task is cancelled,
    /// or the process dies. Returns whether the backend became ready.
    private func waitUntilReady() async -> Bool {
        guard let url = URL(string: "http://localhost:\(httpPort)/api/status") else {
            return false
        }

        // Up to ~60s (120 attempts × 500ms) — first run downloads the MLX model.
        for _ in 0..<120 {
            if Task.isCancelled || stoppedIntentionally { return false }
            if let proc = process, !proc.isRunning { return false }

            if await statusOK(url) { return true }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false
    }

    private nonisolated func statusOK(_ url: URL) async -> Bool {
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Locating Python and the script

    /// Finds `nova_backend/nova.py`. Prefers a path relative to the app bundle's
    /// resources, then falls back to the known development location.
    private func locateNovaScript() throws -> URL {
        let fm = FileManager.default

        if let resourceURL = Bundle.main.resourceURL {
            let bundled = resourceURL
                .appendingPathComponent("nova_backend", isDirectory: true)
                .appendingPathComponent("nova.py")
            if fm.fileExists(atPath: bundled.path) { return bundled }
        }

        let devPath = "/Users/nicholascoppola/Documents/Coding_Projects/Nova/nova_backend/nova.py"
        if fm.fileExists(atPath: devPath) {
            return URL(fileURLWithPath: devPath)
        }

        throw BackendError.scriptNotFound
    }

    /// Finds a Python 3 interpreter. Checks common install locations and the
    /// `PATH` via `/usr/bin/env`.
    private func locatePython() throws -> String {
        let fm = FileManager.default
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]
        for path in candidates where fm.isExecutableFile(atPath: path) {
            return path
        }

        // Fall back to resolving `python3` on PATH.
        if let resolved = resolveOnPath("python3") {
            return resolved
        }

        throw BackendError.pythonNotFound
    }

    private func resolveOnPath(_ name: String) -> String? {
        let which = Process()
        which.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        which.arguments = ["which", name]
        let pipe = Pipe()
        which.standardOutput = pipe
        which.standardError = Pipe()
        do {
            try which.run()
            which.waitUntilExit()
            guard which.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let path = String(decoding: data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return path.isEmpty ? nil : path
        } catch {
            return nil
        }
    }

    enum BackendError: Error {
        case pythonNotFound
        case scriptNotFound
    }
}
