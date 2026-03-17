import Foundation

enum DebugLog {
    nonisolated static func d(_ message: String) {
        #if DEBUG
        print(message)
        #endif
    }
}
