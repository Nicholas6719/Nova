import Foundation

enum DebugLog {
    static func d(_ message: String) {
        #if DEBUG
        print(message)
        #endif
    }
}
