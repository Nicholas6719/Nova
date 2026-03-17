import Foundation

/// Centralized API key resolution. Tries environment variables first (macOS debug),
/// then Info.plist (iOS / production). Returns empty string if not found.
enum APIKeyProvider {
    static var openAIKey: String {
        // 1. Environment variable (macOS debug via Xcode scheme)
        if let envKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"],
           !envKey.isEmpty {
            return envKey
        }
        // 2. Bundled Config.plist (iOS production + macOS production)
        if let url = Bundle.main.url(forResource: "Config", withExtension: "plist"),
           let dict = NSDictionary(contentsOf: url),
           let key = dict["OPENAI_API_KEY"] as? String,
           !key.isEmpty {
            return key
        }
        // 3. Info.plist fallback
        if let plistKey = Bundle.main.infoDictionary?["OPENAI_API_KEY"] as? String,
           !plistKey.isEmpty {
            return plistKey
        }
        // 4. Not found — LLMClient will throw missingAPIKey
        return ""
    }
}
