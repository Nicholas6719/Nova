import Foundation

/// Pure math evaluation. Static, no actors, no UI, no DateFormatter.
/// Handles: single binary ops (8-2, 300*2, 12/3), +/- chains (300-200+2, -5+2).
/// Normalizes Unicode math symbols (× → *, ÷ → /).
enum MathRouter {

    /// Attempt local math evaluation. Returns formatted response or nil.
    static func localMathResponse(for rawInput: String) -> String? {
        let normalized = normalizeMathSymbols(
            rawInput.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        )

        if let result = evaluate(normalized) { return result }

        let stripped = stripMathPrefixes(normalized)
        if stripped != normalized, let result = evaluate(stripped) { return result }

        return nil
    }

    /// True if input looks like an unsupported math expression (digits + operators, no letters).
    /// Used to skip intent detection for mathy strings we can't evaluate locally.
    static func looksLikeMath(_ rawInput: String) -> Bool {
        let s = normalizeMathSymbols(
            rawInput.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        ).replacingOccurrences(of: " ", with: "")
        var hasDigit = false
        var hasOp = false
        let mathChars: Set<Character> = ["+", "-", "*", "/", "(", ")"]
        for ch in s {
            if ch.isNumber || ch == "." { hasDigit = true }
            else if mathChars.contains(ch) { hasOp = true }
            else if ch.isLetter { return false }
        }
        return hasDigit && hasOp
    }

    // MARK: - Private

    private static func evaluate(_ expr: String) -> String? {
        let s = expr.replacingOccurrences(of: " ", with: "")
        guard !s.isEmpty else { return nil }
        if let result = singleBinaryOp(s) { return result }
        if let result = plusMinusChain(s) { return result }
        return nil
    }

    private static func normalizeMathSymbols(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{00D7}", with: "*")
         .replacingOccurrences(of: "\u{00F7}", with: "/")
         .replacingOccurrences(of: "\u{2022}", with: "*")
         .replacingOccurrences(of: "\u{00B7}", with: "*")
    }

    private static func stripMathPrefixes(_ s: String) -> String {
        var result = s
        for prefix in ["nova ", "what's ", "whats ", "what is ", "calculate ", "solve "] {
            if result.hasPrefix(prefix) {
                result = String(result.dropFirst(prefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }
        return result
    }

    /// Parse "number op number" where op is one of +, -, *, /.
    private static func singleBinaryOp(_ s: String) -> String? {
        guard s.rangeOfCharacter(from: .decimalDigits) != nil else { return nil }

        let ops: [Character] = ["+", "-", "*", "/"]
        var found: (Character, Int)? = nil
        for (i, ch) in s.enumerated() {
            if ops.contains(ch) {
                if i == 0 { return nil }
                if found != nil { return nil }
                found = (ch, i)
            }
        }
        guard let (op, idx) = found else { return nil }

        let lhs = String(s.prefix(idx))
        let rhs = String(s.suffix(s.count - idx - 1))
        guard let a = Double(lhs), let b = Double(rhs) else { return nil }

        let result: Double
        switch op {
        case "+": result = a + b
        case "-": result = a - b
        case "*": result = a * b
        case "/":
            if b == 0 { return nil }
            result = a / b
        default: return nil
        }

        let spokenOp: String
        switch op {
        case "+": spokenOp = "plus"
        case "-": spokenOp = "minus"
        case "*": spokenOp = "times"
        case "/": spokenOp = "divided by"
        default: spokenOp = "?"
        }
        return "\(fmt(a)) \(spokenOp) \(fmt(b)) equals \(fmt(result))."
    }

    /// Parse a chain of + and - operations: "300-200+2" → 102, "-5+2" → -3.
    private static func plusMinusChain(_ s: String) -> String? {
        for ch in s {
            guard ch.isNumber || ch == "." || ch == "+" || ch == "-" else { return nil }
        }
        guard !s.isEmpty else { return nil }

        let inner = s.first == "-" ? String(s.dropFirst()) : s
        guard inner.contains("+") || inner.contains("-") else { return nil }

        var result = 0.0
        var currentNum = ""
        var pendingOp: Character = "+"
        var isFirst = true

        for ch in s {
            if (ch == "+" || ch == "-") && !currentNum.isEmpty {
                guard let val = Double(currentNum) else { return nil }
                result = pendingOp == "+" ? result + val : result - val
                currentNum = ""
                pendingOp = ch
                isFirst = false
            } else if ch == "-" && currentNum.isEmpty && isFirst {
                currentNum.append(ch)
            } else if ch.isNumber || ch == "." {
                currentNum.append(ch)
            } else {
                return nil
            }
        }

        guard !currentNum.isEmpty, let val = Double(currentNum) else { return nil }
        result = pendingOp == "+" ? result + val : result - val

        return "That equals \(fmt(result))."
    }

    private static func fmt(_ x: Double) -> String {
        x.truncatingRemainder(dividingBy: 1) == 0 && abs(x) < 1e15
            ? String(Int(x)) : String(x)
    }
}
