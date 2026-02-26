//
//  NovaLogger.swift
//  Nova
//
//  Non-blocking os.Logger for OpenAI/Flow path. Avoids print/DebugLog I/O stalls.
//

import os

enum NovaLogger {
    nonisolated static let log = Logger(subsystem: "com.coppola.nova", category: "Flow")

    @inline(__always)
    nonisolated static func info(_ msg: String) {
        log.info("\(msg, privacy: .public)")
    }

    @inline(__always)
    nonisolated static func error(_ msg: String) {
        log.error("\(msg, privacy: .public)")
    }
}
