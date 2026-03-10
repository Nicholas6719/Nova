//
//  PlatformTools.swift
//  Nova
//
//  Platform-specific tool execution. Uses #if os() for macOS vs iOS.
//

import Foundation

#if os(macOS)
import AppKit
import IOKit.ps
#endif

#if os(iOS)
import UIKit
#endif

enum PlatformTools: Sendable {

    #if os(macOS)
    private static func macOSAppURL(for name: String) -> URL? {
        let candidates = [
            "/Applications/\(name).app",
            "/System/Applications/\(name).app",
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/\(name).app").path
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }
    #endif

    /// Execute open app / settings. Returns spoken response.
    static func executeOpenApp(name: String) async -> NovaToolResult {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        #if os(macOS)
        let workspace = NSWorkspace.shared
        let appNameForDisplay: String
        let appURL: URL?
        if normalized == "settings" || normalized == "system settings" || normalized == "system preferences" {
            appNameForDisplay = "System Settings"
            let settingsPath = "/System/Applications/System Settings.app"
            let prefsPath = "/System/Applications/System Preferences.app"
            if FileManager.default.fileExists(atPath: settingsPath) {
                appURL = URL(fileURLWithPath: settingsPath)
            } else if FileManager.default.fileExists(atPath: prefsPath) {
                appURL = URL(fileURLWithPath: prefsPath)
            } else {
                appURL = nil
            }
        } else {
            appNameForDisplay = normalized.prefix(1).uppercased() + normalized.dropFirst()
            let name = String(appNameForDisplay)
            appURL = Self.macOSAppURL(for: name)
        }
        guard let url = appURL, FileManager.default.fileExists(atPath: url.path) else {
            return .failure(spoken: "I couldn't find an app named \(appNameForDisplay).")
        }
        let launched = workspace.open(url)
        return launched ? .success(spoken: "Opening \(appNameForDisplay).") : .failure(spoken: "I couldn't open \(appNameForDisplay).")
        #elseif os(iOS)
        if normalized == "settings" || normalized == "system settings" {
            if let url = URL(string: UIApplication.openSettingsURLString) {
                let opened = await Task { @MainActor in await UIApplication.shared.open(url) }.value
                return opened ? .success(spoken: "Opening Settings.") : .failure(spoken: "I couldn't open Settings.")
            }
            return .failure(spoken: "I couldn't open Settings.")
        }
        return .failure(spoken: "Opening apps by name is currently available on Mac. I can open Settings here.")
        #else
        return .failure(spoken: "That isn't supported on this device.")
        #endif
    }

    /// Execute quit/close app. Disabled for release. Returns safe message; no execution.
    static func executeQuitApp(name: String) -> NovaToolResult {
        return .failure(spoken: "Closing apps by name isn't available yet.")
    }

    /// Execute battery status. Returns spoken response. Non-blocking on macOS.
    static func executeBatteryStatus(chargingIntent: Bool) async -> NovaToolResult {
        #if os(macOS)
        let result: NovaToolResult = await withTaskGroup(of: NovaToolResult.self) { group in
            group.addTask {
                let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
                let sourcesList = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as [CFTypeRef]
                for ps in sourcesList {
                    guard let info = IOPSGetPowerSourceDescription(snapshot, ps)?.takeUnretainedValue() as? [String: AnyObject] else { continue }
                    let isCharging = (info[kIOPSPowerSourceStateKey] as? String) == kIOPSACPowerValue
                    let capVal = info[kIOPSCurrentCapacityKey]
                    let maxVal = info[kIOPSMaxCapacityKey]
                    let cap = (capVal as? NSNumber)?.intValue ?? (capVal as? Int)
                    let max = (maxVal as? NSNumber)?.intValue ?? (maxVal as? Int)
                    if let c = cap, let m = max, m > 0 {
                        let pct = (c * 100) / m
                        return Self.formatBatteryResponse(pct: pct, isCharging: isCharging, chargingIntent: chargingIntent)
                    }
                    if isCharging {
                        let msg = chargingIntent ? "Yes, your Mac is plugged in and charging." : "Your Mac is plugged in and charging."
                        return .success(spoken: msg)
                    }
                }
                return .failure(spoken: "I couldn't read battery information.")
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                return .failure(spoken: "I couldn't read battery information.")
            }
            let first = await group.next()!
            group.cancelAll()
            return first
        }
        return result
        #elseif os(iOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        let state = UIDevice.current.batteryState
        if level < 0 {
            return .failure(spoken: "I couldn't read battery information.")
        }
        let pct = Int(level * 100)
        let isCharging = (state == .charging || state == .full)
        return Self.formatBatteryResponse(pct: pct, isCharging: isCharging, chargingIntent: chargingIntent, isFull: state == .full)
        #else
        return .failure(spoken: "Battery status isn't available on this device.")
        #endif
    }

    private nonisolated static func formatBatteryResponse(pct: Int, isCharging: Bool, chargingIntent: Bool, isFull: Bool = false) -> NovaToolResult {
        if chargingIntent {
            if isFull {
                return .success(spoken: "Yes, your battery is fully charged at \(pct)%.")
            }
            if isCharging {
                return .success(spoken: "Yes, your battery is charging at \(pct)%.")
            }
            return .success(spoken: "No, your battery is at \(pct)% and not charging.")
        }
        if isCharging {
            if isFull {
                return .success(spoken: "Your battery is at \(pct)% and fully charged.")
            }
            return .success(spoken: "Your battery is at \(pct)% and charging.")
        }
        return .success(spoken: "Your battery is at \(pct)%.")
    }

    /// Execute web search. Opens browser. Returns spoken response.
    static func executeWebSearch(query: String) async -> NovaToolResult {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let searchURLString = "https://www.google.com/search?q=\(encoded)"
        guard let url = URL(string: searchURLString) else {
            return .failure(spoken: "I couldn't create a search URL.")
        }
        #if os(macOS)
        let opened = NSWorkspace.shared.open(url)
        #elseif os(iOS)
        let opened = await Task { @MainActor in await UIApplication.shared.open(url) }.value
        #else
        let opened = false
        #endif
        if opened {
            let cleanQuery = query.trimmingCharacters(in: .whitespaces)
            let display = cleanQuery.prefix(1).uppercased() + cleanQuery.dropFirst()
            return .success(spoken: "Searching the web for \(display).")
        }
        return .failure(spoken: "I couldn't open the browser.")
    }
}
