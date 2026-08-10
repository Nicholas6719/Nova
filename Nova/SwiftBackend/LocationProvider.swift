//
//  LocationProvider.swift
//  Nova
//
//  Supplies the user's location to the Python backend.
//
//  WHY THIS EXISTS IN SWIFT
//  ────────────────────────
//  `maps_engine.py` used to ask CoreLocation directly, from a short-lived
//  python subprocess. That can never work, and it failed silently in the worst
//  possible way: `requestWhenInUseAuthorization()` returned without error, no
//  permission dialog ever appeared, NovaOS never showed up under System
//  Settings › Privacy & Security › Location Services, and the status stayed
//  `notDetermined` (0) forever. Nova then told Nicholas to "enable it under
//  Location Services" — where there was nothing to enable.
//
//  CoreLocation reads `NSLocationWhenInUseUsageDescription` from the Info.plist
//  of the CALLING EXECUTABLE. The caller was a bare miniforge python binary,
//  which has no Info.plist at all, so the request was a no-op. Being a child of
//  NovaOS.app does not lend the child the app's identity.
//
//  This app bundle DOES have the usage string and the
//  com.apple.security.personal-information.location entitlement, so the request
//  has to originate here. Swift acquires the fix and POSTs it to the backend at
//  /api/location; Python caches it and answers distance questions from that.
//
//  Privacy: a coordinate is only ever requested after Nicholas asks a location
//  question, is held in memory (never written to the memory database), and is
//  sent nowhere except localhost. See CLAUDE.md invariant 3.
//

import Combine          // @Published / ObservableObject (MemberImportVisibility is on)
import CoreLocation
import Foundation

@MainActor
final class LocationProvider: NSObject, ObservableObject, CLLocationManagerDelegate {

    /// Published so the UI can show an honest state if we ever surface it.
    @Published private(set) var authorization: CLAuthorizationStatus = .notDetermined
    @Published private(set) var lastFix: CLLocation?

    private let manager = CLLocationManager()
    private let httpPort: Int
    private var httpBase: URL { URL(string: "http://localhost:\(httpPort)")! }

    /// Set while a fix is being pursued, so repeated asks don't stack updates.
    private var isUpdating = false

    init(httpPort: Int = 5001) {
        self.httpPort = httpPort
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        authorization = manager.authorizationStatus
    }

    /// Ask for permission if we've never asked, and start a single update.
    ///
    /// Safe to call repeatedly. When the user has denied access we do NOT
    /// re-prompt (macOS ignores it anyway) — the backend already declines
    /// honestly and points at System Settings, which will now actually list
    /// NovaOS because this request came from the app.
    func requestLocation() {
        switch manager.authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()   // the prompt, at last
        case .authorized, .authorizedAlways:
            startOnce()
        case .denied, .restricted:
            postStatus(available: false, reason: "denied")
        @unknown default:
            break
        }
    }

    private func startOnce() {
        guard !isUpdating else { return }
        isUpdating = true
        manager.requestLocation()          // one fix, then stops on its own
    }

    // MARK: - CLLocationManagerDelegate

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            self.authorization = manager.authorizationStatus
            switch manager.authorizationStatus {
            case .authorized, .authorizedAlways:
                self.startOnce()
            case .denied, .restricted:
                self.postStatus(available: false, reason: "denied")
            default:
                break
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager,
                                     didUpdateLocations locations: [CLLocation]) {
        guard let fix = locations.last else { return }
        Task { @MainActor in
            self.isUpdating = false
            self.lastFix = fix
            self.postFix(fix)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager,
                                     didFailWithError error: Error) {
        Task { @MainActor in
            self.isUpdating = false
            self.postStatus(available: false, reason: "no fix")
        }
    }

    // MARK: - Handing the coordinate to Python

    private func postFix(_ fix: CLLocation) {
        post(body: [
            "available": true,
            "lat": fix.coordinate.latitude,
            "lon": fix.coordinate.longitude,
            "accuracy_m": fix.horizontalAccuracy,
        ])
    }

    private func postStatus(available: Bool, reason: String) {
        post(body: ["available": available, "reason": reason])
    }

    private func post(body: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return }
        var request = URLRequest(url: httpBase.appendingPathComponent("api/location"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = data
        Task { _ = try? await URLSession.shared.data(for: request) }
    }
}
