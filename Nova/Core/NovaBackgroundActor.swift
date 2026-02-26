//
//  NovaBackgroundActor.swift
//  Nova
//
//  Global actor for OpenAI/cache pipeline. Keeps heavy work off MainActor to prevent freezes.
//

import Foundation

/// Background actor for the OpenAI/cache pipeline. Any call into engine or LLM client
/// becomes an actor hop so MainActor yields immediately and cannot freeze.
@globalActor
actor NovaBackgroundActor {
    static let shared = NovaBackgroundActor()
}
