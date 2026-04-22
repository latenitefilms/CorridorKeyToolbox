//
//  OrientationHelper+FxPlug.swift
//  Corridor Key Pro
//
//  FxPlug-aware glue for `OrientationHelper`. Kept in its own file so the
//  helper itself can be compiled inside the Swift Package (which has no
//  FxPlug dependency) and still be reachable with strongly-typed
//  `FxImageOrigin` values from the render pipeline.
//

import Foundation

extension CorridorKeyImageOrigin {
    /// Bridges an `FxImageOrigin` (NSUInteger enum) into our FxPlug-free
    /// mirror. Anything the host surfaces that isn't one of the two known
    /// constants becomes `.unknown`, which the helper then treats as a
    /// "don't flip" signal to leave the image alone.
    init(fxOrigin: FxImageOrigin) {
        switch fxOrigin {
        case FxImageOrigin(kFxImageOrigin_BOTTOM_LEFT):
            self = .bottomLeft
        case FxImageOrigin(kFxImageOrigin_TOP_LEFT):
            self = .topLeft
        default:
            self = .unknown
        }
    }
}
