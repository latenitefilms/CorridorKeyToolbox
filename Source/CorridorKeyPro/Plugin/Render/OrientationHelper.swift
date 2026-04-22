//
//  OrientationHelper.swift
//  Corridor Key Pro
//
//  Keeps the compose quad's texture coordinates consistent between
//  IOSurface-backed source/destination tiles and the plain Metal
//  intermediates produced during MLX inference.
//
//  Hosts surface IOSurfaces with either TOP_LEFT (Final Cut Pro) or
//  BOTTOM_LEFT (Motion) origin. Our MLX alpha/foreground textures are
//  plain Metal allocations, so they end up in whatever orientation the
//  compute kernels populate them with — which follows the source's byte
//  layout because the kernels write at the same `gid` they read. The
//  single case that needs a vertical flip is when Metal's render-target
//  row 0 doesn't line up with IOSurface byte row 0 in the destination's
//  origin convention.
//

import Foundation

/// FxPlug-free mirror of `FxImageOrigin` so the helper can be covered by the
/// Swift Package tests without pulling in the FxPlug framework.
enum CorridorKeyImageOrigin: Sendable, Equatable {
    case topLeft
    case bottomLeft
    case unknown
}

enum OrientationHelper {

    /// Returns `true` when the compose quad should invert its V coordinate
    /// so the final IOSurface comes out right-way-up.
    ///
    /// Empirically, Final Cut Pro renders Motion Template effects with
    /// `TOP_LEFT` source + `TOP_LEFT` destination origins but still needs
    /// the flip because Metal's framebuffer convention treats the first
    /// row as "top" while Final Cut's viewer interprets the IOSurface
    /// bytes with the opposite orientation. Matching Motion's behaviour
    /// (`BOTTOM_LEFT` source + `BOTTOM_LEFT` destination) would likewise
    /// need a flip for the same reason. The only case we see no flip is a
    /// mismatched pair, which we treat as the host telling us it will
    /// perform the orientation swap itself.
    static func composeNeedsVerticalFlip(
        sourceOrigin: CorridorKeyImageOrigin,
        destinationOrigin: CorridorKeyImageOrigin
    ) -> Bool {
        guard sourceOrigin != .unknown, destinationOrigin != .unknown else {
            return false
        }
        return sourceOrigin == destinationOrigin
    }

    /// Human-readable label for the log file so we can tell what FxPlug is
    /// handing us during renders and analysis passes.
    static func describe(_ origin: CorridorKeyImageOrigin) -> String {
        switch origin {
        case .topLeft:
            return "TopLeft"
        case .bottomLeft:
            return "BottomLeft"
        case .unknown:
            return "Unknown"
        }
    }
}
