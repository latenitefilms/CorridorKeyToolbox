//
//  OrientationHelperTests.swift
//  CorridorKeyProLogicTests
//
//  Locks in the vertical-flip contract that the compose render pass relies
//  on. These cases cover the two known hosts (Final Cut Pro and Motion) and
//  any unexpected origin FxPlug might surface in the future.
//

import Testing
@testable import CorridorKeyProLogic

@Suite("OrientationHelper")
struct OrientationHelperTests {

    @Test("Matching TOP_LEFT origins (FCP default) flip V")
    func fcpMatchingOriginsFlip() {
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .topLeft,
            destinationOrigin: .topLeft
        ))
    }

    @Test("Matching BOTTOM_LEFT origins (Motion default) flip V")
    func motionMatchingOriginsFlip() {
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .bottomLeft,
            destinationOrigin: .bottomLeft
        ))
    }

    @Test("Mismatched origins skip the flip so the host can swap orientation itself")
    func mismatchedOriginsNoFlip() {
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .topLeft,
            destinationOrigin: .bottomLeft
        ) == false)
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .bottomLeft,
            destinationOrigin: .topLeft
        ) == false)
    }

    @Test("Unknown origins leave the image untouched")
    func unknownOriginsNoFlip() {
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .unknown,
            destinationOrigin: .unknown
        ) == false)
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .unknown,
            destinationOrigin: .topLeft
        ) == false)
        #expect(OrientationHelper.composeNeedsVerticalFlip(
            sourceOrigin: .topLeft,
            destinationOrigin: .unknown
        ) == false)
    }

    @Test("describe() round-trips to a readable label")
    func describeLabels() {
        #expect(OrientationHelper.describe(.topLeft) == "TopLeft")
        #expect(OrientationHelper.describe(.bottomLeft) == "BottomLeft")
        #expect(OrientationHelper.describe(.unknown) == "Unknown")
    }
}
