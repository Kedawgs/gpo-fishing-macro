"""
GPO Fishing Macro - Detection Module
=====================================
Detects fish position, sweet spot, and progress bar state.
Uses OpenCV for color-based detection.
"""

import cv2
import numpy as np
from config import (
    FISH_MARKER_WHITE,
    FISH_MARKER_GREEN,
    PROGRESS_BAR_COLOR,
    BAR_BACKGROUND_COLOR,
    BLUE_BAR_COLOR,
    MIN_FISH_MARKER_PIXELS,
    MIN_BAR_PIXELS,
    DEBUG_MODE,
    DEAD_ZONE,
    BRAKE_VELOCITY,
    GRAVITY_COMPENSATION,
)


class FishingDetector:
    """
    Detects elements of the GPO fishing minigame UI.

    The fishing UI has:
    - Left bar: Shows fish position (white line) and your sweet spot (dark)
    - Right bar: Progress bar that fills green as you track successfully
    """

    def __init__(self):
        """Initialize the detector."""
        # Store last known positions for smoothing
        self.last_fish_y = None
        self.last_sweet_spot_y = None

        # Velocity tracking for predictive control
        self.prev_sweet_spot_y = None
        self.sweet_spot_velocity = 0  # Positive = moving down, Negative = moving up

        # Control state
        self.is_holding = False
        self.hold_frames = 0  # How long we've been holding/releasing

        # Pulse counter for maintaining position in dead zone
        self.pulse_counter = 0
        self.PULSE_HOLD_FRAMES = 3   # Hold for 3 frames
        self.PULSE_RELEASE_FRAMES = 3  # Release for 3 frames

        # Sweet spot jump filter - reject detection glitches
        self.MAX_SWEET_SPOT_JUMP = 50  # Reject jumps larger than this
        self.warmup_frames = 0  # Frames to skip filtering at start of each fish

        # Brake phase when entering dead zone
        self.in_dead_zone = False
        self.brake_frames = 0
        self.BRAKE_FRAMES_NEEDED = 5  # Hold for 5 frames to brake before pulsing

    def reset_state(self):
        """Reset all tracking state for a new fish. Call when fishing starts."""
        self.last_fish_y = None
        # Set to None so first detection is accepted (no filtering on first frame)
        self.last_sweet_spot_y = None
        self.prev_sweet_spot_y = None
        self.sweet_spot_velocity = 0
        self.is_holding = False
        self.hold_frames = 0
        self.pulse_counter = 0
        self.in_dead_zone = False
        self.brake_frames = 0
        # Allow first few frames to stabilize before filtering
        self.warmup_frames = 5

    def is_fishing_active(self, frame):
        """
        Check if the fishing minigame is currently active.
        Looks for the dark fishing bars on screen.

        Args:
            frame: BGR image from screen capture

        Returns:
            bool: True if fishing bars are visible
        """
        # Convert color bounds to numpy arrays
        lower = np.array(BAR_BACKGROUND_COLOR["lower"], dtype=np.uint8)
        upper = np.array(BAR_BACKGROUND_COLOR["upper"], dtype=np.uint8)

        # Create mask for dark pixels (bar background)
        mask = cv2.inRange(frame, lower, upper)

        # Count dark pixels
        dark_pixel_count = cv2.countNonZero(mask)

        if DEBUG_MODE:
            print(f"[Detector] Dark pixels found: {dark_pixel_count}")

        # If we have enough dark pixels, the bar is probably visible
        return dark_pixel_count > MIN_BAR_PIXELS

    def get_fish_position(self, frame):
        """
        Find the Y position of the fish marker.
        Detects BOTH white (not tracking) and green (tracking correctly) fish.

        Args:
            frame: BGR image from screen capture

        Returns:
            int or None: Y coordinate of fish marker, or None if not found
        """
        # Detect WHITE fish marker (not tracking)
        lower_white = np.array(FISH_MARKER_WHITE["lower"], dtype=np.uint8)
        upper_white = np.array(FISH_MARKER_WHITE["upper"], dtype=np.uint8)
        white_mask = cv2.inRange(frame, lower_white, upper_white)

        # Detect GREEN fish marker (tracking correctly)
        lower_green = np.array(FISH_MARKER_GREEN["lower"], dtype=np.uint8)
        upper_green = np.array(FISH_MARKER_GREEN["upper"], dtype=np.uint8)
        green_mask = cv2.inRange(frame, lower_green, upper_green)

        # Combine both masks - fish can be either color
        combined_mask = cv2.bitwise_or(white_mask, green_mask)

        # Count fish pixels
        fish_pixels = cv2.countNonZero(combined_mask)

        if fish_pixels < MIN_FISH_MARKER_PIXELS:
            if DEBUG_MODE:
                print(f"[Detector] Fish marker not found (only {fish_pixels} pixels)")
            return self.last_fish_y  # Return last known position

        # Find the coordinates of fish pixels
        coords = cv2.findNonZero(combined_mask)

        if coords is None:
            return self.last_fish_y

        # Get the average Y coordinate
        y_coords = coords[:, 0, 1]
        fish_y = int(np.mean(y_coords))

        # Update last known position
        self.last_fish_y = fish_y

        if DEBUG_MODE:
            print(f"[Detector] Fish Y position: {fish_y} ({fish_pixels} pixels)")

        return fish_y

    def get_sweet_spot_position(self, frame):
        """
        Find the Y position of the sweet spot (the section you control).

        The sweet spot is the DARK section between BLUE bars.
        - Blue bars = "not your zone"
        - Gap between blue = your sweet spot

        Args:
            frame: BGR image from screen capture

        Returns:
            int or None: Y coordinate of sweet spot center, or None if not found
        """
        height, width = frame.shape[:2]

        # Use entire frame (capture region should already be just the bar)
        bar_area = frame

        # Detect BLUE pixels (the "not your zone" areas)
        lower_blue = np.array(BLUE_BAR_COLOR["lower"], dtype=np.uint8)
        upper_blue = np.array(BLUE_BAR_COLOR["upper"], dtype=np.uint8)
        blue_mask = cv2.inRange(bar_area, lower_blue, upper_blue)

        # Count blue pixels per row
        blue_per_row = np.sum(blue_mask > 0, axis=1)

        # Find rows with LOW blue count = sweet spot area
        # Threshold: less than 20% of max blue count
        max_blue = np.max(blue_per_row) if np.max(blue_per_row) > 0 else 1
        threshold = max_blue * 0.2

        # Find rows that are NOT blue (sweet spot rows)
        sweet_spot_rows = np.where(blue_per_row < threshold)[0]

        if len(sweet_spot_rows) == 0:
            if DEBUG_MODE:
                print("[Detector] Sweet spot not detected (no gap in blue)")
            return self.last_sweet_spot_y

        # Find the center of the sweet spot
        # Look for the largest continuous gap
        if len(sweet_spot_rows) > 0:
            # Find continuous segments
            gaps = np.diff(sweet_spot_rows)
            segment_breaks = np.where(gaps > 5)[0]

            if len(segment_breaks) == 0:
                # One continuous segment
                sweet_spot_y = int(np.mean(sweet_spot_rows))
            else:
                # Multiple segments - find the largest one
                segments = np.split(sweet_spot_rows, segment_breaks + 1)
                largest_segment = max(segments, key=len)
                sweet_spot_y = int(np.mean(largest_segment))
        else:
            sweet_spot_y = self.last_sweet_spot_y

        # Validate sweet_spot_y - reject sudden large jumps (likely detection glitches)
        # But skip filtering during warmup period (first few frames of each fish)
        if self.warmup_frames > 0:
            self.warmup_frames -= 1
            if DEBUG_MODE:
                print(f"[Detector] Warmup frame - accepting sweet_y={sweet_spot_y} (warmup={self.warmup_frames})")
        elif sweet_spot_y is not None and self.last_sweet_spot_y is not None:
            jump = abs(sweet_spot_y - self.last_sweet_spot_y)
            if jump > self.MAX_SWEET_SPOT_JUMP:
                # Reject this value - likely a detection glitch
                if DEBUG_MODE:
                    print(f"[Detector] REJECTING sweet_y jump: {self.last_sweet_spot_y} -> {sweet_spot_y} (jump={jump})")
                return self.last_sweet_spot_y

        # Update last known position
        if sweet_spot_y is not None:
            self.last_sweet_spot_y = sweet_spot_y

        if DEBUG_MODE:
            print(f"[Detector] Sweet spot Y position: {sweet_spot_y} (gap in blue bars)")

        return sweet_spot_y

    def get_progress(self, frame):
        """
        Get the progress bar fill percentage.

        Args:
            frame: BGR image from screen capture

        Returns:
            float: Progress from 0.0 to 1.0, or 0.0 if not detected
        """
        # Convert color bounds to numpy arrays
        lower = np.array(PROGRESS_BAR_COLOR["lower"], dtype=np.uint8)
        upper = np.array(PROGRESS_BAR_COLOR["upper"], dtype=np.uint8)

        # Look at right portion of frame (where progress bar is)
        height, width = frame.shape[:2]
        right_portion = frame[:, int(width * 0.6):]

        # Create mask for green pixels
        mask = cv2.inRange(right_portion, lower, upper)

        # Count green pixels
        green_pixels = cv2.countNonZero(mask)

        # Estimate total possible pixels in progress bar
        # Rough estimate: progress bar is about 10% of width, full height
        total_bar_pixels = height * (width * 0.1)

        # Calculate fill percentage
        progress = min(1.0, green_pixels / max(1, total_bar_pixels))

        if DEBUG_MODE:
            print(f"[Detector] Progress: {progress:.1%}")

        return progress

    def is_fish_caught(self, frame):
        """
        Check if the fish has been caught (progress bar full or bars gone).

        Args:
            frame: BGR image from screen capture

        Returns:
            bool: True if fish is caught
        """
        # Check if bars are still visible
        if not self.is_fishing_active(frame):
            return True  # Bars gone = fish caught

        # Check if progress bar is nearly full
        progress = self.get_progress(frame)
        if progress > 0.95:
            return True

        return False

    def should_hold_mouse(self, frame):
        """
        Determine if we should hold the mouse button.
        Uses predictive control to account for momentum/acceleration.

        Args:
            frame: BGR image from screen capture

        Returns:
            bool or None: True = hold, False = release, None = can't determine
        """
        fish_y = self.get_fish_position(frame)
        sweet_spot_y = self.get_sweet_spot_position(frame)

        if fish_y is None or sweet_spot_y is None:
            if DEBUG_MODE:
                print("[Detector] Cannot determine - missing position data")
            return None

        # Calculate velocity (how fast sweet spot is moving)
        if self.prev_sweet_spot_y is not None:
            # Smooth the velocity a bit to reduce noise
            new_velocity = sweet_spot_y - self.prev_sweet_spot_y
            self.sweet_spot_velocity = (self.sweet_spot_velocity * 0.5) + (new_velocity * 0.5)
        self.prev_sweet_spot_y = sweet_spot_y

        # For first few frames, be more aggressive (no velocity data yet)
        is_warmup = abs(self.sweet_spot_velocity) < 0.1

        # Calculate distance and direction
        # Apply gravity compensation - aim ABOVE the fish to counteract falling
        # This makes the "target" position slightly above the actual fish
        target_y = fish_y - GRAVITY_COMPENSATION
        distance = target_y - sweet_spot_y  # Positive = target below, Negative = target above

        # Prediction zone - anticipate where sweet spot will be
        # Account for gravity: when released, it falls faster (add gravity factor)
        PREDICTION_FRAMES = 3
        gravity_factor = 2 if not self.is_holding else 0  # Extra fall speed when released
        predicted_sweet_y = sweet_spot_y + (self.sweet_spot_velocity * PREDICTION_FRAMES) + gravity_factor
        predicted_distance = target_y - predicted_sweet_y

        if DEBUG_MODE:
            print(f"[Detector] Fish Y: {fish_y}, Target Y: {target_y}, Sweet Y: {sweet_spot_y}")
            print(f"[Detector] Distance: {distance}, Velocity: {self.sweet_spot_velocity:.1f}")
            print(f"[Detector] Predicted distance: {predicted_distance:.1f}, Gravity comp: {GRAVITY_COMPENSATION}")

        # During warmup, use smaller dead zone for faster initial response
        active_dead_zone = 5 if is_warmup else DEAD_ZONE

        # Smart control logic using config values
        # If fish is above (distance negative)
        if distance < -active_dead_zone:
            self.in_dead_zone = False  # Left the dead zone
            # Fish is above - we need to go up (hold)
            # But if we're already moving up fast, maybe release to slow down
            # (Skip braking during warmup - react immediately)
            if not is_warmup and self.sweet_spot_velocity < -BRAKE_VELOCITY and predicted_distance > -DEAD_ZONE:
                # Moving up fast and will overshoot - release to brake
                if DEBUG_MODE:
                    print("[Detector] Moving up fast, releasing to brake")
                self.is_holding = False
                return False
            else:
                if DEBUG_MODE:
                    print("[Detector] Fish above -> HOLD")
                self.is_holding = True
                return True

        # If fish is below (distance positive)
        elif distance > active_dead_zone:
            self.in_dead_zone = False  # Left the dead zone
            # Fish is below - we need to go down (release)
            # But if we're already falling fast, maybe hold to slow down
            # (Skip braking during warmup - react immediately)
            if not is_warmup and self.sweet_spot_velocity > BRAKE_VELOCITY and predicted_distance < DEAD_ZONE:
                # Falling fast and will overshoot - hold to brake
                if DEBUG_MODE:
                    print("[Detector] Falling fast, holding to brake")
                self.is_holding = True
                return True
            else:
                if DEBUG_MODE:
                    print("[Detector] Fish below -> RELEASE")
                self.is_holding = False
                return False

        # In dead zone - brake first if we were moving, then pulse to maintain
        else:
            # Check if we just entered the dead zone
            if not self.in_dead_zone:
                self.in_dead_zone = True
                self.brake_frames = 0
                self.pulse_counter = 0
                if DEBUG_MODE:
                    print(f"[Detector] ENTERING dead zone, velocity: {self.sweet_spot_velocity:.1f}")

            # If we have downward velocity (falling), brake first
            if self.sweet_spot_velocity > 1.5 and self.brake_frames < self.BRAKE_FRAMES_NEEDED:
                self.brake_frames += 1
                if DEBUG_MODE:
                    print(f"[Detector] BRAKING - holding to stop fall ({self.brake_frames}/{self.BRAKE_FRAMES_NEEDED})")
                self.is_holding = True
                return True

            # If we have upward velocity (rising), brake by releasing
            if self.sweet_spot_velocity < -1.5 and self.brake_frames < self.BRAKE_FRAMES_NEEDED:
                self.brake_frames += 1
                if DEBUG_MODE:
                    print(f"[Detector] BRAKING - releasing to stop rise ({self.brake_frames}/{self.BRAKE_FRAMES_NEEDED})")
                self.is_holding = False
                return False

            # After braking (or if already slow), pulse to maintain position
            self.pulse_counter += 1
            total_cycle = self.PULSE_HOLD_FRAMES + self.PULSE_RELEASE_FRAMES

            # Reset counter if it gets too high
            if self.pulse_counter >= total_cycle:
                self.pulse_counter = 0

            # Hold for first part of cycle, release for second part
            if self.pulse_counter < self.PULSE_HOLD_FRAMES:
                if DEBUG_MODE:
                    print(f"[Detector] PULSE HOLD ({self.pulse_counter}/{self.PULSE_HOLD_FRAMES})")
                self.is_holding = True
                return True
            else:
                if DEBUG_MODE:
                    print(f"[Detector] PULSE RELEASE ({self.pulse_counter - self.PULSE_HOLD_FRAMES}/{self.PULSE_RELEASE_FRAMES})")
                self.is_holding = False
                return False


# Quick test if run directly
if __name__ == "__main__":
    import cv2
    from screen_capture import ScreenCapture

    print("Testing detector...")
    print("Make sure Roblox GPO is open with fishing minigame active!")
    print("")

    capture = ScreenCapture()
    detector = FishingDetector()

    # Test for a few seconds
    import time
    for i in range(30):  # 3 seconds of testing
        frame = capture.grab()

        print(f"\n--- Frame {i+1} ---")
        active = detector.is_fishing_active(frame)
        print(f"Fishing active: {active}")

        if active:
            fish_y = detector.get_fish_position(frame)
            sweet_y = detector.get_sweet_spot_position(frame)
            progress = detector.get_progress(frame)
            should_hold = detector.should_hold_mouse(frame)

            print(f"Fish Y: {fish_y}")
            print(f"Sweet Spot Y: {sweet_y}")
            print(f"Progress: {progress:.1%}")
            print(f"Should hold mouse: {should_hold}")

        time.sleep(0.1)

    capture.close()
    print("\nDone testing!")
