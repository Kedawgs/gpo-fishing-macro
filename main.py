"""
GPO Fishing Macro - Main Entry Point
=====================================
Run this file to start the fishing macro.

Controls:
    F6 - Toggle macro ON/OFF
    F7 - Exit program

Usage:
    1. Open Roblox GPO and go to a fishing spot
    2. Run this script: python main.py
    3. Press F6 to start the macro
    4. Press F6 again to pause, F7 to exit
"""

import time
import keyboard
from enum import Enum

from config import (
    TOGGLE_KEY,
    EXIT_KEY,
    LOOP_DELAY,
    RECAST_DELAY,
    WAIT_FOR_FISH_DELAY,
    DEBUG_MODE,
    SAVE_DEBUG_SCREENSHOTS,
)
from screen_capture import ScreenCapture
from detector import FishingDetector
from mouse_control import MouseController
from debug_capture import DebugCapture


class FishingState(Enum):
    """States for the fishing state machine."""
    IDLE = "idle"           # Waiting for fish to bite / ready to cast
    FISHING = "fishing"     # Actively tracking the fish
    CAUGHT = "caught"       # Fish caught, need to recast


class FishingMacro:
    """
    Main fishing macro controller.

    Handles the state machine and coordinates all modules.
    """

    def __init__(self):
        """Initialize the macro."""
        self.capture = ScreenCapture()
        self.detector = FishingDetector()
        self.mouse = MouseController()

        # Debug capture (if enabled)
        self.debug_capture = DebugCapture() if SAVE_DEBUG_SCREENSHOTS else None

        self.enabled = False
        self.running = True
        self.state = FishingState.IDLE
        self.is_holding = False  # Track current mouse state

        # Set up hotkeys
        keyboard.on_press_key(TOGGLE_KEY, self._on_toggle)
        keyboard.on_press_key(EXIT_KEY, self._on_exit)

        print("=" * 50)
        print("     GPO FISHING MACRO")
        print("=" * 50)
        print("")
        print(f"  Press {TOGGLE_KEY.upper()} to toggle ON/OFF")
        print(f"  Press {EXIT_KEY.upper()} to exit")
        print("")
        if SAVE_DEBUG_SCREENSHOTS:
            print("  DEBUG MODE: Saving frames to debug_frames/")
        print("  Status: PAUSED")
        print("=" * 50)

    def _on_toggle(self, event):
        """Handle toggle hotkey press."""
        self.enabled = not self.enabled
        status = "RUNNING" if self.enabled else "PAUSED"
        print(f"\n[Macro] {status}")

        # Reset state when toggling
        if self.enabled:
            self.state = FishingState.IDLE
        else:
            self.mouse.cleanup()

    def _on_exit(self, event):
        """Handle exit hotkey press."""
        print("\n[Macro] Exiting...")
        self.running = False
        self.enabled = False

    def run(self):
        """Main loop - run the macro."""
        try:
            while self.running:
                if self.enabled:
                    self._tick()
                time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[Macro] Interrupted by user")
        finally:
            self._cleanup()

    def _tick(self):
        """Single tick of the macro logic."""
        # Capture screen
        frame = self.capture.grab()

        # State machine
        if self.state == FishingState.IDLE:
            self._handle_idle(frame)

        elif self.state == FishingState.FISHING:
            self._handle_fishing(frame)

        elif self.state == FishingState.CAUGHT:
            self._handle_caught(frame)

    def _handle_idle(self, frame):
        """Handle IDLE state - waiting for fish."""
        # Check if fishing minigame started (bars appeared)
        if self.detector.is_fishing_active(frame):
            print("[State] Fish on the line! -> FISHING")
            self.detector.reset_state()  # Reset all tracking state for the new fish
            self.state = FishingState.FISHING
        else:
            # No fish yet - could click to cast if needed
            # For now, just wait
            if DEBUG_MODE:
                print("[State] Waiting for fish...")

    def _handle_fishing(self, frame):
        """Handle FISHING state - actively tracking."""
        # Check if fish was caught
        if self.detector.is_fish_caught(frame):
            print("[State] Fish caught! -> CAUGHT")
            self.mouse.release()  # Make sure we release
            self.is_holding = False
            self.state = FishingState.CAUGHT
            return

        # Determine if we should hold or release
        should_hold = self.detector.should_hold_mouse(frame)

        if should_hold is True:
            self.mouse.hold()
            self.is_holding = True
        elif should_hold is False:
            self.mouse.release()
            self.is_holding = False
        # If None, keep current state (couldn't determine)

        # Save debug frame if enabled
        if self.debug_capture:
            fish_y = self.detector.last_fish_y
            sweet_y = self.detector.last_sweet_spot_y
            velocity = self.detector.sweet_spot_velocity
            action = "HOLD" if should_hold else "RELEASE" if should_hold is False else "NONE"
            self.debug_capture.save_frame(frame, fish_y, sweet_y, velocity, action, self.is_holding)

    def _handle_caught(self, frame):
        """Handle CAUGHT state - recast and return to idle."""
        print("[State] Recasting...")

        # Wait a moment
        time.sleep(RECAST_DELAY)

        # Click to recast
        self.mouse.click()

        # Wait for the cast animation
        time.sleep(WAIT_FOR_FISH_DELAY)

        # Back to idle
        print("[State] -> IDLE (waiting for fish)")
        self.state = FishingState.IDLE

    def _cleanup(self):
        """Clean up resources."""
        print("[Macro] Cleaning up...")
        self.mouse.cleanup()
        self.capture.close()
        if self.debug_capture:
            self.debug_capture.close()
        keyboard.unhook_all()
        print("[Macro] Goodbye!")


def main():
    """Entry point."""
    print("")
    print("Starting GPO Fishing Macro...")
    print("")

    # Check that we're not running in a weird environment
    try:
        macro = FishingMacro()
        macro.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("")
        print("Troubleshooting tips:")
        print("1. Try running as Administrator")
        print("2. Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("3. Make sure Roblox is open and visible")
        raise


if __name__ == "__main__":
    main()
