"""
render_thread.py
Async Mitsuba 3 renderer that won't freeze the GUI.
"""
from __future__ import annotations  # PEP 563: Deferred type hint evaluation

import threading
import queue
import numpy as np
from typing import Optional, Callable, TYPE_CHECKING, Any

# Avoid early Mitsuba variant dependency at import time
if TYPE_CHECKING:
    import mitsuba as mi

class RenderThread(threading.Thread):
    """
    Background thread for progressive Mitsuba rendering.
    Pushes pixel data to a thread-safe queue for UI consumption.
    """

    def __init__(
        self,
        scene: Any,  # mi.Scene, but avoid import-time evaluation
        phase_data: np.ndarray,
        output_queue: queue.Queue,
        spp_per_iteration: int = 4,
        max_iterations: int = 256,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Args:
            scene: Pre-loaded Mitsuba scene
            phase_data: Phase function data to inject
            output_queue: Thread-safe queue for pixel data
            spp_per_iteration: Samples per pixel per iteration (progressive)
            max_iterations: Maximum progressive refinement steps
            callback: Optional callback(image_rgb) called after each iteration
        """
        super().__init__(daemon=True)
        self.scene = scene
        self.phase_data = phase_data
        self.output_queue = output_queue
        self.spp_per_iteration = spp_per_iteration
        self.max_iterations = max_iterations
        self.callback = callback
        self.stop_event = threading.Event()

    def stop(self):
        """Signal the thread to stop gracefully."""
        self.stop_event.set()

    def run(self):
        """Main rendering loop (runs in background thread)."""
        try:
            # Import at runtime (after variant is set)
            import mitsuba as mi
            import drjit as dr

            # Update phase function data
            from .preview_scene import update_phase_data
            update_phase_data(self.scene, self.phase_data)

            # Get sensor
            sensor = self.scene.sensors()[0]

            # Initialize accumulator
            film = sensor.film()
            accumulated_img = None

            for iteration in range(self.max_iterations):
                if self.stop_event.is_set():
                    break

                # Render with incremental samples
                # CRITICAL: Use a fresh seed per iteration
                seed = iteration * 12345

                # Render
                img = mi.render(
                    self.scene,
                    sensor=sensor,
                    spp=self.spp_per_iteration,
                    seed=seed
                )

                # Accumulate (progressive refinement)
                if accumulated_img is None:
                    accumulated_img = img
                else:
                    # Weighted average
                    weight = float(iteration) / (iteration + 1)
                    accumulated_img = weight * accumulated_img + (1 - weight) * img

                # Synchronize Dr.Jit computation graph
                # CRITICAL for thread safety
                dr.sync_thread()
                dr.eval(accumulated_img)

                # Convert to CPU NumPy array
                # Handle both spectral and RGB modes
                if isinstance(accumulated_img, mi.Color3f):
                    rgb_array = np.array(accumulated_img)
                else:
                    # If spectral, convert to sRGB
                    rgb_array = np.array(accumulated_img)

                # Ensure shape is [H, W, 3]
                if rgb_array.ndim == 2:
                    rgb_array = np.stack([rgb_array] * 3, axis=-1)

                # Convert linear -> sRGB for display
                rgb_srgb = self._linear_to_srgb(rgb_array)

                # Clamp and convert to 8-bit
                rgb_8bit = (np.clip(rgb_srgb, 0, 1) * 255).astype(np.uint8)

                # Push to queue (non-blocking)
                try:
                    self.output_queue.put_nowait({
                        'iteration': iteration + 1,
                        'total_spp': (iteration + 1) * self.spp_per_iteration,
                        'image': rgb_8bit
                    })
                except queue.Full:
                    pass  # Skip if queue is full (UI is lagging)

                # Optional callback
                if self.callback:
                    self.callback(rgb_8bit)

            # Signal completion
            self.output_queue.put_nowait({'done': True})

        except Exception as e:
            # Push error to queue
            self.output_queue.put_nowait({'error': str(e)})

    @staticmethod
    def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to sRGB for display.

        Args:
            linear: Linear RGB array [H, W, 3] in [0, inf)

        Returns:
            sRGB array [H, W, 3] in [0, 1]
        """
        # Standard sRGB transfer function
        return np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055
        )


class RenderManager:
    """
    High-level manager for render threads.
    Ensures only one render at a time, handles cancellation.
    """

    def __init__(self):
        self.current_thread: Optional[RenderThread] = None
        self.output_queue = queue.Queue(maxsize=4)  # Backpressure limit

    def start_render(
        self,
        scene: Any,  # mi.Scene
        phase_data: np.ndarray,
        spp_per_iteration: int = 4,
        max_iterations: int = 256,
        callback: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Start a new render, canceling any existing render.

        Args:
            scene: Mitsuba scene
            phase_data: Phase function data
            spp_per_iteration: Samples per iteration
            max_iterations: Max progressive steps
            callback: Optional callback per iteration
        """
        # Cancel existing render
        self.cancel_render()

        # Clear queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        # Start new thread
        self.current_thread = RenderThread(
            scene=scene,
            phase_data=phase_data,
            output_queue=self.output_queue,
            spp_per_iteration=spp_per_iteration,
            max_iterations=max_iterations,
            callback=callback
        )
        self.current_thread.start()

    def cancel_render(self):
        """Stop the current render gracefully."""
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.stop()
            self.current_thread.join(timeout=1.0)
        self.current_thread = None

    def get_latest_result(self, timeout: float = 0.01) -> Optional[dict]:
        """
        Non-blocking fetch of latest render result.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Dict with 'image', 'iteration', 'total_spp', or 'error'/'done' keys
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
