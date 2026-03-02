"""
app.py
Main DearPyGui application for Atmospheric Phase Function UI.
"""
import dearpygui.dearpygui as dpg
import numpy as np
from pathlib import Path
import sys
import threading

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CRITICAL: Set Mitsuba variant BEFORE importing any modules that use it
import mitsuba as mi
mi.set_variant('llvm_spectral')  # Set once at module level

# Now safe to import UI modules
from src.ui.state_manager import StateManager
from src.ui.render_thread import RenderManager
from src.ui.bake_thread import BakeManager
from src.ui.preview_scene import create_preview_scene
from src.core.generator.hgfake import save_fake_mie, MieConfig as HGConfig
from src.ui.preview_scene import print_scene_tree

class AtmosphericPhaseApp:
    """
    Main application class for the GUI.
    """

    def __init__(self, megalut_path: str = "assets/MegaLUT_Water.npz"):
        """
        Args:
            megalut_path: Path to MegaLUT file (relative to project root or absolute)
        """
        # Note: Mitsuba variant already set at module level

        # Resolve path relative to project root if not absolute
        if not Path(megalut_path).is_absolute():
            # Project root is 3 levels up from this file: src/ui/app.py
            project_root = Path(__file__).parent.parent.parent
            megalut_path = str(project_root / megalut_path)

        # State manager
        self.state = StateManager(megalut_path)
        self.render_manager = RenderManager()
        self.bake_manager = BakeManager()

        # UI element IDs
        self.polar_plot_id = None
        self.preview_image_id = None
        self.warning_text_id = None
        self.bake_progress_bar_id = None
        self.bake_status_text_id = None

        # Debounce timer for auto-preview
        self._preview_debounce_timer = None
        self._preview_debounce_delay = 0.25  # 250ms

        # Mitsuba scene (pre-loaded)
        self.scene = None
        self._init_mitsuba_scene()

    def _init_mitsuba_scene(self):
        """Pre-compile Mitsuba scene (with error handling)."""
        try:
            print("Initializing Mitsuba scene...")

            # Create dummy phase LUT using existing hgfake.py


            project_root = Path(__file__).parent.parent.parent
            dummy_path = project_root / "assets" / "dummy_phase.bin"

            # Generate if doesn't exist (matching MegaLUT dimensions)
            if not dummy_path.exists():
                print(f"  Creating dummy HG phase LUT at {dummy_path}...")
                hg_config = HGConfig(
                    num_angles=360,        # Match MegaLUT
                    num_wavelengths=8,     # Match MegaLUT
                    min_wavelength=360.0,
                    max_wavelength=830.0,
                    g=0.8                  # Forward scattering
                )
                save_fake_mie(str(dummy_path), hg_config)

            # Create scene using dummy file
            scene_dict = create_preview_scene(
                phase_lut_path=str(dummy_path),
                resolution=(512, 512),
                spp=16
            )
            self.scene = mi.load_dict(scene_dict)

            # Print scene tree for debugging
            print_scene_tree(self.scene)

            print("✓ Scene ready")
        except Exception as e:
            print(f"⚠ Mitsuba scene initialization failed:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {e}")
            import traceback
            traceback.print_exc()
            print("\nPreview rendering will be disabled.")
            self.scene = None

    def _create_line_theme(self, color):
        """Create a theme for colored plot lines."""
        with dpg.theme() as theme_id:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
        return theme_id

    def build_ui(self):
        """Construct the DearPyGui interface."""
        dpg.create_context()
        dpg.create_viewport(title="Atmospheric Phase Function Tool", width=1280, height=720)

        # Main window
        with dpg.window(label="Phase Function Editor", tag="main_window"):
            # Top-level layout: Horizontal split
            with dpg.group(horizontal=True):
                # LEFT COLUMN: Controls
                with dpg.group(width=350):
                    dpg.add_text("Cloud Parameters", color=(100, 200, 255))
                    dpg.add_separator()

                    bounds = self.state.get_bounds()

                    # Effective Radius slider
                    dpg.add_text("Effective Radius (um)")  # Use 'um' instead of Unicode
                    dpg.add_slider_float(
                        tag="slider_reff",
                        default_value=self.state.current_reff,
                        min_value=bounds['reff'][0],
                        max_value=bounds['reff'][1],
                        callback=self._on_slider_change,
                        width=300,
                        format="%.2f um"
                    )

                    # Variance slider
                    dpg.add_text("Variance (spread of droplet sizes)")
                    dpg.add_slider_float(
                        tag="slider_variance",
                        default_value=self.state.current_variance,
                        min_value=bounds['variance'][0],
                        max_value=bounds['variance'][1],
                        callback=self._on_slider_change,
                        width=300,
                        format="%.4f"
                    )

                    dpg.add_separator()
                    dpg.add_text("Note: LWC affects extinction, not phase shape", color=(150, 150, 150))

                    # Warning text (hidden by default)
                    self.warning_text_id = dpg.add_text(
                        "⚠ Geometric Optics Regime",
                        color=(255, 200, 50),
                        show=False
                    )

                    dpg.add_separator()

                    # Action buttons
                    dpg.add_button(
                        label="Preview Render",
                        callback=self._on_preview_click,
                        width=300,
                        height=40
                    )

                    dpg.add_button(
                        label="Bake High-Res LUT",
                        callback=self._on_bake_click,
                        width=300,
                        height=40
                    )

                    # Bake progress
                    self.bake_status_text_id = dpg.add_text("", color=(150, 150, 150))
                    self.bake_progress_bar_id = dpg.add_progress_bar(
                        default_value=0.0,
                        width=300,
                        show=False
                    )

                # RIGHT COLUMN: Visualizations
                with dpg.group():
                    # Top: Phase function plot (log scale, 3 wavelengths)
                    dpg.add_text("Phase Function (Log Scale, 3 Wavelengths)", color=(100, 200, 255))
                    with dpg.plot(label="Phase Plot", height=450, width=-1):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Scattering Angle (degrees)", tag="phase_x_axis")
                        dpg.set_axis_limits("phase_x_axis", 0, 180)

                        # Linear axis since we're plotting log10(phase) values
                        dpg.add_plot_axis(dpg.mvYAxis, label="log10(Phase Function)", tag="phase_y_axis")

                        # Initialize with 3 wavelengths (RGB-like: blue, green, red)
                        angles_deg, phases_3wl = self.state.get_current_phase_multiwavelength()

                        self.polar_plot_ids = []
                        colors = [(0, 100, 255), (0, 200, 100), (255, 50, 50)]  # Blue, Green, Red
                        labels = ["Short λ (Blue)", "Mid λ (Green)", "Long λ (Red)"]

                        for i, (color, label) in enumerate(zip(colors, labels)):
                            line_id = dpg.add_line_series(
                                angles_deg.tolist(),
                                phases_3wl[i].tolist(),
                                label=label,
                                parent="phase_y_axis"
                            )
                            dpg.bind_item_theme(line_id, self._create_line_theme(color))
                            self.polar_plot_ids.append(line_id)

                    dpg.add_separator()

                    # Bottom: Mitsuba preview
                    dpg.add_text("3D Render Preview", color=(100, 200, 255))

                    # Create texture registry
                    with dpg.texture_registry(show=False):
                        # Placeholder image (black)
                        placeholder = np.zeros((512, 512, 4), dtype=np.float32)
                        placeholder[:, :, 3] = 1.0  # Alpha channel
                        dpg.add_raw_texture(
                            width=512,
                            height=512,
                            default_value=placeholder.flatten(),
                            format=dpg.mvFormat_Float_rgba,
                            tag="preview_texture"
                        )

                    self.preview_image_id = dpg.add_image("preview_texture", width=512, height=512)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def _on_slider_change(self, sender, app_data):
        """Callback when any slider moves."""
        # Read slider values
        reff = dpg.get_value("slider_reff")
        variance = dpg.get_value("slider_variance")

        # Update state (LWC doesn't affect phase shape, so we ignore it)
        self.state.set_params(reff, variance, lwc=0.5)

        # Update phase function plot (3 wavelengths, log scale)
        angles_deg, phases_3wl = self.state.get_current_phase_multiwavelength()

        for i, line_id in enumerate(self.polar_plot_ids):
            dpg.set_value(line_id, [angles_deg.tolist(), phases_3wl[i].tolist()])

        # Check geometric optics warning
        threshold_radius = self.state.get_geometric_threshold_radius()
        show_warning = bool(reff >= threshold_radius)  # Ensure bool type
        dpg.configure_item(self.warning_text_id, show=show_warning)

        # Debounced auto-preview trigger
        self._trigger_debounced_preview()

    def _trigger_debounced_preview(self):
        """Trigger preview render after debounce delay (prevents spam on rapid slider dragging)."""
        # Cancel previous timer if exists
        if self._preview_debounce_timer is not None:
            self._preview_debounce_timer.cancel()

        # Only trigger if scene is ready
        if self.scene is None:
            return

        # Start new debounce timer
        self._preview_debounce_timer = threading.Timer(
            self._preview_debounce_delay,
            self._on_preview_click
        )
        self._preview_debounce_timer.start()

    def _on_preview_click(self):
        """Callback when Preview button is clicked."""
        if self.scene is None:
            print("⚠ Preview unavailable: Mitsuba scene failed to initialize")
            return

        try:
            print("Starting preview render...")

            # Get current phase data
            phase_3d = self.state.get_current_phase_3d()

            # Start async render
            self.render_manager.start_render(
                scene=self.scene,
                phase_data=phase_3d,
                spp_per_iteration=4,
                max_iterations=16
            )
        except Exception as e:
            print(f"⚠ Preview render failed: {e}")

    def _on_bake_click(self):
        """Callback when Bake button is clicked."""
        # Use file dialog to get output path
        output_path = dpg.show_item("bake_file_dialog") if dpg.does_item_exist("bake_file_dialog") else None

        # For now, use a default path (file dialog integration can be added)
        # Resolve relative to project root
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "outputs" / f"phase_r{self.state.current_reff:.1f}_v{self.state.current_variance:.3f}.bin"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Starting bake to: {output_path}")

        # Show progress UI
        dpg.configure_item(self.bake_progress_bar_id, show=True)
        dpg.set_value(self.bake_status_text_id, "Baking...")

        # Start bake
        self.bake_manager.start_bake(
            output_path=str(output_path),
            reff=self.state.current_reff,
            variance=self.state.current_variance,
            material=self.state.material,
            num_angles=2048,
            num_wavelengths=64,
            num_samples=256,
            progress_callback=self._on_bake_progress
        )

    def _on_bake_progress(self, status: dict):
        """Callback for bake progress updates."""
        if status.get('status') == 'generating':
            progress = status.get('progress', 0.0)
            dpg.set_value(self.bake_progress_bar_id, progress)
            dpg.set_value(self.bake_status_text_id, f"Generating... {progress*100:.0f}%")
        elif status.get('status') == 'saving':
            dpg.set_value(self.bake_status_text_id, "Saving...")
        elif status.get('status') == 'done':
            dpg.configure_item(self.bake_progress_bar_id, show=False)
            dpg.set_value(self.bake_status_text_id, f"✓ Saved: {status.get('path', '')}")
            print(f"✓ Bake complete: {status.get('path')}")
        elif status.get('status') == 'error':
            dpg.configure_item(self.bake_progress_bar_id, show=False)
            dpg.set_value(self.bake_status_text_id, f"✗ Error: {status.get('error', 'Unknown')}")
            print(f"✗ Bake error: {status.get('error')}")

    def _update_preview_texture(self, image_rgb: np.ndarray):
        """
        Update the preview image texture.

        Args:
            image_rgb: RGB image [H, W, 3] uint8
        """
        # Convert to RGBA float32 for DearPyGui
        h, w = image_rgb.shape[:2]
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = image_rgb.astype(np.float32) / 255.0

        # Update texture
        dpg.set_value("preview_texture", rgba.flatten())

    def run(self):
        """Main event loop."""
        self.build_ui()

        # Main loop with render polling
        while dpg.is_dearpygui_running():
            # Poll render results
            result = self.render_manager.get_latest_result(timeout=0.01)
            if result:
                if 'error' in result:
                    print(f"Render error: {result['error']}")
                elif 'done' in result:
                    print("Render complete")
                elif 'image' in result:
                    self._update_preview_texture(result['image'])

            dpg.render_dearpygui_frame()

        # Cleanup
        self.render_manager.cancel_render()
        self.bake_manager.cancel_bake()
        dpg.destroy_context()


def main():
    """Entry point with error handling."""
    try:
        app = AtmosphericPhaseApp(megalut_path="assets/MegaLUT_Water.npz")
        app.run()
    except FileNotFoundError as e:
        print(f"✗ MegaLUT file not found: {e}")
        print("Please run: python src/backend/lut_generator.py")
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
