import os
import json
import math
from pathlib import Path
import typer
import questionary
from rich.console import Console
from rich.panel import Panel

# --- IMPORTANT: Adjust these imports based on your actual file structure ---
from nimbuscore import (
    Particle,
    DropletComposition,
    DropletShape,
    HexagonalComposition,
    HexagonalHabit,
    CuboctahedralComposition,
    CuboctahedralHabit,
    create_atmospheric_phase, generate_mitsuba_xml
)

app = typer.Typer(help="Nimbus: Wave-Optics Atmospheric Phase Generator", add_completion=False)
console = Console()

CONFIG_DIR = Path("configs")
CONFIG_DIR.mkdir(exist_ok=True)
PHASE_DIR = Path("assets/phases")
PHASE_DIR.mkdir(exist_ok=True, parents=True)

# ==========================================
# PRESETS DICTIONARY
# ==========================================
PRESETS = {
    "water": {
        "Standard Cloud Droplet (Rainbows)": [
            {"shape": "sphere", "weight": 1.0, "radius_mean_um": 10.0, "variance": 0.05}],
        "Fine Mist / Fog (Glory focus)": [{"shape": "sphere", "weight": 1.0, "radius_mean_um": 4.0, "variance": 0.15}],
    },
    "ice": {
        "22° Halo (Random Columns)": [
            {"habit": "tumbling", "weight": 1.0, "length_axis_um": 50.0, "width_axis_um": 20.0, "variance": 0.05}],
        "Sun Dogs (Oriented Plates)": [
            {"habit": "plate", "weight": 1.0, "length_axis_um": 20.0, "width_axis_um": 50.0, "variance": 0.05}],
        "Complex Halo (Plates + Columns + Parry)": [
            {"habit": "tumbling", "weight": 0.4, "length_axis_um": 50.0, "width_axis_um": 20.0, "variance": 0.05},
            {"habit": "plate", "weight": 0.4, "length_axis_um": 20.0, "width_axis_um": 50.0, "variance": 0.05},
            {"habit": "parry", "weight": 0.2, "length_axis_um": 40.0, "width_axis_um": 20.0, "variance": 0.05}
        ]
    },
    "co2_ice": {
        "Standard Martian Dust": [{"habit": "tumbling", "weight": 1.0, "a_axis_um": 800.0, "variance": 0.01}],
        "Martian Cloud Mix": [
            {"habit": "tumbling", "weight": 0.7, "a_axis_um": 800.0, "variance": 0.01},
            {"habit": "plate", "weight": 0.3, "a_axis_um": 800.0, "variance": 0.01}
        ]
    }
}


def load_existing_flow() -> tuple:
    """Handles loading existing JSON configs."""
    configs = list(CONFIG_DIR.glob("*.json"))
    if not configs:
        console.print("[yellow]No saved configurations found in /configs. Let's make a new one.[/yellow]")
        return run_wizard(), None

    choices = [f.name for f in configs] + ["<-- Cancel (Generate New)"]
    choice = questionary.select("\nSelect a saved configuration:", choices=choices).ask()

    if choice == "<-- Cancel (Generate New)":
        return run_wizard(), None

    file_path = CONFIG_DIR / choice
    with open(file_path, "r") as f:
        state = json.load(f)
    console.print(f"[green]Loaded configuration from {file_path}[/green]")
    return state, file_path


def run_wizard() -> dict:
    """The interactive Questionary wizard for building the config."""
    console.print("\n[bold cyan]--- Generator Wizard ---[/bold cyan]")

    particle_choice = questionary.select(
        "What type of atmospheric particle are we baking today?",
        choices=[
            "Water Droplet (Earth Rainbow/Glory)",
            "Hexagonal Ice (Earth Halos)",
            "Cuboctahedral CO2 (Mars Dust/Halos)"
        ]
    ).ask()

    state = {"particle_type": "", "material": "", "resolution": "standard", "composition": []}
    preset_keys = []

    if "Water" in particle_choice:
        state["particle_type"], state["material"] = "droplet", "water"
        preset_keys = list(PRESETS["water"].keys())
    elif "Hexagonal" in particle_choice:
        state["particle_type"], state["material"] = "hexagonal", "ice"
        preset_keys = list(PRESETS["ice"].keys())
    elif "CO2" in particle_choice:
        state["particle_type"], state["material"] = "cuboctahedral", "co2_ice"
        preset_keys = list(PRESETS["co2_ice"].keys())

    # --- PRESET OR CUSTOM ---
    build_mode = questionary.select(
        "How do you want to build the composition?",
        choices=["Use a Curated Preset", "Custom Build"]
    ).ask()

    if build_mode == "Use a Curated Preset":
        preset_choice = questionary.select("Select a preset:", choices=preset_keys).ask()
        state["composition"] = PRESETS[state["material"]][preset_choice]
        if state["particle_type"] != "droplet":
            state["sun_elevation_deg"] = 15.0
            state["air_turbulence_factor"] = 0.5
        console.print(f"[green]Loaded Preset: {preset_choice}[/green]")
    else:
        # --- CUSTOM BUILD LOGIC ---
        if state["particle_type"] == "droplet":
            radius = questionary.text("Enter Mean Radius (μm) [Default: 10.0]:", default="10.0").ask()
            variance = questionary.text("Enter Log-Normal Variance [Default: 0.1]:", default="0.1").ask()
            state["composition"].append(
                {"shape": "sphere", "weight": 1.0, "radius_mean_um": float(radius), "variance": float(variance)})

        elif state["particle_type"] == "hexagonal":
            while True:
                habit = questionary.select("Select Crystal Habit:",
                                           choices=["tumbling", "plate", "column_horizontal", "parry"]).ask()
                weight = float(questionary.text("Fractional Weight (e.g., 0.5 for 50%):", default="1.0").ask())
                length = float(questionary.text("Length (c-axis) in μm:", default="50.0").ask())
                width = float(questionary.text("Width (a-axis) in μm:", default="20.0").ask())
                console.print(
                    "[dim italic]Note (Future Work): Geometric raytracing is size-invariant. Base dimensions won't alter the macroscopic phase function without a coupled diffraction model.[/dim italic]")
                variance = float(questionary.text("Log-Normal Variance (Post-Process Blur):", default="0.05").ask())

                state["composition"].append(
                    {"habit": habit, "weight": weight, "length_axis_um": length, "width_axis_um": width,
                     "variance": variance})
                if not questionary.confirm("Add another crystal habit to this mix?").ask(): break
            state["sun_elevation_deg"] = float(questionary.text("Sun Elevation (Degrees):", default="15.0").ask())
            state["air_turbulence_factor"] = float(
                questionary.text("Air Turbulence Factor (0.0 - 1.0):", default="0.5").ask())

        elif state["particle_type"] == "cuboctahedral":
            while True:
                habit = questionary.select("Select Crystal Habit:", choices=["tumbling", "plate"]).ask()
                weight = float(questionary.text("Fractional Weight (e.g., 0.5 for 50%):", default="1.0").ask())
                size = float(questionary.text("Size (a-axis) in μm:", default="800.0").ask())
                console.print(
                    "[dim italic]Note (Future Work): Geometric raytracing is size-invariant. Base dimensions won't alter the macroscopic phase function without a coupled diffraction model.[/dim italic]")
                variance = float(questionary.text("Log-Normal Variance (Post-Process Blur):", default="0.005").ask())

                state["composition"].append({"habit": habit, "weight": weight, "a_axis_um": size, "variance": variance})
                if not questionary.confirm("Add another CO2 habit to this mix?").ask(): break
            state["sun_elevation_deg"] = float(questionary.text("Sun Elevation (Degrees):", default="15.0").ask())
            state["air_turbulence_factor"] = float(
                questionary.text("Air Turbulence Factor (0.0 - 1.0):", default="0.5").ask())

        # --- WEIGHT VALIDATOR ---
        total_weight = sum(c["weight"] for c in state["composition"])
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            console.print(
                f"\n[bold yellow]Wait up! Your habit weights sum to {total_weight:.2f}, not 1.0.[/bold yellow]")
            if questionary.confirm("Do you want me to automatically normalize them to 1.0?").ask():
                for c in state["composition"]: c["weight"] = c["weight"] / total_weight
                console.print("[green]Weights normalized successfully.[/green]")
            else:
                console.print("[red]Aborting. Please run again and check your math![/red]")
                raise typer.Exit(1)

        # Global Resolution
        res_choice = questionary.select(
            "\nSelect computation resolution:",
            choices=["Quick Draft (3 wavelengths, low angles)", "Standard (16 wavelengths, 1800 angles)",
                     "Publication Quality (32 wavelengths, 4096 angles)"]
        ).ask()

        if "Quick" in res_choice:
            state["resolution"] = "draft"
        elif "Standard" in res_choice:
            state["resolution"] = "standard"
        else:
            state["resolution"] = "publication"

        # --- NEW ENGINE CONTROLS ---
        console.print("\n[bold cyan]--- Engine Output Controls ---[/bold cyan]")
        state["threshold"] = float(
            questionary.text("Forward Peak Threshold (Truncation limit):", default="1.0").ask())
        state["generate_polar"] = questionary.confirm("Generate Polar Plot (.png)?", default=True).ask()

        return state


def build_config_from_state(state: dict):
    """Maps the raw JSON state back into your heavily nested Dataclasses."""
    res_map = {
        "draft": {"num_wavelengths": 3, "num_angles": 720, "num_phi_bins": 180},
        "standard": {"num_wavelengths": 16, "num_angles": 1800, "num_phi_bins": 360},
        "publication": {"num_wavelengths": 32, "num_angles": 4096, "num_phi_bins": 2048},
    }
    res_args = res_map.get(state.get("resolution", "standard"))

    if state["particle_type"] == "droplet":
        comps = [
            DropletComposition(shape=DropletShape(c["shape"]), weight=c["weight"], radius_mean_um=c["radius_mean_um"],
                               variance=c["variance"]) for c in state["composition"]]
        return Particle.droplet(composition=comps, **res_args)
    elif state["particle_type"] == "hexagonal":
        comps = [HexagonalComposition(habit=HexagonalHabit(c["habit"]), weight=c["weight"],
                                      length_axis_um=c["length_axis_um"], width_axis_um=c["width_axis_um"],
                                      variance=c["variance"]) for c in state["composition"]]
        return Particle.hexagonal(composition=comps, sun_elevation_deg=state["sun_elevation_deg"],
                                  air_turbulence_factor=state["air_turbulence_factor"], **res_args)
    elif state["particle_type"] == "cuboctahedral":
        comps = [
            CuboctahedralComposition(habit=CuboctahedralHabit(c["habit"]), weight=c["weight"], a_axis_um=c["a_axis_um"],
                                     variance=c["variance"]) for c in state["composition"]]
        return Particle.cuboctahedral(composition=comps, sun_elevation_deg=state["sun_elevation_deg"],
                                      air_turbulence_factor=state["air_turbulence_factor"], **res_args)
    return None


@app.command()
def start(
        config_file: str = typer.Option(None, "--from-file", "-f", help="Path to a saved JSON config"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print config and commands without running engine")
):
    """Main Entry Point."""
    console.print(Panel.fit("[bold cyan]NimbusCore[/bold cyan] v1.0\n[dim]Electromagnetic Wave-Optics Simulator[/dim]"))

    state, file_path = None, None

    # Fast-track path via CLI arguments
    if config_file:
        file_path = Path(config_file)
        if not file_path.exists():
            console.print(f"[bold red]Error:[/] Could not find {file_path}")
            raise typer.Exit(1)
        with open(file_path, "r") as f:
            state = json.load(f)
        console.print(f"[green]Loaded configuration from {file_path}[/green]")
    else:
        # MAIN MENU ROUTING
        action = questionary.select(
            "Welcome to NimbusCore. What would you like to do?",
            choices=["Generate New Phase Function", "Load Existing Configuration"]
        ).ask()

        if action == "Load Existing Configuration":
            state, file_path = load_existing_flow()
        else:
            state = run_wizard()
            file_path = CONFIG_DIR / f"last_run_{state['particle_type']}.json"
            with open(file_path, "w") as f:
                json.dump(state, f, indent=4)
            console.print(f"\n[dim]State saved to {file_path}[/dim]")

    particle_config = build_config_from_state(state)
    result_path = PHASE_DIR / f"{state['material']}_{state['resolution']}.atmphase"

    # Overwrite protection ONLY applies if they generated a new file, not if they intentionally loaded an old one
    if result_path.exists() and not dry_run and action == "Generate New Phase Function":
        if not questionary.confirm(
                f"\n⚠️  [bold red]Wait![/bold red] {result_path} already exists. Overwrite it?").ask():
            console.print("[yellow]Bake aborted.[/yellow]")
            raise typer.Exit(0)

    if dry_run:
        console.print("\n[bold yellow]--- DRY RUN ACTIVE ---[/bold yellow]")
        console.print(particle_config)
        return

    console.print("\n[bold cyan]Igniting Engine...[/bold cyan]")
    try:
        with console.status("[bold green]Computing phase function... Do not close terminal."):
            # ACTUAL ENGINE CALL
            result_dict = create_atmospheric_phase(
                particle_config,
                threshold=state.get("threshold", 1.0),
                generate_polar=state.get("generate_polar", True)
            )
            result_path_actual = result_dict["filename"]

        console.print(f"\n[bold green]SUCCESS![/bold green] Output resolved at: {result_path_actual}")

        # --- THE PAYOFF WIDGETS ---
        console.print("\n[bold cyan]Mitsuba 3 XML Snippet:[/bold cyan]")
        xml_snippet = generate_mitsuba_xml(result_dict)
        console.print(Panel(xml_snippet, border_style="blue"))

        console.print("\n[bold cyan]Python Fast-Track Command:[/bold cyan]")
        console.print(f"[bold white]python nimbus.py start --from-file {file_path}[/bold white]\n")

    except Exception as e:
        console.print(f"\n[bold red]ENGINE FAILURE:[/bold red] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()