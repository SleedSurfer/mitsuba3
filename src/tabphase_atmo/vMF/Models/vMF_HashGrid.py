# Context: Place this either in a new file (vMF_HashGrid.py) or in your main script
# right where the vMF_KDTree class used to be defined.

import drjit as dr
import mitsuba as mi


class vMF_HashGrid:
    def __init__(self, cell_size: float = 0.05, table_size: int = 1000003) -> None:
        self.cell_size = mi.Float(cell_size)
        self.table_size = mi.UInt32(table_size)

        self.p1 = mi.UInt32(73856093)
        self.p2 = mi.UInt32(19349663)
        self.p3 = mi.UInt32(83492791)

        self.sorted_pos = dr.zeros(mi.Vector3f)
        self.sorted_dir = dr.zeros(mi.Vector3f)
        self.sorted_flux = dr.zeros(mi.Float)

        # --- THE FIX ---
        # Instead of passing the Mitsuba array 'invalid_idx' to dr.full,
        # we pass a plain Python integer. DrJit will correctly broadcast
        # it to the type mi.UInt32.

        # We also use 'self.table_size[0]' to ensure we pass a scalar
        # count to the shape argument.

        self.cell_starts = dr.full(mi.UInt32, 0xFFFFFFFF, shape=self.table_size[0])
        self.cell_ends = dr.full(mi.UInt32, 0xFFFFFFFF, shape=self.table_size[0])

        dr.eval(self.cell_starts, self.cell_ends)

    def to_string(self):
        return f"vMF_HashGrid[cell_size={self.cell_size[0]}, table_size={self.table_size[0]}]"

        # Context: Add this method to the vMF_HashGrid class, right under __init__

    def build(self, c_pos: mi.Vector3f, c_dir: mi.Vector3f, c_weight: mi.Float) -> None:
        """
        Takes the flat arrays of active photons, hashes them, sorts them,
        and builds the $O(1)$ lookup tables.
        """
        num_photons = dr.width(c_pos)
        if num_photons == 0:
            print("[vMF_HashGrid] Warning: Received 0 photons. Skipping build.")
            return

        # -------------------------------------------------------------
        # 1. Map to Grid & Hash
        # -------------------------------------------------------------
        grid_p = dr.floor(c_pos / self.cell_size)
        ix = mi.UInt32(grid_p.x)
        iy = mi.UInt32(grid_p.y)
        iz = mi.UInt32(grid_p.z)

        # Bitwise XOR hash modulo the table size.
        # The cast ensures DrJit compiles this into integer math ops.
        photon_hashes = ((ix * self.p1) ^ (iy * self.p2) ^ (iz * self.p3)) % self.table_size

        # -------------------------------------------------------------
        # 2. Sort the Photons
        # This is where the magic happens. dr.argsort gives us the indices
        # that put the array in ascending hash order.
        # -------------------------------------------------------------
        sorted_indices = dr.argsort(photon_hashes)

        self.sorted_pos = dr.gather(mi.Vector3f, c_pos, sorted_indices)
        self.sorted_dir = dr.gather(mi.Vector3f, c_dir, sorted_indices)
        self.sorted_flux = dr.gather(mi.Float, c_weight, sorted_indices)

        # We also need the sorted hashes to find where the runs start/end
        sorted_hashes = dr.gather(mi.UInt32, photon_hashes, sorted_indices)

        # -------------------------------------------------------------
        # 3. Find the Boundaries (Run-Length Encoding)
        # We compare each photon's hash to its neighbor to detect borders.
        # -------------------------------------------------------------
        photon_indices = dr.arange(mi.UInt32, num_photons)

        # Shift right by 1 (clamped to 0) to compare with previous
        prev_indices = dr.maximum(photon_indices - 1, 0)
        prev_hashes = dr.gather(mi.UInt32, sorted_hashes, prev_indices)
        is_run_start = (sorted_hashes != prev_hashes) | (photon_indices == 0)

        # Shift left by 1 (clamped to max) to compare with next
        next_indices = dr.minimum(photon_indices + 1, num_photons - 1)
        next_hashes = dr.gather(mi.UInt32, sorted_hashes, next_indices)
        is_run_end = (sorted_hashes != next_hashes) | (photon_indices == num_photons - 1)

        # -------------------------------------------------------------
        # 4. Scatter into Lookup Tables
        # Reset the tables first so ghost data from the last pass doesn't linger.
        # -------------------------------------------------------------
        invalid_idx = mi.UInt32(0xFFFFFFFF)
        self.cell_starts = dr.full(mi.UInt32, 0xFFFFFFFF, shape=self.table_size[0])
        self.cell_ends = dr.full(mi.UInt32, 0xFFFFFFFF, shape=self.table_size[0])

        # Compress removes the 'False' lanes, leaving only the indices where a run starts
        starts_to_scatter = dr.compress(is_run_start)
        dr.scatter(
            target=self.cell_starts,
            value=dr.gather(mi.UInt32, photon_indices, starts_to_scatter),
            index=dr.gather(mi.UInt32, sorted_hashes, starts_to_scatter)
        )

        ends_to_scatter = dr.compress(is_run_end)
        # +1 because we want the end index to be exclusive for the gather loop later
        dr.scatter(
            target=self.cell_ends,
            value=dr.gather(mi.UInt32, photon_indices, ends_to_scatter) + 1,
            index=dr.gather(mi.UInt32, sorted_hashes, ends_to_scatter)
        )

        # Lock it in so LLVM actually dispatches the compute graph
        dr.eval(self.cell_starts, self.cell_ends, self.sorted_pos, self.sorted_dir, self.sorted_flux)

    def evaluate_surface_caustics(self, si: mi.SurfaceInteraction3f, radius: mi.Float, active: mi.Bool) -> mi.Spectrum:
        """
        Evaluates the photon density at a surface hit point.
        Returns the raw irradiance from the photon map.
        """
        # 1. Find the base grid cell of the query point
        grid_p = dr.floor(si.p / self.cell_size)

        # 2. Flattened 3x3x3 neighborhood offsets
        # Doing this in Python means DrJit unrolls the outer loop during tracing,
        # which is exactly what we want.
        offsets = [mi.Vector3i(x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]

        r2 = radius * radius
        # Accumulated irradiance (or flux, we'll divide by area at the end)
        accumulated_flux = dr.zeros(mi.Spectrum, dr.width(si.p))

        for offset in offsets:
            neighbor_grid_p = grid_p + offset

            # Hash the neighbor coordinate
            ix = mi.UInt32(neighbor_grid_p.x)
            iy = mi.UInt32(neighbor_grid_p.y)
            iz = mi.UInt32(neighbor_grid_p.z)
            h = ((ix * self.p1) ^ (iy * self.p2) ^ (iz * self.p3)) % self.table_size

            # Look up where this cell's photons start and end in the sorted arrays
            start_idx = dr.gather(mi.UInt32, self.cell_starts, h, active)
            end_idx = dr.gather(mi.UInt32, self.cell_ends, h, active)

            # If the cell is empty, start_idx is 0xFFFFFFFF
            valid_cell = active & (start_idx != 0xFFFFFFFF)

            loop_idx = mi.UInt32(start_idx)

            # DrJit while_loop state. Everything modified inside the loop must be passed here.
            loop_state = (loop_idx, end_idx, accumulated_flux, valid_cell)

            def loop_cond(loop_idx, end_idx, accumulated_flux, valid_cell):
                return valid_cell & (loop_idx < end_idx)

            def loop_body(loop_idx, end_idx, accumulated_flux, valid_cell):
                # Gather the photon data
                p_pos = dr.gather(mi.Vector3f, self.sorted_pos, loop_idx, valid_cell)
                p_dir = dr.gather(mi.Vector3f, self.sorted_dir, loop_idx, valid_cell)
                p_flux = dr.gather(mi.Float, self.sorted_flux, loop_idx, valid_cell)

                # Check if it falls within the search radius
                dist2 = dr.squared_norm(p_pos - si.p)
                inside_radius = valid_cell & (dist2 <= r2)

                # Optionally: evaluate the surface BSDF with the photon's incoming direction.
                # Usually, caustics maps assume a diffuse receiver, so we just calculate irradiance
                # and multiply it by the albedo outside this function.
                # But if you want full BSDF evaluation per photon, you do it here:
                # f_r = si.bsdf().eval(mi.BSDFContext(), si, si.to_local(-p_dir), inside_radius)

                # We'll use a simple uniform kernel (just summing the flux).
                # If you want it smoother, multiply p_flux by (1.0 - dist2/r2).
                accumulated_flux += dr.select(inside_radius, p_flux, 0.0)

                loop_idx += 1
                return (loop_idx, end_idx, accumulated_flux, valid_cell)

            # Execute the loop for this specific neighbor cell
            _, _, accumulated_flux, _ = dr.while_loop(
                state=loop_state,
                cond=loop_cond,
                body=loop_body
            )

        # Normalize by the area of the search disk to convert total flux into Irradiance
        area = dr.pi * r2
        irradiance = accumulated_flux / area

        return irradiance