"""Relaxation WorkChain for Fireball.

This workchain drives one or more FireballCalculation runs to relax a structure.
It monitors the total energy and stops when the energy change between iterations
is below a given threshold or when the maximum number of iterations is reached.

Assumptions:
- Fireball can perform ionic relaxation within a single CalcJob when configured
  appropriately via the input parameters (e.g., quenching/minimization modes).
- If additional iterations are required, we re-launch with the last relaxed
  structure and optionally reuse the previous remote folder for restart files.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_
from aiida.common import AttributeDict

from aiida_fireball.calculations.fireball import FireballCalculation


class FireballRelaxChain(WorkChain):
    """WorkChain that relaxes a structure using FireballCalculation."""

    @classmethod
    def define(cls, spec):  # noqa: D401
        super().define(spec)

        # Inputs
        spec.input("code", valid_type=orm.Code, help="Fireball executable.")
        spec.input("structure", valid_type=orm.StructureData, help="Initial structure to relax.")
        spec.input("parameters", valid_type=orm.Dict, help="Fireball input parameters (namelists).")
        spec.input("kpoints", valid_type=orm.KpointsData, help="K-points for Fireball.")
        spec.input(
            "fdata_remote",
            valid_type=orm.RemoteData,
            help="Remote folder pointing to Fireball Fdata directory to symlink.",
        )
        spec.input("settings", valid_type=orm.Dict, required=False, help="Additional calculation settings.")
        spec.input(
            "relax",
            valid_type=orm.Dict,
            required=False,
            help=(
                "Convenience parameters for relaxation. Any keys provided will be merged into Fireball namelists: "
                "all keys go to OPTION except 'energytol'/'forcetol' which go to QUENCH."
            ),
        )

        # Control of the relaxation loop
        spec.input("max_iterations", valid_type=orm.Int, default=lambda: orm.Int(5), help="Maximum number of iterations.")
        spec.input(
            "energy_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0e-4),
            help="Convergence threshold on |ΔE| in eV between iterations.",
        )
        spec.input(
            "withmpi",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Run CalcJobs with MPI if supported by the code.",
        )
        spec.input(
            "resources",
            valid_type=orm.Dict,
            required=False,
            help="Scheduler resources to use. Defaults to 1 machine, 1 core.",
        )
        spec.input(
            "calcjob_options",
            valid_type=orm.Dict,
            required=False,
            help=(
                "Extra CalcJob metadata.options to propagate to each FireballCalculation (e.g. queue_name, "
                "max_wallclock_seconds, account, qos, prepend_text, append_text, etc.). If a key is also provided "
                "by dedicated inputs (withmpi/resources), those serve as defaults and can be overridden here."
            ),
        )
        spec.input(
            "max_restarts",
            valid_type=orm.Int,
            default=lambda: orm.Int(2),
            help="Maximum number of automatic restarts on recoverable errors (e.g. out-of-walltime).",
        )
        spec.input(
            "retry_on_parse_error",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Retry once on parse/stdout related errors before failing.",
        )
        spec.input(
            "max_parse_retries",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Maximum number of retries for parse/stdout related errors.",
        )
        spec.input(
            "restart_symlink",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=(
                "When True, use symlinks for restart files from the parent folder; when False (default), copy the "
                "files. Copying avoids SFTP symlink limitations on some clusters."
            ),
        )

        # Outputs
        spec.output("relaxed_structure", valid_type=orm.StructureData, help="Final relaxed structure.")
        spec.output("final_output_parameters", valid_type=orm.Dict, help="Output parameters from last calculation.")
        spec.output("final_output_trajectory", valid_type=orm.TrajectoryData, required=False, help="Trajectory from last calculation, if available.")
        spec.output("convergence_info", valid_type=orm.Dict, help="Convergence details: energies, iterations, converged flag.")

        # Exit codes
        spec.exit_code(400, "ERROR_SUB_PROCESS_FAILED", message="A FireballCalculation sub process failed.")

        # Outline
        spec.outline(
            cls.setup,
            while_(cls.should_run_next)(
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.results,
        )

    def setup(self):
        """Initialize the context and prepare first iteration inputs."""
        self.ctx.iteration = 0
        self.ctx.max_iterations = int(self.inputs.max_iterations.value)
        self.ctx.energy_threshold = float(self.inputs.energy_threshold.value)

        self.ctx.current_structure = self.inputs.structure
        self.ctx.previous_energy = None
        self.ctx.energies = []
        self.ctx.converged = False
        self.ctx.parent_folder = None
        self.ctx.restarts = 0
        self.ctx.parse_retries = 0

        # Compute default resources if not provided
        if "resources" in self.inputs:
            self.ctx.resources = self.inputs.resources.get_dict()
        else:
            self.ctx.resources = {"num_machines": 1, "num_cores_per_machine": 1}

    def should_run_next(self):
        """Return True to continue iterations."""
        return (self.ctx.iteration < self.ctx.max_iterations) and (not self.ctx.converged)

    def run_calculation(self):
        """Submit a FireballCalculation for the current structure."""
        self.ctx.iteration += 1

        # Build inputs for CalcJob
        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.structure = self.ctx.current_structure
        # Merge base parameters with relax conveniences
        inputs.parameters = self._build_parameters()
        inputs.kpoints = self.inputs.kpoints
        inputs.fdata_remote = self.inputs.fdata_remote
        if "settings" in self.inputs:
            settings_dict = dict(self.inputs.settings.get_dict())
        else:
            settings_dict = {}

        # On subsequent iterations, try to reuse the previous remote folder via restart files
        if self.ctx.parent_folder is not None:
            # Control restart file handling: symlink vs copy
            # Default is to copy (more portable), can be overridden by input 'restart_symlink'
            use_symlink = bool(self.inputs.restart_symlink.value)
            settings_dict.setdefault("PARENT_FOLDER_SYMLINK", use_symlink)

        if settings_dict:
            inputs.settings = orm.Dict(settings_dict)

        if self.ctx.parent_folder is not None:
            inputs.parent_folder = self.ctx.parent_folder

        # Options
        # Build CalcJob options with precedence: calcjob_options overrides defaults
        options = {}
        if "calcjob_options" in self.inputs:
            try:
                options.update(self.inputs.calcjob_options.get_dict())
            except Exception:  # pragma: no cover - defensive
                pass
        options.setdefault("withmpi", bool(self.inputs.withmpi.value))
        options.setdefault("resources", self.ctx.resources)
        inputs.metadata = {"options": options}

        future = self.submit(FireballCalculation, **inputs)
        self.report(f"Submitted FireballCalculation<{future.pk}> (iteration {self.ctx.iteration})")
        return ToContext(last_calc=future)

    def _build_parameters(self) -> orm.Dict:
        """Merge base parameters with optional relax dict into proper namelists.

        Generic mapping applied:
        - OPTION: all keys from `relax` except 'energytol' and 'forcetol'
        - QUENCH: 'energytol' and 'forcetol' only
        """
        base = dict(self.inputs.parameters.get_dict())
        relax = dict(self.inputs.relax.get_dict()) if "relax" in self.inputs else {}

        if relax:
            for key, value in relax.items():
                if key in ("energytol", "forcetol"):
                    base.setdefault("QUENCH", {})
                    base["QUENCH"][key] = value
                else:
                    base.setdefault("OPTION", {})
                    base["OPTION"][key] = value

        return orm.Dict(dict=base)

    def inspect_calculation(self):
        """Check the calculation result, update convergence status and carry over state."""
        calc = self.ctx.last_calc
        if not calc.is_finished_ok:
            status = calc.exit_status
            self.report(f"FireballCalculation<{calc.pk}> failed with exit_status={status}")

            # Handle out-of-walltime scenarios with a controlled restart
            if status in (400, 340):  # ERROR_OUT_OF_WALLTIME or ERROR_OUT_OF_WALLTIME_INTERRUPTED
                if self.ctx.restarts < int(self.inputs.max_restarts.value):
                    self.ctx.restarts += 1
                    # Keep using the same parent folder for restart files
                    if "remote_folder" in calc.outputs:
                        self.ctx.parent_folder = calc.outputs.remote_folder
                    # Do not count this failed attempt as an iteration
                    self.ctx.iteration = max(0, self.ctx.iteration - 1)
                    self.report(
                        f"Retrying after walltime (restart {self.ctx.restarts}/{int(self.inputs.max_restarts.value)})"
                    )
                    return  # continue loop
                # exceeded restarts
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED

            # Handle parse/stdout issues with a limited retry
            if status in (302, 310, 311, 312):  # missing, read, parse, incomplete
                if bool(self.inputs.retry_on_parse_error.value) and self.ctx.parse_retries < int(self.inputs.max_parse_retries.value):
                    self.ctx.parse_retries += 1
                    self.ctx.iteration = max(0, self.ctx.iteration - 1)
                    self.report(
                        f"Retrying after parse/stdout error (retry {self.ctx.parse_retries}/{int(self.inputs.max_parse_retries.value)})"
                    )
                    return  # continue loop
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED

            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        # Update current structure from output
        if "output_structure" in calc.outputs:
            self.ctx.current_structure = calc.outputs.output_structure

        # Track energy convergence using total_energy_per_atom from output_parameters only
        energy_per_atom = None
        if "output_parameters" in calc.outputs:
            outp = calc.outputs.output_parameters.get_dict()
            energy_per_atom = outp.get("total_energy_per_atom")

        if energy_per_atom is not None:
            self.ctx.energies.append(energy_per_atom)
            if self.ctx.previous_energy is not None:
                delta_e = abs(energy_per_atom - self.ctx.previous_energy)
                self.report(
                    f"Iteration {self.ctx.iteration}: total_energy_per_atom={energy_per_atom:.8f} eV, ΔE={delta_e:.3e} eV"
                )
                if delta_e <= self.ctx.energy_threshold:
                    self.ctx.converged = True
            self.ctx.previous_energy = energy_per_atom

        # Carry over parent_folder for potential restart in next iteration
        if "remote_folder" in calc.outputs:
            self.ctx.parent_folder = calc.outputs.remote_folder

        # Save pointers to final outputs
        self.ctx.final_calc = calc

    # No parameter tuning on restart: relaunch with the same inputs

    def results(self):
        """Expose final results."""
        calc = getattr(self.ctx, "final_calc", None)
        if calc is None:
            # Should not happen, but fail gracefully
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        # Main outputs: store defensively before returning (noop if already stored)
        if "output_structure" in calc.outputs:
            node = calc.outputs.output_structure
            self.out("relaxed_structure", node if node.is_stored else node.store())
        if "output_parameters" in calc.outputs:
            node = calc.outputs.output_parameters
            self.out("final_output_parameters", node if node.is_stored else node.store())
        if "output_trajectory" in calc.outputs:
            node = calc.outputs.output_trajectory
            self.out("final_output_trajectory", node if node.is_stored else node.store())

        # Convergence info
        info = {
            "converged": bool(self.ctx.converged),
            "iterations": int(self.ctx.iteration),
            "energy_threshold": float(self.ctx.energy_threshold),
            "energies": list(self.ctx.energies),
        }
        self.out("convergence_info", orm.Dict(dict=info).store())
