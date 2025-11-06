"""Density of states WorkChain for Fireball.

This workflow runs a single `FireballCalculation` configured to write the
`dens_TOT.dat` file and post-processes it into a handy plot.  It expects the
caller to provide the DOS-specific settings so that the plugin can create the
`dos.optional` helper file (see `FireballCalculation.generate_dos_optional`).
"""

from __future__ import annotations

import base64
import copy
import io
import os
import tempfile
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # Ensure plotting works in headless environments
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction

from aiida_fireball.calculations.fireball import FireballCalculation

# Template used to build the DOS settings block.  The last atom index is filled
# dynamically from the input structure; all other keys can be overridden by
# user input.
DOS_SETTINGS_TEMPLATE: Dict[str, Dict[str, Any]] = {
    "DOS": {
        "first_atom_index": 1,
        "last_atom_index": None,
        "Emin": -10.0,
        "Emax": +10.0,
        "n_energy_steps": 1000,
        "eta": 0.05,
        "iwrttip": 0,
        "Emin_tip": 0.0,
        "Emax_tip": 0.0,
    }
}


class FireballDOSChain(WorkChain):
    """WorkChain that produces a total DOS file and plot from Fireball."""

    @classmethod
    def define(cls, spec):  # noqa: D401
        super().define(spec)

        # Inputs
        spec.input("code", valid_type=orm.Code, help="Fireball executable.")
        spec.input("structure", valid_type=orm.StructureData, help="Structure on which to compute the DOS.")
        spec.input("fdata_remote", valid_type=orm.RemoteData, help="Remote folder containing the Fdata tables.")
        spec.input("kpoints", valid_type=orm.KpointsData, help="K-points for the DOS calculation.")
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            help=(
                "Remote folder from a preceding SCF run. It is required so the plugin can reuse the SCF "
                "results (e.g., Fermi level) when creating the `dos.optional` file."
            ),
        )
        spec.input(
            "parameters",
            valid_type=orm.Dict,
            required=False,
            help="Fireball input parameters (namelists). Optional overrides merged with DOS defaults.",
        )
        spec.input(
            "dos_settings",
            valid_type=orm.Dict,
            required=False,
            help="Overrides for the `DOS` settings block used to build `dos.optional`.",
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            required=False,
            help="Additional calculation settings. The DOS block will be injected automatically.",
        )
        spec.input(
            "plot_labels",
            valid_type=orm.Dict,
            required=False,
            help="Dictionary with keys `x`, `y`, `title` to customise axis labels for the DOS plot.",
        )
        spec.input(
            "calcjob_options",
            valid_type=orm.Dict,
            required=False,
            help="Extra CalcJob metadata options (queue, walltime, etc.).",
        )

        # Outputs
        spec.output("output_parameters", valid_type=orm.Dict, help="Output parameters from the DOS calculation.")
        spec.output("remote_folder", valid_type=orm.RemoteData, help="Remote folder of the DOS calculation.")
        spec.output("retrieved", valid_type=orm.FolderData, help="Retrieved folder containing the raw outputs.")
        spec.output("dos_file", valid_type=orm.SinglefileData, help="The retrieved `dens_TOT.dat` file.")
        spec.output("dos_plot", valid_type=orm.SinglefileData, help="PNG plot generated from `dens_TOT.dat`.")

        # Exit codes
        spec.exit_code(400, "ERROR_SUB_PROCESS_FAILED", message="The FireballCalculation sub process failed.")
        spec.exit_code(401, "ERROR_DOS_FILE_MISSING", message="The `dens_TOT.dat` file was not found in the retrieved folder.")
        spec.exit_code(
            402,
            "ERROR_DOS_FILE_INVALID",
            message="Failed to parse `dens_TOT.dat` – expected at least two columns of numerical data.",
        )
        spec.exit_code(
            403,
            "ERROR_INVALID_DOS_SETTINGS",
            message="Invalid atom indices provided in the DOS settings.",
        )

        # Outline
        spec.outline(
            cls.prepare_inputs,
            cls.run_dos,
            cls.inspect_dos,
            cls.finalise,
        )

    def prepare_inputs(self):
        """Prepare CalcJob inputs and ancillary data."""
        self.ctx.inputs = AttributeDict()
        self.ctx.inputs.code = self.inputs.code
        self.ctx.inputs.structure = self.inputs.structure
        self.ctx.inputs.kpoints = self.inputs.kpoints
        self.ctx.inputs.fdata_remote = self.inputs.fdata_remote
        self.ctx.inputs.parent_folder = self.inputs.parent_folder

        # Build parameters: start from user dict (if provided) and ensure minimal defaults.
        base_parameters: Dict[str, Any] = {}
        if "parameters" in self.inputs:
            base_parameters = copy.deepcopy(self.inputs.parameters.get_dict())
        base_parameters.setdefault("OPTION", {})
        base_parameters["OPTION"].setdefault("ifixcharge", 0)
        base_parameters["OPTION"].setdefault("dt", 0.5)
        base_parameters["OPTION"].setdefault("nstepi", 1)
        base_parameters["OPTION"].setdefault("nstepf", 1)
        base_parameters["OPTION"].setdefault("max_scf_iterations", 200)
        self.ctx.inputs.parameters = orm.Dict(dict=base_parameters)

        # Build DOS settings dynamically from the template and user overrides.
        struct_node: orm.StructureData = self.inputs.structure
        natoms = struct_node.get_ase().get_global_number_of_atoms()
        dos_settings = copy.deepcopy(DOS_SETTINGS_TEMPLATE)
        if "dos_settings" in self.inputs:
            user_settings = self.inputs.dos_settings.get_dict()
            # Accept either a top-level DOS key or a flat dictionary with the parameters.
            user_dos = user_settings.get("DOS", user_settings)
            dos_settings["DOS"].update(user_dos)
        first_index = int(dos_settings["DOS"].get("first_atom_index", 1))
        last_index = dos_settings["DOS"].get("last_atom_index")
        last_index = natoms if last_index in (None, 0) else int(last_index)
        if first_index < 1 or last_index < first_index or last_index > natoms:
            self.report("Invalid DOS atom indices: check first_atom_index/last_atom_index against structure size.")
            return self.exit_codes.ERROR_INVALID_DOS_SETTINGS
        dos_settings["DOS"]["first_atom_index"] = first_index
        dos_settings["DOS"]["last_atom_index"] = last_index
        dos_settings["DOS"].setdefault("n_energy_steps", 1000)

        settings_dict: Dict[str, Any] = {}
        if "settings" in self.inputs:
            settings_dict = copy.deepcopy(self.inputs.settings.get_dict())
        settings_dict.setdefault("DOS", {})
        settings_dict["DOS"].update(dos_settings["DOS"])
        self.ctx.inputs.settings = orm.Dict(dict=settings_dict)

        # Metadata options
        options: Dict[str, Any] = {}
        if "calcjob_options" in self.inputs:
            options.update(self.inputs.calcjob_options.get_dict())
        if options:
            self.ctx.inputs.metadata = {"options": options}

        # Plot labels for matplotlib
        labels = {"x": "Energy (eV)", "y": "Density of states", "title": "Total DOS"}
        if "plot_labels" in self.inputs:
            labels.update(self.inputs.plot_labels.get_dict())
        self.ctx.plot_labels = labels

    def run_dos(self):
        """Submit the FireballCalculation configured for DOS."""
        future = self.submit(FireballCalculation, **self.ctx.inputs)
        return ToContext(dos_calc=future)

    def inspect_dos(self):
        """Handle the finished CalcJob."""
        calc = self.ctx.dos_calc
        if not calc.is_finished_ok:
            self.report(f"DOS FireballCalculation failed with exit status {calc.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        return None

    def finalise(self):
        """Expose CalcJob outputs, extract `dens_TOT.dat`, and build the plot."""
        calc = self.ctx.dos_calc
        outputs = calc.outputs

        if "output_parameters" in outputs:
            self.out("output_parameters", outputs.output_parameters)
        if "remote_folder" in outputs:
            self.out("remote_folder", outputs.remote_folder)
        if "retrieved" in outputs:
            self.out("retrieved", outputs.retrieved)

        retrieved = outputs.retrieved
        try:
            dos_content = retrieved.get_object_content("dens_TOT.dat")
        except Exception:  # pragma: no cover - defensive
            return self.exit_codes.ERROR_DOS_FILE_MISSING

        dos_file = _store_text_file(orm.Str("dens_TOT.dat"), orm.Str(dos_content))
        self.out("dos_file", dos_file)

        try:
            data = np.loadtxt(io.StringIO(dos_content))
        except Exception as exc:  # pragma: no cover - defensive
            self.report(f"Unable to parse dens_TOT.dat: {exc}")
            return self.exit_codes.ERROR_DOS_FILE_INVALID

        if data.ndim == 1:
            # Single column only – cannot produce a x/y plot.
            return self.exit_codes.ERROR_DOS_FILE_INVALID

        if data.shape[1] < 2:
            return self.exit_codes.ERROR_DOS_FILE_INVALID

        x_values = data[:, 0]
        y_values = data[:, 1]

        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        ax.set_xlabel(self.ctx.plot_labels.get("x", "Energy (eV)"))
        ax.set_ylabel(self.ctx.plot_labels.get("y", "Density of states"))
        ax.set_title(self.ctx.plot_labels.get("title", "Total DOS"))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)

        encoded_png = base64.b64encode(buffer.getvalue()).decode("ascii")
        dos_plot = _store_binary_file(orm.Str("dens_TOT.png"), orm.Str(encoded_png))
        self.out("dos_plot", dos_plot)

        return None


@calcfunction
def _store_text_file(filename: orm.Str, content: orm.Str) -> orm.SinglefileData:
    """Persist a text file as SinglefileData via a calcfunction to preserve provenance."""
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write(content.value)
        temp_path = handle.name
    try:
        node = orm.SinglefileData(file=temp_path, filename=filename.value)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return node


@calcfunction
def _store_binary_file(filename: orm.Str, blob_b64: orm.Str) -> orm.SinglefileData:
    """Persist base64-encoded binary content as SinglefileData via a calcfunction."""
    content = base64.b64decode(blob_b64.value.encode("ascii"))
    with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
        handle.write(content)
        temp_path = handle.name
    try:
        node = orm.SinglefileData(file=temp_path, filename=filename.value)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return node
