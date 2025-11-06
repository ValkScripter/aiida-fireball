"""SCF WorkChain for Fireball: runs a single SCF calculation on a relaxed structure."""

from aiida import orm
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida_fireball.calculations.fireball import FireballCalculation

class FireballSCFChain(WorkChain):
    """WorkChain that runs a single SCF FireballCalculation on a relaxed structure."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Inputs
        spec.input("code", valid_type=orm.Code, help="Fireball executable.")
        spec.input("structure", valid_type=orm.StructureData, required=False, help="Structure to run SCF on.")
        # Accept both native Python str and AiiDA Str for user convenience
        spec.input(
            "relax_label",
            valid_type=(str, orm.Str),
            required=False,
            help="Label du WorkChain relax dont on veut la dernière structure optimisée.",
        )
        spec.input("fdata_remote", valid_type=orm.RemoteData, help="Remote Fdata folder.")
        spec.input("kpoints", valid_type=orm.KpointsData, help="K-points for Fireball.")
        spec.input("parameters", valid_type=orm.Dict, required=False, help="Fireball input parameters (namelists). Optional, will be merged with SCF defaults.")
        spec.input("settings", valid_type=orm.Dict, required=False, help="Additional calculation settings.")
        spec.input("calcjob_options", valid_type=orm.Dict, required=False, help="Extra CalcJob metadata.options (queue, walltime, etc.)")
        # Outputs
        spec.output("output_parameters", valid_type=orm.Dict, help="Output parameters from SCF calculation.")
        spec.output("output_structure", valid_type=orm.StructureData, required=False, help="Output structure from SCF calculation.")
        spec.output("output_trajectory", valid_type=orm.TrajectoryData, required=False, help="Trajectory from SCF calculation, if available.")
        spec.output("remote_folder", valid_type=orm.RemoteData, help="Remote folder of SCF calculation.")
        spec.output("retrieved", valid_type=orm.FolderData, help="Retrieved folder of SCF calculation.")
        spec.output("fermi_energy", valid_type=orm.Float, required=False, help="Fermi level from SCF calculation.")
        spec.output(
            "scf_summary",
            valid_type=orm.Dict,
            required=False,
            help="Résumé SCF avec quelques scalaires utiles (par ex. fermi_energy).",
        )
        # Outline
        spec.outline(
            cls.setup_structure,
            cls.run_scf,
            cls.results,
        )

    def setup_structure(self):
        """Détermine la structure à utiliser pour le SCF (directe ou via label relax)."""
        if "structure" in self.inputs:
            self.ctx.structure = self.inputs.structure
        elif "relax_label" in self.inputs:
            from aiida.orm import WorkflowNode, QueryBuilder
            # Normalize label to native string in case an AiiDA Str was provided
            label_inp = self.inputs.relax_label
            label_value = label_inp.value if isinstance(label_inp, orm.Str) else label_inp
            qb = QueryBuilder()
            qb.append(WorkflowNode, filters={"label": label_value})
            result = qb.first()
            if not result:
                raise ValueError(f"Aucun WorkChain relax trouvé avec le label {label_value}")
            relax_wc = result[0]
            # On prend l'output 'relaxed_structure' du WorkChain relax
            self.ctx.structure = relax_wc.outputs.relaxed_structure
        else:
            raise ValueError("Il faut fournir soit 'structure', soit 'relax_label' en entrée.")

    def run_scf(self):
        # Merge user parameters with SCF defaults
        base = {}
        if "parameters" in self.inputs:
            base = dict(self.inputs.parameters.get_dict())
        # SCF defaults (can be overridden by user)
        base.setdefault("OPTION", {})
        base["OPTION"].setdefault("ifixcharge", 0)
        base["OPTION"].setdefault("dt", 0.5)
        base["OPTION"].setdefault("nstepi", 1)
        base["OPTION"].setdefault("nstepf", 1)
        base["OPTION"].setdefault("max_scf_iterations", 200)
        parameters = orm.Dict(dict=base)
        # Build inputs
        inputs = {
            "code": self.inputs.code,
            "structure": self.ctx.structure,
            "fdata_remote": self.inputs.fdata_remote,
            "kpoints": self.inputs.kpoints,
            "parameters": parameters,
        }
        if "settings" in self.inputs:
            inputs["settings"] = self.inputs.settings
        # No parent_folder: do NOT use CHARGES from previous calc
        # Options
        options = {}
        if "calcjob_options" in self.inputs:
            options.update(self.inputs.calcjob_options.get_dict())
        inputs["metadata"] = {"options": options} if options else {}
        future = self.submit(FireballCalculation, **inputs)
        return ToContext(scf_calc=future)

    def results(self):
        calc = self.ctx.scf_calc
        # Outputs: propagate all main outputs
        if "output_parameters" in calc.outputs:
            self.out("output_parameters", calc.outputs.output_parameters)
            # Expose Fermi level as a direct output if present
            outp = calc.outputs.output_parameters.get_dict()
            if "fermi_energy" in outp:
                # Create fermi_energy via a calcfunction so it's a derived, stored node
                fermi = _extract_fermi_energy(calc.outputs.output_parameters)
                self.out("fermi_energy", fermi)
            # Also build a small summary dict for convenience
            try:
                summary = _build_scf_summary(calc.outputs.output_parameters)
                self.out("scf_summary", summary)
            except Exception:
                # summary is optional; ignore if it fails
                pass
        if "output_structure" in calc.outputs:
            self.out("output_structure", calc.outputs.output_structure)
        if "output_trajectory" in calc.outputs:
            self.out("output_trajectory", calc.outputs.output_trajectory)
        if "remote_folder" in calc.outputs:
            self.out("remote_folder", calc.outputs.remote_folder)
        if "retrieved" in calc.outputs:
            self.out("retrieved", calc.outputs.retrieved)

@calcfunction
def _extract_fermi_energy(parameters: orm.Dict) -> orm.Float:
    """Extract and return the Fermi energy as a Float via a calcfunction to preserve provenance."""
    data = parameters.get_dict()
    value = data.get("fermi_energy", None)
    if value is None:
        raise ValueError("'fermi_energy' not found in parameters")
    return orm.Float(value)

@calcfunction
def _build_scf_summary(parameters: orm.Dict) -> orm.Dict:
    """Build a compact summary Dict from output_parameters.
    Currently includes fermi_energy if present; can be extended with other scalars.
    """
    data = parameters.get_dict()
    summary: dict = {}
    if "fermi_energy" in data:
        summary["fermi_energy"] = data["fermi_energy"]
    # Add other simple scalars if available without heavy arrays
    for key in ("total_energy", "total_energy_per_atom", "scf_iterations"):
        if key in data and isinstance(data[key], (int, float)):
            summary[key] = data[key]
    return orm.Dict(dict=summary)

