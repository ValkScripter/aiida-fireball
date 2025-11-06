"""Workflow de convergence en points k pour Fireball."""

from aiida import orm
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.common.exceptions import NotExistentAttributeError
from aiida_fireball.workflows.scf import FireballSCFChain

class FireballKpointsChain(WorkChain):
    """Workflow pour la convergence en k-points avec FireballSCFChain."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=orm.StructureData, help="Structure à calculer.")
        # Accepte int natif ou AiiDA Int
        spec.input("k_start", valid_type=(int, orm.Int), help="Valeur initiale du maillage k (ex: 1 pour [1,1,1]).")
        spec.input("k_stop", valid_type=(int, orm.Int), help="Valeur finale du maillage k (ex: 5 pour [5,5,5]).")
        spec.input("k_step", valid_type=(int, orm.Int), default=1, help="Pas d'incrémentation du maillage k.")
        # Expose explicit SCF inputs to avoid type confusion
        spec.input("code", valid_type=orm.Code, help="Code Fireball à utiliser.")
        spec.input("fdata_remote", valid_type=orm.RemoteData, help="Dossier Fdata distant.")
        spec.input("parameters", valid_type=orm.Dict, required=False, help="Paramètres Fireball (Dict).")
        spec.input("settings", valid_type=orm.Dict, required=False, help="Settings additionnels (Dict).")
        spec.input("calcjob_options", valid_type=orm.Dict, required=False, help="Options de soumission CalcJob (Dict).")
        # Accepte float natif ou AiiDA Float
        spec.input(
            "convergence_criterion",
            valid_type=(float, orm.Float),
            required=False,
            default=0.001,
            help="Critère de convergence en eV (par défaut 1 meV).",
        )
        # Note: k_axis removed. By default the mesh is isotropic [k,k,k].
        # Optional: allow fixing some components, e.g. [k,k,1] -> fixed_components = [None, None, 1]
        spec.input(
            "fixed_components",
            valid_type=(list, orm.List),
            required=False,
            help="Optional list of length 3 with int or None to fix components, e.g. [None, None, 1]",
        )
        spec.output("kpoints_energies", valid_type=orm.Dict, help="Dictionnaire {k: energy_per_total_atom}.")
        spec.output("kpoints_converged", valid_type=orm.Dict, help="Dictionnaire avec le k optimal et l'énergie associée.")
        spec.outline(
            cls.run_kpoints_scan,
            cls.analyze_convergence,
        )

    @staticmethod
    def _as_int(value) -> int:
        return int(value.value) if isinstance(value, orm.Int) else int(value)

    @staticmethod
    def _as_float(value) -> float:
        return float(value.value) if isinstance(value, orm.Float) else float(value)

    def run_kpoints_scan(self):
        """Soumet un SCF pour chaque valeur de k et stocke les futures."""
        k_start = self._as_int(self.inputs.k_start)
        k_stop = self._as_int(self.inputs.k_stop)
        k_step = self._as_int(self.inputs.k_step)
        self.ctx.k_list = list(range(k_start, k_stop + 1, k_step))
        self.ctx.scf_futures = {}
        # Determine fixed components if provided
        fixed = None
        if "fixed_components" in self.inputs:
            fixed_inp = self.inputs.fixed_components
            # support orm.List or native list
            if isinstance(fixed_inp, orm.List):
                fixed = [None if v is None else int(v) for v in fixed_inp.get_list()]
            else:
                fixed = [None if v is None else int(v) for v in fixed_inp]

        for k in self.ctx.k_list:
            # Default isotropic mesh [k,k,k]
            kmesh = [k, k, k]
            # Apply fixed components if provided (use their values instead of k)
            if fixed is not None:
                for idx in range(3):
                    if fixed[idx] is not None:
                        kmesh[idx] = fixed[idx]
            kpoints = orm.KpointsData()
            kpoints.set_kpoints_mesh(kmesh)
            scf_builder = FireballSCFChain.get_builder()
            scf_builder.structure = self.inputs.structure
            scf_builder.kpoints = kpoints
            scf_builder.code = self.inputs.code
            scf_builder.fdata_remote = self.inputs.fdata_remote
            if "parameters" in self.inputs:
                scf_builder.parameters = self.inputs.parameters
            if "settings" in self.inputs:
                scf_builder.settings = self.inputs.settings
            if "calcjob_options" in self.inputs:
                scf_builder.calcjob_options = self.inputs.calcjob_options
            future = self.submit(scf_builder)
            self.ctx.scf_futures[k] = future
        return ToContext(**{f"scf_{k}": fut for k, fut in self.ctx.scf_futures.items()})

    def analyze_convergence(self):
        """Analyse les énergies et détermine le k optimal."""
        energies = {}
        for k in self.ctx.k_list:
            scf = getattr(self.ctx, f"scf_{k}")
            # On suppose que le SCF expose scf_summary avec total_energy_per_atom
            e = None
            try:
                summary = scf.outputs.scf_summary
            except NotExistentAttributeError:
                summary = None

            if summary is not None:
                d = summary.get_dict()
                e = d.get("total_energy_per_atom", None)

            if e is None:
                try:
                    out_params = scf.outputs.output_parameters
                except NotExistentAttributeError:
                    out_params = None
                if out_params is not None:
                    d = out_params.get_dict()
                    e = d.get("total_energy_per_atom", None)
            if e is not None:
                energies[k] = e

        # Construire un Dict stocké via un calcfunction (clés en str)
        k_list = orm.List(list=[int(k) for k in energies.keys()])
        e_list = orm.List(list=[float(v) for v in energies.values()])
        energies_dict = _build_kpoints_energies_from_lists(k_list, e_list)
        self.out("kpoints_energies", energies_dict)

        # Recherche du k optimal
        k_opt = None
        criterion = self._as_float(self.inputs.convergence_criterion)
        for k1, k2 in zip(sorted(energies)[:-1], sorted(energies)[1:]):
            if abs(energies[k2] - energies[k1]) < criterion:
                k_opt = k2
                break
        if k_opt is None:
            result = {"k_opt": None}
        else:
            result = {"k_opt": int(k_opt), "energy": float(energies[k_opt])}
        result_dict_input = orm.Dict(dict=result)
        result_dict = _build_kpoints_converged_from_dict(result_dict_input)
        self.out("kpoints_converged", result_dict)


@calcfunction
def _build_kpoints_energies_from_lists(k_list: orm.List, e_list: orm.List) -> orm.Dict:
    """Build a stored Dict mapping str(k) -> energy from two List inputs."""
    keys = [str(int(k)) for k in k_list.get_list()]
    vals = [float(v) for v in e_list.get_list()]
    return orm.Dict(dict=dict(zip(keys, vals)))


@calcfunction
def _build_kpoints_converged_from_dict(data: orm.Dict) -> orm.Dict:
    """Return a stored Dict for the converged k-point information (pass-through)."""
    return orm.Dict(dict=data.get_dict())
