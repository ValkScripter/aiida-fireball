import numpy
from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict

from aiida_fireball.calculations.fireball import FireballCalculation


class TransportScanWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure")
        spec.input("parameters")
        spec.input("kpoints")
        spec.input("fdata_remote")
        spec.input("trans_params", valid_type=Dict)
        spec.input("Emin", valid_type=float)
        spec.input("Emax", valid_type=float)
        spec.input("step", valid_type=float)
        spec.input("bias_values", required=False)  # Ajouté pour le scan sur le bias
        spec.outline(
            cls.init_scan,
            cls.run_next,
            cls.collect_results,
        )
        spec.output("transmission", valid_type=Dict)

    def init_scan(self):
        self.ctx.energies = numpy.arange(self.inputs.Emin, self.inputs.Emax + self.inputs.step, self.inputs.step)
        self.ctx.index = 0
        self.ctx.results = []

    def run_next(self):
        if self.ctx.index >= len(self.ctx.energies):
            return
        E = self.ctx.energies[self.ctx.index]
        trans_content = FireballCalculation.generate_trans_optional_for_energy(self.inputs.trans_params.get_dict(), E)
        with open("trans.optional", "w") as f:
            f.write(trans_content)
        # Générer bias.optional si présent
        if hasattr(self.inputs, "bias_values"):
            bias = self.inputs.bias_values[self.ctx.index]
            bias_content = FireballCalculation.generate_bias_optional({"bias": bias})
            with open("bias.optional", "w") as f:
                f.write(bias_content)
        params = self.inputs.parameters.get_dict().copy()
        params.setdefault("TRANSPORT", {}).setdefault("TRANS", {})
        params["TRANSPORT"]["TRANS"]["Ebottom"] = E
        params["TRANSPORT"]["TRANS"]["Etop"] = E
        future = self.submit(
            FireballCalculation,
            structure=self.inputs.structure,
            parameters=Dict(params),
            kpoints=self.inputs.kpoints,
            fdata_remote=self.inputs.fdata_remote,
        )
        # Stocke chaque calc avec un nom unique
        result = {f"calc_{self.ctx.index}": future}
        self.ctx.index += 1
        return ToContext(**result)

    def collect_results(self):
        for i, E in enumerate(self.ctx.energies):
            calc = self.ctx.get(f"calc_{i}", None)
            go_value = None
            if calc is not None:
                try:
                    retrieved = calc.outputs.retrieved
                    with retrieved.open("conductance.dat", "r") as handle:
                        for line in handle:
                            if line.strip().startswith("Go ="):
                                go_value = float(line.split()[2].replace("E", "e"))
                                break
                except Exception:
                    go_value = calc.outputs.output_parameters.get_dict().get("transmission", None)
            self.ctx.results.append((E, go_value))
        self.out("transmission", Dict(dict={"data": self.ctx.results}))
