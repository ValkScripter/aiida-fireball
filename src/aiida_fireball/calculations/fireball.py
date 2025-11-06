"""General `CalcJob` for Fireball calculations"""

import os

import numpy
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.orm import BandsData, Dict, KpointsData, RemoteData, StructureData, TrajectoryData

from .utils import _lowercase_dict, _uppercase_dict, conv_to_fortran, convert_input_to_namelist_entry
from .validation import validate_cgopt_params, validate_dos_params, validate_fixed_coords


class FireballCalculation(CalcJob):
    """General `CalcJob` for Fireball calculations"""

    _PREFIX = "aiida"
    _DEFAULT_INPUT_FILE = "fireball.in"
    _DEFAULT_OUTPUT_FILE = "aiida.out"
    _DEFAULT_BAS_FILE = "aiida.bas"
    _DEFAULT_LVS_FILE = "aiida.lvs"
    _DEFAULT_KPTS_FILE = "aiida.kpts"
    _FDATA_SUBFOLDER = "./Fdata/"
    _CRASH_FILE = "CRASH"

    # Blocked keywords that are to be specified in the subclass:
    _blocked_keywords = {
        "OPTION": {
            "basisfile": _DEFAULT_BAS_FILE,
            "lvsfile": _DEFAULT_LVS_FILE,
            "kptpreference": _DEFAULT_KPTS_FILE,
            "verbosity": 3,
        }
    }

    # Additional files that should always be retrieved for the specific plugin
    _internal_retrieve_list = []

    # In restarts, will copy not symlink
    _default_symlink_usage = False

    # In restarts, it will copy the following files from the parent folder
    _restart_files_list = ["CHARGES", "*restart*"]

    # In restarts, it will copy the previous folder in the following one
    _restart_copy_to = "./"

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # Inputs
        spec.input("structure", valid_type=StructureData, help="The input structure.")
        spec.input("parameters", valid_type=Dict, help="The input parameters.")
        spec.input("kpoints", valid_type=KpointsData, help="The input kpoints.")
        spec.input("fdata_remote", valid_type=RemoteData, help="Remote folder containing the Fdata files.")
        spec.input("settings", valid_type=Dict, required=False, help="Additional input parameters.")
        spec.input("metadata.options.parser_name", valid_type=str, default="fireball.fireball")
        spec.input("metadata.options.input_filename", valid_type=str, default=cls._DEFAULT_INPUT_FILE)
        spec.input("metadata.options.output_filename", valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)
        spec.input("parent_folder", valid_type=RemoteData, required=False, help="The parent remote folder to restart from.")
        spec.input("metadata.options.resources", valid_type=dict, default=lambda: {"num_machines": 1, "num_cores_per_machine": 1})
        # spec.inputs["metadata"]["options"]["resources"].default = lambda: {
        #     "num_machines": 1,
        #     "num_cores_per_machine": 1,
        # }
        spec.inputs.validator = cls.validate_inputs

        # Outputs
        spec.output(
            "output_parameters",
            valid_type=Dict,
            help="The `output_parameters` output node of the successful calculation.",
        )
        spec.output(
            "output_structure",
            valid_type=StructureData,
            required=False,
            help="The `output_structure` output node of the successful calculation if present.",
        )
        spec.output("output_trajectory", valid_type=TrajectoryData, required=False)
        spec.output(
            "output_band",
            valid_type=BandsData,
            required=False,
            help="The `output_band` output node of the successful calculation if present.",
        )
        spec.output("output_kpoints", valid_type=KpointsData, required=False)
        spec.output("output_atomic_occupations", valid_type=Dict, required=False)
        spec.default_output_node = "output_parameters"

        # Exit codes
        spec.exit_code(301, "ERROR_NO_RETRIEVED_TEMPORARY_FOLDER", message="The retrieved temporary folder could not be accessed.")
        spec.exit_code(
            302,
            "ERROR_OUTPUT_STDOUT_MISSING",
            message="The retrieved folder did not contain the required stdout output file.",
        )
        spec.exit_code(310, "ERROR_OUTPUT_STDOUT_READ", message="The stdout output file could not be read.")
        spec.exit_code(311, "ERROR_OUTPUT_STDOUT_PARSE", message="The stdout output file could not be parsed.")
        spec.exit_code(
            312,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The stdout output file was incomplete probably because the calculation got interrupted.",
        )
        spec.exit_code(400, "ERROR_OUT_OF_WALLTIME", message="The calculation stopped prematurely because it ran out of walltime.")
        spec.exit_code(
            340,
            "ERROR_OUT_OF_WALLTIME_INTERRUPTED",
            message="The calculation stopped prematurely because it ran out of walltime but the job was killed by the "
            "scheduler before the files were safely written to disk for a potential restart.",
        )
        spec.exit_code(350, "ERROR_UNEXPECTED_PARSER_EXCEPTION", message="The parser raised an unexpected exception: {exception}")

    @classmethod
    def validate_inputs(cls, value, port_namespace):  # pylint: disable=too-many-branches
        """Validate the entire inputs' namespace."""

        # Wrapping processes may choose to exclude certain input ports, in which case we can't validate. If the ports
        # have been excluded, and so are no longer part of the ``port_namespace``, skip the validation.
        # if any(key not in port_namespace for key in ("fdata_remote", "structure")):
        #     return None

        # At this point, both ports are part of the namespace, and both are required so return an error message if any
        # of the two is missing.
        # for key in ("fdata_remote", "structure"):
        #     if key not in value:
        #         return f"required value was not provided for the `{key}` namespace."

        messages = []

        if "settings" in value:
            settings: dict = _uppercase_dict(value["settings"].get_dict(), dict_name="settings")
        else:
            settings = {}

        if "parameters" in value:
            parameters: dict = _uppercase_dict(value["parameters"].get_dict(), dict_name="parameters")
        else:
            parameters = {}

        if "OPTION" in parameters and "iquench" in parameters["OPTION"] and parameters["OPTION"]["iquench"] in [-5, -4]:
            if "CGOPT" not in settings:
                settings.setdefault("CGOPT", {})

        # Validate the FIXED_COORDS setting
        messages.extend(validate_fixed_coords(value, settings, parameters))

        # Validate the DOS settings
        messages.extend(validate_dos_params(value, settings, parameters))

        # Validate the CGOPT settings
        messages.extend(validate_cgopt_params(value, settings, parameters))

        # Update settings with the new values
        value["settings"] = Dict(settings)
        # Update parameters with the new values
        value["parameters"] = Dict(parameters)

        return "\n".join(messages) if messages else None

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation job for submission by generating input files and parameters."""

        settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name="settings") if "settings" in self.inputs else {}

        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        # Create Fdata subfolder
        folder.get_subfolder(self._FDATA_SUBFOLDER, create=True)

        # Symlink all files in the Fdata remote folder
        remote_symlink_list.append(
            (
                self.inputs.fdata_remote.computer.uuid,
                os.path.join(self.inputs.fdata_remote.get_remote_path(), "*"),
                self._FDATA_SUBFOLDER,
            )
        )

        def write_in_folder(folder: Folder, filename: str, content: str):
            """Helper function to write content to a file in the given folder."""
            with folder.open(filename, "w") as handle:
                handle.write(content)

        # Write the input file
        write_in_folder(folder, self.metadata.options.input_filename, self.generate_input(self.inputs.parameters.get_dict()))

        # Write the bas file
        write_in_folder(folder, self._DEFAULT_BAS_FILE, self.generate_bas(self.inputs.structure))

        # Write the lvs file
        write_in_folder(folder, self._DEFAULT_LVS_FILE, self.generate_lvs(self.inputs.structure))

        # Write the kpts file
        write_in_folder(folder, self._DEFAULT_KPTS_FILE, self.generate_kpts(self.inputs.kpoints, self.inputs.structure))

        # Write the constraints file if the FIXED_COORDS setting is provided
        if "FIXED_COORDS" in settings and settings["FIXED_COORDS"] is not None:
            fixed_coords = settings.pop("FIXED_COORDS")
            fixed_coords = numpy.array(fixed_coords)
            write_in_folder(folder, "FRAGMENTS", self.generate_constraints(self.inputs.structure, fixed_coords))

        # Write the dos.optional file if the DOS setting is provided
        if "DOS" in settings and settings["DOS"] is not None:
            dos_params: dict = settings.pop("DOS")
            parent_output_parameters = self.inputs.parent_folder.creator.outputs.output_parameters.get_dict()
            write_in_folder(folder, "dos.optional", self.generate_dos_optional(dos_params, parent_output_parameters.get("fermi_energy", 0.0)))

        # Write the cgopt.optional file if the CGOPT setting is provided
        if "CGOPT" in settings and settings["CGOPT"] is not None:
            cgopt_params: dict = settings.pop("CGOPT")
            write_in_folder(folder, "cgopt.optional", self.generate_cgopt_optional(cgopt_params))

        # operations for restart
        symlink = settings.pop("PARENT_FOLDER_SYMLINK", self._default_symlink_usage)  # a boolean
        if symlink:
            if "parent_folder" in self.inputs:
                for file_name in self._restart_files_list:
                    remote_symlink_list.append(
                        (
                            self.inputs.parent_folder.computer.uuid,
                            os.path.join(self.inputs.parent_folder.get_remote_path(), file_name),
                            self._restart_copy_to,
                        )
                    )
        elif "parent_folder" in self.inputs:
            for file_name in self._restart_files_list:
                remote_copy_list.append(
                    (
                        self.inputs.parent_folder.computer.uuid,
                        os.path.join(self.inputs.parent_folder.get_remote_path(), file_name),
                        self._restart_copy_to,
                    )
                )

        # Prepare the code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = []  # Fireball reads directly from the input file 'fireball.in'
        codeinfo.stdout_name = self.inputs.metadata.options.output_filename

        # Prepare the calculation info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]

        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list

        # Retrieve by default the output file and any additional files specified in the settings
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self.metadata.options.output_filename)
        calcinfo.retrieve_list.append(self._CRASH_FILE)
        calcinfo.retrieve_list.extend(settings.pop("ADDITIONAL_RETRIEVE_LIST", []))
        calcinfo.retrieve_list.extend(self._internal_retrieve_list)

        # Retrieve temporary files
        calcinfo.retrieve_temporary_list = []
        calcinfo.retrieve_temporary_list.append("answer.*")

        return calcinfo

    @classmethod
    def generate_input(cls, parameters):
        """Generate the input data for the calculation."""
        file_lines = []

        input_params = _uppercase_dict(parameters, dict_name="parameters")
        input_params = {k: _lowercase_dict(v, dict_name=k) for k, v in input_params.items()}

        blocked_keywords = _uppercase_dict(cls._blocked_keywords, dict_name="blocked_keywords")
        blocked_keywords = {k: _lowercase_dict(v, dict_name=k) for k, v in blocked_keywords.items()}

        # Check if there are blocked keywords in the input parameters
        for namelist_name, namelist in blocked_keywords.items():
            for key, value in namelist.items():
                if key in input_params.get(namelist_name, {}):
                    raise ValueError(f"Cannot specify the '{key}' keyword in the '{namelist_name}' namelist.")

        # Add keywords from the blocked keywords which have values that are not None to the input parameters
        for namelist_name, namelist in blocked_keywords.items():
            for key, value in namelist.items():
                if value is not None:
                    if namelist_name not in input_params:
                        input_params[namelist_name] = {}
                    input_params[namelist_name][key] = value

        # Write the namelists
        for namelist_name, namelist in sorted(input_params.items()):
            file_lines.append(f"&{namelist_name}")
            for key, value in sorted(namelist.items()):
                file_lines.append(convert_input_to_namelist_entry(key, value)[:-1])
            file_lines.append("&END")

        return "\n".join(file_lines) + "\n"

    @classmethod
    def generate_bas(cls, structure: StructureData):
        """Generate the bas file for the calculation (atomic positions)."""
        ase_structure = structure.get_ase()
        file_lines = []
        file_lines.append(f"\t{len(ase_structure):3d}")

        for atom in ase_structure:
            file_lines.append(
                f"{atom.number:3d} \
{conv_to_fortran(atom.position[0])} \
{conv_to_fortran(atom.position[1])} \
{conv_to_fortran(atom.position[2])}"
            )

        return "\n".join(file_lines) + "\n"

    @classmethod
    def generate_lvs(cls, structure: StructureData):
        """Generate the lvs file for the calculation (lattice vectors)."""
        ase_structure = structure.get_ase()
        file_lines = []
        for vector in ase_structure.cell.array:
            file_lines.append(f"{conv_to_fortran(vector[0])} {conv_to_fortran(vector[1])} {conv_to_fortran(vector[2])}")

        return "\n".join(file_lines) + "\n"

    @classmethod
    def generate_kpts(cls, kpoints: KpointsData, structure: StructureData):
        """Generate the kpts file for the calculation (write list of cartesian k-points)."""
        file_lines = []
        temp_kpoints = KpointsData()
        temp_kpoints.set_cell_from_structure(structure)
        if "kpoints" in kpoints.get_arraynames():
            scaled_kpoints = kpoints.get_kpoints()
        else:
            scaled_kpoints = kpoints.get_kpoints_mesh(print_list=True)
        temp_kpoints.set_kpoints(scaled_kpoints, cartesian=False, weights=[1.0 / len(scaled_kpoints)] * len(scaled_kpoints))
        cartesian_kpoints, weights = temp_kpoints.get_kpoints(cartesian=True, also_weights=True)
        file_lines.append(f"\t{len(cartesian_kpoints):5d}")
        for kpt, weight in zip(cartesian_kpoints, weights):
            file_lines.append(f"{conv_to_fortran(kpt[0])} {conv_to_fortran(kpt[1])} {conv_to_fortran(kpt[2])}\t{weight:.10f}")

        return "\n".join(file_lines) + "\n"

    def generate_constraints(self, structure: StructureData, fixed_coords: numpy.ndarray):
        """Generate the constraints file for the calculation."""
        ase_structure = structure.get_ase()
        file_lines = []
        file_lines.append("0")
        file_lines.append("1")
        file_lines.append(f"{len(ase_structure):3d}")

        for i, fix in zip(range(len(ase_structure)), fixed_coords):
            file_lines.append(f"{i + 1:3d} {int(fix[0]):1d} {int(fix[1]):1d} {int(fix[2]):1d}")

        return "\n".join(file_lines) + "\n"

    def generate_dos_optional(self, dos_params: dict, fermi_energy: float) -> str:
        """Generate the content of the file dos.optional"""
        file_lines = []
        file_lines.append("1.0")
        file_lines.append(f"{dos_params['first_atom_index']:3d}\t{dos_params['last_atom_index']:3d}\t! First and last atom index")
        file_lines.append(f"{dos_params['n_energy_steps']}\t! Number of energy steps")
        file_lines.append(
            f"{dos_params['Emin'] + fermi_energy:.6f}\t{(dos_params['Emax'] - dos_params['Emin']) / dos_params['n_energy_steps']}\t! Emin and dE"
        )
        file_lines.append(f"{dos_params['iwrttip']:1d}\t! iwrttip=1 writes the file tip_e_str.inp")
        file_lines.append(f"{dos_params['Emin_tip']:6f}\t{dos_params['Emax_tip']:6f}\t! Emin_tip and Emax_tip")
        file_lines.append(f"{dos_params['eta']:.6f}\t! eta")

        return "\n".join(file_lines) + "\n"

    def generate_cgopt_optional(self, cgopt_params: dict) -> str:
        """Generate the content of the file cgopt.optional"""
        file_lines = []
        file_lines.append(f"{conv_to_fortran(cgopt_params['drmax'])} \t! drmax = Maximum atomic displacement")
        file_lines.append(f"{conv_to_fortran(cgopt_params['dummy'])} \t! dummy = Scale to reduce the search step if e1 < e2")
        file_lines.append(f"{conv_to_fortran(cgopt_params['energy_tol'])} \t! energy_tol = Energy tolerance for the search")
        file_lines.append(f"{conv_to_fortran(cgopt_params['force_tol'])} \t! force_tol = Force tolerance for the search")
        file_lines.append(f"{conv_to_fortran(cgopt_params['max_steps'])} \t! max_steps = Maximum number of CG steps")
        file_lines.append(f"{conv_to_fortran(cgopt_params['min_int_steps'])} \t! min_int_steps = Minimum number of steps in the CG loop")
        file_lines.append(f"{conv_to_fortran(cgopt_params['switch_MD'])} \t! switch_MD = Number of FIRE downhill steps after BFGS minimization fails")

        return "\n".join(file_lines) + "\n"
