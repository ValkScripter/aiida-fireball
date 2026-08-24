# -*- coding: utf-8 -*-
"""Tests for the `FireballCalculation` class."""

import io
import os
import re

import numpy as np
import pytest
from aiida import orm
from aiida.common import datastructures

# from aiida.common.exceptions import InputValidationError
from aiida.plugins import CalculationFactory
from aiida_fireball.calculations.fireball import FireballCalculation


@pytest.fixture(autouse=True)
def add_fireball_entry_point(entry_points):
    """Add the `FireballCalculation` entry point in function scope."""
    entry_points.add(FireballCalculation, "aiida.calculations:fireball.fireball")


@pytest.fixture
def generate_structure_with_tips():
    """Return a `StructureData` with two Au(79) tip groups flanking a central molecule."""

    def _generate_structure_with_tips():
        structure = orm.StructureData(cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
        structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Au", name="Au")
        structure.append_atom(position=(0.0, 0.0, 1.0), symbols="Au", name="Au")  # last atom of first tip -> z1
        structure.append_atom(position=(0.0, 0.0, 5.0), symbols="C", name="C")  # molecule
        structure.append_atom(position=(0.0, 0.0, 9.0), symbols="Au", name="Au")  # first atom of second tip -> z2
        structure.append_atom(position=(0.0, 0.0, 10.0), symbols="Au", name="Au")
        return structure

    return _generate_structure_with_tips


@pytest.fixture
def generate_structure_with_different_tips():
    """Return a `StructureData` with two tip groups of different elements (Pt and Ag) flanking a molecule."""

    def _generate_structure_with_different_tips():
        structure = orm.StructureData(cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
        structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Pt", name="Pt")
        structure.append_atom(position=(0.0, 0.0, 1.0), symbols="Pt", name="Pt")  # last atom of first tip -> z1
        structure.append_atom(position=(0.0, 0.0, 5.0), symbols="C", name="C")  # molecule
        structure.append_atom(position=(0.0, 0.0, 9.0), symbols="Ag", name="Ag")  # first atom of second tip -> z2
        structure.append_atom(position=(0.0, 0.0, 10.0), symbols="Ag", name="Ag")
        return structure

    return _generate_structure_with_different_tips


def test_calculation():
    """Test the `FireballCalculation` load."""
    calc = CalculationFactory("fireball.fireball")
    assert issubclass(calc, FireballCalculation)


@pytest.mark.parametrize(
    ["symlink_restart", "mesh"],
    [
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_fireball_default(
    fixture_sandbox,
    generate_calc_job,
    generate_inputs_fireball,
    file_regression,
    symlink_restart: bool,
    generate_kpoints_mesh,
    generate_kpoints,
    mesh: bool,
):
    """Test a default `FireballCalculation`."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    if symlink_restart:
        inputs["settings"] = orm.Dict(dict={"PARENT_FOLDER_SYMLINK": symlink_restart})
    if mesh:
        inputs["kpoints"] = generate_kpoints_mesh((3, 3, 1))
    else:
        inputs["kpoints"] = generate_kpoints(kpts=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = []
    remote_symlink_list = [(inputs["fdata_remote"].computer.uuid, inputs["fdata_remote"].get_remote_path(), "./Fdata")]

    if symlink_restart:
        remote_symlink_list.extend(
            [
                (
                    inputs["parent_folder"].computer.uuid,
                    os.path.join(inputs["parent_folder"].get_remote_path(), "CHARGES"),
                    "./",
                ),
                (
                    inputs["parent_folder"].computer.uuid,
                    os.path.join(inputs["parent_folder"].get_remote_path(), "*restart*"),
                    "./",
                ),
            ]
        )

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == cmdline_params
    assert sorted(calc_info.remote_symlink_list) == sorted(remote_symlink_list)

    with fixture_sandbox.open("fireball.in") as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(["fireball.in", "aiida.bas", "aiida.lvs", "aiida.kpts"])
    file_regression.check(input_written, encoding="utf-8", extension=".in")

    # Check the content of the bas file
    with fixture_sandbox.open("aiida.bas") as handle:
        bas_written = handle.read()
    file_regression.check(bas_written, encoding="utf-8", extension=".bas")

    # Check the content of the lvs file
    with fixture_sandbox.open("aiida.lvs") as handle:
        lvs_written = handle.read()
    file_regression.check(lvs_written, encoding="utf-8", extension=".lvs")

    # Check the content of the kpts file
    with fixture_sandbox.open("aiida.kpts") as handle:
        kpts_written = handle.read()
    file_regression.check(kpts_written, encoding="utf-8", extension=".kpts")


def test_fireball_fixed_coords(fixture_sandbox, generate_calc_job, generate_inputs_fireball, file_regression):
    """Test a `FireballCalculation` where the `fixed_coords` setting was provided."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["settings"] = orm.Dict(dict={"FIXED_COORDS": [[True, True, False], [False, True, False]]})
    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "FRAGMENTS" in fixture_sandbox.get_content_list()

    with fixture_sandbox.open("FRAGMENTS") as handle:
        input_written = handle.read()

    file_regression.check(input_written, encoding="utf-8", extension=".fragments")


@pytest.mark.parametrize(
    ["fixed_coords", "error_message"],
    [
        ([[True, True], [False, True]], "The `fixed_coords` setting must be a list of lists with length 3."),
        (
            [[True, True, 1], [False, True, False]],
            "All elements in the `fixed_coords` setting lists must be either `True` or `False`.",
        ),
        ([[True, True, False]], "Input structure has 2 sites, but fixed_coords has length 1"),
    ],
)
def test_fireball_fixed_coords_validation(fixture_sandbox, generate_calc_job, generate_inputs_fireball, fixed_coords, error_message):
    """Test the validation for the `fixed_coords` setting."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["settings"] = orm.Dict(dict={"FIXED_COORDS": fixed_coords})

    with pytest.raises(ValueError, match=error_message):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_missing_inputs(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test a `FireballCalculation` with missing required inputs."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    del inputs["fdata_remote"]
    error_message = "Error occurred validating port 'inputs.fdata_remote': "
    error_message += "required value was not provided for 'fdata_remote'"

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_blocked_keywords(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test a `FireballCalculation` with blocked keywords."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["basisfile"] = "test.bas"
    inputs["parameters"] = orm.Dict(parameters)

    error_message = "Cannot specify the 'basisfile' keyword in the 'OPTION' namelist."

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_dos_settings_invalid_key(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test a `FireballCalculation` with DOS settings."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["settings"] = orm.Dict(dict={"DOS": {"first_atom_index": 1, "last_atom_index": 2, "invalid_key": 1}})

    error_message = "Error occurred validating port 'inputs': \
Invalid key 'invalid_key' in the 'DOS' namelist. Valid keys are: \
['first_atom_index', 'last_atom_index', 'Emin', 'Emax', 'n_energy_steps', 'eta', 'iwrttip', 'Emin_tip', 'Emax_tip']"

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


@pytest.mark.parametrize(
    ["dos_params", "error_message"],
    [
        (
            {"first_atom_index": 0, "last_atom_index": 2},
            "Invalid value for 'first_atom_index' in the 'DOS' namelist. It must be between 1 and 2",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 0},
            "Invalid value for 'last_atom_index' in the 'DOS' namelist. It must be between 1 and 2 and greater than 'first_atom_index'",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "n_energy_steps": 0},
            "Invalid value for 'n_energy_steps' in the 'DOS' namelist. It must be greater than 0",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "Emin": 0.0, "Emax": -5.0},
            "Invalid values for 'Emin' and 'Emax' in the 'DOS' namelist. 'Emin' must be less than 'Emax'",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "iwrttip": 2},
            "Invalid value for 'iwrttip' in the 'DOS' namelist. It must be either 0 or 1",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "iwrttip": 1, "Emin_tip": 0.0, "Emax_tip": -5.0},
            "Invalid values for 'Emin_tip' and 'Emax_tip' in the 'DOS' namelist. 'Emin_tip' must be less than 'Emax_tip'",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "eta": -0.1},
            "Invalid value for 'eta' in the 'DOS' namelist. It must be greater than 0",
        ),
        (
            {"first_atom_index": 1, "last_atom_index": 2, "eta": 0.0},
            "Invalid value for 'eta' in the 'DOS' namelist. It must be greater than 0",
        ),
    ],
)
def test_fireball_dos_settings_invalid_value(fixture_sandbox, generate_calc_job, generate_inputs_fireball, dos_params, error_message):
    """Test a `FireballCalculation` with DOS settings."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["settings"] = orm.Dict(dict={"DOS": dos_params})

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_dos_settings(
    fixture_sandbox,
    generate_calc_job,
    generate_inputs_fireball,
    generate_calc_job_node,
    fixture_localhost,
    file_regression,
):
    """Test a `FireballCalculation` with DOS settings."""
    entry_point_name = "fireball.fireball"

    node = generate_calc_job_node(entry_point_name, fixture_localhost, test_name="test_fireball_dos_settings")

    inputs = generate_inputs_fireball()
    inputs["parent_folder"] = node.outputs.remote_folder
    inputs["settings"] = orm.Dict(dict={"DOS": {}})

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "dos.optional" in fixture_sandbox.get_content_list()

    # Check the content of the dos.optional file
    with fixture_sandbox.open("dos.optional") as handle:
        input_written = handle.read()
    file_regression.check(input_written, encoding="utf-8", extension=".dos")

    # Check the content of the fireball.in file
    with fixture_sandbox.open("fireball.in") as handle:
        input_written = handle.read()
    file_regression.check(input_written, encoding="utf-8", extension=".in")


def test_fireball_retrieve_list(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test that `CHARGES` and `conductance.dat` are always included in the `retrieve_list`."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "CHARGES" in calc_info.retrieve_list
    assert "conductance.dat" in calc_info.retrieve_list


def test_fireball_bias(fixture_sandbox, generate_calc_job, generate_inputs_fireball, generate_structure_with_tips, file_regression):
    """Test a `FireballCalculation` with `OPTION.ibias = 1` writes `bias.optional`."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["structure"] = generate_structure_with_tips()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["ibias"] = 1
    inputs["parameters"] = orm.Dict(parameters)
    inputs["bias"] = orm.Float(0.5)

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "bias.optional" in fixture_sandbox.get_content_list()

    with fixture_sandbox.open("bias.optional") as handle:
        bias_written = handle.read()
    file_regression.check(bias_written, encoding="utf-8", extension=".bias")


def test_fireball_bias_different_tip_elements(
    fixture_sandbox, generate_calc_job, generate_inputs_fireball, generate_structure_with_different_tips, file_regression
):
    """Test that `ibias = 1` works when the two tips are made of different, non-gold elements (Pt and Ag)."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["structure"] = generate_structure_with_different_tips()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["ibias"] = 1
    inputs["parameters"] = orm.Dict(parameters)
    inputs["bias"] = orm.Float(0.5)

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "bias.optional" in fixture_sandbox.get_content_list()

    with fixture_sandbox.open("bias.optional") as handle:
        bias_written = handle.read()
    file_regression.check(bias_written, encoding="utf-8", extension=".bias")


def test_fireball_bias_missing_input(fixture_sandbox, generate_calc_job, generate_inputs_fireball, generate_structure_with_tips):
    """Test that `ibias = 1` without a `bias` input raises a validation error."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    inputs["structure"] = generate_structure_with_tips()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["ibias"] = 1
    inputs["parameters"] = orm.Dict(parameters)

    error_message = "The `bias` input is required when `ibias` is set to 1 in the `OPTION` namelist."

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_bias_missing_tips(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test that `ibias = 1` with a structure lacking two Au tip groups raises a validation error."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()  # default structure is 2D-graphene, no Au atoms
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["ibias"] = 1
    inputs["parameters"] = orm.Dict(parameters)
    inputs["bias"] = orm.Float(0.5)

    error_message = "Could not find two separate tip groups flanking a central molecule to compute the bias z1/z2 positions."

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_initial_charges(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test that providing `initial_charges` adds it to the `local_copy_list` as `CHARGES`."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    charges_file = orm.SinglefileData(file=io.BytesIO(b"dummy charges"), filename="CHARGES")
    inputs["initial_charges"] = charges_file

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert (charges_file.uuid, "CHARGES", "CHARGES") in calc_info.local_copy_list


def _generate_transport_settings():
    """Return a valid `settings` dict for `OPTION.itrans = 1`."""
    return {
        "TRANS": {
            "energy": -0.5,
            "imaginary_part": 0.001,
        },
        "INTERACTION": {
            "sample1": {"interval": [1, 2], "n_atoms_tip": 1, "tip_atoms": [1]},
            "sample2": {"interval": [1, 2], "n_atoms_tip": 1, "tip_atoms": [2]},
        },
        "ETA": {
            "eta_value": 0.001,
            "interval": [1, 2],
        },
    }


def test_fireball_transport(fixture_sandbox, generate_calc_job, generate_inputs_fireball, file_regression):
    """Test that `itrans = 1` writes `trans.optional`, `interaction.optional` and `eta.optional` together."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["itrans"] = 1
    inputs["parameters"] = orm.Dict(parameters)
    inputs["settings"] = orm.Dict(_generate_transport_settings())

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    content_list = fixture_sandbox.get_content_list()
    assert "trans.optional" in content_list
    assert "interaction.optional" in content_list
    assert "eta.optional" in content_list

    with fixture_sandbox.open("trans.optional") as handle:
        trans_written = handle.read()
    file_regression.check(trans_written, encoding="utf-8", extension=".trans")

    with fixture_sandbox.open("interaction.optional") as handle:
        interaction_written = handle.read()
    file_regression.check(interaction_written, encoding="utf-8", extension=".interaction")

    with fixture_sandbox.open("eta.optional") as handle:
        eta_written = handle.read()
    file_regression.check(eta_written, encoding="utf-8", extension=".eta")


def test_fireball_interaction_large_system(fixture_sandbox, generate_calc_job, generate_inputs_fireball, file_regression):
    """Test `interaction.optional` generation for a larger, more realistic system (75 atoms total)."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["itrans"] = 1
    inputs["parameters"] = orm.Dict(parameters)

    # `n_atoms` for sample2 is derived from the total atom count of the structure, so the
    # structure must actually contain the 75 atoms the settings below refer to.
    structure = orm.StructureData(cell=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    for i in range(75):
        structure.append_atom(position=(0.0, 0.0, float(i)), symbols="C", name="C")
    inputs["structure"] = structure

    settings = _generate_transport_settings()
    tip_atoms_1 = list(range(1, 6))  # 5 tip atoms at the start of sample1
    tip_atoms_2 = list(range(71, 76))  # 5 tip atoms at the end of sample2
    settings["INTERACTION"] = {
        "sample1": {"interval": [1, 40], "n_atoms_tip": len(tip_atoms_1), "tip_atoms": tip_atoms_1},
        "sample2": {"interval": [41, 75], "n_atoms_tip": len(tip_atoms_2), "tip_atoms": tip_atoms_2},
    }
    inputs["settings"] = orm.Dict(settings)

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert "interaction.optional" in fixture_sandbox.get_content_list()

    with fixture_sandbox.open("interaction.optional") as handle:
        interaction_written = handle.read()
    file_regression.check(interaction_written, encoding="utf-8", extension=".interaction")


def test_fireball_transport_missing_settings(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test that `itrans = 1` without the `TRANS`/`INTERACTION`/`ETA` settings raises a validation error."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["itrans"] = 1
    inputs["parameters"] = orm.Dict(parameters)

    error_message = "The `settings['TRANS']` dictionary is required when `itrans` is set to 1 in the `OPTION` namelist."

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


def test_fireball_transport_inconsistent_tip_atoms(fixture_sandbox, generate_calc_job, generate_inputs_fireball):
    """Test that a mismatched `n_atoms_tip` vs `tip_atoms` length raises a validation error."""
    entry_point_name = "fireball.fireball"

    inputs = generate_inputs_fireball()
    parameters = inputs["parameters"].get_dict()
    parameters.setdefault("OPTION", {})["itrans"] = 1
    inputs["parameters"] = orm.Dict(parameters)

    settings = _generate_transport_settings()
    settings["INTERACTION"]["sample1"]["n_atoms_tip"] = 2  # inconsistent with the single tip atom provided
    inputs["settings"] = orm.Dict(settings)

    error_message = (
        "The declared `n_atoms_tip` (2) for `settings['INTERACTION']['sample1']` does not match the number of `tip_atoms` provided (1)."
    )

    with pytest.raises(ValueError, match=re.escape(error_message)):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)
