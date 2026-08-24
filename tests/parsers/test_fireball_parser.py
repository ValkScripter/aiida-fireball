"""Tests for the `FireballParser`."""

import os

import pytest
from aiida import orm
from aiida.common import AttributeDict
from aiida.parsers import Parser
from aiida_fireball.calculations.fireball import FireballCalculation
from aiida_fireball.parsers.fireball import FireballParser


@pytest.fixture(autouse=True)
def add_fireball_entry_points(entry_points):
    """Add the `FireballCalculation` and `FireballParser` entry points in function scope."""
    entry_points.add(FireballParser, "aiida.parsers:fireball.fireball")
    entry_points.add(FireballCalculation, "aiida.calculations:fireball.fireball")


@pytest.fixture
def generate_inputs(generate_structure, fixture_code, generate_kpoints_mesh, generate_remote_data, fixture_localhost):
    """Return a dictionary with the minimum required inputs for a `FireballCalculation`."""

    def _generate_inputs():
        from aiida.orm import Dict

        parameters = Dict(
            {
                "OPTION": {
                    "nstepi": 1,
                    "nstepf": 100,
                    "ifixcharge": 0,
                    "iquench": -3,
                },
                "OUTPUT": {
                    "iwrtxyz": 1,
                },
            }
        )
        structure = generate_structure("2D-graphene")
        inputs = {
            "code": fixture_code("fireball.fireball"),
            "structure": structure,
            "kpoints": generate_kpoints_mesh((3, 3, 1)),
            "parameters": parameters,
            "fdata_remote": generate_remote_data(computer=fixture_localhost, remote_path="/path/to/fdata"),
            # "parent_folder": generate_remote_data(computer=fixture_localhost, remote_path="/path/to/parent"),
            "metadata": {
                "options": {
                    "resources": {"num_machines": 1, "num_cores_per_machine": 4},
                    "max_wallclock_seconds": 1800,
                    "withmpi": False,
                }
            },
        }
        return AttributeDict(inputs)

    return _generate_inputs


# pylint: disable=redefined-outer-name
def test_fireball_default(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs, data_regression):
    """Test a `fireball` calculation.

    The output is created by running a simple Fireball calculation. This test should test the
    standard parsing of the stdout content and any other relevant output files.
    """
    name = "default"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    retrieve_temporary_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "fireball", name)
    retrieve_temporary_list = [
        "answer.bas",
        "answer.xyz",
    ]

    node = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        name,
        generate_inputs(),
        retrieve_temporary=(retrieve_temporary_folder, retrieve_temporary_list),
    )
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False, retrieved_temporary_folder=retrieve_temporary_folder)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.collection.get_logs_for(node), [log.message for log in orm.Log.collection.get_logs_for(node)]
    assert "output_parameters" in results
    assert "output_structure" in results
    assert "output_trajectory" in results

    output_parameters = results["output_parameters"].get_dict()
    output_structure = results["output_structure"].base.attributes.all
    output_trajectory = results["output_trajectory"].base.attributes.all

    for key, value in output_parameters.items():
        if isinstance(value, float):
            output_parameters[key] = float(value)

    data_regression.check(
        {
            "output_parameters": output_parameters,
            "output_structure": output_structure,
            "output_trajectory": output_trajectory,
        }
    )


def test_fireball_no_retrieved_temporary_folder(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs):
    """Test a `fireball` calculation without a retrieved temporary folder."""
    name = "no_retrieved_temporary_folder"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs())
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_failed, calcfunction.process_state
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER.status


def test_fireball_output_charges(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs, tmp_path):
    """Test that a `CHARGES` file present in the retrieved folder is parsed into `output_charges`."""
    name = "charges"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs())
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False, retrieved_temporary_folder=str(tmp_path))

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "output_charges" in results
    assert isinstance(results["output_charges"], orm.SinglefileData)
    assert results["output_charges"].get_content() == "dummy charges content\n"


def test_fireball_no_output_charges(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs):
    """Test that no `output_charges` is produced when `CHARGES` is absent from the retrieved folder."""
    name = "default"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    retrieve_temporary_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "fireball", name)
    retrieve_temporary_list = ["answer.bas", "answer.xyz"]

    node = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        name,
        generate_inputs(),
        retrieve_temporary=(retrieve_temporary_folder, retrieve_temporary_list),
    )
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False, retrieved_temporary_folder=retrieve_temporary_folder)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "output_charges" not in results


def test_fireball_output_conductance(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs, tmp_path):
    """Test that a `conductance.dat` file present in the retrieved folder is parsed into `output_conductance`."""
    name = "conductance"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs())
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False, retrieved_temporary_folder=str(tmp_path))

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "output_conductance" in results
    assert isinstance(results["output_conductance"], orm.Dict)

    output_conductance = results["output_conductance"].get_dict()
    assert output_conductance["steps"] == [{"step": 1, "energy": 10.0, "transmission": 0.00091731}]
    assert output_conductance["conductance_quantum_go"] == pytest.approx(0.91730568e-03)


def test_fireball_no_output_conductance(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs):
    """Test that no `output_conductance` is produced when `conductance.dat` is absent from the retrieved folder."""
    name = "default"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    retrieve_temporary_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "fireball", name)
    retrieve_temporary_list = ["answer.bas", "answer.xyz"]

    node = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        name,
        generate_inputs(),
        retrieve_temporary=(retrieve_temporary_folder, retrieve_temporary_list),
    )
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False, retrieved_temporary_folder=retrieve_temporary_folder)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "output_conductance" not in results
