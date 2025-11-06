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


def test_fireball_parser_transport(
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    generate_inputs,
    tmp_path,
):
    """Test le parsing d'un calcul Fireball avec transport (eta/trans/interaction) et CHARGES."""

    name = "transport"
    entry_point_calc_job = "fireball.fireball"
    entry_point_parser = "fireball.fireball"

    # Prépare le dossier temporaire avec les fichiers nécessaires
    retrieve_temporary_folder = tmp_path / "fireball" / name
    retrieve_temporary_folder.mkdir(parents=True, exist_ok=True)

    # Fichier answer.bas minimal (structure finale)
    (retrieve_temporary_folder / "answer.bas").write_text(
        """2
C 0.0 0.0 0.0
C 1.0 0.0 0.0
"""
    )
    # Fichier CHARGES minimal
    (retrieve_temporary_folder / "CHARGES").write_text("0.1\n0.2\n")
    # Fichier eta.optional
    (retrieve_temporary_folder / "eta.optional").write_text("0.05\n1\n1 5\n")
    # Fichier trans.optional
    (retrieve_temporary_folder / "trans.optional").write_text("TRUE\nFALSE\nTRUE\n1\n-2.0\n2.0\n4\n0.02\n")
    # Fichier interaction.optional
    (retrieve_temporary_folder / "interaction.optional").write_text("0\n10\n1\n1 10\n2\n2,3\n0\n8\n1\n11 18\n3\n12,13,14\n")

    retrieve_temporary_list = [
        "answer.bas",
        "CHARGES",
        "eta.optional",
        "trans.optional",
        "interaction.optional",
    ]

    node = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        name,
        generate_inputs(),
        retrieve_temporary=(str(retrieve_temporary_folder), retrieve_temporary_list),
    )
    parser: Parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(
        node,
        store_provenance=False,
        retrieved_temporary_folder=str(retrieve_temporary_folder),
    )

    # Vérifie que le parsing a réussi
    assert calcfunction.is_finished_ok

    # Vérifie la présence de la structure finale
    assert "output_structure" in results
    structure = results["output_structure"]
    assert structure.get_formula() == "C2"

    # Vérifie la présence des fichiers de transport parsés (optionnel selon ton parser)
    # Par exemple, si tu exposes les résultats comme Dict :
    assert "output_eta" in results or "eta" in results
    assert "output_trans" in results or "trans" in results
    assert "output_interaction" in results or "interaction" in results

    # Vérifie la présence du fichier CHARGES dans le dossier récupéré
    with node.outputs.retrieved.open("CHARGES") as handle:
        charges_content = handle.read()
        assert "0.1" in charges_content and "0.2" in charges_content

    # Copie les fichiers générés dans le dossier courant pour inspection
    import os

    output_dir = os.path.join(os.getcwd(), "fireball_test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    for fname in ["answer.bas", "CHARGES", "eta.optional", "trans.optional", "interaction.optional"]:
        src = retrieve_temporary_folder / fname
        if src.exists():
            with open(src, "r") as fsrc, open(os.path.join(output_dir, fname), "w") as fdst:
                fdst.write(fsrc.read())
