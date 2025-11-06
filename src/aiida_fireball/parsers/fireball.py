"""Parser implementation for the FireballCalculation calculation job class."""

import os
import re
from typing import Optional, Tuple

import numpy as np
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ExitCode
from aiida.parsers import Parser
from ase import Atoms

from . import get_logging_container
from .raw import parse_raw_stdout


class FireballParser(Parser):
    """`Parser` implementation for the `FireballCalculation` calculation job class."""

    success_string = "(FIREBALL RUNTIME)|(That`sall for now)"

    def parse(self, **kwargs) -> Optional[ExitCode]:
        """Parse outputs and store results in the database."""
        logs = get_logging_container()

        # Parse the stdout content
        parsed_data, logs = self.parse_stdout(logs)
        self.emit_logs(logs, ignore=None)

        # Absolute path to the retrieved temporary folder
        retrieved_temporary_folder: str = kwargs.get("retrieved_temporary_folder", None)
        if not retrieved_temporary_folder:
            return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER)

        # Parse output structure from 'answer.bas' file in the retrieved_temporary_folder
        # and store it in the 'output_structure' output node
        output_structure, logs = self.parse_output_structure(retrieved_temporary_folder, parsed_data.get("rescale_factor", 1.0), logs)
        self.emit_logs(logs, ignore=None)
        if output_structure:
            self.out("output_structure", output_structure)

        # Add the volume of the output structure to the output parameters
        if output_structure:
            output_volume = output_structure.get_cell_volume()
            parsed_data["volume"] = output_volume
        self.out("output_parameters", orm.Dict(parsed_data))

        # Parse output trajectory from 'answer.xyz' file in the retrieved_temporary_folder
        # and store it in the 'output_trajectory' output node
        output_trajectory, logs = self.parse_output_trajectory(retrieved_temporary_folder, parsed_data.get("rescale_factor", 1.0), logs)
        self.emit_logs(logs, ignore=None)
        if output_trajectory:
            self.out("output_trajectory", output_trajectory)

    def parse_stdout(self, logs: AttributeDict) -> Tuple[dict, AttributeDict]:
        """Parse the stdout content of a Fireball calculation."""
        output_filename = self.node.get_option("output_filename")

        if output_filename not in self.retrieved.base.repository.list_object_names():
            logs.error.append("ERROR_OUTPUT_STDOUT_MISSING")
            return {}, logs

        try:
            with self.retrieved.open(output_filename, "r") as handle:
                stdout = handle.read()
        except OSError as exception:
            logs.error.append("ERROR_OUTPUT_STDOUT_READ")
            logs.error.append(exception)
            return {}, logs

        try:
            parsed_data, logs = self._parse_stdout_base(stdout, logs)
        except (ValueError, KeyError, OSError) as exception:
            logs.error.append("ERROR_OUTPUT_STDOUT_PARSE")
            logs.error.append(exception)
            return {}, logs

        return parsed_data, logs

    @classmethod
    def _parse_stdout_base(cls, stdout: str, logs: AttributeDict) -> Tuple[dict, AttributeDict]:
        """
        This function only checks for basic content like FIREBALL RUNTIME

        :param stdout: the stdout content as a string.
        :returns: tuple of two dictionaries, with the parsed data and log messages, respectively.
        """

        if not re.search(cls.success_string, stdout):
            logs.error.append("ERROR_OUTPUT_STDOUT_INCOMPLETE")

        parsed_data = parse_raw_stdout(stdout)

        return parsed_data, logs

    def parse_output_structure(
        self, retrieved_temporary_folder: str, rescale_factor: float, logs: AttributeDict
    ) -> tuple[orm.StructureData | None, AttributeDict]:
        """Parse the output structure from the 'answer.bas' file in the retrieved temporary folder.
        rescale_factor: used to rescale the input structure cell to the output structure cell.
        the answer.bas file contains the atomic positions of the output structure (already scaled).
        """
        answer_bas_file = os.path.join(retrieved_temporary_folder, "answer.bas")

        if not os.path.isfile(answer_bas_file):
            logs.error.append("ERROR_OUTPUT_STRUCTURE_NOT_FOUND")
            return None, logs

        with open(answer_bas_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
            numbers = []
            positions = []
            natoms = int(lines.pop(0).strip())
            for _ in range(natoms):
                line = lines.pop(0).strip()
                number, *coords = line.split()[:4]
                number = int(number)
                numbers.append(number)
                positions.append([float(coord.strip()) for coord in coords])

        # Create the structure
        input_structure: orm.StructureData = self.node.inputs.structure
        cell = np.array(input_structure.cell) * rescale_factor
        ase_structure = Atoms(numbers=numbers, positions=positions, cell=cell)
        ase_structure.set_pbc(input_structure.pbc)
        structure = orm.StructureData(ase=ase_structure)

        return structure, logs

    def parse_output_trajectory(
        self, retrieved_temporary_folder: str, rescale_factor: float, logs: AttributeDict
    ) -> tuple[orm.TrajectoryData | None, AttributeDict]:
        """Parse the output trajectory from the 'answer.xyz' file in the retrieved temporary folder if it exists.
        rescale_factor: used to rescale the input structure cell to the output structure cells.
        the answer.xyz file contains the atomic positions of the output structures (already scaled).
        """
        answer_xyz_file = os.path.join(retrieved_temporary_folder, "answer.xyz")

        if not os.path.isfile(answer_xyz_file):
            # logs.error.append("ERROR_OUTPUT_TRAJECTORY_NOT_FOUND")
            return None, logs

        # pylint: disable=line-too-long
        comment_match = re.compile(
            r"\s*ETOT =\s*(?P<energy>[+-]?(\w|\.)*)\s*eV; T =\s*(?P<temperature>(\w|\.)*)\s*K; Time =\s*(?P<time>(\w|\.)*)\s*fs"
        )

        with open(answer_xyz_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
            images: list[Atoms] = []
            energies: list[float | None] = []
            temperatures: list[float | None] = []
            times: list[float | None] = []
            while len(lines) > 0:
                symbols: list[str] = []
                positions: list[list[float]] = []
                natoms = int(lines.pop(0))
                comment = lines.pop(0)  # Comment line with energy, temperature, and time
                match = comment_match.match(comment)
                if match:
                    energies.append(float(match.group("energy")))
                    temperatures.append(float(match.group("temperature")))
                    times.append(float(match.group("time")))
                else:
                    energies.append(None)
                    temperatures.append(None)
                    times.append(None)
                for _ in range(natoms):
                    line = lines.pop(0)
                    symbol, *coords = line.split()[:4]
                    symbol = symbol.lower().capitalize()
                    symbols.append(symbol)
                    positions.append([float(coord.strip()) for coord in coords])
                images.append(Atoms(symbols=symbols, positions=positions))

        # Create the trajectory
        symbols: list[str] = images[0].get_chemical_symbols()
        positions: np.ndarray = np.array([image.get_positions() for image in images])
        cells: np.ndarray = np.array([np.array(self.node.inputs.structure.cell) * rescale_factor for _ in range(len(images))])
        times: np.ndarray = np.array(times)
        temperatures: np.ndarray = np.array(temperatures)
        energies: np.ndarray = np.array(energies)
        trajectory = orm.TrajectoryData()
        trajectory.set_trajectory(
            symbols=symbols,
            positions=positions,
            cells=cells,
            times=times,
        )
        trajectory.set_array("temperatures", temperatures)
        trajectory.set_array("energies", energies)

        return trajectory, logs

    def emit_logs(self, logs: list[AttributeDict] | tuple[AttributeDict] | AttributeDict, ignore: Optional[list] = None) -> None:
        """Emit the messages in one or multiple "log dictionaries" through the logger of the parser.

        A log dictionary is expected to have the following structure: each key must correspond to a log level of the
        python logging module, e.g. `error` or `warning` and its values must be a list of string messages. The method
        will loop over all log dictionaries and emit the messages it contains with the log level indicated by the key.

        Example log dictionary structure::

            logs = {
                'warning': ['Could not parse the `etot_threshold` variable from the stdout.'],
                'error': ['Self-consistency was not achieved']
            }

        :param logs: log dictionaries
        :param ignore: list of log messages to ignore
        """
        ignore = ignore or []

        if not isinstance(logs, (list, tuple)):
            logs = [logs]

        for logs in logs:
            for level, messages in logs.items():
                for message in messages:
                    stripped = message.strip()

                    if stripped in ignore:
                        continue

                    getattr(self.logger, level)(stripped)

    def exit(self, exit_code: ExitCode | None = None, logs: AttributeDict | None = None) -> ExitCode:
        """Log all messages in the ``logs`` as well as the ``exit_code`` message and return the correct exit code.

        This is a utility function if one wants to return from the parse method and automically add the ``logs`` and
        exit message associated to and exit code as a log message to the node: e.g.
        ``return self._exit(self.exit_codes.LABEL))``

        If no ``exit_code`` is provided, the method will check if an ``exit_status`` has already been set on the node
        and return the corresponding ``ExitCode`` in this case. If not, ``ExitCode(0)`` is returned.

        :param logs: log dictionaries
        :param exit_code: an ``ExitCode``
        :return: The correct exit code
        """
        if logs:
            self.emit_logs(logs)

        if exit_code is not None:
            self.logger.error(exit_code.message)
        elif self.node.exit_status is not None:
            exit_code = ExitCode(self.node.exit_status, self.node.exit_message)
        else:
            exit_code = ExitCode(0)

        return exit_code
