"""Validation functions for the FireballCalculation inputs."""

from typing import Optional

import numpy


def validate_fixed_coords(value, settings: dict, parameters: dict) -> list[str]:
    """Validate the ``fixed_coords`` input port.

    :param value: The entire inputs namespace.
    :param settings: The settings dictionary.
    :param parameters: The parameters dictionary.
    :return: A list of error messages, empty if no errors.
    """
    # Validate the FIXED_COORDS setting
    messages = []

    fixed_coords = settings.get("FIXED_COORDS", None)

    if fixed_coords is not None:
        fixed_coords = numpy.array(fixed_coords)

        if len(fixed_coords.shape) != 2 or fixed_coords.shape[1] != 3:
            messages.append("The `fixed_coords` setting must be a list of lists with length 3.")

        if fixed_coords.dtype != bool:
            messages.append("All elements in the `fixed_coords` setting lists must be either `True` or `False`.")

        if "structure" in value:
            nb_sites = len(value["structure"].sites)

            if len(fixed_coords) != nb_sites:
                messages.append(f"Input structure has {nb_sites} sites, but fixed_coords has length {len(fixed_coords)}")
    return messages


def validate_dos_params(value, settings: dict, parameters: dict) -> list[str]:
    """Validate the ``dos_params`` input port.

    :param value: The entire inputs namespace.
    :param settings: The settings dictionary.
    :param parameters: The parameters dictionary.
    :return: A list of error messages, empty if no errors.
    """
    messages = []

    dos_params: Optional[dict] = settings.get("DOS", None)

    if dos_params is not None:
        parameters.setdefault("OUTPUT", {}).setdefault("iwrtdos", 1)
        valid_keys = [
            "first_atom_index",
            "last_atom_index",
            "Emin",
            "Emax",
            "n_energy_steps",
            "eta",
            "iwrttip",
            "Emin_tip",
            "Emax_tip",
        ]
        defaults = {
            "first_atom_index": 1,
            "last_atom_index": len(value["structure"].sites),
            "eta": 0.1,
            "n_energy_steps": 100,
            "Emin": -5.0,
            "Emax": 5.0,
            "iwrttip": 0,  # writes the file tip_e_str.inp
            "Emin_tip": 0.0,
            "Emax_tip": 0.0,
        }
        types = [
            int,
            int,
            float,
            int,
            float,
            float,
            int,
            float,
            float,
        ]
        # Emin and Emax are in eV and are relative to the Fermi level:
        # conversion to Fireball format will be performed
        # There will be (n_energy_steps + 1) energy points in the output DOS file
        for key in dos_params:
            if key not in valid_keys:
                messages.append(f"Invalid key '{key}' in the 'DOS' namelist. Valid keys are: {valid_keys}")

        for key, default in defaults.items():
            dos_params.setdefault(key, default)

        for key, val, type_ in zip(dos_params.keys(), dos_params.values(), types):
            try:
                dos_params[key] = type_(val)
            except ValueError:
                messages.append(f"Invalid value for '{key}' in the 'DOS' namelist. It must be a {type_}")

        if dos_params["first_atom_index"] < 1 or dos_params["first_atom_index"] > len(value["structure"].sites):
            messages.append(f"Invalid value for 'first_atom_index' in the 'DOS' namelist. It must be between 1 and {len(value['structure'].sites)}")
        if (
            dos_params["last_atom_index"] < 1
            or dos_params["last_atom_index"] > len(value["structure"].sites)
            or dos_params["last_atom_index"] < dos_params["first_atom_index"]
        ):
            messages.append(
                f"Invalid value for 'last_atom_index' in the 'DOS' namelist. \
It must be between 1 and {len(value['structure'].sites)} and greater than 'first_atom_index'"
            )
        if dos_params["n_energy_steps"] < 1:
            messages.append("Invalid value for 'n_energy_steps' in the 'DOS' namelist. It must be greater than 0")
        if dos_params["eta"] <= 0.0:
            messages.append("Invalid value for 'eta' in the 'DOS' namelist. It must be greater than 0")
        if dos_params["iwrttip"] not in [0, 1]:
            messages.append("Invalid value for 'iwrttip' in the 'DOS' namelist. It must be either 0 or 1")
        if dos_params["Emin_tip"] > dos_params["Emax_tip"]:
            messages.append("Invalid values for 'Emin_tip' and 'Emax_tip' in the 'DOS' namelist. 'Emin_tip' must be less than 'Emax_tip'")
        if dos_params["Emin"] > dos_params["Emax"]:
            messages.append("Invalid values for 'Emin' and 'Emax' in the 'DOS' namelist. 'Emin' must be less than 'Emax'")

        settings["DOS"] = dos_params

    return messages


def validate_cgopt_params(value, settings: dict, parameters: dict) -> list[str]:
    """Validate the ``cgopt_params`` input port.

    :param value: The entire inputs namespace.
    :param settings: The settings dictionary.
    :param parameters: The parameters dictionary.
    :return: A list of error messages, empty if no errors.
    """
    messages = []

    cgopt_params: Optional[dict] = settings.get("CGOPT", None)

    if cgopt_params is not None:
        valid_keys = [
            "drmax",
            "dummy",
            "energy_tol",
            "force_tol",
            "max_steps",
            "min_int_steps",
            "switch_MD",
        ]
        defaults = {
            "drmax": 0.1,
            "dummy": 0.1,
            "energy_tol": 1.0e-06,
            "force_tol": 1.0e-4,
            "max_steps": 1000,
            "min_int_steps": 0,
            "switch_MD": 0,
        }
        types = [
            float,
            float,
            float,
            float,
            int,
            int,
            int,
        ]
        for key in cgopt_params:
            if key not in valid_keys:
                messages.append(f"Invalid key '{key}' in the 'CGOPT' namelist. Valid keys are: {valid_keys}")

        for key, val in defaults.items():
            cgopt_params.setdefault(key, val)

        for key, val, type_ in zip(cgopt_params.keys(), cgopt_params.values(), types):
            try:
                cgopt_params[key] = type_(val)
            except ValueError:
                messages.append(f"Invalid value for '{key}' in the 'CGOPT' namelist. It must be a {type_}")

        if cgopt_params["drmax"] <= 0.0:
            messages.append("Invalid value for 'drmax' in the 'CGOPT' namelist. It must be greater than 0")

        if cgopt_params["dummy"] <= 0.0 or cgopt_params["dummy"] >= 1.0:
            messages.append("Invalid value for 'dummy' in the 'CGOPT' namelist. It must be between 0 and 1")

        if cgopt_params["energy_tol"] <= 0.0:
            messages.append("Invalid value for 'energy_tol' in the 'CGOPT' namelist. It must be greater than 0")

        if cgopt_params["force_tol"] <= 0.0:
            messages.append("Invalid value for 'force_tol' in the 'CGOPT' namelist. It must be greater than 0")

        if cgopt_params["max_steps"] < 1:
            messages.append("Invalid value for 'max_steps' in the 'CGOPT' namelist. It must be greater than 0")

        if cgopt_params["min_int_steps"] < 0:
            messages.append("Invalid value for 'min_int_steps' in the 'CGOPT' namelist. It must be greater than or equal to 0")

        if cgopt_params["switch_MD"] < 0:
            messages.append("Invalid value for 'switch_MD' in the 'CGOPT' namelist. It must be greater than or equal to 0")

        settings["CGOPT"] = cgopt_params

    return messages
