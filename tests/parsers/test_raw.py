"""Tests for the pares_raw functions"""

import pytest

from aiida_fireball.parsers.raw import parse_raw_stdout


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("FIREBALL RUNTIME : 123.456 [sec]", 123.456),
        ("FIREBALL RUNTIME : 0.0 [sec]", 0.0),
        ("\n  FIREBALL RUNTIME :   0.147854000000000      [sec]", 0.147854),
    ],
)
def test_parse_raw_stdout_wall_time(stdout, expected):
    result = parse_raw_stdout(stdout)
    assert "wall_time_seconds" in result
    assert result["wall_time_seconds"] == expected


def test_parse_raw_stdout_no_wall_time():
    stdout = "Some other output without wall time"
    result = parse_raw_stdout(stdout)
    assert "wall_time_seconds" not in result


def test_parse_raw_stdout_invalid_wall_time():
    stdout = "FIREBALL RUNTIME : abc.def [sec]"
    result = parse_raw_stdout(stdout)
    assert "wall_time_seconds" not in result


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("ETOT = 123.456", 123.456),
        ("   ETOT = -123.456", -123.456),
        ("ETOT = 0.0", 0.0),
        ("Some text\n       ETOT =          +0.147854 \n  Some more text", 0.147854),
        ("Some text iter 1\n   ETOT =     -425.98734   \n  Some more text iter n \n   ETOT = -2456.4465234  \n Final text\n", -2456.4465234),
    ],
)
def test_parse_raw_stdout_total_energy(stdout, expected):
    result = parse_raw_stdout(stdout)
    assert "energy" in result
    assert result["energy"] == expected
    assert result["energy_units"] == "eV"


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("Fermi Level = 5.4321", 5.4321),
        ("   Fermi Level = -5.4321", -5.4321),
        ("Fermi Level = 0.0", 0.0),
        ("Some text\n       Fermi Level =          +0.1234 \n  Some more text", 0.1234),
    ],
)
def test_parse_raw_stdout_fermi_energy(stdout, expected):
    result = parse_raw_stdout(stdout)
    assert "fermi_energy" in result
    assert result["fermi_energy"] == expected
    assert result["fermi_energy_units"] == "eV"


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("qztot = 10.1234", 10.1234),
        ("   qztot = 10.1234", 10.1234),
        ("qztot = 0.0", 0.0),
        ("Some text\n       qztot =          0.5678 \n  Some more text", 0.5678),
    ],
)
def test_parse_raw_stdout_number_of_electrons(stdout, expected):
    result = parse_raw_stdout(stdout)
    assert "number_of_electrons" in result
    assert result["number_of_electrons"] == expected


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("energy tolerance = 1.2345E-06 [eV]", 1.2345e-06),
        ("   energy tolerance = 1.2345E-06 [eV]", 1.2345e-06),
        ("energy tolerance =  1.00 [eV]", 1.0),
        ("Some text\n       energy tolerance =          1.2345E-06 [eV]\n  Some more text", 1.2345e-06),
    ],
)
def test_parse_raw_stdout_energy_tolerance(stdout, expected):
    result = parse_raw_stdout(stdout)
    assert "energy_tolerance" in result
    assert result["energy_tolerance"] == expected
