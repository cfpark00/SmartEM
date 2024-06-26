import pytest


def test_import_smartem():
    assert __import__("smartem")


def test_import_smartem_smartem():
    assert __import__("smartem.smartem")


def test_import_smartem_tools():
    assert __import__("smartem.tools")


def test_import_smartem_online():
    assert __import__("smartem.online")


def test_import_smartem_offline():
    assert __import__("smartem.offline")


def test_import_smartem_online_microscope():
    assert __import__("smartem.online.microscope")


def test_import_smartem_online_get_rescan_maps():
    assert __import__("smartem.online.get_rescan_maps")
