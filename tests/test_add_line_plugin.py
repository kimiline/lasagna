"""Tests for the add_line plugin's "Load points" button, driving the real plugin with a stand-in Lasagna.

Run from the repo root: python -m pytest tests -q
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets

from lasagna.ingredients.sparsepoints import sparsepoints
from lasagna.plugins.annotation_plugins import add_line_plugin

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Rows as sparsepoints.save writes them: "Z,X,Y" floats of whole voxels, surface first
TRACK = [(312, 140, 20), (313, 141, 40), (315, 143, 60), (318, 146, 100)]


class Ingredient:
    def __init__(self, data):
        self._data = data

    def addToPlots(self):
        pass

    def raw_data(self):
        return np.asarray(self._data)


class Axis:
    currentSlice = 0

    def listNamedItemsInPlotWidget(self):
        pass


class StubLasagna:
    def __init__(self):
        self.hooks = {"axisClicked": [], "updateMainWindowOnMouseMove_End": []}
        self.ingredients = {}
        self.axes2D = [Axis()]

    def addIngredient(self, objectName, kind, data):
        self.ingredients[objectName] = Ingredient(data)

    def returnIngredientByName(self, name):
        return self.ingredients[name]

    def removeIngredientByName(self, name):
        self.ingredients.pop(name, None)

    def update_2D_plot_ingredients_in_axes(self):
        pass


@pytest.fixture
def plugin(monkeypatch):
    # Where the panel sits in the main window is not under test (the stub has no Points tab)
    monkeypatch.setattr(add_line_plugin.plugin, "embedInPointsTab", lambda self: None, raising=False)
    lasagna = StubLasagna()
    p = add_line_plugin.plugin(lasagna)
    p.fitType_comboBox.setCurrentText("piecewise")
    yield p
    p.close()


def save_like_lasagna(path, points):
    """Write points with the real sparsepoints.save, as right-click > Save does."""
    holder = type("Holder", (), {"objectName": "x", "raw_data": lambda self: np.asarray(points, float)})()
    sparsepoints.save(holder, path=str(path))


def click(p, pos, button):
    p.lasagna.mousePositionInStack = list(pos)
    p.hook_updateMainWindowOnMouseMove_End()
    p.lasagna.last_button_click_in_axis = button
    p.hook_axisClicked(None)


def test_load_fills_table_in_file_order_and_names_line(plugin, tmp_path):
    path = tmp_path / "2026-07-14_KM_074_001_probe00a_pts.csv"
    save_like_lasagna(path, TRACK)
    assert plugin.load_points(str(path))
    assert plugin.get_points_coord().tolist() == [list(map(float, t)) for t in TRACK]
    assert plugin.num_points == 4 and plugin.numPoints_textLabel.text() == "n pts: 4"
    assert plugin.name_lineEdit.text() == "2026-07-14_KM_074_001_probe00a"
    fit = np.asarray(plugin.fit["fit_coords"])
    assert len(fit) and np.all(np.diff(fit[:, 2]) >= 0)  # piecewise line drawn, surface to tip


def test_extend_insert_and_save_round_trip(plugin, tmp_path):
    path = tmp_path / "2026-07-14_KM_074_001_probe00a_pts.csv"
    save_like_lasagna(path, TRACK)
    plugin.load_points(str(path))
    click(plugin, (320, 148, 130), 1)  # nothing marked: goes at the end (extends the bottom)
    click(plugin, (318, 146, 100), 2)  # mark DV 100, the deeper point of the 60-100 gap
    click(plugin, (316, 144, 80), 1)  # goes in just before the marked point
    assert plugin.get_points_coord()[:, 2].tolist() == [20, 40, 60, 80, 100, 130]

    plugin.add_line()  # "Add it!"
    layer = plugin.lasagna.ingredients["2026-07-14_KM_074_001_probe00a_pts"]
    out = tmp_path / "saved.csv"
    save_like_lasagna(out, layer.raw_data())
    lines = out.read_text().splitlines()
    assert lines[0] == "312.0,140.0,20.0" and lines[-1] == "320.0,148.0,130.0" and len(lines) == 6
    assert plugin.tableWidget.rowCount() == 0  # Add it! clears the table, as before


def test_load_replaces_existing_points(plugin, tmp_path):
    click(plugin, (1, 2, 3), 1)
    path = tmp_path / "t_pts.csv"
    save_like_lasagna(path, TRACK)
    plugin.load_points(str(path))
    assert plugin.tableWidget.rowCount() == 4 and plugin.num_points == 4


@pytest.mark.parametrize("content", ["", "\n", "1.0,2.0\n3.0,4.0\n", "1.0,2.0,3.0,4.0\n", "a,b,c\n"])
def test_bad_files_leave_table_alone(plugin, tmp_path, content):
    click(plugin, (1, 2, 3), 1)
    path = tmp_path / "bad_pts.csv"
    path.write_text(content)
    assert not plugin.load_points(str(path))
    assert plugin.get_points_coord().tolist() == [[1.0, 2.0, 3.0]]
    assert plugin.name_lineEdit.text() == "ManualLine"


def test_missing_file(plugin, tmp_path):
    assert not plugin.load_points(str(tmp_path / "nope.csv"))


def test_single_point_and_non_integer_values(plugin, tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("10.4,20.6,30.0\n")
    assert plugin.load_points(str(path))
    assert plugin.get_points_coord().tolist() == [[10.0, 21.0, 30.0]]
    assert plugin.name_lineEdit.text() == "one"


def test_button_sits_above_add_it(plugin):
    layout = plugin.verticalLayout
    assert layout.indexOf(plugin.load_pushButton) == layout.indexOf(plugin.add_pushButton) - 1


@pytest.fixture
def fake_prefs(monkeypatch):
    written = {}
    monkeypatch.setattr(add_line_plugin.preferences, "readPreference", lambda key: "")
    monkeypatch.setattr(add_line_plugin.preferences, "preferenceWriter", lambda k, v: written.update({k: v}))
    return written


@pytest.mark.parametrize("answer, loaded", [(QtWidgets.QMessageBox.No, False), (QtWidgets.QMessageBox.Yes, True)])
def test_dialog_asks_before_replacing(plugin, tmp_path, monkeypatch, fake_prefs, answer, loaded):
    path = tmp_path / "t_pts.csv"
    save_like_lasagna(path, TRACK)
    click(plugin, (1, 2, 3), 1)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: answer)
    plugin.load_points_dialog()
    assert (plugin.num_points == 4) is loaded
    assert (fake_prefs.get("lastLoadDir") == str(tmp_path) + os.sep) is loaded


def test_dialog_cancel_does_nothing(plugin, monkeypatch, fake_prefs):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    plugin.load_points_dialog()
    assert plugin.tableWidget.rowCount() == 0 and not fake_prefs
