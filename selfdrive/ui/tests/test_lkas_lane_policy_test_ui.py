from types import SimpleNamespace

from openpilot.selfdrive.modeld.lkas_lane_policy import (
  LANE_LOCK_MODE_PARAM,
  format_lkas_lane_policy_test_hud,
  lane_lock_hud_label,
)
from openpilot.selfdrive.ui.onroad.starpilot.lkas_lane_policy_test_ui import (
  TEST_UI_PARAM,
  lkas_lane_policy_test_hud_lines,
)


class _FakeParams:
  def __init__(self, values):
    self.values = values

  def get_bool(self, key, default=False):
    if key not in self.values:
      return default
    return bool(self.values[key])

  def get(self, key, encoding=None, default=None):
    return self.values.get(key, default)


def test_hud_lines_include_kill_switch_oem_and_lock_mode():
  lines = format_lkas_lane_policy_test_hud(True, True, "full-lane midpoint (lane-policy on)", False)
  assert lines[0] == "LP ON  OEM ON  VIA off"
  assert lines[1] == "LOCK full midpoint"


def test_hud_labels_cover_off_path_unavailable_locking_and_full():
  assert lane_lock_hud_label("lane-policy off") == "off"
  assert lane_lock_hud_label("lane-policy fallback: path unavailable") == "path unavailable"
  assert lane_lock_hud_label("locking (lane-policy on)") == "locking"
  assert lane_lock_hud_label("full-lane midpoint (lane-policy on)") == "full midpoint"


def test_overlay_hidden_when_test_ui_param_is_off():
  params = _FakeParams({
    TEST_UI_PARAM: False,
    "LkasLanePolicy": True,
    "LkasLanePolicyViaLkas": True,
    LANE_LOCK_MODE_PARAM: "locking (lane-policy on)",
  })
  car_state = SimpleNamespace(lkasEnabled=True)
  assert lkas_lane_policy_test_hud_lines(params, car_state) is None


def test_overlay_reads_params_and_carstate_when_enabled():
  params = _FakeParams({
    TEST_UI_PARAM: True,
    "LkasLanePolicy": True,
    "LkasLanePolicyViaLkas": True,
    LANE_LOCK_MODE_PARAM: "lane-policy fallback: path unavailable",
  })
  car_state = SimpleNamespace(lkasEnabled=False)
  assert lkas_lane_policy_test_hud_lines(params, car_state) == [
    "LP ON  OEM off  VIA ON",
    "LOCK path unavailable",
  ]


def test_overlay_marks_oem_missing_when_carstate_absent():
  params = _FakeParams({
    TEST_UI_PARAM: True,
    "LkasLanePolicy": False,
    "LkasLanePolicyViaLkas": False,
    LANE_LOCK_MODE_PARAM: "lane-policy off",
  })
  assert lkas_lane_policy_test_hud_lines(params, None) == [
    "LP off  OEM n/a  VIA off",
    "LOCK off",
  ]
