"""Testing-only onroad readout for LKAS lane-policy lock.

Delete this file and the `_render_lkas_lane_policy_test_ui` hook to remove the HUD.
"""
from __future__ import annotations

from openpilot.selfdrive.modeld.lkas_lane_policy import (
  LANE_LOCK_MODE_PARAM,
  format_lkas_lane_policy_test_hud,
)

TEST_UI_PARAM = "LkasLanePolicyTestUi"


def lkas_lane_policy_test_ui_enabled(params) -> bool:
  return bool(params.get_bool(TEST_UI_PARAM, default=True))


def lkas_lane_policy_test_hud_lines(params, car_state) -> list[str] | None:
  if not lkas_lane_policy_test_ui_enabled(params):
    return None
  oem = None
  if car_state is not None:
    oem = bool(getattr(car_state, "lkasEnabled", False))
  lock_mode = params.get(LANE_LOCK_MODE_PARAM, encoding="utf-8", default="") or None
  return format_lkas_lane_policy_test_hud(
    lane_policy_on=bool(params.get_bool("LkasLanePolicy")),
    oem_lkas_on=oem,
    lock_mode=lock_mode,
    via_lkas_on=bool(params.get_bool("LkasLanePolicyViaLkas")),
  )
