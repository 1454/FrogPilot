import numpy as np
import pytest
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld import lkas_lane_policy as policy
from openpilot.selfdrive.modeld.lkas_lane_policy import LANE_CHANGE_LEFT


def _reset_policy():
  policy.reset_lane_lock()
  policy._lane_lock_error_logged = False
  policy._lane_lock_mode = None
  policy._lane_lock_last_log_time = 0.0


@pytest.fixture(autouse=True)
def _clean_policy_state():
  _reset_policy()
  yield
  _reset_policy()


def _straight_lane_output(left_prob=0.99, right_prob=0.99, center_y=0.0, width=3.6, path_y=None):
  n = len(ModelConstants.X_IDXS)
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  left = np.full(n, center_y + width / 2.0)
  right = np.full(n, center_y - width / 2.0)
  lane_lines = np.zeros((1, 4, n, 2), dtype=np.float64)
  lane_lines[0, 1, :, 0] = left
  lane_lines[0, 2, :, 0] = right

  plan = np.zeros((1, n, 6), dtype=np.float64)
  plan[0, :, 0] = x
  plan[0, :, 1] = 0.0 if path_y is None else path_y

  desire = np.zeros((1, 8), dtype=np.float64)
  return {
    "lane_lines": lane_lines,
    "lane_lines_prob": np.array([[0.0, left_prob, right_prob, 0.0]], dtype=np.float64),
    "desire_state": desire,
    "plan": plan,
  }


def test_lkas_off_returns_raw_e2e_and_clears_state():
  out = policy.apply_lane_lock(_straight_lane_output(), 0.012, 20.0, lane_policy_enabled=False)
  assert out == pytest.approx(0.012)
  assert policy._lane_lock_weight == 0.0
  assert policy._lane_lock_has_lane_curvature is False


def test_blinker_releases_immediately_to_e2e():
  policy.apply_lane_lock(_straight_lane_output(), 0.0, 20.0, lane_policy_enabled=True)
  out = policy.apply_lane_lock(_straight_lane_output(), 0.02, 20.0, blinkers_active=True, lane_policy_enabled=True)
  assert out == pytest.approx(0.02)
  assert policy._lane_lock_weight == 0.0


def test_lane_change_intent_releases_to_e2e():
  model = _straight_lane_output()
  model["desire_state"][0, LANE_CHANGE_LEFT] = 0.4
  out = policy.apply_lane_lock(model, 0.03, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.03)
  assert policy._lane_lock_weight == 0.0


def test_full_lane_engages_when_lkas_on_and_lines_are_clean():
  model = _straight_lane_output(center_y=0.4)
  first = policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is True
  assert first != pytest.approx(0.0)
  for _ in range(80):
    policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_weight == pytest.approx(1.0, abs=0.02)


def test_stop_and_go_still_eligible_when_lkas_on():
  model = _straight_lane_output()
  policy.apply_lane_lock(model, 0.0, 0.2, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is True


def test_low_confidence_does_not_engage():
  model = _straight_lane_output(left_prob=0.4, right_prob=0.4)
  out = policy.apply_lane_lock(model, 0.011, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.011)
  assert policy._lane_lock_full_active is False
