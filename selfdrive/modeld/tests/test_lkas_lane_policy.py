import numpy as np
import pytest
from cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld import lkas_lane_policy as policy
from openpilot.selfdrive.modeld.lkas_lane_policy import (
  INNER_LEFT_PROB_INDEX,
  INNER_RIGHT_PROB_INDEX,
  LANE_CHANGE_LEFT,
  LANE_CHANGE_RIGHT,
  LANE_LINES_PROB_WIDTH,
)


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


def _lane_probs(left=0.99, right=0.99, far_left=0.1, far_right=0.1):
  probs = np.zeros((1, LANE_LINES_PROB_WIDTH), dtype=np.float64)
  probs[0, 1] = far_left
  probs[0, INNER_LEFT_PROB_INDEX] = left
  probs[0, INNER_RIGHT_PROB_INDEX] = right
  probs[0, 7] = far_right
  return probs


def _straight_lane_output(left_prob=0.99, right_prob=0.99, center_y=0.0, width=3.6,
                          v_ego=20.0, path_y=None, far_left_prob=0.1, far_right_prob=0.1):
  n = ModelConstants.IDX_N
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  left = np.full(n, center_y + width / 2.0)
  right = np.full(n, center_y - width / 2.0)
  lane_lines = np.zeros((1, ModelConstants.NUM_LANE_LINES, n, 2), dtype=np.float64)
  lane_lines[0, 1, :, 0] = left
  lane_lines[0, 2, :, 0] = right

  plan = np.zeros((1, n, ModelConstants.PLAN_WIDTH), dtype=np.float64)
  plan[0, :, 0] = v_ego * np.asarray(ModelConstants.T_IDXS, dtype=np.float64)
  plan[0, :, 1] = 0.0 if path_y is None else path_y

  desire = np.zeros((1, ModelConstants.DESIRE_PRED_WIDTH), dtype=np.float64)
  return {
    "lane_lines": lane_lines,
    "lane_lines_prob": _lane_probs(left_prob, right_prob, far_left_prob, far_right_prob),
    "desire_state": desire,
    "plan": plan,
  }


def test_desire_indices_match_cereal():
  assert LANE_CHANGE_LEFT == int(log.Desire.laneChangeLeft)
  assert LANE_CHANGE_RIGHT == int(log.Desire.laneChangeRight)


def test_inner_probs_use_8_wide_odd_slots():
  model = _straight_lane_output(left_prob=0.91, right_prob=0.93, far_left_prob=0.99)
  assert policy.inner_lane_line_probs(model) == pytest.approx((0.91, 0.93))


def test_wrong_prob_width_does_not_engage():
  model = _straight_lane_output()
  model["lane_lines_prob"] = np.array([[0.0, 0.99, 0.99, 0.0]], dtype=np.float64)
  out = policy.apply_lane_lock(model, 0.02, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.02)
  assert policy._lane_lock_full_active is False


def test_outer_line_confidence_does_not_engage_without_inner_lines():
  model = _straight_lane_output(left_prob=0.2, right_prob=0.2, far_left_prob=0.99, far_right_prob=0.99)
  out = policy.apply_lane_lock(model, 0.015, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.015)
  assert policy._lane_lock_full_active is False


def test_lkas_off_returns_raw_e2e_and_clears_state():
  policy.apply_lane_lock(_straight_lane_output(), 0.0, 20.0, lane_policy_enabled=True)
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


def test_full_lane_engages_when_policy_on_and_lines_are_clean():
  model = _straight_lane_output(center_y=0.4)
  first = policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is True
  assert first != pytest.approx(0.0)
  for _ in range(80):
    policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_weight == pytest.approx(1.0, abs=0.02)


def test_invalid_geometry_clears_hold_hysteresis():
  model = _straight_lane_output()
  policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is True
  wide = _straight_lane_output(width=6.0)
  policy.apply_lane_lock(wide, 0.0, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is False
  hold_only = _straight_lane_output(left_prob=0.88, right_prob=0.88)
  policy.apply_lane_lock(hold_only, 0.011, 20.0, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is False


def test_width_change_gate_blocks_engage():
  model = _straight_lane_output()
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  fit = (x >= 5.0) & (x <= 35.0)
  model["lane_lines"][0, 1, fit, 0] = np.linspace(1.4, 2.0, int(np.count_nonzero(fit)))
  model["lane_lines"][0, 2, fit, 0] = -1.8
  out = policy.apply_lane_lock(model, 0.01, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.01)
  assert policy._lane_lock_full_active is False


def test_extreme_path_disagreement_resets_to_e2e():
  model = _straight_lane_output(path_y=np.full(ModelConstants.IDX_N, 1.2))
  out = policy.apply_lane_lock(model, 0.04, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.04)
  assert policy._lane_lock_weight == 0.0


def test_crawl_speed_releases_on_short_plan_horizon():
  model = _straight_lane_output(v_ego=0.2)
  policy.apply_lane_lock(model, 0.0, 0.2, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is False


def test_plan_horizon_allows_lock_at_1_2_mps():
  model = _straight_lane_output(v_ego=1.2)
  policy.apply_lane_lock(model, 0.0, 1.2, lane_policy_enabled=True)
  assert policy._lane_lock_full_active is True


def test_low_confidence_does_not_engage():
  model = _straight_lane_output(left_prob=0.4, right_prob=0.4)
  out = policy.apply_lane_lock(model, 0.011, 20.0, lane_policy_enabled=True)
  assert out == pytest.approx(0.011)
  assert policy._lane_lock_full_active is False


def test_mode_updates_even_when_log_is_rate_limited():
  policy.log_lane_lock_mode("lane-policy off")
  policy.log_lane_lock_mode("lane-policy fallback: blinker")
  assert policy._lane_lock_mode == "lane-policy fallback: blinker"


def test_get_action_from_model_applies_policy_before_smoothing():
  modeld = pytest.importorskip("openpilot.selfdrive.modeld.modeld")
  model = _straight_lane_output(center_y=0.4)
  model["action"] = np.array([[0.0, 0.0]], dtype=np.float32)
  prev = log.ModelDataV2.Action(desiredCurvature=0.0, desiredAcceleration=0.0, shouldStop=False)
  raw = policy.apply_lane_lock(model, 0.0, 20.0, lane_policy_enabled=True)
  _reset_policy()
  action = modeld.get_action_from_model(
    model, prev, 0.2, 0.73, 20.0, True, False, True, False, None,
    is_v16=True, blinkers_active=False, lane_policy_enabled=True,
  )
  assert action.desiredCurvature != pytest.approx(0.0)
  assert np.sign(action.desiredCurvature) == np.sign(raw)
