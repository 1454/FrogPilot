"""LKAS-button full-lane midpoint policy for modeld.

When the kill switch is on and carState.lkasEnabled is set, a clean stable
pair of inner lane lines gets full midpoint authority. Otherwise the exact
upstream E2E curvature is returned. This is an A/B switch, not a blend.

There is no explicit speed gate, but valid_plan requires a 12 m lookahead
inside the plan horizon. Real plan x is about v_ego * T_IDXS, so the lock
cannot hold below about 1.2 m/s.

Ported from morrislee/gm1500 lkas-lp-toggle, then corrected for Dom's
8-wide lane_lines_prob layout.
"""
from __future__ import annotations

import time

import numpy as np
from cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

LANE_CHANGE_LEFT = int(log.Desire.laneChangeLeft)
LANE_CHANGE_RIGHT = int(log.Desire.laneChangeRight)
LANE_LINES_PROB_WIDTH = 2 * ModelConstants.NUM_LANE_LINES
INNER_LEFT_PROB_INDEX = 3
INNER_RIGHT_PROB_INDEX = 5

# No explicit speed gate. Lookahead is still bounded to a stable 12 m, and the
# plan-horizon check is the effective low-speed release (about 1.2 m/s).
LANE_LOCK_ENTER_LINE_PROB = 0.92                # both lines to engage
LANE_LOCK_HOLD_LINE_PROB = 0.85                 # both lines to remain active
LANE_LOCK_MIN_LANE_WIDTH = 2.8                  # m
LANE_LOCK_MAX_LANE_WIDTH = 4.7                  # m
LANE_LOCK_MIN_WIDTH_EDGE = 0.15                 # m inside accepted range
LANE_LOCK_MAX_WIDTH_CHANGE = 0.18               # m across fitted horizon
LANE_LOCK_MAX_PATH_DISAGREEMENT = 0.75          # m, extreme safety release
LANE_LOCK_ENGAGE_TIME = 0.50                    # seconds
LANE_LOCK_RELEASE_TIME = 0.15                   # seconds
LANE_LOCK_CURVATURE_TIME = 0.25                 # seconds
LANE_LOCK_MAX_LANE_CHANGE_PROB = 0.10
LANE_LOCK_LOG_INTERVAL = 1.0                    # seconds

_lane_lock_weight = 0.0
_lane_lock_lane_curvature = 0.0
_lane_lock_has_lane_curvature = False
_lane_lock_full_active = False
_lane_lock_error_logged = False
_lane_lock_mode = None
_lane_lock_last_log_time = 0.0


def reset_lane_lock() -> None:
  """Discard any lane target so the caller immediately receives raw E2E."""
  global _lane_lock_weight, _lane_lock_lane_curvature
  global _lane_lock_has_lane_curvature, _lane_lock_full_active, _lane_lock_mode
  _lane_lock_weight = 0.0
  _lane_lock_lane_curvature = 0.0
  _lane_lock_has_lane_curvature = False
  _lane_lock_full_active = False
  _lane_lock_mode = None


def log_lane_lock_mode(mode: str) -> None:
  """Record every mode change. Rate-limit only the emitted rlog line."""
  global _lane_lock_mode, _lane_lock_last_log_time
  if mode == _lane_lock_mode:
    return
  now = time.monotonic()
  if now - _lane_lock_last_log_time >= LANE_LOCK_LOG_INTERVAL:
    cloudlog.info(f"lkas-lp-toggle: {mode}")
    _lane_lock_last_log_time = now
  _lane_lock_mode = mode


def inner_lane_line_probs(model_output: dict[str, np.ndarray]) -> tuple[float, float] | None:
  """Return inner left/right probs from the 8-wide pairs-per-line layout."""
  probs = np.asarray(model_output['lane_lines_prob'][0], dtype=np.float64).reshape(-1)
  if probs.size != LANE_LINES_PROB_WIDTH:
    return None
  return float(probs[INNER_LEFT_PROB_INDEX]), float(probs[INNER_RIGHT_PROB_INDEX])


def apply_lane_lock(model_output: dict[str, np.ndarray], e2e_curvature: float, v_ego: float,
                    blinkers_active: bool = False, lane_policy_enabled: bool = False) -> float:
  """
  With lane policy off, return the exact upstream E2E curvature and clear state.

  With lane policy on, use the fitted midpoint of the two inner lane lines only
  after a high-confidence, stable-geometry gate passes. Weak lines release
  smoothly; blinkers, lane-change intent, and extreme disagreement release
  immediately to upstream E2E.
  """
  global _lane_lock_weight, _lane_lock_lane_curvature
  global _lane_lock_has_lane_curvature, _lane_lock_full_active
  global _lane_lock_error_logged

  if not lane_policy_enabled:
    reset_lane_lock()
    log_lane_lock_mode("lane-policy off")
    return float(e2e_curvature)

  if blinkers_active:
    reset_lane_lock()
    log_lane_lock_mode("lane-policy fallback: blinker")
    return float(e2e_curvature)

  target_weight = 0.0
  candidate_lane_curvature = None
  fallback_reason = "lane-policy fallback: lane data unavailable"

  try:
    left_y = model_output['lane_lines'][0, 1, :, 0].astype(np.float64)
    right_y = model_output['lane_lines'][0, 2, :, 0].astype(np.float64)
    probs = inner_lane_line_probs(model_output)
    if probs is None:
      reset_lane_lock()
      log_lane_lock_mode("lane-policy fallback: lane-prob layout")
      return float(e2e_curvature)
    left_prob, right_prob = probs
    desire_state = model_output['desire_state'][0]
    lane_change_prob = float(desire_state[LANE_CHANGE_LEFT] + desire_state[LANE_CHANGE_RIGHT])
    if lane_change_prob > LANE_LOCK_MAX_LANE_CHANGE_PROB:
      reset_lane_lock()
      log_lane_lock_mode("lane-policy fallback: lane-change intent")
      return float(e2e_curvature)

    x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
    lookahead = float(np.clip(1.5 * v_ego, 12.0, 30.0))
    fit = (x >= 5.0) & (x <= 35.0)
    lane_width = left_y - right_y
    valid_lane_geometry = (
      np.count_nonzero(fit) >= 3 and
      np.isfinite(left_prob) and
      np.isfinite(right_prob) and
      np.all(np.isfinite(left_y[fit])) and
      np.all(np.isfinite(right_y[fit])) and
      np.all((lane_width[fit] >= LANE_LOCK_MIN_LANE_WIDTH) &
             (lane_width[fit] <= LANE_LOCK_MAX_LANE_WIDTH))
    )

    if not valid_lane_geometry:
      _lane_lock_full_active = False
      fallback_reason = "lane-policy fallback: lane geometry"
    else:
      mean_width = float(np.mean(lane_width[fit]))
      width_change = float(np.ptp(lane_width[fit]))
      width_edge_distance = min(mean_width - LANE_LOCK_MIN_LANE_WIDTH,
                                LANE_LOCK_MAX_LANE_WIDTH - mean_width)
      required_line_prob = (LANE_LOCK_HOLD_LINE_PROB if _lane_lock_full_active
                            else LANE_LOCK_ENTER_LINE_PROB)
      full_lane_ready = (
        min(left_prob, right_prob) >= required_line_prob and
        width_edge_distance >= LANE_LOCK_MIN_WIDTH_EDGE and
        width_change <= LANE_LOCK_MAX_WIDTH_CHANGE
      )

      if not full_lane_ready:
        _lane_lock_full_active = False
        if min(left_prob, right_prob) < required_line_prob:
          fallback_reason = "lane-policy fallback: lane confidence"
        else:
          fallback_reason = "lane-policy fallback: lane-width stability"
      else:
        # y = ax^2 + bx + c. The curvature includes center offset and
        # heading, so it actively converges the vehicle to lane midpoint.
        a, b, c = np.polyfit(x[fit], 0.5 * (left_y[fit] + right_y[fit]), 2)
        slope = 2.0 * a * lookahead + b
        lane_geometry_curvature = 2.0 * a / ((1.0 + slope * slope) ** 1.5)
        lane_curvature = lane_geometry_curvature + 2.0 * b / lookahead + 2.0 * c / (lookahead * lookahead)

        plan = model_output['plan'][0, :, Plan.POSITION]
        plan_x, plan_y = plan[:, 0], plan[:, 1]
        plan_horizon = (plan_x >= 0.0) & (plan_x <= max(lookahead + 5.0, 20.0))
        horizon_x, horizon_y = plan_x[plan_horizon], plan_y[plan_horizon]
        valid_plan = (
          horizon_x.size >= 2 and
          np.all(np.isfinite(horizon_x)) and
          np.all(np.isfinite(horizon_y)) and
          horizon_x[0] <= lookahead <= horizon_x[-1] and
          np.all(np.diff(horizon_x) > 0.0)
        )

        if not valid_plan or not np.isfinite(lane_curvature):
          _lane_lock_full_active = False
          fallback_reason = "lane-policy fallback: path unavailable"
        else:
          e2e_y_at_lookahead = float(np.interp(lookahead, horizon_x, horizon_y))
          lane_y_at_lookahead = float(np.polyval((a, b, c), lookahead))
          path_disagreement = abs(e2e_y_at_lookahead - lane_y_at_lookahead)
          if path_disagreement > LANE_LOCK_MAX_PATH_DISAGREEMENT:
            reset_lane_lock()
            log_lane_lock_mode("lane-policy fallback: extreme path disagreement")
            return float(e2e_curvature)

          _lane_lock_full_active = True
          target_weight = 1.0
          candidate_lane_curvature = float(lane_curvature)
          fallback_reason = ""

  except (KeyError, IndexError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
    _lane_lock_full_active = False
    fallback_reason = "lane-policy fallback: lane-policy input error"
    if not _lane_lock_error_logged:
      cloudlog.warning(f"lkas-lp-toggle input error: {type(err).__name__}: {err}")
      _lane_lock_error_logged = True

  time_constant = LANE_LOCK_ENGAGE_TIME if target_weight > _lane_lock_weight else LANE_LOCK_RELEASE_TIME
  _lane_lock_weight += (target_weight - _lane_lock_weight) * DT_MDL / time_constant
  _lane_lock_weight = float(np.clip(_lane_lock_weight, 0.0, 1.0))

  if candidate_lane_curvature is not None:
    if not _lane_lock_has_lane_curvature:
      # Start from the current E2E command, then filter toward lane center.
      # This is a transition limiter, not a permanent cap on lane authority.
      _lane_lock_lane_curvature = float(e2e_curvature)
      _lane_lock_has_lane_curvature = True
    alpha = min(DT_MDL / LANE_LOCK_CURVATURE_TIME, 1.0)
    _lane_lock_lane_curvature += alpha * (candidate_lane_curvature - _lane_lock_lane_curvature)
    _lane_lock_error_logged = False
  elif _lane_lock_weight <= 1e-3:
    _lane_lock_has_lane_curvature = False

  if not _lane_lock_has_lane_curvature:
    log_lane_lock_mode(fallback_reason)
    return float(e2e_curvature)

  if candidate_lane_curvature is not None:
    log_lane_lock_mode("full-lane midpoint (lane-policy on)")
  else:
    log_lane_lock_mode(fallback_reason)
  return float(e2e_curvature + _lane_lock_weight * (_lane_lock_lane_curvature - e2e_curvature))
