"""Accept pandad 29-bit wire addresses for Vector-marked DBC IDs.

CANPacker emits the DBC address with bit 31 set. Live panda frames use the
raw 29-bit id. This file is runnable without pycapnp so Windows can prove
the GM HUD wire address 0x4C0000 is parsed.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

_OPENDBC_ROOT = Path(__file__).resolve().parents[3]
if str(_OPENDBC_ROOT) not in sys.path:
  sys.path.insert(0, str(_OPENDBC_ROOT))


def _stub_pkg(name: str) -> types.ModuleType:
  if name in sys.modules:
    return sys.modules[name]
  mod = types.ModuleType(name)
  mod.__path__ = []
  sys.modules[name] = mod
  return mod


def _stub_mod(name: str, **attrs) -> None:
  if name in sys.modules:
    return
  mod = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(mod, key, value)
  sys.modules[name] = mod


def _install_capnp_free_imports() -> None:
  car = sys.modules.get("opendbc.car")
  if car is not None and getattr(car, "__file__", None):
    return

  def _unused_checksum(*_args, **_kwargs):
    return 0

  _stub_pkg("opendbc.car")
  _stub_pkg("opendbc.car.honda")
  _stub_pkg("opendbc.car.toyota")
  _stub_pkg("opendbc.car.subaru")
  _stub_pkg("opendbc.car.chrysler")
  _stub_pkg("opendbc.car.hyundai")
  _stub_pkg("opendbc.car.volkswagen")
  _stub_pkg("opendbc.car.tesla")
  _stub_pkg("opendbc.car.body")
  _stub_pkg("opendbc.car.psa")
  _stub_mod("opendbc.car.honda.hondacan", honda_checksum=_unused_checksum)
  _stub_mod("opendbc.car.toyota.toyotacan", toyota_checksum=_unused_checksum)
  _stub_mod("opendbc.car.subaru.subarucan", subaru_checksum=_unused_checksum)
  _stub_mod("opendbc.car.chrysler.chryslercan", chrysler_checksum=_unused_checksum,
            fca_giorgio_checksum=_unused_checksum)
  _stub_mod("opendbc.car.hyundai.hyundaicanfd", hkg_can_fd_checksum=_unused_checksum)
  _stub_mod("opendbc.car.volkswagen.mlbcan", volkswagen_mlb_checksum=_unused_checksum)
  _stub_mod("opendbc.car.volkswagen.mqbcan", volkswagen_meb_alt_crc_checksum=_unused_checksum,
            volkswagen_mqb_meb_checksum=_unused_checksum, xor_checksum=_unused_checksum)
  _stub_mod("opendbc.car.tesla.teslacan", tesla_checksum=_unused_checksum)
  _stub_mod("opendbc.car.body.bodycan", body_checksum=_unused_checksum)
  _stub_mod("opendbc.car.psa.psacan", psa_checksum=_unused_checksum)
  _stub_mod("opendbc.car.carlog", carlog=logging.getLogger("carlog"))


def _import_can():
  try:
    from opendbc.can import CANPacker, CANParser
    from opendbc.can.parser import alt_dbc_address
    return CANPacker, CANParser, alt_dbc_address
  except ModuleNotFoundError as exc:
    if "capnp" not in str(exc):
      raise
    _install_capnp_free_imports()
    from opendbc.can import CANPacker, CANParser
    from opendbc.can.parser import alt_dbc_address
    return CANPacker, CANParser, alt_dbc_address


CANPacker, CANParser, alt_dbc_address = _import_can()

LKAS_HUD_DBC = "gm_global_a_lowspeed_1818125"
LKAS_HUD_MSG = "Lane_Departure_Warning_LS"
LKAS_HUD_SIGNAL = "LnKpAstDisbldIO"
LKAS_HUD_DBC_ADDR = 0x804C0000
LKAS_HUD_WIRE_ADDR = 0x4C0000


def oem_lkas_from_hud_disabled(disabled_io: int) -> bool:
  return int(disabled_io) == 0


def resolve_lkas_enabled(hud_enabled, oem_lkas_seen, oem_lkas_enabled, session_enabled) -> bool:
  if hud_enabled is not None:
    return hud_enabled
  if oem_lkas_seen:
    return oem_lkas_enabled
  return session_enabled


def test_alt_dbc_address_maps_hud_ids():
  assert alt_dbc_address(LKAS_HUD_WIRE_ADDR) == LKAS_HUD_DBC_ADDR
  assert alt_dbc_address(LKAS_HUD_DBC_ADDR) == LKAS_HUD_WIRE_ADDR


def test_hud_live_panda_wire_address_updates_lkas():
  packer = CANPacker(LKAS_HUD_DBC)
  parser = CANParser(LKAS_HUD_DBC, [(LKAS_HUD_MSG, 0)], 0)
  marked_addr, enabled_dat, bus = packer.make_can_msg(LKAS_HUD_MSG, 0, {LKAS_HUD_SIGNAL: 0})
  _, disabled_dat, _ = packer.make_can_msg(LKAS_HUD_MSG, 0, {LKAS_HUD_SIGNAL: 1})
  assert marked_addr == LKAS_HUD_DBC_ADDR
  wire_addr = marked_addr & 0x1FFFFFFF
  assert wire_addr == LKAS_HUD_WIRE_ADDR

  updated = parser.update([0, [(wire_addr, enabled_dat, bus)]])
  assert LKAS_HUD_DBC_ADDR in updated
  assert parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL] == 0
  hud_enabled = oem_lkas_from_hud_disabled(parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL])
  assert hud_enabled is True
  assert resolve_lkas_enabled(hud_enabled, False, False, False) is True

  updated = parser.update([1, [(wire_addr, disabled_dat, bus)]])
  assert LKAS_HUD_DBC_ADDR in updated
  assert parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL] == 1
  hud_enabled = oem_lkas_from_hud_disabled(parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL])
  assert hud_enabled is False
  assert resolve_lkas_enabled(hud_enabled, True, True, True) is False


def test_hud_packer_marked_address_still_parses():
  packer = CANPacker(LKAS_HUD_DBC)
  parser = CANParser(LKAS_HUD_DBC, [(LKAS_HUD_MSG, 0)], 0)
  enabled_msg = packer.make_can_msg(LKAS_HUD_MSG, 0, {LKAS_HUD_SIGNAL: 0})
  assert enabled_msg[0] == LKAS_HUD_DBC_ADDR
  parser.update([0, [enabled_msg]])
  assert parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL] == 0
  assert oem_lkas_from_hud_disabled(parser.vl[LKAS_HUD_MSG][LKAS_HUD_SIGNAL]) is True


if __name__ == "__main__":
  test_alt_dbc_address_maps_hud_ids()
  test_hud_packer_marked_address_still_parses()
  test_hud_live_panda_wire_address_updates_lkas()
  print("HUD wire address 0x4C0000 accepted")
