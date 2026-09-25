"""Тесты мониторинга RX, регулятора мощности и эхо-профиля.

Запуск (B200 не нужен, UHD импортируется лениво)::

    E:\\MySoftware\\SDR_Scan\\.venv\\Scripts\\python.exe tests\\test_monitor.py

Проверяются: ``rms_dbfs``, сходимость/пределы ``PowerRegulator``,
``delay_profile`` на синтетическом двухлучевом канале, ``estimate_noise_floor``,
а также импорт config/cli/gui и ``_collect()`` в offscreen-режиме.
"""

from __future__ import annotations

import math
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from gnss_sim.power import (  # noqa: E402
    PowerRegulator,
    delay_profile,
    estimate_noise_floor,
    rms_dbfs,
)

_TESTS = []


def _test(fn):
    _TESTS.append(fn)
    return fn


# ----------------------------------------------------------------------
@_test
def test_rms_dbfs() -> None:
    tone = np.exp(2j * np.pi * 0.1 * np.arange(4096)).astype(np.complex64)
    assert abs(rms_dbfs(tone) - 0.0) < 1e-3, rms_dbfs(tone)

    half = (0.5 * tone).astype(np.complex64)
    expected = 20.0 * math.log10(0.5)
    assert abs(rms_dbfs(half) - expected) < 1e-3

    assert rms_dbfs(np.zeros(0, dtype=np.complex64)) == float("-inf")


@_test
def test_regulator_bounded_step() -> None:
    reg = PowerRegulator(target_dbfs=0.0, step_db=1.0, ema=1.0,
                         deadband_db=0.0, initial_gain=0.0)
    assert reg.update(-50.0) == 1.0
    assert reg.update(50.0) == 0.0


@_test
def test_regulator_convergence() -> None:
    reg = PowerRegulator(target_dbfs=-30.0, step_db=1.0, ema=0.3,
                         gain_min=-30.0, gain_max=30.0, initial_gain=0.0)
    measured = -60.0
    gain = reg.gain
    for _ in range(200):
        gain = reg.update(measured)
        measured = -60.0 + gain  # линейная модель: 1 дБ усиления = 1 дБ уровня
    assert abs(measured - (-30.0)) <= 0.6, measured
    assert reg.converged()
    assert reg.gain_min <= gain <= reg.gain_max
    st = reg.state()
    assert st["updates"] == 200 and st["converged"] is True


@_test
def test_regulator_limits() -> None:
    reg = PowerRegulator(target_dbfs=-100.0, gain_min=-30.0, gain_max=30.0,
                         initial_gain=0.0)
    measured = -60.0
    gain = reg.gain
    for _ in range(500):
        gain = reg.update(measured)
        measured = -60.0 + gain
    assert gain == -30.0, gain
    assert measured >= -90.0 - 1e-9
    # И обратный предел (не может улететь вверх).
    reg2 = PowerRegulator(target_dbfs=100.0, gain_min=-30.0, gain_max=30.0)
    g2 = 0.0
    for _ in range(500):
        g2 = reg2.update(-60.0 + g2)
    assert g2 == 30.0, g2


@_test
def test_regulator_snr() -> None:
    reg = PowerRegulator(target_snr_db=20.0, initial_gain=0.0)
    noise = -80.0
    measured = -50.0
    snr = measured - noise
    for _ in range(400):
        gain = reg.update(measured, noise)
        measured = -50.0 + gain
        snr = measured - noise
    assert abs(snr - 20.0) <= 0.6, snr
    assert reg.state()["snr_db"] is not None


@_test
def test_delay_profile_two_taps() -> None:
    rng = np.random.default_rng(1234)
    fs = 1.0e6
    n = 8192
    d = 37
    tx = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    rx = tx.copy()
    rx[d:] = rx[d:] + 0.5 * tx[:-d]

    delays, levels = delay_profile(tx.astype(np.complex64),
                                   rx.astype(np.complex64), fs,
                                   max_delay=200e-6)
    assert delays.size > 0
    assert abs(levels[0]) < 0.2  # основной луч нормирован в 0 дБ

    mask = delays > 5.0 / fs
    i = int(np.argmax(levels[mask]))
    echo_delay = delays[mask][i]
    echo_level = levels[mask][i]
    assert abs(echo_delay - d / fs) < 1.5 / fs, echo_delay
    assert abs(echo_level - 20.0 * math.log10(0.5)) < 1.0, echo_level


@_test
def test_estimate_noise_floor() -> None:
    rng = np.random.default_rng(7)
    sigma = 0.1
    blocks = [(rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
              * sigma for _ in range(41)]
    # E|x|^2 = 2*sigma^2 -> 10log10(0.02) = -16.99 дБ
    expected = 10.0 * math.log10(2.0 * sigma * sigma)
    floor = estimate_noise_floor(blocks)
    assert abs(floor - expected) < 0.5, (floor, expected)
    assert estimate_noise_floor([]) == float("-inf")


@_test
def test_uhd_duplex_lazy_and_channel_guard() -> None:
    from gnss_sim.uhd_duplex import UhdDuplex
    from gnss_sim.uhd_tx import TxError
    # Без железа: проверяем, что одинаковые каналы отвергаются до открытия USRP.
    try:
        UhdDuplex(args="type=b200", tx_channel=0, rx_channel=0)
    except TxError as exc:
        assert "канал" in str(exc).lower() or "uhd" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("ожидался TxError для одинаковых TX/RX каналов")


@_test
def test_config_defaults() -> None:
    from gnss_sim.config import SimConfig
    cfg = SimConfig()
    assert cfg.monitor is False
    assert cfg.tx_power_target_dbfs == -30.0
    assert cfg.rx_channel == 1 and cfg.rx_antenna == "RX2"
    assert cfg.rx_gain == 30.0 and cfg.rx_center_freq == 0.0
    assert cfg.echo_analysis is False and cfg.tx_power_auto is True
    assert "monitor" in cfg.to_dict()


@_test
def test_cli_parser() -> None:
    from gnss_sim.cli import build_parser
    args = build_parser().parse_args([
        "--monitor", "--tx-power-target", "-40", "--no-tx-power-auto",
        "--rx-channel", "0", "--rx-ant", "RX2", "--rx-gain", "20",
        "--rx-freq", "1575.42e6", "--echo",
    ])
    assert args.monitor is True
    assert args.tx_power_target == -40.0
    assert args.tx_power_auto is False
    assert args.rx_channel == 0 and args.rx_antenna == "RX2"
    assert args.rx_gain == 20.0 and args.rx_freq == 1575.42e6
    assert args.echo is True
    # По умолчанию авторегулятор включён.
    assert build_parser().parse_args([]).tx_power_auto is True


@_test
def test_gui_collect_headless() -> None:
    from PyQt5 import QtWidgets
    from gnss_sim.gui import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.cb_monitor.setChecked(True)
    win.sp_target.setValue(-35.0)
    win.cb_tx_auto.setChecked(False)
    win.sp_rx_ch.setValue(0)
    win.ed_rx_ant.setText("RX2")
    win.sp_rx_gain.setValue(25.0)
    win.sp_rx_freq.setValue(1.6e9)
    win.cb_echo.setChecked(True)
    cfg = win._collect()
    assert cfg.monitor is True
    assert cfg.tx_power_target_dbfs == -35.0
    assert cfg.tx_power_auto is False
    assert cfg.rx_channel == 0 and cfg.rx_antenna == "RX2"
    assert cfg.rx_gain == 25.0 and cfg.rx_center_freq == 1.6e9
    assert cfg.echo_analysis is True
    win.close()
    del app


# ----------------------------------------------------------------------
def main() -> int:
    failures = 0
    for fn in _TESTS:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
