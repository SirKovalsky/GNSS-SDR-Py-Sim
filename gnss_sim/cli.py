"""Command line interface: ``python -m gnss_sim [options]``."""

from __future__ import annotations

import argparse
import os
import sys
import time

from .config import SimConfig
from .runner import SimulationRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gnss_sim",
        description="Генератор сигналов GPS L1 C/A + L1C (IQ-файл или USRP B210).")
    p.add_argument("-e", "--nav", default="",
                   help="RINEX navigation file (пусто = автоскачивание)")
    p.add_argument("-l", "--location", default="35.681298,139.766247,10.0",
                   help="static Lat,Lon,Hgt (deg, deg, m)")
    p.add_argument("-x", "--motion", default="",
                   help="user motion file (10 Hz CSV, ECEF or Lat,Lon,Hgt)")
    p.add_argument("-t", "--start", default="now",
                   help="start time 'YYYY/MM/DD,hh:mm:ss' or 'now'")
    p.add_argument("-d", "--duration", type=float, default=120.0,
                   help="duration [s] (0 = until stopped)")
    p.add_argument("--band", default="l1",
                   choices=["l1", "b1i", "all", "wide"],
                   help="полоса/сессия: l1 — L1/E1, центр 1575.42 МГц, "
                        "2.6 Мвыб/с (по умолч.); b1i — только BeiDou B1I, "
                        "центр 1561.098 МГц, 4.092 Мвыб/с; all — единый поток "
                        "всех систем (L1/E1 + B1I), центр/fs вычисляются "
                        "(≈1571.33 МГц / 25 Мвыб/с); wide — синоним all")
    p.add_argument("--combine", action="store_true",
                   help="явно объединить L1+B1I в один широкий поток "
                        "(≈1571.33 МГц / ~25 Мвыб/с, боковые лепестки BOC(6,1) "
                        "сохраняются). По умолчанию при L1+B1I выбирается узкая "
                        "полоса L1 (2.6 Мвыб/с), а B1I отключается")
    p.add_argument("-s", "--sample-rate", type=float, default=None,
                   help="sampling frequency [Hz] (по умолчанию — из --band)")
    p.add_argument("-f", "--center-freq", type=float, default=None,
                   help="TX center frequency [Hz] (по умолчанию — из --band)")
    p.add_argument("-o", "--output", default="",
                   help="output IQ file (empty = no file)")
    p.add_argument("--iq-input", default="",
                   help="готовый IQ-файл для повторного использования "
                        "(генерация пропускается; .json рядом задаёт "
                        "sample_rate/format/center)")
    p.add_argument("--iq-ram", action="store_true",
                   help="загрузить готовый IQ-файл целиком в RAM и "
                        "зацикливать из памяти (по умолчанию — с диска)")
    p.add_argument("--format", default="cs16",
                   choices=["cf32", "cs16", "cs8", "cs4"],
                   help="IQ file format (default cs16)")
    p.add_argument("--scale", type=float, default=10000.0,
                   help="float->integer sample scale")
    p.add_argument("--no-ca", action="store_true", help="disable GPS L1 C/A")
    p.add_argument("--no-l1c", action="store_true", help="disable GPS L1C")
    p.add_argument("--no-galileo", action="store_true",
                   help="disable Galileo E1")
    p.add_argument("--no-qzss", action="store_true", help="disable QZSS L1")
    p.add_argument("--no-sbas", action="store_true", help="disable SBAS L1")
    p.add_argument("--no-beidou", action="store_true",
                   help="отключить BeiDou B1I")
    p.add_argument("--auto-b1i", dest="auto_b1i", action="store_true",
                   default=True,
                   help="авто-подбор fs/центра под B1I вместе с L1 (по умолч. вкл)")
    p.add_argument("--no-auto-b1i", dest="auto_b1i", action="store_false",
                   help="не менять fs/центр; B1I вне полосы будет пропущен")
    p.add_argument("--no-auto-download", action="store_true",
                   help="не скачивать RINEX автоматически")
    p.add_argument("--source", default="auto",
                   choices=["auto", "bkg", "cddis"],
                   help="источник эфемерид: auto/bkg/cddis (default auto)")
    p.add_argument("--nav-mode", dest="nav_mode", default="auto",
                   choices=["auto", "merged"],
                   help="режим эфемерид: auto или merged (мультисистемный "
                        "G/E/J/C; merged публикуется только за прошлые сутки)")
    p.add_argument("--cddis-user", default="",
                   help="логин Earthdata для CDDIS")
    p.add_argument("--cddis-password", default="",
                   help="пароль Earthdata для CDDIS")
    p.add_argument("--cddis-token", default="",
                   help="bearer-токен Earthdata (вместо логина/пароля)")
    p.add_argument("--l1c-data", default="zeros", choices=["zeros", "cnav2"],
                   help="L1Cd data source")
    p.add_argument("--b1i-data", dest="b1i_data", default="d1",
                   choices=["d1", "placeholder"],
                   help="данные BeiDou B1I: d1 — реальное сообщение D1 "
                        "(NH20+BCH), placeholder — постоянный +1 (по умолч. d1)")
    p.add_argument("--el-mask", type=float, default=5.0,
                   help="elevation mask [deg]")
    p.add_argument("--amp", type=float, default=None,
                   help="общий масштаб амплитуды (один на все каналы, <1). "
                        "По умолчанию — исторический номинал 0.15 (проверен "
                        "на ZED-F9P). Задайте явно, чтобы получить ровно этот "
                        "множитель")
    p.add_argument("--headroom", type=float, nargs="?", const=0.7,
                   default=None, metavar="TARGET",
                   help="ВКЛЮЧИТЬ anti-clip: целевой пик композитного baseband "
                        "как доля полной шкалы формата (без значения — 0.7). "
                        "Учитывает масштаб cs16 (32767/output_scale), поэтому "
                        "обычная сцена не ослабляется. По умолчанию выключено — "
                        "исторический уровень 0.15")
    p.add_argument("--no-headroom", action="store_true",
                   help="отключить автоматический запас по уровню (anti-clip)")
    p.add_argument("--no-iono", action="store_true",
                   help="disable ionospheric delay")
    p.add_argument("--tx", action="store_true",
                   help="stream to USRP B210 instead of / in addition to file")
    p.add_argument("--uhd-args", default="type=b200", help="UHD device arguments")
    p.add_argument("--tx-channel", type=int, default=0, help="TX channel")
    p.add_argument("--tx-gain", type=float, default=10.0, help="TX gain [dB]")
    p.add_argument("--tx-antenna", default="TX/RX", help="TX antenna")
    p.add_argument("--tx-bandwidth", type=float, default=0.0,
                   help="TX analog bandwidth [Hz] (0 = выбрать автоматически в UHD)")
    p.add_argument("--clock-source", default="internal",
                   help="clock source: internal/external/gpsdo")
    p.add_argument("--monitor", action="store_true",
                   help="одновременный приём RX на другом канале (контроль TX)")
    p.add_argument("--tx-power-target", type=float, default=-30.0,
                   help="целевой уровень RX [dBFS] для авторегулятора")
    p.add_argument("--tx-power-auto", dest="tx_power_auto",
                   action="store_true", default=True,
                   help="автоматически подстраивать TX gain (по умолч. вкл)")
    p.add_argument("--no-tx-power-auto", dest="tx_power_auto",
                   action="store_false",
                   help="не трогать TX gain")
    p.add_argument("--rx-channel", type=int, default=1,
                   help="канал приёма (должен отличаться от TX!)")
    p.add_argument("--rx-ant", dest="rx_antenna", default="RX2",
                   help="антенна/порт RX")
    p.add_argument("--rx-gain", type=float, default=30.0, help="RX gain [дБ]")
    p.add_argument("--rx-freq", type=float, default=0.0,
                   help="частота RX [Гц] (0 = как TX; для шумового пола)")
    p.add_argument("--echo", action="store_true",
                   help="оценивать профиль задержек (эхо в камере)")
    p.add_argument("--tx-check", dest="tx_check", action="store_true",
                   default=False,
                   help="контроль передачи: принять RX на другом канале и "
                        "предупредить, если сигнал не обнаружен над шумом")
    p.add_argument("--tx-check-margin", type=float, default=6.0,
                   help="порог обнаружения передачи [дБ над шумом] "
                        "(по умолч. 6)")
    p.add_argument("--block-ms", type=float, default=500.0,
                   help="generation block size [ms]")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda"],
                   help="синтез: auto/cpu/cuda")
    p.add_argument("--no-loop", action="store_true",
                   help="не зацикливать: писать полную длительность")
    p.add_argument("--loop-seconds", type=float, default=0.0,
                   help="длина зацикливаемого сегмента [с] (0=авто)")
    p.add_argument("--tx-jitter", type=float, default=0.0,
                   help="буфер джиттера синтез->UHD [с] (0=авто, обычно 20)")
    p.add_argument("--memory-budget", type=float, default=0.0,
                   help="бюджет RAM для сегмента [ГБ] (0=60%% доступной)")
    p.add_argument("--list-devices", action="store_true",
                   help="list USRP devices and exit")
    p.add_argument("--uhd-log-level", default=None,
                   help="уровень логов UHD (UHD_LOG_LEVEL): fatal/error/warning/"
                        "info/debug. По умолчанию info: строки UHD видны в "
                        "консоли (stdout), но stderr остаётся чистым, поэтому "
                        "PowerShell не красит их оранжевым; нативный маркер "
                        "underflow «U» отфильтровывается отдельно")
    return p


def main(argv: list[str] | None = None) -> int:
    # The Windows console often uses a legacy code page (cp866/cp1251) that
    # cannot encode the log's ``≥``/``→``/``·`` glyphs.  A raw ``print`` then
    # raised ``UnicodeEncodeError`` inside ``run()`` and aborted the run before
    # TX (observed during the COM3 investigation).  Never let encoding kill a
    # run: replace unencodable characters instead.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - reconfigure is best effort
            pass

    # UHD logs ``[INFO]``/``[WARNING]`` to stderr; on Windows PowerShell paints
    # any native stderr output orange/red and turns it into
    # ``NativeCommandError`` even when the process exit code is 0.  Ask UHD for
    # a *visible* level (``info`` by default) and capture its native stderr (the
    # bare ``U``/``O`` markers bypass UHD_LOG_LEVEL); the capture forwards every
    # line to stdout, so UHD text is visible without orange PowerShell stderr.
    from .nativelog import install_native_stderr_filter, quiet_uhd

    quiet_uhd()
    install_native_stderr_filter()

    args = build_parser().parse_args(argv)
    if args.uhd_log_level:
        os.environ["UHD_LOG_LEVEL"] = str(args.uhd_log_level)

    if args.list_devices:
        from .uhd_tx import list_devices, uhd_version
        print(f"UHD: {uhd_version()}")
        try:
            for dev in list_devices(args.uhd_args):
                print(" ", dict(dev))
        except Exception as exc:  # noqa: BLE001
            print("  ошибка:", exc)
        return 0

    from .config import band_preset
    preset = band_preset(args.band) or band_preset("l1")
    fs = preset.fs if args.sample_rate is None else args.sample_rate
    center = (preset.center_freq if args.center_freq is None
              else args.center_freq)

    lat, lon, hgt = (float(x) for x in args.location.split(","))
    cfg = SimConfig(
        nav_file=args.nav, lat=lat, lon=lon, height=hgt,
        motion_file=args.motion, start_text=args.start, duration=args.duration,
        fs=fs, center_freq=center, band=args.band,
        fs_override=args.sample_rate is not None,
        center_override=args.center_freq is not None,
        combine=args.combine,
        enable_ca=not args.no_ca, enable_l1c=not args.no_l1c,
        enable_galileo=not args.no_galileo, enable_qzss=not args.no_qzss,
        enable_sbas=not args.no_sbas, enable_beidou=not args.no_beidou,
        auto_download=not args.no_auto_download,
        download_source=args.source, cddis_user=args.cddis_user,
        cddis_password=args.cddis_password, cddis_token=args.cddis_token,
        nav_mode=args.nav_mode,
        el_mask=args.el_mask, amp_scale=args.amp,
        headroom=args.headroom is not None and not args.no_headroom,
        headroom_target=(0.7 if args.headroom is None else args.headroom),
        iono_enable=not args.no_iono, l1c_data=args.l1c_data,
        b1i_data=args.b1i_data, auto_b1i=args.auto_b1i,
        output=args.output, output_format=args.format, output_scale=args.scale,
        iq_input=args.iq_input, iq_in_ram=args.iq_ram,
        use_usrp=args.tx, uhd_args=args.uhd_args, tx_channel=args.tx_channel,
        tx_gain=args.tx_gain, tx_antenna=args.tx_antenna,
        tx_bandwidth=args.tx_bandwidth, clock_source=args.clock_source,
        block_ms=args.block_ms, backend=args.backend,
        loop=not args.no_loop, loop_seconds=args.loop_seconds,
        tx_jitter_seconds=args.tx_jitter,
        memory_budget_gb=args.memory_budget,
        monitor=args.monitor, tx_power_target_dbfs=args.tx_power_target,
        tx_power_auto=args.tx_power_auto, rx_channel=args.rx_channel,
        rx_antenna=args.rx_antenna, rx_gain=args.rx_gain,
        rx_center_freq=args.rx_freq, echo_analysis=args.echo,
        tx_check=args.tx_check, tx_check_margin_db=args.tx_check_margin,
    )

    last = [0.0]

    def progress(frac, sim_s, wall, rate):
        if wall - last[0] >= 0.5 or frac >= 1.0:
            last[0] = wall
            pct = f"{frac*100:5.1f}%" if frac else "  --  "
            print(f"\r{pct}  {sim_s:7.2f} с / {wall:6.2f} с  ({rate:4.2f}x)",
                  end="", flush=True)

    def finished(err):
        print()
        if err:
            print("Ошибка:", err)

    runner = SimulationRunner(cfg, log=lambda m: print(m, flush=True),
                              progress=progress, finished=finished)
    try:
        runner.prepare()
    except Exception as exc:  # noqa: BLE001 - подготовка может упасть (UHD/RINEX)
        # Diagnostics stay on stdout; native stderr is captured separately.
        print(f"Ошибка: {exc}", flush=True)
        return 1
    runner.start()
    try:
        while runner.is_running():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nОстановка...")
        runner.stop()
        while runner.is_running():
            time.sleep(0.1)
    return 0 if runner.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
