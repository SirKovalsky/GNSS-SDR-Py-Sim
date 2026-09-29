#!/usr/bin/env bash
# run_player_bg.sh — запуск плеера (передача готового IQ-файла на USRP) в фоне.
#
# Зачем: на удалённом Linux-хосте по SSH обычный процесс получает SIGHUP и
# умирает при закрытии сессии. Скрипт отвязывает плеер от терминала (nohup,
# stdin из /dev/null, вывод в лог), пишет PID-файл, так что передача
# продолжается после выхода из SSH.
#
# Плеер — это штатный CLI-режим playback: `python -m gnss_sim --iq-input FILE
# --tx`. Нативный C++ помощник (native/tx_player.cpp) собран под Windows/MSVC
# (windows.h), поэтому на Linux используется Python-путь UHD.
#
# Использование:
#   scripts/run_player_bg.sh start IQ.cs16 [доп. аргументы gnss_sim ...]
#   scripts/run_player_bg.sh stop
#   scripts/run_player_bg.sh restart IQ.cs16 [доп. аргументы ...]
#   scripts/run_player_bg.sh status
#   scripts/run_player_bg.sh log [N]        # последние N строк (по умолч. 40)
#
# Примеры:
#   scripts/run_player_bg.sh start out.cs16 --uhd-args type=b200 --tx-gain 18
#   scripts/run_player_bg.sh start out.cs16 --uhd-args type=b200,serial=XXXX
#   scripts/run_player_bg.sh log 100
#   scripts/run_player_bg.sh stop
#
# Переменные окружения:
#   PYTHON         интерпретатор (по умолч. <repo>/.venv/bin/python, иначе python3)
#   GNSS_BG_LOG    путь к логу (по умолч. <repo>/player.log)
#   GNSS_BG_PID    путь к PID  (по умолч. <repo>/player.pid)
#   GNSS_BG_EXTRA  доп. аргументы gnss_sim одной строкой (перед аргументами CLI)
#
# Возврат: 0 — успех; 1 — плеер уже/не запущен или не удержался; 2 — ошибка вызова.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${GNSS_BG_LOG:-$ROOT/player.log}"
PIDFILE="${GNSS_BG_PID:-$ROOT/player.pid}"

pick_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        printf '%s\n' "$PYTHON"
    elif [[ -x "$ROOT/.venv/bin/python" ]]; then
        printf '%s\n' "$ROOT/.venv/bin/python"
    else
        command -v python3
    fi
}

is_running() {
    [[ -f "$PIDFILE" ]] || return 1
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

do_start() {
    local iq="${1:-}"
    shift || true
    if [[ -z "$iq" ]]; then
        echo "ошибка: укажите IQ-файл: $0 start IQ.cs16 [аргументы]" >&2
        exit 2
    fi
    if is_running; then
        echo "плеер уже запущен (PID $(cat "$PIDFILE"))" >&2
        exit 1
    fi
    local py
    py="$(pick_python)"
    local -a args=(-m gnss_sim --iq-input "$iq" --tx)
    # GNSS_BG_EXTRA намеренно разворачивается по словам (это строка аргументов).
    # shellcheck disable=SC2206
    [[ -n "${GNSS_BG_EXTRA:-}" ]] && args+=(${GNSS_BG_EXTRA})
    args+=("$@")

    cd "$ROOT"
    : > "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start: $py ${args[*]}" >> "$LOG"
    echo "запуск: $py ${args[*]}"

    # nohup игнорирует SIGHUP и делает процесс невосприимчивым к закрытию SSH;
    # все дескрипторы уводятся из терминала, поэтому сессия закрывается сразу.
    nohup "$py" "${args[@]}" >>"$LOG" 2>&1 </dev/null &
    echo $! > "$PIDFILE"

    sleep 1
    if is_running; then
        echo "плеер запущен (PID $(cat "$PIDFILE"))"
        echo "лог: $LOG"
    else
        echo "плеер не удержался — смотрите лог: $LOG" >&2
        rm -f "$PIDFILE"
        tail -n 20 "$LOG" >&2 || true
        exit 1
    fi
}

do_stop() {
    if ! is_running; then
        echo "плеер не запущен"
        rm -f "$PIDFILE"
        return 0
    fi
    local pid
    pid="$(cat "$PIDFILE")"
    echo "остановка плеера (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "не завершился штатно — SIGKILL" >&2
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "остановлен"
}

do_status() {
    if is_running; then
        local pid
        pid="$(cat "$PIDFILE")"
        echo "плеер работает (PID $pid)"
        ps -o pid,etime,cmd -p "$pid" || true
        echo "лог: $LOG"
    else
        echo "плеер не запущен"
        return 1
    fi
}

do_log() {
    local n="${1:-40}"
    if [[ ! -f "$LOG" ]]; then
        echo "лог пуст: $LOG"
        return 0
    fi
    tail -n "$n" "$LOG"
}

cmd="${1:-}"
shift || true
case "$cmd" in
    start)   do_start "$@" ;;
    stop)    do_stop ;;
    restart) do_stop; do_start "$@" ;;
    status)  do_status ;;
    log|logs) do_log "$@" ;;
    *)
        echo "usage: $0 {start IQ.cs16 [args]|stop|restart IQ.cs16 [args]|status|log [N]}" >&2
        exit 2
        ;;
esac
