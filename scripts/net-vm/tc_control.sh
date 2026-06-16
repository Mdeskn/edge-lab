#!/bin/bash

# Traffic Control Helper for Triton experiments
# Interface used by router
IFACE="ens18"

show_usage() {
    echo ""
    echo "Usage:"
    echo "  sudo ./tc_control.sh clear"
    echo "  sudo ./tc_control.sh show"
    echo ""
    echo "  sudo ./tc_control.sh tbf <rate> <burst> <latency> [duration_seconds]"
    echo "  Example:"
    echo "    sudo ./tc_control.sh tbf 1gbit 2mbit 50ms"
    echo "    sudo ./tc_control.sh tbf 500mbit 1mbit 50ms 20"
    echo ""
    echo "  sudo ./tc_control.sh netem_tbf <delay> <jitter> <rate> <burst> <latency> [duration_seconds]"
    echo "  Example:"
    echo "    sudo ./tc_control.sh netem_tbf 50ms 20ms 1gbit 2mbit 50ms"
    echo "    sudo ./tc_control.sh netem_tbf 10ms 5ms 1gbit 2mbit 50ms 20"
    echo ""
    echo "  sudo ./tc_control.sh netem_loss_tbf <delay> <jitter> <loss_pct> <rate> <burst> <latency> [duration_seconds]"
    echo "  Example:"
    echo "    sudo ./tc_control.sh netem_loss_tbf 0ms 0ms 2% 5mbit 256kb 50ms"
    echo ""
    echo "Notes:"
    echo "  tbf latency is queue latency, not fixed artificial delay."
    echo "  netem adds artificial delay/jitter, but it behaved badly for your Triton/gRPC setup."
    echo ""
}

clear_tc() {
    echo "[INFO] Clearing tc rules on $IFACE ..."
    sudo tc qdisc del dev "$IFACE" root 2>/dev/null
    echo "[INFO] Current qdisc:"
    tc qdisc show dev "$IFACE"
}

show_tc() {
    echo "[INFO] Current qdisc on $IFACE:"
    tc qdisc show dev "$IFACE"
    echo ""
    echo "[INFO] Detailed statistics:"
    tc -s qdisc show dev "$IFACE"
}

run_with_optional_timer() {
    local duration="$1"

    echo ""
    echo "[INFO] Current qdisc:"
    tc qdisc show dev "$IFACE"

    if [[ -n "$duration" ]]; then
        echo ""
        echo "[INFO] Rule will run for $duration seconds..."
        sleep "$duration"
        echo "[INFO] Time finished. Clearing tc rules..."
        clear_tc
    else
        echo ""
        echo "[INFO] Rule will stay active until you run:"
        echo "       sudo ./tc_control.sh clear"
    fi
}

apply_tbf() {
    local rate="$1"
    local burst="$2"
    local latency="$3"
    local duration="$4"

    if [[ -z "$rate" || -z "$burst" || -z "$latency" ]]; then
        echo "[ERROR] Missing arguments for tbf."
        show_usage
        exit 1
    fi

    echo "[INFO] Applying TBF bandwidth shaping:"
    echo "       interface = $IFACE"
    echo "       rate      = $rate"
    echo "       burst     = $burst"
    echo "       latency   = $latency"

    sudo tc qdisc replace dev "$IFACE" root tbf rate "$rate" burst "$burst" latency "$latency"

    run_with_optional_timer "$duration"
}

apply_netem_tbf() {
    local delay="$1"
    local jitter="$2"
    local rate="$3"
    local burst="$4"
    local latency="$5"
    local duration="$6"

    if [[ -z "$delay" || -z "$jitter" || -z "$rate" || -z "$burst" || -z "$latency" ]]; then
        echo "[ERROR] Missing arguments for netem_tbf."
        show_usage
        exit 1
    fi

    echo "[INFO] Applying NETEM delay/jitter + TBF bandwidth shaping:"
    echo "       interface = $IFACE"
    echo "       delay     = $delay"
    echo "       jitter    = $jitter"
    echo "       rate      = $rate"
    echo "       burst     = $burst"
    echo "       latency   = $latency"

    sudo tc qdisc replace dev "$IFACE" root handle 1: netem delay "$delay" "$jitter"
    sudo tc qdisc add dev "$IFACE" parent 1:1 handle 10: tbf rate "$rate" burst "$burst" latency "$latency"

    run_with_optional_timer "$duration"
}

apply_netem_loss_tbf() {
    local delay="$1"
    local jitter="$2"
    local loss="$3"
    local rate="$4"
    local burst="$5"
    local latency="$6"
    local duration="$7"

    if [[ -z "$delay" || -z "$jitter" || -z "$loss" || -z "$rate" || -z "$burst" || -z "$latency" ]]; then
        echo "[ERROR] Missing arguments for netem_loss_tbf."
        show_usage
        exit 1
    fi

    echo "[INFO] Applying NETEM delay/jitter/loss + TBF bandwidth shaping:"
    echo "       interface = $IFACE"
    echo "       delay     = $delay"
    echo "       jitter    = $jitter"
    echo "       loss      = $loss"
    echo "       rate      = $rate"
    echo "       burst     = $burst"
    echo "       latency   = $latency"

    sudo tc qdisc replace dev "$IFACE" root handle 1: netem delay "$delay" "$jitter" loss "$loss"
    sudo tc qdisc add dev "$IFACE" parent 1:1 handle 10: tbf rate "$rate" burst "$burst" latency "$latency"

    run_with_optional_timer "$duration"
}

if [[ "$EUID" -ne 0 ]]; then
    echo "[ERROR] Please run with sudo:"
    echo "       sudo ./tc_control.sh ..."
    exit 1
fi

MODE="$1"

case "$MODE" in
    clear)
        clear_tc
        ;;
    show)
        show_tc
        ;;
    tbf)
        apply_tbf "$2" "$3" "$4" "$5"
        ;;
    netem_tbf)
        apply_netem_tbf "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    netem_loss_tbf)
        apply_netem_loss_tbf "$2" "$3" "$4" "$5" "$6" "$7" "$8"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
