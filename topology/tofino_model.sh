#!/bin/bash
# Tofino 2 model test bed for the compiled TNA program (p4/tna/policysplit.p4).
# Needs root for the driver's hugepage DMA pool, the veth pairs and raw
# packet I/O; everything else (scripts/tofino_model.py: table programming
# over BF-RT gRPC, counters, replay jobs) runs unprivileged.
#
#   sudo bash topology/tofino_model.sh up      # hugepages, veths, model, switchd, packet agent
#   sudo bash topology/tofino_model.sh agent   # restart only the packet agent
#   sudo bash topology/tofino_model.sh down    # stop everything, remove veths
#   sudo env MODEL_FLAGS=--time-disable bash topology/tofino_model.sh up
#                                              # model clock stands still (flood alert test)
#
# Environment: SDE (install dir of Open P4 Studio), WORK (runtime directory,
# default tmp/tofino under the repository).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SDE=${SDE:-$ROOT/../tofino-sde/open-p4studio/install}
B=$ROOT/p4/tna/build/tofino2
W=${WORK:-$ROOT/tmp/tofino}
MODEL_FLAGS=${MODEL_FLAGS:-}
OWNER=${SUDO_UID:-1000}:${SUDO_GID:-1000}
export PATH=$SDE/bin:/usr/sbin:/sbin:$PATH LD_LIBRARY_PATH=$SDE/lib
SYSCTL=/usr/sbin/sysctl; IP=/usr/bin/ip; ETHTOOL=/usr/sbin/ethtool
# veth pair i: model side veth(2i), test side veth(2i+1). Device ports:
# 8 to 11 front ports, 3 inspection, 2 controller.
VETHS="0 1 2 3 4 125"
declare -A DEVPORT=([0]=8 [1]=9 [2]=10 [3]=11 [4]=3 [125]=2)

agent_start() {
  nohup python3 "$ROOT/scripts/tofino_model.py" --work "$W" agent > "$W/agent.out" 2>&1 &
  echo $! > "$W/agent.pid"
}

up() {
  [[ $(id -u) -eq 0 ]] || { echo "run with sudo"; exit 1; }
  mkdir -p "$W/logs" "$W/inbox" "$W/outbox"
  # Hugepages need 2 MB contiguous blocks; free the page cache and compact
  # memory first, then retry the reservation a few times.
  want=192
  for attempt in 1 2 3 4 5 6; do
    have=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
    [[ $have -ge $want ]] && break
    sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
    sleep 2
    echo $want > /proc/sys/vm/nr_hugepages
    sleep 1
  done
  echo "hugepages: $(grep HugePages_Total /proc/meminfo)"
  if [[ $(grep HugePages_Total /proc/meminfo | awk '{print $2}') -lt 96 ]]; then
    echo "hugepage reservation failed; close memory-hungry programs and retry"; exit 1
  fi
  for i in $VETHS; do
    a=veth$((i*2)); b=veth$((i*2+1))
    $IP link show $a &>/dev/null || $IP link add $a type veth peer name $b
    for d in $a $b; do
      $IP link set $d mtu 10240 up
      $ETHTOOL -K $d rx off tx off sg off tso off gso off gro off 2>/dev/null || true
      $SYSCTL -qw net.ipv6.conf.$d.disable_ipv6=1 || true
    done
  done
  {
    echo '{"PortToVeth":['
    first=1
    for i in $VETHS; do
      [[ $first -eq 1 ]] || echo ','
      first=0
      printf '{"device_port":%d,"veth1":%d,"veth2":%d}' ${DEVPORT[$i]} $((i*2)) $((i*2+1))
    done
    echo ']}'
  } > "$W/ports.json"
  cat > "$W/policysplit.conf" <<EOF
{"chip_list":[{"chip_family":"tofino2","instance":0,
  "pcie_sysfs_prefix":"/sys/devices/pci0000:00/0000:00:03.0/0000:05:00.0",
  "sds_fw_path":"share/tofino_sds_fw/avago/firmware"}],
 "p4_devices":[{"device-id":0,"p4_programs":[{"program-name":"policysplit",
  "bfrt-config":"$B/bfrt.json",
  "p4_pipelines":[{"p4_pipeline_name":"pipe","context":"$B/pipe/context.json",
   "config":"$B/pipe/tofino2.bin","pipe_scope":[0,1,2,3],"path":"$B"}]}],
  "agent0":"lib/libpltfm_mgr.so"}],
 "switch_options":[{"device-id":0}]}
EOF
  cd "$W"
  nohup nice -n 10 tofino-model -d 1 -k 1 -f "$W/ports.json" \
    --p4-target-config "$W/policysplit.conf" --install-dir "$SDE" --chip-type 5 \
    --log-dir "$W/logs" --no-cli --logs-disable $MODEL_FLAGS > "$W/model.out" 2>&1 &
  echo $! > "$W/model.pid"
  echo "model flags: ${MODEL_FLAGS:-none}" > "$W/model.flags"
  sleep 8
  nohup nice -n 10 bf_switchd --install-dir "$SDE" --conf-file "$W/policysplit.conf" \
    --status-port 7777 --background < /dev/null > "$W/switchd.out" 2>&1 &
  echo $! > "$W/switchd.pid"
  agent_start
  sleep 20
  chown -R "$OWNER" "$W"
  echo "model $(cat "$W/model.pid") switchd $(cat "$W/switchd.pid") agent $(cat "$W/agent.pid")"
  tail -3 "$W/switchd.out"
}

agent_restart() {
  [[ -f $W/agent.pid ]] && kill "$(cat "$W/agent.pid")" 2>/dev/null
  pkill -f "tofino_model.py --work $W agent"
  sleep 1
  agent_start
  sleep 1
  chown -R "$OWNER" "$W"
  echo "agent $(cat "$W/agent.pid")"; cat "$W/agent.out"
}

down() {
  for p in agent switchd model; do
    [[ -f $W/$p.pid ]] && kill "$(cat "$W/$p.pid")" 2>/dev/null
  done
  sleep 2
  pkill -f tofino-model; pkill -f bf_switchd
  pkill -f "tofino_model.py --work $W agent"; pkill -f "$W/agent.py"
  for i in $VETHS; do $IP link del veth$((i*2)) 2>/dev/null; done
  chown -R "$OWNER" "$W"
  echo down
}

case ${1:-} in
  up) up ;;
  agent) agent_restart ;;
  down) down ;;
  *) echo "usage: $0 up|agent|down"; exit 1 ;;
esac
