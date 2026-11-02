#!/usr/bin/env bash
# Compile the TNA port with the open-p4studio SDE and keep the compiler's
# resource logs under compiled/<target>/, which scripts/plot_results.py reads.
# Usage: ./build.sh [tofino|tofino2 ...]   (default: tofino2 tofino)
set -u
SDE=${SDE:-$(cd "$(dirname "$0")/../.." && pwd)/../tofino-sde/open-p4studio/install}
export PATH=/usr/bin:$PATH LD_LIBRARY_PATH=$SDE/lib
cd "$(dirname "$0")"
for target in "${@:-tofino2 tofino}"; do
  arch=tna; [[ $target == tofino2 ]] && arch=t2na
  rm -rf "build/$target" && mkdir -p "build/$target" "compiled/$target"
  nice -n 10 "$SDE/bin/p4c" --target "$target" --arch "$arch" -g --verbose 2 \
    -o "build/$target" policysplit.p4 > "compiled/$target/compile.log" 2>&1
  status=$?
  echo "$target exit $status"
  grep -E "error|^[0-9]+ errors" "compiled/$target/compile.log" | head -20
  logs="build/$target/pipe/logs"
  if [[ $status -eq 0 && -d $logs ]]; then
    cp "$logs/metrics.json" "$logs/mau.resources.log" "$logs/table_summary.log" \
       "compiled/$target/"
    gzip -c "$logs/resources.json" > "compiled/$target/resources.json.gz"
  fi
done
