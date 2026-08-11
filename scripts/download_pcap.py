#!/usr/bin/env python3
"""
download_pcap.py — Download CIC-IoT-2023 PCAPs (1 per missing group) + supplementary.

The CIC research portal requires a free registration. Log in with a browser,
copy your own session cookie from the request headers, and pass it here.
Never commit that cookie.

Usage:
    python scripts/download_pcap.py --cookie "$CIC_COOKIE"
"""

import argparse
import os
import re
import sys
import time
import requests
from urllib.parse import unquote, quote


BASE = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
BROWSE = BASE + "browse.php"
DOWNLOAD = BASE + "download.php"

GROUP_MAP = {
    "Benign_Final": "Benign",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon", "Recon-PortScan": "Recon",
    "BrowserHijacking": "Web", "CommandInjection": "Web",
    "SqlInjection": "Web", "Uploading_Attack": "Web",
    "VulnerabilityScan": "Web", "XSS": "Web",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce",
    "Backdoor_Malware": "Web",
}

PREFERRED = {
    "DoS": "DoS-SYN_Flood",
    "Mirai": "Mirai-greeth_flood",
    "Recon": "Recon-PortScan",
    "Spoofing": "DNS_Spoofing",
    "BruteForce": "DictionaryBruteForce",
}


def get_session(cookie_str):
    s = requests.Session()
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:148.0) Gecko/20100101 Firefox/148.0"
    )
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            s.cookies.set(k.strip(), v.strip())
    return s


def list_files(session, path):
    """Returns (files, folders). files = [(filename, dl_path), ...]. folders = [path, ...]."""
    resp = session.get(BROWSE, params={"p": path})
    if "Registration required" in resp.text:
        print(f"  [!] Auth failed for {path}")
        return [], []

    files = []
    for m in re.finditer(r'href="download\.php\?file=([^"]+)"', resp.text):
        dl_path = unquote(m.group(1))
        fname = dl_path.split("/")[-1]
        files.append((fname, dl_path))

    folders = []
    for m in re.finditer(r'href="browse\.php\?p=([^"]+)"', resp.text):
        p = unquote(m.group(1))
        if p != path and p.startswith(path) and p != "":
            folders.append(p)

    return files, folders


def download_file(session, dl_path, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 500_000:
        mb = os.path.getsize(output_path) / 1e6
        print(f"  [SKIP] {output_path} ({mb:.1f} MB)")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = DOWNLOAD + "?file=" + quote(dl_path, safe="/")
    print(f"  [GET]  {dl_path}")

    try:
        resp = session.get(url, stream=True, timeout=60)
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            print(f"  [FAIL] Got HTML — cookie may have expired")
            return False

        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        t0 = time.time()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=128 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    speed = downloaded / (time.time() - t0 + 0.01) / 1e6
                    print(
                        f"\r  [{pct:5.1f}%] {downloaded/1e6:.1f}/{total/1e6:.1f} MB "
                        f"({speed:.1f} MB/s)   ",
                        end="", flush=True,
                    )
        print()

        with open(output_path, "rb") as f:
            hdr = f.read(15)
        if b"<!DOCTYPE" in hdr or b"<html" in hdr:
            print(f"  [FAIL] File is HTML, removing")
            os.remove(output_path)
            return False

        mb = os.path.getsize(output_path) / 1e6
        print(f"  [OK]   {mb:.1f} MB → {output_path}")
        return True

    except Exception as e:
        print(f"  [ERR]  {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cookie", required=True)
    ap.add_argument("--pcap-dir", default="dataset_cic_iot_2023/PCAP")
    ap.add_argument(
        "--supp-dir", default="dataset_cic_iot_2023/Supplementary Materials")
    ap.add_argument("--max-per-category", type=int, default=1)
    ap.add_argument("--all-categories", action="store_true",
                    help="Download from ALL categories, not just missing groups")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = get_session(args.cookie)

    # --- Check local PCAPs ---
    existing_groups = set()
    existing_cats = set()
    for cat_dir in sorted(os.listdir(args.pcap_dir)):
        cat_path = os.path.join(args.pcap_dir, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        pcaps = [
            f for f in os.listdir(cat_path)
            if f.endswith(".pcap")
            and os.path.getsize(os.path.join(cat_path, f)) > 500_000
        ]
        if pcaps:
            group = GROUP_MAP.get(cat_dir, "Unknown")
            existing_groups.add(group)
            existing_cats.add(cat_dir)
            print(f"  [LOCAL] {cat_dir} ({group}): {len(pcaps)} pcap(s)")

    all_groups = set(GROUP_MAP.values())
    missing = all_groups - existing_groups
    print(f"\n[*] Have: {sorted(existing_groups)}")
    print(f"[*] Missing: {sorted(missing)}")

    # --- List remote categories ---
    print(f"\n[*] Listing remote PCAP categories...")
    resp = session.get(BROWSE, params={"p": "PCAP"})
    if "Registration required" in resp.text:
        print("[!] Cookie expired. Get a new one from your browser.")
        sys.exit(1)

    remote_cats = []
    for m in re.finditer(r'href="browse\.php\?p=PCAP%2F([^"]+)"', resp.text):
        remote_cats.append(unquote(m.group(1)))
    print(f"[*] {len(remote_cats)} remote categories")

    # --- Decide what to download ---
    to_process = []
    if args.all_categories:
        for cat in remote_cats:
            if cat not in existing_cats:
                to_process.append(cat)
    else:
        for group in sorted(missing):
            if group in PREFERRED and PREFERRED[group] in remote_cats:
                to_process.append(PREFERRED[group])
            else:
                for cat in remote_cats:
                    if GROUP_MAP.get(cat) == group:
                        to_process.append(cat)
                        break

    print(f"\n[*] Will download from {len(to_process)} categories:")
    for cat in to_process:
        print(f"     {cat} ({GROUP_MAP.get(cat, '?')})")

    if args.dry_run:
        print("\n[DRY RUN]")
        return

    # --- Download PCAPs ---
    ok = fail = 0
    for cat in to_process:
        print(f"\n{'='*60}")
        print(f"Category: {cat} ({GROUP_MAP.get(cat, '?')})")

        files, _ = list_files(session, f"PCAP/{cat}")
        if not files:
            print(f"  [!] No files found")
            fail += 1
            continue

        print(f"  {len(files)} file(s): {[f[0] for f in files]}")
        for fname, dl_path in files[: args.max_per_category]:
            out = os.path.join(args.pcap_dir, cat, fname)
            if download_file(session, dl_path, out):
                ok += 1
            else:
                fail += 1

    # --- Supplementary Materials ---
    print(f"\n{'='*60}")
    print("[*] Supplementary Materials...")

    supp_files, supp_folders = list_files(session, "Supplementary Materials")
    for fname, dl_path in supp_files:
        out = os.path.join(args.supp_dir, fname)
        download_file(session, dl_path, out)

    for folder in supp_folders:
        sf, _ = list_files(session, folder)
        folder_name = folder.split("/")[-1]
        print(f"  Subfolder: {folder_name} ({len(sf)} files)")
        for fname, dl_path in sf:
            out = os.path.join(args.supp_dir, folder_name, fname)
            download_file(session, dl_path, out)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"[*] DONE: {ok} downloaded, {fail} failed")

    # Re-check coverage
    for cat_dir in os.listdir(args.pcap_dir):
        cat_path = os.path.join(args.pcap_dir, cat_dir)
        if os.path.isdir(cat_path):
            pcaps = [
                f for f in os.listdir(cat_path)
                if f.endswith(".pcap")
                and os.path.getsize(os.path.join(cat_path, f)) > 500_000
            ]
            if pcaps:
                existing_groups.add(GROUP_MAP.get(cat_dir, "Unknown"))

    still = all_groups - existing_groups
    if still:
        print(f"[!] Still missing: {sorted(still)}")
    else:
        print("[*] All 8 groups now covered!")


if __name__ == "__main__":
    main()
