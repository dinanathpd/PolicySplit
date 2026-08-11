"""
mininet_topo.py — Mininet topology for Split-Mamba experiments.

Topology:
    attacker1 (h1) ---\
    attacker2 (h2) ----+--- [P4 Switch s1] --- victim (h4)
    benign    (h3) ---/         |
                           controller (h5)

Usage:
    sudo python topology/mininet_topo.py
"""

import sys
import os
import time

try:
    from mininet.net import Mininet
    from mininet.topo import Topo
    from mininet.link import TCLink
    from mininet.cli import CLI
    from mininet.log import setLogLevel
except ImportError:
    print("Mininet not installed. Run: sudo apt-get install mininet")
    sys.exit(1)

# Try to import P4-specific Mininet extensions
try:
    from p4_mininet import P4Switch, P4Host
    HAS_P4_MININET = True
except ImportError:
    HAS_P4_MININET = False


class DDoSTopology(Topo):
    """
    Simple topology with 3 sources, 1 victim, 1 controller,
    all connected through a single P4 switch.
    """

    def build(self, p4_json=None):
        # Add switch
        if HAS_P4_MININET and p4_json:
            switch = self.addSwitch(
                "s1",
                sw_path="simple_switch_grpc",
                json_path=p4_json,
                thrift_port=9090,
                grpc_port=9559,
                device_id=0,
            )
        else:
            switch = self.addSwitch("s1")

        # Add hosts
        attacker1 = self.addHost(
            "h1", ip="10.0.1.1/24", mac="00:00:00:00:01:01")
        attacker2 = self.addHost(
            "h2", ip="10.0.1.2/24", mac="00:00:00:00:01:02")
        benign = self.addHost("h3", ip="10.0.1.3/24", mac="00:00:00:00:01:03")
        victim = self.addHost("h4", ip="10.0.2.1/24", mac="00:00:00:00:02:01")
        controller = self.addHost(
            "h5", ip="10.0.3.1/24", mac="00:00:00:00:03:01")

        # Links
        self.addLink(attacker1, switch, bw=10000, delay="1ms")
        self.addLink(attacker2, switch, bw=10000, delay="1ms")
        self.addLink(benign, switch, bw=10000, delay="1ms")
        self.addLink(victim, switch, bw=10000, delay="1ms")
        self.addLink(controller, switch, bw=1000, delay="1ms")


def start_network(p4_json=None):
    """Create and start the Mininet network."""
    setLogLevel("info")

    topo = DDoSTopology(p4_json=p4_json)

    host_cls = P4Host if HAS_P4_MININET else None
    switch_cls = P4Switch if HAS_P4_MININET else None

    net = Mininet(
        topo=topo,
        host=host_cls,
        switch=switch_cls,
        link=TCLink,
        controller=None,
    )

    net.start()

    # Configure routes
    victim = net.get("h4")
    victim.cmd("ip route add 10.0.1.0/24 via 10.0.2.254")

    controller = net.get("h5")
    controller.cmd("ip route add 10.0.1.0/24 via 10.0.3.254")
    controller.cmd("ip route add 10.0.2.0/24 via 10.0.3.254")

    print("\n" + "=" * 60)
    print("  Split-Mamba Testbed Ready")
    print("  Hosts: h1 (attacker), h2 (attacker), h3 (benign),")
    print("         h4 (victim), h5 (controller)")
    print("=" * 60)
    print("\nStart controller on h5:")
    print("  h5 python controller/sdn_controller.py &")
    print("\nReplay traffic:")
    print("  h3 tcpreplay --intf1=h3-eth0 dataset/PCAP/Benign.pcap &")
    print("  h1 tcpreplay --intf1=h1-eth0 dataset/PCAP/DDoS-SYN.pcap")
    print("=" * 60 + "\n")

    time.sleep(2)
    return net


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-json", type=str, default=None,
                        help="Path to compiled P4 JSON")
    args = parser.parse_args()

    net = start_network(args.p4_json)
    CLI(net)
    net.stop()
