"""PolicySplit report receiver and controller work queue."""

from controller.p4runtime_client import P4RuntimeClient
from controller.telemetry_parser import parse_telemetry_packet
import argparse
import logging
import queue
import threading
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("SDNController")


class SDNController:
    """
    Two-thread controller:
      Thread 1: Packet listener (Scapy sniff on UDP 9999)
    Thread 2: Report worker (policy verdicts and deferred flows)
    """

    def __init__(
        self,
        mamba_model_path: str | None = None,
        switch_addr: str = "127.0.0.1:9559",
        listen_port: int = 9999,
        attack_threshold: float = 0.7,
        num_classes: int = 8,
    ):
        self.listen_port = listen_port
        self.attack_threshold = attack_threshold
        self.num_classes = num_classes
        self.attack_classes = set(range(1, num_classes))  # 0 = Benign

        # Packet queue
        self.packet_queue: queue.Queue = queue.Queue(maxsize=1000)

        # The current report carries five raw statistics, not the packet
        # series expected by the legacy Mamba engines. A fallback model must
        # therefore provide a report-compatible adapter before it is loaded.
        self.mamba_engine = None
        if mamba_model_path:
            raise ValueError(
                "the current 44-byte report has no packet sequence; legacy "
                "Mamba/ONNX models are not report-compatible")

        # P4Runtime client
        self.p4_client = P4RuntimeClient(switch_addr)

        # Statistics
        self.stats = {
            "packets_received": 0,
            "policy_drops": 0,
            "deferred": 0,
            "rules_installed": 0,
        }

    def packet_listener(self):
        """Thread 1: Capture telemetry packets via Scapy."""
        try:
            from scapy.all import sniff

            def handler(pkt):
                data = parse_telemetry_packet(pkt)
                if data:
                    try:
                        self.packet_queue.put(data, block=False)
                    except queue.Full:
                        logger.warning("Queue full — dropping telemetry")

            logger.info("Listening for telemetry on UDP port %d…",
                        self.listen_port)
            sniff(
                filter=f"udp port {self.listen_port}",
                prn=handler,
                store=0,
            )
        except ImportError:
            logger.error("scapy not installed. pip install scapy")
        except Exception as e:
            logger.error("Listener error: %s", e)

    def inference_worker(self):
        """Consume reports without inventing unavailable sequence features."""
        logger.info("Report worker started.")

        while True:
            try:
                data = self.packet_queue.get(timeout=1)
                self.stats["packets_received"] += 1

                if self.mamba_engine is None:
                    if data["reason"] == 2 and data["dp_class"] in self.attack_classes:
                        self.stats["policy_drops"] += 1
                        logger.info(
                            "POLICY-DROP flow=%d class=%d conf=%d",
                            data["flow_id"], data["dp_class"],
                            data["dp_confidence"],
                        )
                    else:
                        self.stats["deferred"] += 1
                        logger.info(
                            "DEFER flow=%d reason=%d packets=%d",
                            data["flow_id"], data["reason"], data["pkt_count"],
                        )
                    self.packet_queue.task_done()
                    continue

            except queue.Empty:
                continue

    def stats_reporter(self):
        """Periodically log statistics."""
        while True:
            time.sleep(30)
            logger.info(
                "STATS | rx=%d policy_drops=%d rules=%d deferred=%d",
                self.stats["packets_received"],
                self.stats["policy_drops"],
                self.stats["rules_installed"],
                self.stats["deferred"],
            )

    def run(self):
        """Start all controller threads."""
        threads = [
            threading.Thread(target=self.packet_listener,
                             daemon=True, name="listener"),
            threading.Thread(target=self.inference_worker,
                             daemon=True, name="report-worker"),
            threading.Thread(target=self.stats_reporter,
                             daemon=True, name="stats"),
        ]

        for t in threads:
            t.start()

        logger.info("SDN Controller running. Press Ctrl+C to stop.")
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("Shutting down controller…")


def main():
    parser = argparse.ArgumentParser(description="PolicySplit SDN controller")
    parser.add_argument("--model", type=str,
                        default=None,
                        help="reserved for a report-compatible fallback model")
    parser.add_argument("--switch", type=str, default="127.0.0.1:9559")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--num-classes", type=int, default=8)
    args = parser.parse_args()

    controller = SDNController(
        mamba_model_path=args.model,
        switch_addr=args.switch,
        listen_port=args.port,
        attack_threshold=args.threshold,
        num_classes=args.num_classes,
    )
    controller.run()


if __name__ == "__main__":
    main()
