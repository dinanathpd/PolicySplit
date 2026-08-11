"""Parse the 44-byte PolicySplit report emitted by the P4 switch."""

try:
    from scapy.all import Packet, IntField, ByteField, ShortField
    from scapy.all import bind_layers, UDP

    class TelemetryHeader(Packet):
        """Field-for-field representation of ``report_t`` in headers.p4."""
        name = "PolicySplitReport"
        fields_desc = [
            IntField("flow_id", 0),
            IntField("src_addr", 0),
            IntField("dst_addr", 0),
            ShortField("src_port", 0),
            ShortField("dst_port", 0),
            ByteField("protocol", 0),
            ByteField("dp_class", 0),
            ByteField("dp_confidence", 0),
            ByteField("reason", 0),
            IntField("pkt_count", 0),
            IntField("byte_count", 0),
            ShortField("src_diversity", 0),
            ShortField("dstport_fanout", 0),
            IntField("mean_iat_us", 0),
            IntField("mean_len", 0),
            ShortField("min_len", 0),
            ShortField("max_len", 0),
        ]

    # Bind to UDP port 9999
    bind_layers(UDP, TelemetryHeader, dport=9999)

except ImportError:
    TelemetryHeader = None


def parse_telemetry_packet(packet) -> dict | None:
    """Extract the report fields from a Scapy packet."""
    if TelemetryHeader is None:
        raise ImportError("scapy is required. pip install scapy")

    if not packet.haslayer(TelemetryHeader):
        return None

    telem = packet[TelemetryHeader]

    return {
        name: int(getattr(telem, name))
        for name in (
            "flow_id", "src_addr", "dst_addr", "src_port", "dst_port",
            "protocol", "dp_class", "dp_confidence", "reason",
            "pkt_count", "byte_count", "src_diversity",
            "dstport_fanout", "mean_iat_us", "mean_len", "min_len",
            "max_len",
        )
    }
