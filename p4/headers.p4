/* ================================================================
 * headers.p4 - Header and metadata definitions.
 * ================================================================ */

#ifndef HEADERS_P4
#define HEADERS_P4

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length;
    bit<16> checksum;
}

/* Escalation report emitted towards the controller. It carries the
 * per-flow state the controller needs to re-key the flow and the
 * data-plane verdict it is being asked to revisit. */
header report_t {
    bit<32> flow_id;
    bit<32> src_addr;
    bit<32> dst_addr;
    bit<16> src_port;
    bit<16> dst_port;
    bit<8>  protocol;
    bit<8>  dp_class;          // data-plane verdict
    bit<8>  dp_confidence;     // confidence in 1/255 units
    bit<8>  reason;            // 1 = defer, 2 = policy verdict, 3 = diversity
    bit<32> pkt_count;
    bit<32> byte_count;
    bit<16> src_diversity;     // distinct sources seen towards this victim
    bit<16> dstport_fanout;    // distinct destination ports from this source
    bit<32> mean_iat_us;
    bit<32> mean_len;
    bit<16> min_len;
    bit<16> max_len;
}

/* Every field the escalated clone must carry out of ingress has to be
 * named in the clone field list, otherwise egress observes a zeroed
 * metadata struct and the report is emitted with meaningless contents. */
struct metadata_t {
    @field_list(1) bit<32> flow_slot;
    @field_list(1) bit<32> flow_id;
    @field_list(1) bit<32> pkt_count;
    @field_list(1) bit<32> byte_count;
    @field_list(1) bit<32> mean_iat_us;
    @field_list(1) bit<32> mean_len;
    @field_list(1) bit<16> min_len;
    @field_list(1) bit<16> max_len;
    @field_list(1) bit<16> src_diversity;
    @field_list(1) bit<16> dstport_fanout;
    @field_list(1) bit<16> src_port;
    @field_list(1) bit<16> dst_port;
    @field_list(1) bit<8>  dp_class;
    @field_list(1) bit<8>  dp_confidence;
    @field_list(1) bit<8>  escalation_reason;

    bit<32> victim_slot;
    bit<32> source_slot;
    bit<8>  len_bucket;
    bit<8>  iat_bucket;
    bit<8>  byte_bucket;
    bit<8>  minlen_bucket;
    bit<8>  maxlen_bucket;
    bit<9>  output_port;
    bool    escalate;
}

struct headers_t {
    ethernet_t  ethernet;
    ipv4_t      ipv4;
    tcp_t       tcp;
    udp_t       udp;
    report_t    report;
}

#endif  // HEADERS_P4
