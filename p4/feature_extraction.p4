/* ================================================================
 * feature_extraction.p4 - PolicySplit data plane.
 *
 * Pipeline:
 *   1. canonical bidirectional flow state in registers
 *   2. distinct-source and distinct-port estimation by linear counting
 *   3. drop admission in a ternary match-action table whose entries are
 *      policy-aligned cells emitted by the offline compiler
 *   4. escalation of cells absent from that policy to the controller as a
 *      well-formed UDP report
 *   5. mitigation from controller-installed exact-match rules
 * ================================================================ */

#include <core.p4>
#include <v1model.p4>
#include "headers.p4"
#include "parsers.p4"

#define FLOW_TABLE_SIZE      16384
#define DIVERSITY_TABLE_SIZE 4096
#define SKETCH_WIDTH         32
#define CLONE_SESSION_ID     100

const bit<8>  CLASS_UNKNOWN = 255;
const bit<8>  CONF_THRESHOLD = 179;   // defensive guard; policy entries exceed 0.89
// Packets a flow must contribute before the switch will classify it.
// Must equal DECISION_PACKETS in scripts/train_rf.py: the policy table is
// keyed on the register values as they stand at exactly this packet.
// Overridable at compile time so a measurement build can observe more
// flows than the deployed threshold admits.
#ifndef MIN_PKTS
#define MIN_PKTS 10
#endif
const bit<32> MIN_PKTS_TO_CLASSIFY = MIN_PKTS;
const bit<48> FLOW_TIMEOUT_US = 60000000;
// A spoofed flood spreads its packets over many single-packet flows, so
// no flow ever accumulates enough state to be classified. The victim-side
// sketch is what makes such a flood observable, and it triggers a report
// once the number of distinct sources addressing one victim crosses this
// level.
const bit<16> DIVERSITY_ALERT = 16;

// ---- per-flow state ----
register<bit<32>>(FLOW_TABLE_SIZE) flow_pkts;
register<bit<32>>(FLOW_TABLE_SIZE) flow_bytes;
register<bit<48>>(FLOW_TABLE_SIZE) flow_last_ts;
register<bit<32>>(FLOW_TABLE_SIZE) flow_iat_ewma;
register<bit<32>>(FLOW_TABLE_SIZE) flow_len_ewma;
register<bit<16>>(FLOW_TABLE_SIZE) flow_min_len;
register<bit<16>>(FLOW_TABLE_SIZE) flow_max_len;
register<bit<32>>(FLOW_TABLE_SIZE) flow_tag;
// 0 = observing, 1 = deferred, 2 = admitted drop.
register<bit<2>>(FLOW_TABLE_SIZE)  flow_decision;

// ---- source-diversity sketches ----
// One bitmap slot per victim; a source contributes a single bit, so the
// population count of the bitmap estimates the number of distinct
// sources addressing that victim. The count is maintained incrementally
// because the target has no population-count primitive.
register<bit<32>>(DIVERSITY_TABLE_SIZE) victim_src_bitmap;
register<bit<16>>(DIVERSITY_TABLE_SIZE) victim_src_count;
register<bit<32>>(DIVERSITY_TABLE_SIZE) source_port_bitmap;
register<bit<16>>(DIVERSITY_TABLE_SIZE) source_port_count;
register<bit<8>>(DIVERSITY_TABLE_SIZE)  victim_reported;

// ---- counters exposed to the control plane ----
counter(8, CounterType.packets) class_verdicts;
counter(3, CounterType.packets) pipeline_events;   // 0 fwd, 1 drop, 2 escalate

control SplitMambaIngress(inout headers_t hdr,
                          inout metadata_t meta,
                          inout standard_metadata_t standard_metadata) {

    // ---------------- feature extraction ----------------

    action extract_l4_ports() {
        meta.src_port = 0;
        meta.dst_port = 0;
        if (hdr.tcp.isValid()) {
            meta.src_port = hdr.tcp.srcPort;
            meta.dst_port = hdr.tcp.dstPort;
        } else if (hdr.udp.isValid()) {
            meta.src_port = hdr.udp.srcPort;
            meta.dst_port = hdr.udp.dstPort;
        }
    }

    action compute_slots() {
                // Match the extractor's canonical bidirectional 5-tuple.
                if (hdr.ipv4.srcAddr < hdr.ipv4.dstAddr ||
                        (hdr.ipv4.srcAddr == hdr.ipv4.dstAddr &&
                         meta.src_port <= meta.dst_port)) {
                        hash(meta.flow_slot, HashAlgorithm.crc32, (bit<32>)0,
                                 { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                                     meta.src_port, meta.dst_port, hdr.ipv4.protocol },
                                 (bit<32>)FLOW_TABLE_SIZE);
                        hash(meta.flow_id, HashAlgorithm.crc32, (bit<32>)0,
                                 { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                                     meta.src_port, meta.dst_port, hdr.ipv4.protocol },
                                 (bit<32>)0xFFFFFFFF);
                } else {
                        hash(meta.flow_slot, HashAlgorithm.crc32, (bit<32>)0,
                                 { hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                                     meta.dst_port, meta.src_port, hdr.ipv4.protocol },
                                 (bit<32>)FLOW_TABLE_SIZE);
                        hash(meta.flow_id, HashAlgorithm.crc32, (bit<32>)0,
                                 { hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                                     meta.dst_port, meta.src_port, hdr.ipv4.protocol },
                                 (bit<32>)0xFFFFFFFF);
                }
        hash(meta.victim_slot, HashAlgorithm.crc16, (bit<32>)0,
             { hdr.ipv4.dstAddr }, (bit<32>)DIVERSITY_TABLE_SIZE);
        hash(meta.source_slot, HashAlgorithm.crc16, (bit<32>)0,
             { hdr.ipv4.srcAddr }, (bit<32>)DIVERSITY_TABLE_SIZE);
    }

    action update_flow_state() {
        bit<32> pkts;
        bit<32> bytes;
        bit<48> prev_ts;
        bit<32> ewma;
        bit<32> len_ewma;
        bit<16> min_len;
        bit<16> max_len;
        bit<32> tag;

        flow_pkts.read(pkts, meta.flow_slot);
        flow_bytes.read(bytes, meta.flow_slot);
        flow_last_ts.read(prev_ts, meta.flow_slot);
        flow_iat_ewma.read(ewma, meta.flow_slot);
        flow_len_ewma.read(len_ewma, meta.flow_slot);
        flow_min_len.read(min_len, meta.flow_slot);
        flow_max_len.read(max_len, meta.flow_slot);
        flow_tag.read(tag, meta.flow_slot);

        if (tag != meta.flow_id ||
            (prev_ts != 0 &&
             standard_metadata.ingress_global_timestamp - prev_ts >
             FLOW_TIMEOUT_US)) {
            pkts = 0;
            bytes = 0;
            prev_ts = 0;
            ewma = 0;
            len_ewma = 0;
            min_len = 0;
            max_len = 0;
            flow_tag.write(meta.flow_slot, meta.flow_id);
            flow_decision.write(meta.flow_slot, 0);
        }

        pkts = pkts + 1;
        bytes = bytes + (bit<32>)hdr.ipv4.totalLen;

        if (min_len == 0 || hdr.ipv4.totalLen < min_len) {
            min_len = hdr.ipv4.totalLen;
        }
        if (hdr.ipv4.totalLen > max_len) {
            max_len = hdr.ipv4.totalLen;
        }

        // The pipeline has no divider, so both running means are tracked
        // as exponentially weighted moving averages with a power-of-two
        // weight, which reduces to two shifts and an add.
        if (len_ewma == 0) {
            len_ewma = (bit<32>)hdr.ipv4.totalLen;
        } else {
            len_ewma = len_ewma - (len_ewma >> 3) +
                       ((bit<32>)hdr.ipv4.totalLen >> 3);
        }

        if (prev_ts != 0) {
            bit<32> delta =
                (bit<32>)(standard_metadata.ingress_global_timestamp - prev_ts);
            if (ewma == 0) {
                ewma = delta;
            } else {
                ewma = ewma - (ewma >> 3) + (delta >> 3);
            }
        }

        flow_pkts.write(meta.flow_slot, pkts);
        flow_bytes.write(meta.flow_slot, bytes);
        flow_last_ts.write(meta.flow_slot,
                           standard_metadata.ingress_global_timestamp);
        flow_iat_ewma.write(meta.flow_slot, ewma);
        flow_len_ewma.write(meta.flow_slot, len_ewma);
        flow_min_len.write(meta.flow_slot, min_len);
        flow_max_len.write(meta.flow_slot, max_len);

        meta.pkt_count = pkts;
        meta.byte_count = bytes;
        meta.mean_iat_us = ewma;
        meta.mean_len = len_ewma;
        meta.min_len = min_len;
        meta.max_len = max_len;
    }

    action update_source_diversity() {
        bit<32> victim_bits;
        bit<16> victim_n;
        bit<32> src_hash;

        hash(src_hash, HashAlgorithm.crc32, (bit<32>)0,
             { hdr.ipv4.srcAddr }, (bit<32>)SKETCH_WIDTH);
        victim_src_bitmap.read(victim_bits, meta.victim_slot);
        victim_src_count.read(victim_n, meta.victim_slot);

        bit<32> mask = (bit<32>)1 << (bit<8>)src_hash;
        if ((victim_bits & mask) == 0) {
            victim_src_bitmap.write(meta.victim_slot, victim_bits | mask);
            victim_n = victim_n + 1;
            victim_src_count.write(meta.victim_slot, victim_n);
        }
        meta.src_diversity = victim_n;
    }

    action update_port_fanout() {
        bit<32> port_bits;
        bit<16> port_n;
        bit<32> port_hash;

        hash(port_hash, HashAlgorithm.crc32, (bit<32>)0,
             { meta.dst_port }, (bit<32>)SKETCH_WIDTH);
        source_port_bitmap.read(port_bits, meta.source_slot);
        source_port_count.read(port_n, meta.source_slot);

        bit<32> mask = (bit<32>)1 << (bit<8>)port_hash;
        if ((port_bits & mask) == 0) {
            source_port_bitmap.write(meta.source_slot, port_bits | mask);
            port_n = port_n + 1;
            source_port_count.write(meta.source_slot, port_n);
        }
        meta.dstport_fanout = port_n;
    }

    // Quantisation to the buckets the table was fitted on. The cut points
    // are fitted on the training partition by scripts/train_rf.py, which
    // also emits this action, so both quantizers use the same representable
    // cut points. This does not make offline context counts equivalent to
    // the causal per-victim sketches maintained above.
#include "generated/bucketise.p4inc"

    // ---------------- forwarding and mitigation ----------------

    action forward(bit<9> port) {
        standard_metadata.egress_spec = port;
        meta.output_port = port;
    }

    action drop_packet() {
        mark_to_drop(standard_metadata);
    }

    table ipv4_lpm {
        key = { hdr.ipv4.dstAddr: lpm; }
        actions = { forward; drop_packet; NoAction; }
        size = 1024;
        default_action = NoAction();
    }

    // Populated by the controller once a flow has been adjudicated.
    direct_counter(CounterType.packets) mitigation_hits;

    table mitigation {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: ternary;
            meta.dst_port:    ternary;
        }
        actions = { drop_packet; NoAction; }
        size = 4096;
        default_action = NoAction();
        counters = mitigation_hits;
    }

    // ---------------- classification ----------------

    action set_verdict(bit<8> cls, bit<8> conf) {
        meta.dp_class = cls;
        meta.dp_confidence = conf;
    }

    action unknown_verdict() {
        meta.dp_class = CLASS_UNKNOWN;
        meta.dp_confidence = 0;
    }

    // Entries are exact ten-packet cells admitted by the drop policy.
    // Every absent cell falls through to controller escalation.
    table flow_classifier {
        key = {
            meta.byte_bucket:   ternary;
            meta.len_bucket:    ternary;
            meta.iat_bucket:    ternary;
            meta.minlen_bucket: ternary;
            meta.maxlen_bucket: ternary;
        }
        actions = { set_verdict; unknown_verdict; }
        size = 8192;
        default_action = unknown_verdict();
    }

    action escalate_to_controller(bit<8> reason) {
        meta.escalation_reason = reason;
        meta.escalate = true;
        clone_preserving_field_list(CloneType.I2E, CLONE_SESSION_ID, 1);
    }

    apply {
        if (!hdr.ipv4.isValid()) {
            return;
        }

        extract_l4_ports();
        compute_slots();
        update_flow_state();
        update_source_diversity();
        update_port_fanout();
        bucketise();

        ipv4_lpm.apply();

        // Mitigation installed by the controller takes precedence over
        // any data-plane verdict.
        if (mitigation.apply().hit) {
            pipeline_events.count(1);
            return;
        }

        meta.escalate = false;

        bit<2> decision;
        flow_decision.read(decision, meta.flow_slot);

        bit<8> victim_done;
        victim_reported.read(victim_done, meta.victim_slot);

        if (decision == 2) {
            drop_packet();
            pipeline_events.count(1);
        } else if (decision == 1) {
            if (meta.src_diversity >= DIVERSITY_ALERT && victim_done == 0) {
                victim_reported.write(meta.victim_slot, 1);
                escalate_to_controller(3);
                pipeline_events.count(2);
            } else {
                pipeline_events.count(0);
            }
        } else if (meta.pkt_count == MIN_PKTS_TO_CLASSIFY) {
            flow_classifier.apply();
            if (meta.dp_class == CLASS_UNKNOWN ||
                meta.dp_confidence < CONF_THRESHOLD) {
                flow_decision.write(meta.flow_slot, 1);
                escalate_to_controller(1);
                pipeline_events.count(2);
            } else {
                flow_decision.write(meta.flow_slot, 2);
                class_verdicts.count((bit<32>)meta.dp_class);
                escalate_to_controller(2);
                drop_packet();
                pipeline_events.count(1);
            }
        } else if (meta.pkt_count > MIN_PKTS_TO_CLASSIFY) {
            // A slot replacement that missed the boundary must fail closed.
            flow_decision.write(meta.flow_slot, 1);
            escalate_to_controller(1);
            pipeline_events.count(2);
        } else if (meta.src_diversity >= DIVERSITY_ALERT && victim_done == 0) {
            victim_reported.write(meta.victim_slot, 1);
            escalate_to_controller(3);
            pipeline_events.count(2);
        } else {
            // Not enough observations yet; forward and keep accumulating.
            pipeline_events.count(0);
        }
    }
}

control SplitMambaEgress(inout headers_t hdr,
                         inout metadata_t meta,
                         inout standard_metadata_t standard_metadata) {

    action build_report() {
        hdr.report.setValid();
        hdr.report.flow_id        = meta.flow_id;
        hdr.report.src_addr       = hdr.ipv4.srcAddr;
        hdr.report.dst_addr       = hdr.ipv4.dstAddr;
        hdr.report.src_port       = meta.src_port;
        hdr.report.dst_port       = meta.dst_port;
        hdr.report.protocol       = hdr.ipv4.protocol;
        hdr.report.dp_class       = meta.dp_class;
        hdr.report.dp_confidence  = meta.dp_confidence;
        hdr.report.reason         = meta.escalation_reason;
        hdr.report.pkt_count      = meta.pkt_count;
        hdr.report.byte_count     = meta.byte_count;
        hdr.report.src_diversity  = meta.src_diversity;
        hdr.report.dstport_fanout = meta.dstport_fanout;
        hdr.report.mean_iat_us    = meta.mean_iat_us;
        hdr.report.mean_len       = meta.mean_len;
        hdr.report.min_len        = meta.min_len;
        hdr.report.max_len        = meta.max_len;

        // The clone is re-cast as a UDP datagram carrying only the
        // report. Any transport header from the original packet is
        // removed so that the emitted frame is self-consistent.
        hdr.tcp.setInvalid();
        hdr.udp.setValid();
        hdr.udp.srcPort  = REPORT_UDP_PORT;
        hdr.udp.dstPort  = REPORT_UDP_PORT;
        hdr.udp.length   = 8 + 44;     // UDP header plus report_t
        hdr.udp.checksum = 0;          // optional for IPv4, and the
                                       // controller does not verify it

        hdr.ipv4.protocol = PROTO_UDP;
        hdr.ipv4.ihl      = 5;
        hdr.ipv4.totalLen = 20 + 8 + 44;
        hdr.ipv4.ttl      = 64;
    }

    apply {
        if (standard_metadata.instance_type == 1) {   // ingress clone
            build_report();
        }
    }
}

V1Switch(
    SplitMambaParser(),
    SplitMambaVerifyChecksum(),
    SplitMambaIngress(),
    SplitMambaEgress(),
    SplitMambaComputeChecksum(),
    SplitMambaDeparser()
) main;
