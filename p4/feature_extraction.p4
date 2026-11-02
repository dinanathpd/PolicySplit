/* ================================================================
 * feature_extraction.p4 - PolicySplit data plane.
 *
 * Pipeline:
 *   1. mitigation from controller-installed rules, ahead of all state
 *   2. canonical bidirectional flow state in a two-choice register table
 *   3. distinct-source and distinct-port estimation by linear counting,
 *      and a per-victim count of new flows in a one-second window
 *   4. fast-path admission in a ternary match-action table whose entries
 *      are policy-aligned cells emitted by the offline compiler; an entry
 *      names the action (drop, per-flow rate limit, or redirect)
 *   5. escalation of cells absent from that policy to the controller as a
 *      well-formed UDP report carrying twelve flow registers, and flood
 *      alerts when a victim's new-flow count crosses a power of two
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
// Fast-path actions named by the policy entries.
const bit<8>  ACT_DROP = 1;
const bit<8>  ACT_RATE_LIMIT = 2;
const bit<8>  ACT_REDIRECT = 3;
const bit<9>  INSPECT_PORT = 3;
// flow_decision values: 0 observing, 1 deferred, 2 drop, 3 rate-limit,
// 4 redirect (the fast-path action plus one).
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
// A flood also shows as flow-table pressure: its single-packet flows each
// claim a slot, so the number of slots claimed towards one destination in
// a window separates a flood from ordinary traffic by orders of magnitude.
// The count is kept per victim over a one-second tumbling window and a
// report goes to the controller each time it reaches a power of two from
// 256 up. The 256 level is the table's own capacity: 16,384 slots over the
// 60 s flow timeout is 273 new flows per second.
const bit<48> NEW_FLOW_WINDOW_US = 1000000;

// ---- per-flow state ----
// Each flow may live in one of two slots, chosen by two hashes of its
// canonical key; a new flow takes a free or expired slot, or else evicts
// the less recently used one. This keeps collisions rare at high flow
// arrival rates without adding memory.
register<bit<32>>(FLOW_TABLE_SIZE) flow_pkts;
register<bit<32>>(FLOW_TABLE_SIZE) flow_bytes;
register<bit<48>>(FLOW_TABLE_SIZE) flow_last_ts;
register<bit<32>>(FLOW_TABLE_SIZE) flow_iat_ewma;
register<bit<32>>(FLOW_TABLE_SIZE) flow_len_ewma;
register<bit<16>>(FLOW_TABLE_SIZE) flow_min_len;
register<bit<16>>(FLOW_TABLE_SIZE) flow_max_len;
register<bit<32>>(FLOW_TABLE_SIZE) flow_tag;
register<bit<3>>(FLOW_TABLE_SIZE)  flow_decision;
// Registers carried only by the report, for the controller adjudicator.
register<bit<8>>(FLOW_TABLE_SIZE)  flow_init_orient;
register<bit<48>>(FLOW_TABLE_SIZE) flow_first_ts;
register<bit<16>>(FLOW_TABLE_SIZE) flow_first_len;
register<bit<16>>(FLOW_TABLE_SIZE) flow_fwd_pkts;
register<bit<32>>(FLOW_TABLE_SIZE) flow_fwd_bytes;
register<bit<32>>(FLOW_TABLE_SIZE) flow_iat_min;
register<bit<32>>(FLOW_TABLE_SIZE) flow_iat_max;
register<bit<32>>(FLOW_TABLE_SIZE) flow_len_dev;

// Per-flow token buckets for the rate-limit action.
meter(FLOW_TABLE_SIZE, MeterType.packets) flow_meter;

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
// New flows (slot claims) towards each victim in the current window.
register<bit<32>>(DIVERSITY_TABLE_SIZE) victim_new_flows;
register<bit<48>>(DIVERSITY_TABLE_SIZE) victim_window_ts;

// ---- counters exposed to the control plane ----
counter(8, CounterType.packets) class_verdicts;
// 0 fwd, 1 drop, 2 escalate, 3 rate-limited and passed, 4 rate-limited
// and dropped, 5 redirected to the inspection port
counter(6, CounterType.packets) pipeline_events;

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
        // Match the extractor's canonical bidirectional 5-tuple. The first
        // slot hashes the key in that order, the second hashes it with the
        // ports first; flow_id is the tag stored in whichever slot is used.
        if (hdr.ipv4.srcAddr < hdr.ipv4.dstAddr ||
                (hdr.ipv4.srcAddr == hdr.ipv4.dstAddr &&
                 meta.src_port <= meta.dst_port)) {
            meta.orient = 1;
            hash(meta.slot_a, HashAlgorithm.crc32, (bit<32>)0,
                 { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                   meta.src_port, meta.dst_port, hdr.ipv4.protocol },
                 (bit<32>)FLOW_TABLE_SIZE);
            hash(meta.slot_b, HashAlgorithm.crc32, (bit<32>)0,
                 { meta.src_port, meta.dst_port,
                   hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.ipv4.protocol },
                 (bit<32>)FLOW_TABLE_SIZE);
            hash(meta.flow_id, HashAlgorithm.crc32, (bit<32>)0,
                 { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                   meta.src_port, meta.dst_port, hdr.ipv4.protocol },
                 (bit<32>)0xFFFFFFFF);
        } else {
            meta.orient = 0;
            hash(meta.slot_a, HashAlgorithm.crc32, (bit<32>)0,
                 { hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                   meta.dst_port, meta.src_port, hdr.ipv4.protocol },
                 (bit<32>)FLOW_TABLE_SIZE);
            hash(meta.slot_b, HashAlgorithm.crc32, (bit<32>)0,
                 { meta.dst_port, meta.src_port,
                   hdr.ipv4.dstAddr, hdr.ipv4.srcAddr, hdr.ipv4.protocol },
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

    action select_slot() {
        bit<32> tag_a;
        bit<32> tag_b;
        bit<48> ts_a;
        bit<48> ts_b;
        bit<48> now = standard_metadata.ingress_global_timestamp;
        flow_tag.read(tag_a, meta.slot_a);
        flow_tag.read(tag_b, meta.slot_b);
        flow_last_ts.read(ts_a, meta.slot_a);
        flow_last_ts.read(ts_b, meta.slot_b);
        bool free_a = tag_a == 0 ||
                      (ts_a != 0 && now - ts_a > FLOW_TIMEOUT_US);
        bool free_b = tag_b == 0 ||
                      (ts_b != 0 && now - ts_b > FLOW_TIMEOUT_US);
        if (tag_a == meta.flow_id) {
            meta.flow_slot = meta.slot_a;
        } else if (tag_b == meta.flow_id) {
            meta.flow_slot = meta.slot_b;
        } else if (free_a) {
            meta.flow_slot = meta.slot_a;
        } else if (free_b) {
            meta.flow_slot = meta.slot_b;
        } else if (ts_a <= ts_b) {
            meta.flow_slot = meta.slot_a;
        } else {
            meta.flow_slot = meta.slot_b;
        }
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
        bit<8>  init_orient;
        bit<48> first_ts;
        bit<16> first_len;
        bit<16> fwd_pkts;
        bit<32> fwd_bytes;
        bit<32> iat_min;
        bit<32> iat_max;
        bit<32> len_dev;
        bit<48> now = standard_metadata.ingress_global_timestamp;
        bit<16> len = hdr.ipv4.totalLen;

        flow_pkts.read(pkts, meta.flow_slot);
        flow_bytes.read(bytes, meta.flow_slot);
        flow_last_ts.read(prev_ts, meta.flow_slot);
        flow_iat_ewma.read(ewma, meta.flow_slot);
        flow_len_ewma.read(len_ewma, meta.flow_slot);
        flow_min_len.read(min_len, meta.flow_slot);
        flow_max_len.read(max_len, meta.flow_slot);
        flow_tag.read(tag, meta.flow_slot);
        flow_init_orient.read(init_orient, meta.flow_slot);
        flow_first_ts.read(first_ts, meta.flow_slot);
        flow_first_len.read(first_len, meta.flow_slot);
        flow_fwd_pkts.read(fwd_pkts, meta.flow_slot);
        flow_fwd_bytes.read(fwd_bytes, meta.flow_slot);
        flow_iat_min.read(iat_min, meta.flow_slot);
        flow_iat_max.read(iat_max, meta.flow_slot);
        flow_len_dev.read(len_dev, meta.flow_slot);

        if (tag != meta.flow_id ||
            (prev_ts != 0 && now - prev_ts > FLOW_TIMEOUT_US)) {
            pkts = 0;
            bytes = 0;
            prev_ts = 0;
            ewma = 0;
            len_ewma = 0;
            min_len = 0;
            max_len = 0;
            fwd_pkts = 0;
            fwd_bytes = 0;
            iat_min = 0;
            iat_max = 0;
            len_dev = 0;
            flow_tag.write(meta.flow_slot, meta.flow_id);
            flow_decision.write(meta.flow_slot, 0);
            meta.new_flow = 1;
        }

        pkts = pkts + 1;
        bytes = bytes + (bit<32>)len;
        if (pkts == 1) {
            // Direction is judged against the flow's first packet.
            init_orient = meta.orient;
            first_ts = now;
            first_len = len;
        }
        if (meta.orient == init_orient) {
            fwd_pkts = fwd_pkts + 1;
            fwd_bytes = fwd_bytes + (bit<32>)len;
        }

        if (min_len == 0 || len < min_len) {
            min_len = len;
        }
        if (len > max_len) {
            max_len = len;
        }

        // Deviation from the running mean length, taken before the mean
        // absorbs this packet; no update on the first packet.
        bit<32> dev;
        if ((bit<32>)len > len_ewma) {
            dev = (bit<32>)len - len_ewma;
        } else {
            dev = len_ewma - (bit<32>)len;
        }
        if (pkts > 1) {
            if (len_dev == 0) {
                len_dev = dev;
            } else {
                len_dev = len_dev - (len_dev >> 3) + (dev >> 3);
            }
        }

        // The pipeline has no divider, so both running means are tracked
        // as exponentially weighted moving averages with a power-of-two
        // weight, which reduces to two shifts and an add.
        if (len_ewma == 0) {
            len_ewma = (bit<32>)len;
        } else {
            len_ewma = len_ewma - (len_ewma >> 3) + ((bit<32>)len >> 3);
        }

        if (prev_ts != 0) {
            bit<32> delta = (bit<32>)(now - prev_ts);
            if (ewma == 0) {
                ewma = delta;
            } else {
                ewma = ewma - (ewma >> 3) + (delta >> 3);
            }
            if (pkts == 2 || delta < iat_min) {
                iat_min = delta;
            }
            if (delta > iat_max) {
                iat_max = delta;
            }
        }

        flow_pkts.write(meta.flow_slot, pkts);
        flow_bytes.write(meta.flow_slot, bytes);
        flow_last_ts.write(meta.flow_slot, now);
        flow_iat_ewma.write(meta.flow_slot, ewma);
        flow_len_ewma.write(meta.flow_slot, len_ewma);
        flow_min_len.write(meta.flow_slot, min_len);
        flow_max_len.write(meta.flow_slot, max_len);
        flow_init_orient.write(meta.flow_slot, init_orient);
        flow_first_ts.write(meta.flow_slot, first_ts);
        flow_first_len.write(meta.flow_slot, first_len);
        flow_fwd_pkts.write(meta.flow_slot, fwd_pkts);
        flow_fwd_bytes.write(meta.flow_slot, fwd_bytes);
        flow_iat_min.write(meta.flow_slot, iat_min);
        flow_iat_max.write(meta.flow_slot, iat_max);
        flow_len_dev.write(meta.flow_slot, len_dev);

        meta.pkt_count = pkts;
        meta.byte_count = bytes;
        meta.mean_iat_us = ewma;
        meta.mean_len = len_ewma;
        meta.min_len = min_len;
        meta.max_len = max_len;
        meta.fwd_pkts = fwd_pkts;
        meta.fwd_bytes = fwd_bytes;
        meta.iat_min_us = iat_min;
        meta.iat_max_us = iat_max;
        meta.duration_us = (bit<32>)(now - first_ts);
        meta.len_dev = len_dev;
        meta.first_len = first_len;
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

    // Slot claims towards this packet's destination in the current
    // one-second window; applied only to packets that claimed a slot.
    action update_new_flow_rate() {
        bit<32> n;
        bit<48> window_ts;
        bit<48> now = standard_metadata.ingress_global_timestamp;
        victim_new_flows.read(n, meta.victim_slot);
        victim_window_ts.read(window_ts, meta.victim_slot);
        if (window_ts == 0 || now - window_ts > NEW_FLOW_WINDOW_US) {
            n = 0;
            victim_window_ts.write(meta.victim_slot, now);
        }
        n = n + 1;
        victim_new_flows.write(meta.victim_slot, n);
        meta.victim_new_flows = n;
    }

    action flood_alert(bit<8> level) {
        meta.alert_level = level;
        meta.flood_alert = true;
    }

    // The count passes each level exactly once per window, so an exact
    // match on the powers of two yields one report per level crossed.
    table flood_levels {
        key = { meta.victim_new_flows: exact; }
        actions = { flood_alert; NoAction; }
        const entries = {
            256   : flood_alert(8);
            512   : flood_alert(9);
            1024  : flood_alert(10);
            2048  : flood_alert(11);
            4096  : flood_alert(12);
            8192  : flood_alert(13);
            16384 : flood_alert(14);
            32768 : flood_alert(15);
        }
        default_action = NoAction();
        size = 16;
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

    // Populated by the controller once a flow has been adjudicated or a
    // flood alert has named a source. All three fields are ternary, so an
    // entry can name a flow, a source-victim pair, or a victim port for
    // every source.
    direct_counter(CounterType.packets) mitigation_hits;

    table mitigation {
        key = {
            hdr.ipv4.srcAddr: ternary;
            hdr.ipv4.dstAddr: ternary;
            meta.dst_port:    ternary;
        }
        actions = { drop_packet; NoAction; }
        size = 4096;
        default_action = NoAction();
        counters = mitigation_hits;
    }

    // ---------------- classification ----------------

    action set_verdict(bit<8> cls, bit<8> conf, bit<8> act) {
        meta.dp_class = cls;
        meta.dp_confidence = conf;
        meta.dp_action = act;
    }

    action unknown_verdict() {
        meta.dp_class = CLASS_UNKNOWN;
        meta.dp_confidence = 0;
        meta.dp_action = 0;
    }

    // Entries are exact ten-packet cells admitted by the policy, each
    // naming its action. Every absent cell falls through to escalation.
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

        meta.escalate = false;
        meta.new_flow = 0;
        meta.flood_alert = false;
        meta.alert_level = 0;

        extract_l4_ports();

        // Mitigation installed by the controller comes first: a blocked
        // packet touches no flow or victim state, so a blocked source
        // neither churns the flow table nor consumes the alert levels.
        if (mitigation.apply().hit) {
            pipeline_events.count(1);
            return;
        }

        compute_slots();
        select_slot();
        update_flow_state();
        if (meta.new_flow == 1) {
            update_new_flow_rate();
            flood_levels.apply();
        }
        update_source_diversity();
        update_port_fanout();
        bucketise();

        ipv4_lpm.apply();

        bit<3> decision;
        flow_decision.read(decision, meta.flow_slot);

        bit<8> victim_done;
        victim_reported.read(victim_done, meta.victim_slot);

        if (decision == 0 && meta.pkt_count == MIN_PKTS_TO_CLASSIFY) {
            flow_classifier.apply();
            if (meta.dp_class == CLASS_UNKNOWN ||
                meta.dp_confidence < CONF_THRESHOLD ||
                meta.dp_action == 0 || meta.dp_action > ACT_REDIRECT) {
                flow_decision.write(meta.flow_slot, 1);
                decision = 1;
                escalate_to_controller(1);
            } else {
                decision = (bit<3>)meta.dp_action + 1;
                flow_decision.write(meta.flow_slot, decision);
                class_verdicts.count((bit<32>)meta.dp_class);
                escalate_to_controller(2);
            }
        } else if (decision == 0 && meta.pkt_count > MIN_PKTS_TO_CLASSIFY) {
            // A slot replacement that missed the boundary must fail closed.
            flow_decision.write(meta.flow_slot, 1);
            decision = 1;
            escalate_to_controller(1);
        }

        if (decision == (bit<3>)ACT_DROP + 1) {
            drop_packet();
            pipeline_events.count(1);
        } else if (decision == (bit<3>)ACT_RATE_LIMIT + 1) {
            bit<2> color;
            flow_meter.execute_meter<bit<2>>(meta.flow_slot, color);
            if (color == 2) {           // red: over the committed rate
                drop_packet();
                pipeline_events.count(4);
            } else {
                pipeline_events.count(3);
            }
        } else if (decision == (bit<3>)ACT_REDIRECT + 1) {
            standard_metadata.egress_spec = INSPECT_PORT;
            pipeline_events.count(5);
        } else if (meta.escalate) {
            pipeline_events.count(2);
        } else if (meta.flood_alert) {
            // The packet that carried the count over a level names one
            // source of the surge; the controller blocks that pair and,
            // after enough distinct sources, the victim's port.
            escalate_to_controller(4);
            pipeline_events.count(2);
        } else if (meta.src_diversity >= DIVERSITY_ALERT && victim_done == 0) {
            victim_reported.write(meta.victim_slot, 1);
            escalate_to_controller(3);
            pipeline_events.count(2);
        } else {
            // Deferred, or not enough observations yet: forward.
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
        hdr.report.fwd_pkts       = meta.fwd_pkts;
        hdr.report.fwd_bytes      = meta.fwd_bytes;
        hdr.report.iat_min_us     = meta.iat_min_us;
        hdr.report.iat_max_us     = meta.iat_max_us;
        hdr.report.duration_us    = meta.duration_us;
        hdr.report.len_dev        = meta.len_dev;
        hdr.report.first_len      = meta.first_len;
        hdr.report.dp_action      = meta.dp_action;
        hdr.report.alert_level    = meta.alert_level;

        // The clone is re-cast as a UDP datagram carrying only the
        // report. Any transport header from the original packet is
        // removed so that the emitted frame is self-consistent.
        hdr.tcp.setInvalid();
        hdr.udp.setValid();
        hdr.udp.srcPort  = REPORT_UDP_PORT;
        hdr.udp.dstPort  = REPORT_UDP_PORT;
        hdr.udp.length   = 8 + 70;     // UDP header plus report_t
        hdr.udp.checksum = 0;          // optional for IPv4, and the
                                       // controller does not verify it

        hdr.ipv4.protocol = PROTO_UDP;
        hdr.ipv4.ihl      = 5;
        hdr.ipv4.totalLen = 20 + 8 + 70;
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
