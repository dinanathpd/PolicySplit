/* ================================================================
 * policysplit.p4 - PolicySplit data plane for the Tofino Native
 * Architecture (TNA), ported from feature_extraction.p4 (v1model, BMv2).
 *
 * The port keeps the structures of the BMv2 program: a two-choice
 * per-flow table, the per-victim and per-source sketches, the quantised
 * admission table, the mitigation table, the per-flow meter and the
 * escalation of a cloned packet rewritten as a report. It is written for
 * the resource report of the Tofino compiler (bf-p4c), so that stage,
 * SRAM, TCAM and stateful-ALU figures describe this program and not an
 * estimate. The deviations a stateful pipeline forces are these.
 *
 *  1. A register array is accessed once per packet, so the two candidate
 *     slots of a flow are probed in two tag banks of half the depth. The
 *     probe of bank A claims a matching or free slot; only a busy bank A
 *     sends the packet to bank B, which evicts its occupant when it does
 *     not match. BMv2 evicts the less recently used of the two. Every other
 *     per-flow register is one array over both banks, indexed by the bank
 *     bit and the slot the probe chose.
 *  2. A stateful operation returns either a stored value or a code, not
 *     both, so the tag register answers the probe with a code and a
 *     separate timestamp register returns the interval since the previous
 *     arrival. Idle slots are freed by a control-plane sweep that clears
 *     tags whose timestamp is older than the flow timeout; BMv2 reclaims
 *     them in band on the next packet.
 *  3. The stateful ALU has neither a shift nor an exact divider, so the
 *     recurrence v - (v >> 3) + (x >> 3) of the running means cannot run
 *     inside the register, and a register is accessed in one stage only,
 *     so a value read early cannot be written back late in the same pass.
 *     The first pass reads the three means (inter-arrival time, length,
 *     length deviation), the match-action ALUs compute the new values
 *     exactly as the BMv2 program does, and the packet is resubmitted
 *     with the new values, its slot and what the first pass decided in
 *     the 128-bit resubmit header; the second pass stores them and
 *     forwards or drops the packet as the first pass decided. A flow's
 *     packets up to and including its decision packet make this second
 *     pass. Later packets, and packets the mitigation table drops, do
 *     not, because the means are read only at the decision packet. The
 *     length extrema and the first length are 32-bit registers, because
 *     the stateful ALU takes the 32-bit length operand; BMv2 keeps them
 *     in 16 bits.
 *  4. The verdict register is accessed once, after the admission lookup,
 *     with the lookup result as its input; the first-packet reset and the
 *     deferral of a slot that missed its decision packet are folded into
 *     the same operation, and only the decision packet is reported. The
 *     confidence threshold is applied by the controller when it installs
 *     an entry, so the entry carries the verdict (action code plus one,
 *     or deferral) that the BMv2 program derives in the pipeline from the
 *     class, confidence and action fields. Three 8-bit registers keep the
 *     class, confidence and action of the lookup for the report.
 *  5. The per-victim and per-source bitmaps are 1-bit registers indexed by
 *     the slot and the hashed bit, which returns the old bit directly.
 *  6. Time is the 48-bit ingress MAC timestamp in nanoseconds with its
 *     low byte dropped, so one unit is 256 ns; the flow timeout and the
 *     report fields use that unit.
 *  7. The clone is an ingress mirror whose 47-byte header carries the
 *     register values to egress in report order at the narrowest widths
 *     that hold them; egress adds the flow key and zero padding and emits
 *     the 70-byte UDP report of the BMv2 program, byte for byte. The
 *     deparser does not mirror a packet it resubmits, so a report asked
 *     for by a packet that makes the second pass (the decision packet,
 *     or an alert before it) is mirrored on that second pass, which reads
 *     the stored flow and sketch values back; the report's duration then
 *     includes the resubmission. The reason, or the flood level, travels
 *     in the resubmit header.
 *  8. One table at the end of the pipeline sets the deparser's mirror,
 *     resubmit and drop flags from the phase, the verdict, the meter
 *     colour and the alert, tells the second pass whether to drop or
 *     redirect, and its direct counter replaces the event counter of the
 *     BMv2 program, so the exclusive outcome paths do not take a stage
 *     each.
 *  9. The per-victim new-flow counter keeps the window start and the
 *     count in one 64-bit register entry, so a single stateful operation
 *     restarts a stale window or advances a live one; it runs on the
 *     packets that claim a slot. The mitigation table, with all three
 *     key fields ternary, is applied before any stateful operation, as
 *     in the BMv2 program.
 *
 * Build (open-p4studio SDE, bf-p4c 9.13.4): ./build.sh, which runs
 *   p4c --target tofino2 --arch t2na -g --verbose 2 -o build/tofino2 policysplit.p4
 * and keeps the compiler's resource logs under compiled/tofino2/. The
 * program is written for Tofino 2: its resubmit header is 128 bits, which
 * Tofino 1 limits to 64 (the compiler rejects it, compiled/tofino/), and
 * its table placement takes 13 stages where Tofino 1 has 12.
 *
 * Values the controller applies when it installs entries, which the BMv2
 * program holds as constants: the confidence threshold 179 of 255 for a
 * verdict, the decision packet 10, and the flow timeout of 60 s (234,375,000
 * units of 256 ns) for the idle sweep.
 * ================================================================ */

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#define FLOW_TABLE_SIZE      16384
#define BANK_SIZE            8192     // two tag banks hold the 16,384 slots
#define DIVERSITY_TABLE_SIZE 4096
#define BITMAP_SIZE          131072   // DIVERSITY_TABLE_SIZE x 32 bits

typedef bit<13> slot_t;
typedef bit<14> idx_t;      // bank bit ++ slot
typedef bit<12> victim_t;
typedef bit<17> bitslot_t;

const bit<16> TYPE_IPV4        = 0x0800;
const bit<8>  PROTO_TCP        = 6;
const bit<8>  PROTO_UDP        = 17;
const bit<16> REPORT_UDP_PORT  = 9999;
const bit<8>  CLASS_UNKNOWN    = 255;
const bit<8>  ACT_DROP         = 1;
const bit<8>  ACT_RATE_LIMIT   = 2;
const bit<8>  ACT_REDIRECT     = 3;
const PortId_t INSPECT_PORT    = 3;
const PortId_t CONTROLLER_PORT = 2;
const MirrorId_t REPORT_SESSION = 100;
const bit<16> DIVERSITY_ALERT = 16;
// New-flow window of the per-victim flood counter: one second in 256 ns
// units. A report goes out each time the count reaches a power of two
// from 256 up (flood_levels).
const bit<32> NEW_FLOW_WINDOW = 3906250;
// flow_decision: 0 observing, 1 deferred, 2 drop, 3 rate limit, 4 redirect.
const bit<8>  DEC_DEFER = 1;
// The verdict register writes when its value is below the sentinel. Stored
// verdicts are 0 to 4, so 0xFF always writes, 1 writes an undecided slot
// and 0 never writes.
const bit<8>  SENT_ALWAYS    = 0xFF;
const bit<8>  SENT_UNDECIDED = 1;
const bit<8>  SENT_NEVER     = 0;
// What the second pass does with the packet.
const bit<8>  ENF_NONE     = 0;
const bit<8>  ENF_DROP     = 1;
const bit<8>  ENF_REDIRECT = 2;

header ethernet_h { bit<48> dstAddr; bit<48> srcAddr; bit<16> etherType; }
header ipv4_h {
    bit<4> version; bit<4> ihl; bit<8> diffserv; bit<16> totalLen;
    bit<16> identification; bit<3> flags; bit<13> fragOffset; bit<8> ttl;
    bit<8> protocol; bit<16> hdrChecksum; bit<32> srcAddr; bit<32> dstAddr;
}
header tcp_h {
    bit<16> srcPort; bit<16> dstPort; bit<32> seqNo; bit<32> ackNo;
    bit<4> dataOffset; bit<4> res; bit<8> flags; bit<16> window;
    bit<16> checksum; bit<16> urgentPtr;
}
header udp_h { bit<16> srcPort; bit<16> dstPort; bit<16> length; bit<16> checksum; }

// The mirror payload carries the report fields in report order and at the
// narrowest widths that hold them at the decision packet, so that it stays
// within the 48 bytes the mirror buffer accepts. It has no type byte: only
// mirrored packets reach the controller port, and the egress parser selects
// on that port. Egress parses it into the segments below and emits them
// unchanged between constant padding headers, which widens each field to
// the width of the 70-byte BMv2 report.
header mirror_h {
    bit<8>  dp_class; bit<8> dp_confidence; bit<8> reason;
    bit<8>  pkt_count; bit<24> byte_count;
    bit<8>  src_diversity; bit<8> dstport_fanout;
    bit<32> mean_iat; bit<32> mean_len; bit<16> min_len; bit<16> max_len;
    bit<16> fwd_pkts; bit<24> fwd_bytes; bit<32> iat_min; bit<32> iat_max;
    bit<32> duration; bit<32> len_dev; bit<16> first_len; bit<8> dp_action;
    bit<8>  alert_level;
}
header mverdict_h { bit<8> dp_class; bit<8> dp_confidence; bit<8> reason; }
header mpkts_h   { bit<8> pkt_count; }
header mbytes_h  { bit<24> byte_count; }
header mdiv_h    { bit<8> src_diversity; }
header mfan_h    { bit<8> dstport_fanout; }
header mstats_h  { bit<32> mean_iat; bit<32> mean_len; bit<16> min_len;
                   bit<16> max_len; bit<16> fwd_pkts; }
header mtail_h   { bit<24> fwd_bytes; bit<32> iat_min; bit<32> iat_max;
                   bit<32> duration; bit<32> len_dev; bit<16> first_len;
                   bit<8> dp_action; bit<8> alert_level; }
// Report bytes egress adds: the flow key, and zero padding that widens the
// narrow mirror fields to the report widths.
header rkey_h { bit<32> flow_id; bit<32> src_addr; bit<32> dst_addr;
                bit<16> src_port; bit<16> dst_port; bit<8> protocol; }
header pad8_h  { bit<8> z; }
header pad24_h { bit<24> z; }
// The resubmit header: the second pass stores the three means, enforces
// the first pass's verdict and, when the first pass asked for a report,
// mirrors the packet. It fills the 128-bit resubmit data exactly. The
// enforcement code is 1 for drop and 2 for redirect; the report code is
// the reason (1 to 3) or, for a flood alert, the level (8 to 15).
header resub_h {
    bit<8>  enforce; bit<8> rcode; bit<16> idx;
    bit<32> iat; bit<32> len; bit<32> dev;
}

struct headers_t {
    resub_h    resub;
    mverdict_h mverdict;
    mpkts_h    mpkts;
    mbytes_h   mbytes;
    mdiv_h     mdiv;
    mfan_h     mfan;
    mstats_h   mstats;
    mtail_h    mtail;
    ethernet_h ethernet;
    ipv4_h     ipv4;
    tcp_h      tcp;
    udp_h      udp;
    rkey_h     rkey;
    pad24_h    pad_pkts;
    pad8_h     pad_bytes;
    pad8_h     pad_div;
    pad8_h     pad_fan;
    pad8_h     pad_fwd;
}

// The per-victim new-flow counter: window start and count in one 64-bit
// register entry, so one stateful operation resets or advances both.
struct window_t { bit<32> ts; bit<32> n; }

struct ig_metadata_t {
    bit<32> now;            // 256 ns units
    bit<16> src_port; bit<16> dst_port; bit<16> port_min;
    bit<32> addr_lo;
    bit<8>  orient;
    slot_t  slot_a; slot_t slot_b; bit<32> flow_id;
    slot_t  slot_b_f; bit<32> tag_f;
    slot_t  slot_b_r; bit<32> tag_r;
    victim_t victim_slot; victim_t source_slot;
    bit<5>  src_bit; bit<5> port_bit;
    bitslot_t victim_bitslot; bitslot_t source_bitslot;
    bit<8>  code; idx_t idx; bit<1> second; bit<8> phase;
    bit<32> delta;
    bit<32> pkts; bit<32> bytes; bit<32> vid_len; bit<32> len32;
    bit<8>  fwd_state;
    bit<32> min_len; bit<32> max_len; bit<8> init_orient; bit<32> first_len;
    bit<16> min_len16; bit<16> max_len16; bit<16> first_len16;
    bit<32> first_ts; bit<16> fwd_pkts; bit<32> fwd_bytes;
    bit<32> iat_min; bit<32> iat_max;
    bit<32> iat_prev; bit<32> len_prev; bit<32> dev_prev;
    bit<32> iat_acc; bit<32> len_acc; bit<32> dev_acc;
    bit<32> iat_in; bit<32> len_in; bit<32> len_hi; bit<32> len_lo;
    bit<8>  byte_bucket; bit<8> len_bucket; bit<8> iat_bucket;
    bit<8>  minlen_bucket; bit<8> maxlen_bucket;
    bit<8>  dp_class; bit<8> dp_confidence; bit<8> dp_action;
    bit<8>  new_decision; bit<8> sentinel; bit<8> decision; bit<1> decided_now;
    bit<1>  old_src_bit; bit<1> old_port_bit; bit<1> alert; bit<8> alert_now;
    bit<32> new_flows; bit<8> alert_level;
    bit<16> src_diversity; bit<16> dstport_fanout;
    bit<8>  reason;
    bit<8>  pkts8; bit<24> bytes24; bit<24> fwd_bytes24; bit<8> diversity8; bit<8> fanout8;
    bit<8>  color;
    MirrorId_t mirror_session;
}

struct eg_metadata_t { bit<32> addr_lo; bit<16> port_min; }

// ---------------------------------------------------------------- ingress
parser IgParser(packet_in pkt, out headers_t hdr, out ig_metadata_t meta,
                out ingress_intrinsic_metadata_t ig_intr_md) {
    state start {
        pkt.extract(ig_intr_md);
        meta.src_port = 0;
        meta.dst_port = 0;
        transition select(ig_intr_md.resubmit_flag) {
            1: parse_resubmit;
            default: parse_port_metadata;
        }
    }
    state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
    // The resubmit data takes the place of the port metadata; on Tofino 2
    // the padding that follows it is skipped.
    state parse_resubmit {
        pkt.extract(hdr.resub);
#if __TARGET_TOFINO__ == 2
        pkt.advance(PORT_METADATA_SIZE - 128);
#endif
        transition parse_ethernet;
    }
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4_word;
            default: accept;
        }
    }
    state parse_ipv4_word {
        // The first word of the IPv4 header, read whole: its low half is the
        // length, which the byte counters take as a 32-bit operand after one
        // mask, so no action casts a 16-bit field.
        meta.vid_len = pkt.lookahead<bit<32>>();
        transition parse_ipv4;
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP: parse_tcp;
            PROTO_UDP: parse_udp;
            default: accept;
        }
    }
    state parse_tcp {
        pkt.extract(hdr.tcp);
        meta.src_port = hdr.tcp.srcPort;
        meta.dst_port = hdr.tcp.dstPort;
        transition accept;
    }
    state parse_udp {
        pkt.extract(hdr.udp);
        meta.src_port = hdr.udp.srcPort;
        meta.dst_port = hdr.udp.dstPort;
        transition accept;
    }
}

// Each quantised key is alone in an 8-bit container, so the five bucket
// tables write in one stage and the admission key is five whole bytes.
@pa_container_size("ingress", "meta.byte_bucket", 8)
@pa_container_size("ingress", "meta.len_bucket", 8)
@pa_container_size("ingress", "meta.iat_bucket", 8)
@pa_container_size("ingress", "meta.minlen_bucket", 8)
@pa_container_size("ingress", "meta.maxlen_bucket", 8)
control Ingress(inout headers_t hdr, inout ig_metadata_t meta,
                in ingress_intrinsic_metadata_t ig_intr_md,
                in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
                inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
                inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // ---- hashes: the key is symmetric in direction, so both directions
    // of a flow share the slots and the tag. Both orders of the key are
    // hashed at once and the order of the BMv2 key is selected a stage
    // later: the lower address leads with its own port; only when the
    // addresses are equal does the lower port lead. As in the BMv2
    // program, bank A's slot is the low bits of the tag; bank B's slot
    // hashes the key with the ports first. One action per hash, because
    // an action may carry at most 32 bits of hash output.
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_tag_f;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_tag_r;
    Hash<slot_t>(HashAlgorithm_t.CRC32) hash_slot_b_f;
    Hash<slot_t>(HashAlgorithm_t.CRC32) hash_slot_b_r;
    Hash<victim_t>(HashAlgorithm_t.CRC16) hash_victim;
    Hash<victim_t>(HashAlgorithm_t.CRC16) hash_source;
    Hash<bit<5>>(HashAlgorithm_t.CRC32) hash_src_bit;
    Hash<bit<5>>(HashAlgorithm_t.CRC32) hash_port_bit;

    action canonical_key() {
        meta.addr_lo = min(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr);
        meta.port_min = min(meta.src_port, meta.dst_port);
        meta.now = ig_intr_md.ingress_mac_tstamp[39:8];
        meta.len32 = meta.vid_len & 0x0000FFFF;
    }
    action hash_tag_forward() {
        meta.tag_f = hash_tag_f.get({hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                                     meta.src_port, meta.dst_port,
                                     hdr.ipv4.protocol});
    }
    action hash_slot_forward() {
        meta.slot_b_f = hash_slot_b_f.get({meta.src_port, meta.dst_port,
                                           hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                                           hdr.ipv4.protocol});
    }
    action hash_tag_reverse() {
        meta.tag_r = hash_tag_r.get({hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                                     meta.dst_port, meta.src_port,
                                     hdr.ipv4.protocol});
    }
    action hash_slot_reverse() {
        meta.slot_b_r = hash_slot_b_r.get({meta.dst_port, meta.src_port,
                                           hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                                           hdr.ipv4.protocol});
    }
    action use_forward() {
        meta.orient = 1;
        meta.flow_id = meta.tag_f; meta.slot_a = meta.tag_f[12:0];
        meta.slot_b = meta.slot_b_f;
    }
    action use_reverse() {
        meta.orient = 0;
        meta.flow_id = meta.tag_r; meta.slot_a = meta.tag_r[12:0];
        meta.slot_b = meta.slot_b_r;
    }
    action hash_sketch_slots() {
        meta.victim_slot = hash_victim.get({hdr.ipv4.dstAddr});
        meta.source_slot = hash_source.get({hdr.ipv4.srcAddr});
    }
    action hash_sketch_bits() {
        meta.src_bit = hash_src_bit.get({hdr.ipv4.srcAddr});
        meta.port_bit = hash_port_bit.get({meta.dst_port});
    }

    // ---- two tag banks. The tag register answers the probe with a code
    // (2 match, 1 claimed, 0 busy). Bank A claims a free slot; bank B takes
    // its slot over when it does not hold the flow. Idle slots are freed by
    // the control plane, which clears tags whose timestamp is older than
    // the flow timeout.
    Register<bit<32>, slot_t>(BANK_SIZE) tag_a;
    Register<bit<32>, slot_t>(BANK_SIZE) tag_b;
    RegisterAction<bit<32>, slot_t, bit<8>>(tag_a) probe_a = {
        void apply(inout bit<32> v, out bit<8> rv) {
            if (v == meta.flow_id) {
                rv = 2;
            } else if (v == 0) {
                v = meta.flow_id;
                rv = 1;
            } else {
                rv = 0;
            }
        }
    };
    RegisterAction<bit<32>, slot_t, bit<8>>(tag_b) probe_b = {
        void apply(inout bit<32> v, out bit<8> rv) {
            if (v == meta.flow_id) {
                rv = 2;
            } else {
                v = meta.flow_id;
                rv = 1;
            }
        }
    };
    // Bank B's probe forms the slot index in its own stage.
    action probe_bank_b() {
        meta.code = probe_b.execute(meta.slot_b);
        meta.idx = 1w1 ++ meta.slot_b;
    }
    action use_bank_a() { meta.idx = 1w0 ++ meta.slot_a; }
    // Every other per-flow register is one array over both banks, indexed
    // by the bank bit and the slot the probe chose. The timestamp register
    // returns the interval since the previous arrival and stores this one.
    Register<bit<32>, idx_t>(FLOW_TABLE_SIZE) last_ts;
    RegisterAction<bit<32>, idx_t, bit<32>>(last_ts) stamp = {
        void apply(inout bit<32> v, out bit<32> rv) { rv = meta.now - v; v = meta.now; }
    };
    // A claimed slot (code 1) starts the flow, as the reset of the BMv2
    // program does. The read action is for the second pass, which reads
    // the stored values back for the report.
#define FLOW_REGISTER(T, NAME, BODY)                                        \
    Register<T, idx_t>(FLOW_TABLE_SIZE) NAME;                                \
    RegisterAction<T, idx_t, T>(NAME) NAME##_act = {                         \
        void apply(inout T v, out T rv) { BODY } };                         \
    RegisterAction<T, idx_t, T>(NAME) NAME##_read = {                        \
        void apply(inout T v, out T rv) { rv = v; } };

    FLOW_REGISTER(bit<32>, pkts,
        if (meta.code == 1) { v = 1; } else { v = v + 1; } rv = v;)
    FLOW_REGISTER(bit<32>, bytes,
        if (meta.code == 1) { v = meta.len32; } else { v = v + meta.len32; } rv = v;)
    // Length extrema and first length in 32-bit registers: the stateful ALU
    // takes the 32-bit length operand (BMv2 keeps them in 16 bits).
    FLOW_REGISTER(bit<32>, min_len,
        if (meta.code == 1 || meta.len32 < v) { v = meta.len32; } rv = v;)
    FLOW_REGISTER(bit<32>, max_len,
        if (meta.code == 1 || meta.len32 > v) { v = meta.len32; } rv = v;)
    FLOW_REGISTER(bit<8>, init_orient,
        if (meta.code == 1) { v = meta.orient; } rv = v;)
    FLOW_REGISTER(bit<32>, first_len,
        if (meta.code == 1) { v = meta.len32; } rv = v;)
    FLOW_REGISTER(bit<32>, first_ts,
        if (meta.code == 1) { v = meta.now; } rv = v;)
    // One code tells the direction counters what to do (2 first packet,
    // 1 packet in the first direction, 0 other direction), because a
    // stateful operation takes two inputs.
    FLOW_REGISTER(bit<16>, fwd_pkts,
        if (meta.fwd_state == 2) { v = 1; } else if (meta.fwd_state == 1) { v = v + 1; } rv = v;)
    FLOW_REGISTER(bit<32>, fwd_bytes,
        if (meta.fwd_state == 2) { v = meta.len32; } else if (meta.fwd_state == 1) { v = v + meta.len32; } rv = v;)
    FLOW_REGISTER(bit<32>, iat_min,
        if (meta.second == 1 || meta.delta < v) { v = meta.delta; } rv = v;)
    FLOW_REGISTER(bit<32>, iat_max,
        if (meta.second == 1 || meta.delta > v) { v = meta.delta; } rv = v;)

    // Running means: the first pass reads them, the second pass (the
    // resubmitted packet) stores the values the first pass computed.
#define EWMA_REGISTER(NAME, FIELD)                                          \
    Register<bit<32>, idx_t>(FLOW_TABLE_SIZE) NAME;                          \
    RegisterAction<bit<32>, idx_t, bit<32>>(NAME) NAME##_read = {            \
        void apply(inout bit<32> v, out bit<32> rv) { rv = v; } };          \
    RegisterAction<bit<32>, idx_t, bit<32>>(NAME) NAME##_write = {           \
        void apply(inout bit<32> v) { v = hdr.resub.FIELD; } };

    EWMA_REGISTER(iat_mean, iat)
    EWMA_REGISTER(len_mean, len)
    EWMA_REGISTER(dev_mean, dev)

    // ---- verdict: one access after the admission lookup. A new flow
    // resets it; an undecided flow at its decision packet takes the lookup
    // result, and one past it (a slot that missed the packet) is deferred.
    // The three cases are one comparison against a sentinel, because the
    // stateful ALU takes two inputs.
    Register<bit<8>, idx_t>(FLOW_TABLE_SIZE) decision;
    RegisterAction<bit<8>, idx_t, bit<8>>(decision) decide = {
        void apply(inout bit<8> v, out bit<8> rv) {
            if (v < meta.sentinel) {
                v = meta.new_decision;
            }
            rv = v;
        }
    };
    // The admission lookup's class, confidence and action, stored at the
    // decision packet for the report the second pass sends.
#define VERDICT_REGISTER(NAME, FIELD)                                       \
    Register<bit<8>, idx_t>(FLOW_TABLE_SIZE) NAME;                           \
    RegisterAction<bit<8>, idx_t, bit<8>>(NAME) NAME##_write = {             \
        void apply(inout bit<8> v) { v = meta.FIELD; } };                   \
    RegisterAction<bit<8>, idx_t, bit<8>>(NAME) NAME##_read = {              \
        void apply(inout bit<8> v, out bit<8> rv) { rv = v; } };

    VERDICT_REGISTER(verdict_class, dp_class)
    VERDICT_REGISTER(verdict_confidence, dp_confidence)
    VERDICT_REGISTER(verdict_action, dp_action)

    // ---- sketches: one bit per (slot, hashed source or port), a count
    // per slot, and one alert flag per victim.
    Register<bit<1>, bitslot_t>(BITMAP_SIZE) victim_src_bits;
    Register<bit<1>, bitslot_t>(BITMAP_SIZE) source_port_bits;
    Register<bit<16>, victim_t>(DIVERSITY_TABLE_SIZE) victim_src_count;
    Register<bit<16>, victim_t>(DIVERSITY_TABLE_SIZE) source_port_count;
    Register<bit<8>, victim_t>(DIVERSITY_TABLE_SIZE) victim_reported;
    RegisterAction<bit<1>, bitslot_t, bit<1>>(victim_src_bits) set_src_bit = {
        void apply(inout bit<1> v, out bit<1> rv) { rv = v; v = 1; } };
    RegisterAction<bit<1>, bitslot_t, bit<1>>(source_port_bits) set_port_bit = {
        void apply(inout bit<1> v, out bit<1> rv) { rv = v; v = 1; } };
    RegisterAction<bit<16>, victim_t, bit<16>>(victim_src_count) count_src = {
        void apply(inout bit<16> v, out bit<16> rv) {
            if (meta.old_src_bit == 0) { v = v + 1; } rv = v; } };
    RegisterAction<bit<16>, victim_t, bit<16>>(source_port_count) count_port = {
        void apply(inout bit<16> v, out bit<16> rv) {
            if (meta.old_port_bit == 0) { v = v + 1; } rv = v; } };
    RegisterAction<bit<16>, victim_t, bit<16>>(victim_src_count) read_src = {
        void apply(inout bit<16> v, out bit<16> rv) { rv = v; } };
    RegisterAction<bit<16>, victim_t, bit<16>>(source_port_count) read_port = {
        void apply(inout bit<16> v, out bit<16> rv) { rv = v; } };
    // Returns 1 once per victim, on the first packet at or above the alert.
    RegisterAction<bit<8>, victim_t, bit<8>>(victim_reported) report_victim = {
        void apply(inout bit<8> v, out bit<8> rv) {
            if (meta.alert == 1 && v == 0) { v = 1; rv = 1; } else { rv = 0; } } };
    // Slot claims towards the victim in the current one-second window:
    // a stale window restarts at one, a live one counts up.
    Register<window_t, victim_t>(DIVERSITY_TABLE_SIZE) victim_new_flows;
    RegisterAction<window_t, victim_t, bit<32>>(victim_new_flows) count_new_flow = {
        void apply(inout window_t v, out bit<32> rv) {
            if (meta.now - v.ts > NEW_FLOW_WINDOW) {
                v.ts = meta.now;
                v.n = 1;
            } else {
                v.n = v.n + 1;
            }
            rv = v.n;
        }
    };
    action flood_alert(bit<8> level) { meta.alert_level = level; }
    // The count passes each level once per window, so an exact match on
    // the powers of two yields one report per level crossed.
    table flood_levels {
        key = { meta.new_flows : exact; }
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

    Meter<idx_t>(FLOW_TABLE_SIZE, MeterType_t.PACKETS) flow_meter;

    // ---- running-mean arithmetic: v - (v >> 3) + (x >> 3), as in the BMv2
    // program, seeded by the first observation. Each dependent step is its
    // own action, so each takes its own stage. Every mean is accumulated in
    // place in one field, the length deviation |len - mean| is max - min
    // (len_hi then holds the deviation and len_lo its shifted copy), and the
    // seeding cases are folded into the operands so that no step follows
    // the sum.
    action prepare_terms() {
        meta.iat_acc = meta.iat_prev >> 3;
        meta.len_acc = meta.len_prev >> 3;
        meta.dev_acc = meta.dev_prev >> 3;
        meta.iat_in = meta.delta >> 3;
        meta.len_in = meta.len32 >> 3;
        meta.len_hi = max(meta.len32, meta.len_prev);
        meta.len_lo = min(meta.len32, meta.len_prev);
    }
    action base_terms() {
        meta.iat_acc = meta.iat_prev - meta.iat_acc;
        meta.len_acc = meta.len_prev - meta.len_acc;
        meta.dev_acc = meta.dev_prev - meta.dev_acc;
        meta.len_hi = meta.len_hi - meta.len_lo;
    }
    // Second packet: the inter-arrival mean starts from this interval.
    action base_terms_seed_iat() {
        meta.iat_acc = 0;
        meta.iat_in = meta.delta;
        meta.len_acc = meta.len_prev - meta.len_acc;
        meta.dev_acc = meta.dev_prev - meta.dev_acc;
        meta.len_hi = meta.len_hi - meta.len_lo;
    }
    // First packet: the length mean starts from this length, the
    // inter-arrival mean and the deviation from zero.
    action base_terms_new() {
        meta.iat_acc = 0; meta.iat_in = 0;
        meta.len_acc = 0; meta.len_in = meta.len32;
        meta.dev_acc = 0; meta.len_hi = 0;
    }
    action dev_term() { meta.len_lo = meta.len_hi >> 3; }
    action seed_dev() { meta.dev_acc = 0; meta.len_lo = meta.len_hi; }
    action zero_dev() { meta.len_lo = 0; }
    // The two means the admission key reads are summed one stage before
    // the deviation, whose term needs the extra subtraction.
    action sum_means() {
        meta.iat_acc = meta.iat_acc + meta.iat_in;
        meta.len_acc = meta.len_acc + meta.len_in;
    }
    action sum_dev() { meta.dev_acc = meta.dev_acc + meta.len_lo; }

    // Where the flow stands relative to its decision packet: 0 before it,
    // 1 at it, 2 past it. Constant entries, so the test needs no comparison
    // of two 32-bit operands. The same action sets what the verdict
    // register will do: a new flow resets it, the decision packet writes
    // the admission result (deferral until the lookup says otherwise), a
    // slot past its decision packet is deferred, and nothing else writes.
    // The same action clears the admission fields, which only the
    // decision packet's lookup sets, so an alert report carries zeros.
    action phase_new() {
        meta.phase = 0; meta.decided_now = 0; meta.reason = 0;
        meta.sentinel = SENT_ALWAYS; meta.new_decision = 0;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0; }
    action phase_observing() {
        meta.phase = 0; meta.decided_now = 0; meta.reason = 0;
        meta.sentinel = SENT_NEVER; meta.new_decision = 0;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0; }
    action phase_decision() {
        meta.phase = 1; meta.decided_now = 1; meta.reason = 1;
        meta.sentinel = SENT_UNDECIDED; meta.new_decision = DEC_DEFER;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0; }
    action phase_past() {
        meta.phase = 2; meta.decided_now = 0; meta.reason = 1;
        meta.sentinel = SENT_UNDECIDED; meta.new_decision = DEC_DEFER;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0; }
    table decision_phase {
        key = { meta.code : ternary; meta.pkts : ternary; }
        actions = { phase_new; phase_observing; phase_decision; phase_past; }
        const entries = {
            (8w1, _)                       : phase_new();
            (_, 32w10 &&& 32w0xFFFFFFFF)   : phase_decision();
            (_, 32w0  &&& 32w0xFFFFFFF8)   : phase_observing();
            (_, 32w8  &&& 32w0xFFFFFFFE)   : phase_observing();
        }
        default_action = phase_past();
        size = 8;
    }

    // ---- quantisation to the cells of the compiled table. The cut points
    // are installed by the controller as ternary prefix sets (a range of a
    // 32-bit field expands to at most 62 prefixes), because a range match
    // on this target is limited to 20-bit keys and a key slice would split
    // the 32-bit arithmetic operands.
    action set_byte_bucket(bit<8> b) { meta.byte_bucket = b; }
    action set_len_bucket(bit<8> b) { meta.len_bucket = b; }
    action set_iat_bucket(bit<8> b) { meta.iat_bucket = b; }
    action set_minlen_bucket(bit<8> b) { meta.minlen_bucket = b; }
    action set_maxlen_bucket(bit<8> b) { meta.maxlen_bucket = b; }
    table bucket_bytes {
        key = { meta.bytes : ternary; } actions = { set_byte_bucket; }
        default_action = set_byte_bucket(0); size = 512; }
    table bucket_len {
        key = { meta.len_acc : ternary; } actions = { set_len_bucket; }
        default_action = set_len_bucket(0); size = 512; }
    table bucket_iat {
        key = { meta.iat_acc : ternary; } actions = { set_iat_bucket; }
        default_action = set_iat_bucket(0); size = 512; }
    table bucket_minlen {
        key = { meta.min_len : ternary; } actions = { set_minlen_bucket; }
        default_action = set_minlen_bucket(0); size = 512; }
    table bucket_maxlen {
        key = { meta.max_len : ternary; } actions = { set_maxlen_bucket; }
        default_action = set_maxlen_bucket(0); size = 512; }

    // ---- forwarding, mitigation and admission.
    action forward(PortId_t port) { ig_tm_md.ucast_egress_port = port; }
    action drop_packet() { ig_dprsr_md.drop_ctl = 1; }
    action redirect() { ig_tm_md.ucast_egress_port = INSPECT_PORT; }
    table ipv4_lpm {
        key = { hdr.ipv4.dstAddr : lpm; }
        actions = { forward; drop_packet; NoAction; }
        size = 1024; default_action = NoAction();
    }
    DirectCounter<bit<32>>(CounterType_t.PACKETS) mitigation_hits;
    action mitigate() { mitigation_hits.count(); drop_packet(); }
    action mitigation_miss() { mitigation_hits.count(); }
    // All three fields ternary: an entry names a flow, a source-victim
    // pair, or a victim port for every source.
    table mitigation {
        key = { hdr.ipv4.srcAddr : ternary; hdr.ipv4.dstAddr : ternary;
                meta.dst_port : ternary; }
        actions = { mitigate; mitigation_miss; }
        size = 4096; default_action = mitigation_miss();
        counters = mitigation_hits;
    }
    // An entry carries its verdict: the action code plus one when the
    // controller finds the confidence at or above the threshold and the
    // action one the pipeline can take (reason 2), deferral otherwise
    // (reason 1). The lookup runs at the decision packet only.
    action set_verdict(bit<8> cls, bit<8> conf, bit<8> act, bit<8> dec, bit<8> rsn) {
        meta.dp_class = cls; meta.dp_confidence = conf; meta.dp_action = act;
        meta.new_decision = dec; meta.reason = rsn; }
    action unknown_verdict() {
        meta.dp_class = CLASS_UNKNOWN; meta.dp_confidence = 0; meta.dp_action = 0; }
    table flow_classifier {
        key = { meta.byte_bucket : ternary; meta.len_bucket : ternary;
                meta.iat_bucket : ternary; meta.minlen_bucket : ternary;
                meta.maxlen_bucket : ternary; }
        actions = { set_verdict; unknown_verdict; }
        size = 8192; default_action = unknown_verdict();
    }
    Counter<bit<32>, bit<3>>(8, CounterType_t.PACKETS) class_verdicts;

    // ---- outcome: one table sets the deparser's mirror, resubmit and
    // drop flags from the phase, the alerts, the verdict and the meter
    // colour, and its direct counter counts each outcome. Packets up to
    // the decision packet are resubmitted and enforced in the second
    // pass, which the entry tells what to do; later packets are enforced
    // here. A resubmitted packet is not mirrored, so the reports of the
    // first two phases are sent by the second pass, which the report
    // code tells what to send, and only the alert of a packet past its
    // decision is mirrored here. As in the BMv2 program, a flood alert
    // takes precedence over the diversity alert. Meter colour 3 is red.
    DirectCounter<bit<32>>(CounterType_t.PACKETS) outcome_hits;
    action outcome_flags(MirrorType_t mirror, ResubmitType_t resub, bit<3> drop,
                         bit<8> enforce) {
        outcome_hits.count();
        ig_dprsr_md.mirror_type = mirror;
        ig_dprsr_md.resubmit_type = resub;
        ig_dprsr_md.drop_ctl = drop;
        hdr.resub.enforce = enforce;
        hdr.resub.rcode = meta.reason;
    }
    action outcome_alert(MirrorType_t mirror, ResubmitType_t resub) {
        outcome_hits.count();
        ig_dprsr_md.mirror_type = mirror;
        ig_dprsr_md.resubmit_type = resub;
        ig_dprsr_md.drop_ctl = 0;
        hdr.resub.enforce = ENF_NONE;
        hdr.resub.rcode = 3;
        meta.reason = 3;
    }
    action outcome_flood() {
        outcome_hits.count();
        ig_dprsr_md.mirror_type = 0;
        ig_dprsr_md.resubmit_type = 1;
        ig_dprsr_md.drop_ctl = 0;
        hdr.resub.enforce = ENF_NONE;
        hdr.resub.rcode = meta.alert_level;
    }
    table outcome {
        key = { meta.phase : ternary; meta.alert_level : ternary;
                meta.alert_now : ternary; meta.decision : ternary;
                meta.color : ternary; }
        actions = { outcome_flags; outcome_alert; outcome_flood; }
        const entries = {
            (8w1, _, _, 8w2, _)       : outcome_flags(0, 1, 0, ENF_DROP);      // decision packet, drop
            (8w1, _, _, 8w3, 8w3)     : outcome_flags(0, 1, 0, ENF_DROP);      // decision packet, rate limit, red
            (8w1, _, _, 8w4, _)       : outcome_flags(0, 1, 0, ENF_REDIRECT);  // decision packet, redirect
            (8w1, _, _, _, _)         : outcome_flags(0, 1, 0, ENF_NONE);      // decision packet: report
            (8w0, 8w8 &&& 8w0xF8, _, _, _) : outcome_flood();                 // observing, flood alert
            (8w0, _, 8w1, _, _)       : outcome_alert(0, 1);                    // observing, victim alert
            (8w0, _, _, _, _)         : outcome_flags(0, 1, 0, ENF_NONE);      // observing
            (8w2, _, _, 8w2, _)       : outcome_flags(0, 0, 1, ENF_NONE);      // drop
            (8w2, _, _, 8w3, 8w3)     : outcome_flags(0, 0, 1, ENF_NONE);      // rate limit, red
            (8w2, _, _, 8w3, _)       : outcome_flags(0, 0, 0, ENF_NONE);      // rate limit, pass
            (8w2, _, _, 8w4, _)       : outcome_flags(0, 0, 0, ENF_NONE);      // redirect
            (8w2, _, 8w1, _, _)       : outcome_alert(1, 0);                    // victim alert
        }
        default_action = outcome_flags(0, 0, 0, ENF_NONE);
        counters = outcome_hits;
        size = 16;
    }
    // The mirror fields that do not depend on the outcome.
    action report_fields() {
        meta.mirror_session = REPORT_SESSION;
        meta.first_ts = meta.now - meta.first_ts;   // the duration, in place
        meta.pkts8 = meta.pkts[7:0];
        meta.bytes24 = meta.bytes[23:0];
        meta.fwd_bytes24 = meta.fwd_bytes[23:0];
        meta.diversity8 = meta.src_diversity[7:0];
        meta.fanout8 = meta.dstport_fanout[7:0];
        meta.min_len16 = meta.min_len[15:0];
        meta.max_len16 = meta.max_len[15:0];
        meta.first_len16 = meta.first_len[15:0];
    }
    // What the second pass needs besides the outcome: the slot and the
    // three means. The header is emitted only when the outcome table
    // sets the resubmit flag.
    action resubmit_fields() {
        hdr.resub.setValid();
        hdr.resub.idx = (bit<16>)meta.idx;
        hdr.resub.iat = meta.iat_acc;
        hdr.resub.len = meta.len_acc;
        hdr.resub.dev = meta.dev_acc;
    }
    // Second pass: the slot from the header, then the reason and the
    // alert level from the report code, and the means the report carries.
    action second_pass_slot() { meta.idx = hdr.resub.idx[13:0]; }
    action set_report() {
        meta.reason = hdr.resub.rcode;
        meta.alert_level = 0;
        meta.iat_acc = hdr.resub.iat;
        meta.len_acc = hdr.resub.len;
        meta.dev_acc = hdr.resub.dev;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0;
        ig_dprsr_md.mirror_type = 1;
    }
    action set_report_flood() {
        meta.reason = 4;
        meta.alert_level = hdr.resub.rcode;
        meta.iat_acc = hdr.resub.iat;
        meta.len_acc = hdr.resub.len;
        meta.dev_acc = hdr.resub.dev;
        meta.dp_class = 0; meta.dp_confidence = 0; meta.dp_action = 0;
        ig_dprsr_md.mirror_type = 1;
    }

    apply {
        ipv4_lpm.apply();
        if (!hdr.ipv4.isValid()) { return; }
        // The key and hash computations are stateless and serve both passes.
        canonical_key();
        hash_tag_forward();
        hash_slot_forward();
        hash_tag_reverse();
        hash_slot_reverse();
        hash_sketch_slots();
        hash_sketch_bits();
        if (hdr.ipv4.srcAddr == hdr.ipv4.dstAddr) {
            if (meta.src_port == meta.port_min) { use_forward(); } else { use_reverse(); }
        } else if (hdr.ipv4.srcAddr == meta.addr_lo) {
            use_forward();
        } else {
            use_reverse();
        }
        // Second pass: store the means the first pass computed, send the
        // report it asked for from the stored values, then enforce the
        // verdict it carried.
        if (hdr.resub.isValid()) {
            second_pass_slot();
            iat_mean_write.execute(meta.idx);
            len_mean_write.execute(meta.idx);
            dev_mean_write.execute(meta.idx);
            if (hdr.resub.rcode != 0) {
                if (hdr.resub.rcode >= 8) { set_report_flood(); } else { set_report(); }
                meta.pkts = pkts_read.execute(meta.idx);
                meta.bytes = bytes_read.execute(meta.idx);
                meta.min_len = min_len_read.execute(meta.idx);
                meta.max_len = max_len_read.execute(meta.idx);
                meta.first_len = first_len_read.execute(meta.idx);
                meta.first_ts = first_ts_read.execute(meta.idx);
                meta.fwd_pkts = fwd_pkts_read.execute(meta.idx);
                meta.fwd_bytes = fwd_bytes_read.execute(meta.idx);
                meta.iat_min = iat_min_read.execute(meta.idx);
                meta.iat_max = iat_max_read.execute(meta.idx);
                meta.src_diversity = read_src.execute(meta.victim_slot);
                meta.dstport_fanout = read_port.execute(meta.source_slot);
                // The admission fields belong to the decision packet's
                // report; an alert report keeps the zeros.
                if (hdr.resub.rcode < 3) {
                    meta.dp_class = verdict_class_read.execute(meta.idx);
                    meta.dp_confidence = verdict_confidence_read.execute(meta.idx);
                    meta.dp_action = verdict_action_read.execute(meta.idx);
                }
                report_fields();
            }
            if (hdr.resub.enforce == ENF_REDIRECT) {
                redirect();
            } else if (hdr.resub.enforce == ENF_DROP) {
                drop_packet();
            }
            hdr.resub.setInvalid();
            return;
        }
        meta.victim_bitslot = meta.victim_slot ++ meta.src_bit;
        meta.source_bitslot = meta.source_slot ++ meta.port_bit;

        // Mitigation ahead of every stateful operation: a blocked packet is
        // dropped and counted by its entry and touches no flow or victim
        // state. The key and hash computations above are stateless.
        if (mitigation.apply().hit) { return; }

        // Two-choice probe: bank A first, bank B when A is busy. A claimed
        // slot counts as a new flow towards the packet's destination.
        meta.code = probe_a.execute(meta.slot_a);
        if (meta.code == 0) { probe_bank_b(); } else { use_bank_a(); }
        meta.alert_level = 0;
        if (meta.code == 1) {
            meta.new_flows = count_new_flow.execute(meta.victim_slot);
            flood_levels.apply();
        }

        // The stored means first: the arithmetic that follows them is the
        // longest chain of the pipeline. Then the counters and the
        // first-packet fields.
        meta.iat_prev = iat_mean_read.execute(meta.idx);
        meta.len_prev = len_mean_read.execute(meta.idx);
        meta.dev_prev = dev_mean_read.execute(meta.idx);
        meta.delta = stamp.execute(meta.idx);
        meta.pkts = pkts_act.execute(meta.idx);
        meta.bytes = bytes_act.execute(meta.idx);
        meta.min_len = min_len_act.execute(meta.idx);
        meta.max_len = max_len_act.execute(meta.idx);
        meta.init_orient = init_orient_act.execute(meta.idx);
        meta.first_len = first_len_act.execute(meta.idx);
        meta.first_ts = first_ts_act.execute(meta.idx);
        if (meta.pkts == 2) { meta.second = 1; } else { meta.second = 0; }
        if (meta.code == 1) { meta.fwd_state = 2; }
        else if (meta.orient == meta.init_orient) { meta.fwd_state = 1; }
        else { meta.fwd_state = 0; }
        decision_phase.apply();

        // Direction counts and inter-arrival extrema.
        meta.fwd_pkts = fwd_pkts_act.execute(meta.idx);
        meta.fwd_bytes = fwd_bytes_act.execute(meta.idx);
        meta.iat_min = iat_min_act.execute(meta.idx);
        meta.iat_max = iat_max_act.execute(meta.idx);

        // The running means. The stored value is from a stale slot on a new
        // flow, so the first packet seeds every mean.
        prepare_terms();
        if (meta.code == 1) {
            base_terms_new();
        } else if (meta.iat_prev == 0) {
            base_terms_seed_iat();
        } else {
            base_terms();
        }
        sum_means();
        if (meta.code == 1) {
            zero_dev();
        } else if (meta.dev_prev == 0) {
            seed_dev();
        } else {
            dev_term();
        }
        sum_dev();
        resubmit_fields();

        // Sketches.
        meta.old_src_bit = set_src_bit.execute(meta.victim_bitslot);
        meta.old_port_bit = set_port_bit.execute(meta.source_bitslot);
        meta.src_diversity = count_src.execute(meta.victim_slot);
        meta.dstport_fanout = count_port.execute(meta.source_slot);
        if (meta.src_diversity >= DIVERSITY_ALERT) { meta.alert = 1; } else { meta.alert = 0; }

        // Admission at the decision packet, on the registers as updated by
        // this packet, as in the BMv2 program.
        bucket_bytes.apply();
        bucket_len.apply();
        bucket_iat.apply();
        bucket_minlen.apply();
        bucket_maxlen.apply();
        if (meta.phase == 1) {
            flow_classifier.apply();
            verdict_class_write.execute(meta.idx);
            verdict_confidence_write.execute(meta.idx);
            verdict_action_write.execute(meta.idx);
        }
        meta.decision = decide.execute(meta.idx);
        report_fields();
        if (meta.decided_now == 1 && meta.reason == 2) {
            class_verdicts.count(meta.dp_class[2:0]);
        }

        // The meter, the redirect and the diversity alert; drops are
        // applied by the outcome table or in the second pass, because the
        // drop bit would also cancel the resubmission. As in the BMv2
        // program, the diversity alert (reason 3) is raised once per
        // victim by a packet of an observing or deferred flow, and a
        // flood alert on the same packet takes precedence (the outcome
        // table), so the diversity alert waits for a later packet.
        meta.alert_now = 0;
        meta.color = 0;
        if (meta.decision == ACT_RATE_LIMIT + 1) {
            meta.color = flow_meter.execute(meta.idx);
        } else if (meta.decision == ACT_REDIRECT + 1) {
            redirect();
        } else if (meta.decided_now == 0 && meta.decision != ACT_DROP + 1
                   && meta.alert_level == 0) {
            meta.alert_now = report_victim.execute(meta.victim_slot);
        }
        outcome.apply();
    }
}

control IgDeparser(packet_out pkt, inout headers_t hdr, in ig_metadata_t meta,
                   in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    Mirror() mirror;
    Resubmit() resubmit;
    apply {
        if (ig_dprsr_md.resubmit_type == 1) {
            resubmit.emit(hdr.resub);
        }
        if (ig_dprsr_md.mirror_type == 1) {
            mirror.emit<mirror_h>(meta.mirror_session, {
                meta.dp_class, meta.dp_confidence, meta.reason,
                meta.pkts8, meta.bytes24, meta.diversity8, meta.fanout8,
                meta.iat_acc, meta.len_acc, meta.min_len16, meta.max_len16,
                meta.fwd_pkts, meta.fwd_bytes24, meta.iat_min, meta.iat_max,
                meta.first_ts, meta.dev_acc, meta.first_len16, meta.dp_action,
                meta.alert_level });
        }
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }
}

// ----------------------------------------------------------------- egress
parser EgParser(packet_in pkt, out headers_t hdr, out eg_metadata_t meta,
                out egress_intrinsic_metadata_t eg_intr_md) {
    state start {
        pkt.extract(eg_intr_md);
        // Only the mirror session's packets reach the controller port, and
        // every one of them carries the mirror payload.
        transition select(eg_intr_md.egress_port) {
            CONTROLLER_PORT: parse_mirror;
            default: parse_ethernet;
        }
    }
    state parse_mirror {
        pkt.extract(hdr.mverdict);
        pkt.extract(hdr.mpkts);
        pkt.extract(hdr.mbytes);
        pkt.extract(hdr.mdiv);
        pkt.extract(hdr.mfan);
        pkt.extract(hdr.mstats);
        pkt.extract(hdr.mtail);
        transition parse_ethernet;
    }
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP: parse_tcp;
            PROTO_UDP: parse_udp;
            default: accept;
        }
    }
    state parse_tcp { pkt.extract(hdr.tcp); transition accept; }
    state parse_udp { pkt.extract(hdr.udp); transition accept; }
}

control Egress(inout headers_t hdr, inout eg_metadata_t meta,
               in egress_intrinsic_metadata_t eg_intr_md,
               in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
               inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
               inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_tag_f;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_tag_r;
    // The flow key of the report, from the mirrored packet's own headers,
    // in the order of the BMv2 key (see the ingress hashes).
    action key_fields(bit<16> sp, bit<16> dp) {
        hdr.rkey.setValid();
        hdr.rkey.src_addr = hdr.ipv4.srcAddr;
        hdr.rkey.dst_addr = hdr.ipv4.dstAddr;
        hdr.rkey.src_port = sp;
        hdr.rkey.dst_port = dp;
        hdr.rkey.protocol = hdr.ipv4.protocol;
        meta.addr_lo = min(hdr.ipv4.srcAddr, hdr.ipv4.dstAddr);
        meta.port_min = min(sp, dp);
    }
    action tag_forward() {
        hdr.rkey.flow_id = hash_tag_f.get({hdr.ipv4.srcAddr, hdr.ipv4.dstAddr,
                                           hdr.rkey.src_port, hdr.rkey.dst_port,
                                           hdr.ipv4.protocol});
    }
    action tag_reverse() {
        hdr.rkey.flow_id = hash_tag_r.get({hdr.ipv4.dstAddr, hdr.ipv4.srcAddr,
                                           hdr.rkey.dst_port, hdr.rkey.src_port,
                                           hdr.ipv4.protocol});
    }
    action padding() {
        hdr.pad_pkts.setValid();  hdr.pad_pkts.z = 0;
        hdr.pad_bytes.setValid(); hdr.pad_bytes.z = 0;
        hdr.pad_div.setValid();   hdr.pad_div.z = 0;
        hdr.pad_fan.setValid();   hdr.pad_fan.z = 0;
        hdr.pad_fwd.setValid();   hdr.pad_fwd.z = 0;
    }
    // The clone is re-cast as a UDP datagram that carries only the report.
    action cast_as_udp() {
        hdr.tcp.setInvalid();
        hdr.udp.setValid();
        hdr.udp.srcPort  = REPORT_UDP_PORT;
        hdr.udp.dstPort  = REPORT_UDP_PORT;
        hdr.udp.length   = 8 + 70;
        hdr.udp.checksum = 0;
        hdr.ipv4.protocol = PROTO_UDP;
        hdr.ipv4.ihl      = 5;
        hdr.ipv4.totalLen = 20 + 8 + 70;
        hdr.ipv4.ttl      = 64;
    }
    apply {
        if (hdr.mverdict.isValid()) {
            if (hdr.tcp.isValid()) {
                key_fields(hdr.tcp.srcPort, hdr.tcp.dstPort);
            } else if (hdr.udp.isValid()) {
                key_fields(hdr.udp.srcPort, hdr.udp.dstPort);
            } else {
                key_fields(0, 0);
            }
            if (hdr.ipv4.srcAddr == hdr.ipv4.dstAddr) {
                if (hdr.rkey.src_port == meta.port_min) { tag_forward(); } else { tag_reverse(); }
            } else if (hdr.ipv4.srcAddr == meta.addr_lo) {
                tag_forward();
            } else {
                tag_reverse();
            }
            padding();
            cast_as_udp();
        }
    }
}

control EgDeparser(packet_out pkt, inout headers_t hdr, in eg_metadata_t meta,
                   in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    Checksum() ipv4_checksum;
    apply {
        hdr.ipv4.hdrChecksum = ipv4_checksum.update({
            hdr.ipv4.version, hdr.ipv4.ihl, hdr.ipv4.diffserv, hdr.ipv4.totalLen,
            hdr.ipv4.identification, hdr.ipv4.flags, hdr.ipv4.fragOffset,
            hdr.ipv4.ttl, hdr.ipv4.protocol, hdr.ipv4.srcAddr, hdr.ipv4.dstAddr});
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
        // The 70-byte report: key, verdict, then each mirrored field widened
        // by the padding header in front of it.
        pkt.emit(hdr.rkey);
        pkt.emit(hdr.mverdict);
        pkt.emit(hdr.pad_pkts);
        pkt.emit(hdr.mpkts);
        pkt.emit(hdr.pad_bytes);
        pkt.emit(hdr.mbytes);
        pkt.emit(hdr.pad_div);
        pkt.emit(hdr.mdiv);
        pkt.emit(hdr.pad_fan);
        pkt.emit(hdr.mfan);
        pkt.emit(hdr.mstats);
        pkt.emit(hdr.pad_fwd);
        pkt.emit(hdr.mtail);
    }
}

Pipeline(IgParser(), Ingress(), IgDeparser(), EgParser(), Egress(), EgDeparser()) pipe;
Switch(pipe) main;
