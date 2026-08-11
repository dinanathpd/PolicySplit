/* ================================================================
 * parsers.p4 - Parser, deparser and checksum controls.
 * ================================================================ */

#ifndef PARSERS_P4
#define PARSERS_P4

#include "headers.p4"

const bit<16> TYPE_IPV4      = 0x0800;
const bit<8>  PROTO_TCP      = 6;
const bit<8>  PROTO_UDP      = 17;
const bit<16> REPORT_UDP_PORT = 9999;

parser SplitMambaParser(packet_in packet,
                        out headers_t hdr,
                        inout metadata_t meta,
                        inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default:   accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP: parse_tcp;
            PROTO_UDP: parse_udp;
            default:   accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
}

control SplitMambaDeparser(packet_out packet, in headers_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.report);
    }
}

control SplitMambaVerifyChecksum(inout headers_t hdr,
                                 inout metadata_t meta) {
    apply { }
}

control SplitMambaComputeChecksum(inout headers_t hdr,
                                  inout metadata_t meta) {
    apply {
        /* The escalation clone rewrites the IPv4 length and protocol, so
         * the header checksum has to be recomputed or the controller
         * host drops the report before it reaches the socket. */
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.totalLen,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.fragOffset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr
            },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16
        );
    }
}

#endif  // PARSERS_P4
