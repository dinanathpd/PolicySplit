"""
p4runtime_client.py — P4Runtime gRPC client for installing flow rules.
"""

import socket
import logging

logger = logging.getLogger(__name__)


class P4RuntimeClient:
    """
    Simplified P4Runtime client for installing drop/forward rules
    on the BMv2 switch via gRPC.

    For full implementation, requires:
        pip install p4runtime-shell grpcio grpcio-tools protobuf
    """

    def __init__(self, switch_addr: str = "127.0.0.1:9559", device_id: int = 0):
        self.switch_addr = switch_addr
        self.device_id = device_id
        self._connected = False

        try:
            import grpc
            from p4.v1 import p4runtime_pb2_grpc

            self.channel = grpc.insecure_channel(switch_addr)
            self.stub = p4runtime_pb2_grpc.P4RuntimeStub(self.channel)
            self._connected = True
            logger.info("Connected to P4Runtime at %s", switch_addr)
        except ImportError:
            logger.warning(
                "P4Runtime dependencies not installed. "
                "Run: pip install p4runtime-shell grpcio"
            )
        except Exception as e:
            logger.warning("Could not connect to P4Runtime: %s", e)

    def install_drop_rule(
        self,
        flow_id: int,
        src_ip: str,
        dst_ip: str = None,
        src_port: int = None,
        dst_port: int = None,
        protocol: int = None,
    ):
        """Install a drop rule for a specific source IP on the blocklist table."""
        if not self._connected:
            logger.warning("P4Runtime not connected — skipping rule install")
            return False

        try:
            from p4.v1 import p4runtime_pb2

            table_entry = p4runtime_pb2.TableEntry()
            # table_id would come from P4Info — placeholder
            table_entry.table_id = 1

            # Match on source IP
            match = table_entry.match.add()
            match.field_id = 1
            match.exact.value = socket.inet_aton(src_ip)

            # Action: drop
            action = table_entry.action.action
            action.action_id = 1  # drop_packet action ID from P4Info

            # Write request
            request = p4runtime_pb2.WriteRequest()
            request.device_id = self.device_id
            update = request.updates.add()
            update.type = p4runtime_pb2.Update.INSERT
            update.entity.table_entry.CopyFrom(table_entry)

            self.stub.Write(request)
            logger.info("Installed drop rule for flow %d (src=%s)",
                        flow_id, src_ip)
            return True

        except Exception as e:
            logger.error("Failed to install rule: %s", e)
            return False

    def remove_drop_rule(self, src_ip: str):
        """Remove a previously installed drop rule."""
        if not self._connected:
            return False

        try:
            from p4.v1 import p4runtime_pb2

            table_entry = p4runtime_pb2.TableEntry()
            table_entry.table_id = 1
            match = table_entry.match.add()
            match.field_id = 1
            match.exact.value = socket.inet_aton(src_ip)

            request = p4runtime_pb2.WriteRequest()
            request.device_id = self.device_id
            update = request.updates.add()
            update.type = p4runtime_pb2.Update.DELETE
            update.entity.table_entry.CopyFrom(table_entry)

            self.stub.Write(request)
            logger.info("Removed drop rule for src=%s", src_ip)
            return True
        except Exception as e:
            logger.error("Failed to remove rule: %s", e)
            return False
