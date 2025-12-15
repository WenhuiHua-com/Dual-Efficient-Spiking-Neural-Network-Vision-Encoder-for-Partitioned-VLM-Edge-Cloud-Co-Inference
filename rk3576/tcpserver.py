import socket
import struct

class TcpServer:
    MAX_PACKET_LEN = 10 * 1024 * 1024
    MAGIC_NUMBER = 0x0D000721
    socket_ = []
    ip_ = []
    port_ = []

    def __init__(self):
        pass

    def Set(self, ip, port):
        self.ip_ = ip
        self.port_ = port

    def Start(self):
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_.connect((self.ip_, self.port_))

    def Stop(self):
        self.socket_.close()

    def SendPacket(self, packet):
        packet_size = len(packet)
        if(packet_size > self.MAX_PACKET_LEN):
            print('Error: Packet size exceeds the maximum size 10MB.')
            return
        
        header = struct.pack('<II', self.MAGIC_NUMBER, packet_size)
        data = header + packet
        self.socket_.send(data)
        print(f'Success send {packet_size} bytes data')
        