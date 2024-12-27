from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto import Random
import binascii
import SocketServer
from secret import flag

pad = lambda s: s + (16 - len(s) % 16) * chr(16 - len(s) % 16)
key = Random.get_random_bytes(16)
prefix = Random.get_random_bytes(4)
suffix = Random.get_random_bytes(4)


def enc(plaintext):
    count = Counter.new(64, prefix=prefix, suffix=suffix)
    cipher = AES.new(key, AES.MODE_CTR, counter=count)
    return cipher.encrypt(pad(plaintext + flag))


class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass


class EncHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        self.request.sendall("Welcome to flag getting system\n")
        while 1:
            self.request.sendall("plaintext>")
            plaintext = self.request.recv(1024).strip()
            ciphertext = binascii.hexlify(enc(plaintext))
            self.request.sendall("result>" + ciphertext + '\n')


if __name__ == "__main__":
    HOST, PORT = "0.0.0.0", 7777
    server = ThreadedTCPServer((HOST, PORT), EncHandler)
    server.serve_forever()
