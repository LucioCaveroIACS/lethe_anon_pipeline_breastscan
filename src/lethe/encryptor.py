import hmac
from hashlib import sha256

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


class IdentifierEncryptor:
    def __init__(self, site_id: str, master_key: str):
        """
        Port of the BREASTSCAN encryption scheme (originally a Lua function
        using openssl.cipher / openssl.hmac).

        Key material is derived from the master key (the secret) via HMAC-SHA256:
          k_enc = HMAC(master_key, "enc_key")
          k_mac = HMAC(master_key, "mac_key")
          k_iv  = HMAC(master_key, "iv_key")

        The site_id argument is kept for interface compatibility with the
        existing pipeline but plays no role in the encryption itself.
        """
        self._master_key = master_key.encode()
        self._k_enc = self._hmac(self._master_key, b"enc_key")
        self._k_mac = self._hmac(self._master_key, b"mac_key")
        self._k_iv = self._hmac(self._master_key, b"iv_key")

    @staticmethod
    def _hmac(key: bytes, msg: bytes) -> bytes:
        return hmac.new(key, msg, sha256).digest()

    def encrypt(self, identifier: str) -> bytes:
        plaintext = identifier.encode()

        iv = self._hmac(self._k_iv, plaintext)[:16]

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(algorithms.AES(self._k_enc), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        signature = self._hmac(self._k_mac, iv + ciphertext)

        return signature + iv + ciphertext

    def decrypt(self, encrypted_data: bytes) -> str:
        signature = encrypted_data[:32]
        iv = encrypted_data[32:48]
        ciphertext = encrypted_data[48:]

        expected = self._hmac(self._k_mac, iv + ciphertext)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("MAC verification failed")

        cipher = Cipher(algorithms.AES(self._k_enc), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        return data.decode()