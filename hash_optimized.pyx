import hashlib
from libc.stdint cimport uint32_t, uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize

def cython_hash_calculation(bytes header_bin, uint32_t nonce_start, uint32_t batch_size, uint64_t target):
    cdef:
        uint32_t i
        uint32_t nonce
        bytes full_header
        bytes hash1, hash2
        uint64_t hash_int
        list results = []
    
    for i in range(batch_size):
        nonce = nonce_start + i
        full_header = header_bin + nonce.to_bytes(4, byteorder='little')
        hash1 = hashlib.sha256(full_header).digest()
        hash2 = hashlib.sha256(hash1).digest()
        hash_int = int.from_bytes(hash2, byteorder='little')
        
        if hash_int <= target:
            results.append((nonce, hash2.hex()))
            if len(results) >= 2:
                return results, nonce_start + i + 1
                
    return results, nonce_start + batch_size 