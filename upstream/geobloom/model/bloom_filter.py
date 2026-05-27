'''
    Prepare Bloom Filter hash function
    This is a vanilla Bloom filter, suggesting great room of improvements.
'''

import hashlib
import xxhash
import jieba_fast

from tqdm import tqdm
from struct import pack, unpack

def make_hashfuncs(num_slices, num_bits):
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4
    else:
        fmt_code, chunk_size = 'H', 2
    total_hash_bits = 8 * num_slices * chunk_size
    if total_hash_bits > 384:
        hashfn = hashlib.sha512
    elif total_hash_bits > 256:
        hashfn = hashlib.sha384
    elif total_hash_bits > 160:
        hashfn = hashlib.sha256
    elif total_hash_bits > 128:
        hashfn = hashlib.sha1
    else:
        hashfn = xxhash.xxh128

    fmt = fmt_code * (hashfn().digest_size // chunk_size)
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1
    salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range(0, num_salts))

    def _hash_maker(key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        else:
            key = str(key).encode('utf-8')

        i = 0
        for salt in salts:
            h = salt.copy()
            h.update(key)
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits
                i += 1
                if i >= num_slices:
                    return

    return _hash_maker, hashfn


def ngram_split(text, n=3):
    ngrams = set()
    for k in range(1, n + 1):
        for i in range(len(text) - k + 1):
            ngrams.add(text[i:i + k])
    return ngrams

def hybrid_split(text, dict_tokenizer, ngram_tokenizer):
    # We don't segment the address into 2-gram. 
    # The address (fields[2]) is only cut by jieba_fast to mitigate unexpected split.
    fields = text.split(',')
    tokens = ngram_tokenizer(fields[0], 2)
    if len(fields) > 1:
        tokens.update(ngram_tokenizer(fields[1], 2))
    for field in fields:
        tokens.update(dict_tokenizer(field))
    return tokens

def query_split(text, dict_tokenizer, ngram_tokenizer):
    tokens = ngram_tokenizer(text, 2)
    tokens.update(dict_tokenizer(text))
    return tokens

def load_data(file_dir, num_slices, num_bits, is_query=True, dict_tokenizer=jieba_fast.lcut_for_search, ngram_tokenizer=ngram_split):
    bloom_filters = []
    locs = []
    truths = []

    # For Beijing and Shanghai, we use anonymized hash function and location.
    if 'Beijing' in file_dir or 'Shanghai' in file_dir:
        try:
            from encrypt_func import safe_hash_func, safe_address_func
            hash_func_inner = lambda x: safe_hash_func(x, num_slices, num_bits)
            address_func_inner = safe_address_func
        except:
            hash_func_inner, _ = make_hashfuncs(num_slices, num_bits)
            address_func_inner = lambda x, y: (x, y)
    else:
        hash_func_inner, _ = make_hashfuncs(num_slices, num_bits)
        address_func_inner = lambda x, y: (x, y)

    def hash_func(t):
        hash_list = list(hash_func_inner(t))
        for i in range(1, num_slices):
            hash_list[i] += i * num_bits
        return set(hash_list)

    with open(file_dir, 'r',) as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading '+ ('query' if is_query else 'POI') + ' data'):
            line = line.strip().lower().split('\t')
            if is_query:
                text = query_split(line[0], dict_tokenizer, ngram_tokenizer)
            else:
                text = hybrid_split(line[0], dict_tokenizer, ngram_tokenizer)
            bloom_filter = set()
            for t in text:
                bloom_filter.update(hash_func(t))
            bloom_filter = list(bloom_filter)
            bloom_filter.sort()
            bloom_filters.append(bloom_filter)
            x, y = address_func_inner(float(line[1]), float(line[2]))
            locs.append([x, y])
            if is_query:
                truths.append([int(x) for x in line[3].split(',')])
    return bloom_filters, locs, truths