import os
import regex as re
import copy
from collections import defaultdict
import multiprocessing as mp
import heapq
from functools import reduce
from .pretokenization_example import find_chunk_boundaries

class ReverseLexOrderPair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair
        
    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair > other.pair
    
    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair

class BPETokenizer:
    def __init__(self, counts: dict[tuple, int], vocab_size, token_initial: list[bytes]):
        self.counts = counts
        self.vocab_size = vocab_size
        self.token_initial = token_initial
        
        self.vocab = {i: token for i, token in enumerate(self.token_initial)}
    
    def _push_pair(self, pair: tuple[bytes, bytes], freq: int):
        heapq.heappush(self.pair_heap, (-freq, ReverseLexOrderPair(pair), pair))
    
    def _pop_pair(self) -> tuple[bytes, bytes]:
        freq, _, pair_top = heapq.heappop(self.pair_heap)
        return -freq, pair_top
    
    def _build_stats_key(self):
        '''
        To update `self.pair_heap` and `self.tmp`.
        '''
        self.tmp = {}
        self.stats = defaultdict(int)
        self.pair_heap = []
        for key, freq in self.counts.items():
            for i in range(len(key)-1):
                pair = (key[i], key[i+1])
                self.stats[pair] += freq
                self.tmp[pair] = self.tmp.get(pair, []) + [key]
        for pair, freq in self.stats.items():
            self._push_pair(pair, freq)
            
    def _build_new_key(self, key_old, pair: tuple[bytes, bytes], token_new) -> tuple:
        key_new = []
        i = 0
        while i < len(key_old):
            if i < len(key_old) - 1 and key_old[i] == pair[0] and key_old[i+1] == pair[1]:
                key_new.append(token_new)
                i+=2
            else:
                key_new.append(key_old[i])
                i+=1
        return tuple(key_new)
            
    def _merge_pair(self, pair: tuple[bytes, bytes]):
        key_to_merge = list(set(self.tmp.pop(pair)))
        token_new = pair[0] + pair[1]
        for key in key_to_merge:
            # assert key in self.counts.keys(), f"Key {key} not found in self.counts.keys()"
            if key not in self.counts.keys(): continue
            old_freq = self.counts.pop(key)
            key_new = self._build_new_key(key, pair, token_new)
            self.counts[key_new] = self.counts.get(key_new, 0) + old_freq

            for i in range(len(key)-1):
                pair_to_modify = (key[i], key[i+1])
                self.stats[pair_to_modify] -= old_freq
                
            for i in range(len(key_new)-1): # Add freq of new key
                pair_to_modify = (key_new[i], key_new[i+1])
                self.stats[pair_to_modify] += old_freq
                self.tmp[pair_to_modify] = self.tmp.get(pair_to_modify, []) + [key_new]
                    
        self.pair_heap = []
        for pair, freq in self.stats.items():
            self._push_pair(pair, freq)
                        
    def train(self):
        merges = []
        self._build_stats_key()

        while len(self.pair_heap) > 0 and len(merges) + len(self.token_initial) < self.vocab_size:
            freq, pair_top = self._pop_pair()
            if freq <= 0: break
            self._merge_pair(pair_top)
            
            merges.append(pair_top)
            self.vocab[len(self.token_initial) + len(merges) - 1] = pair_top[0] + pair_top[1]
        return self.vocab, merges
    
def pre_tokenize(chunk: str, special_pattern: re.Pattern, PAT: str) -> dict:
    counts = {}
    chunks_sub = special_pattern.split(chunk)
    for chunk_sub in chunks_sub:
        if not chunk_sub: continue
        data_iter = re.finditer(PAT, chunk_sub)

        for match in data_iter: 
            token = match.group()
            tpl = tuple(bytes([b]) for b in token.encode('utf-8'))
            counts[tpl] = counts.get(tpl, 0) + 1
    return counts
    
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    special_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    token_initial  = [tok.encode('utf-8') for tok in special_tokens] + [bytes([i]) for i in range(256)]
    counts = []
    num_processes = 4
    pool = mp.Pool(processes=num_processes)
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            counts.append(pool.apply_async(pre_tokenize, args=(chunk, special_pattern, PAT)))
            del chunk
    pool.close()
    pool.join()
    
    def merge_dict(A:dict, B:dict):
        for k, v in B.items():
            A[k] = A.get(k, 0) + v
        return A
    counts = [c.get() for c in counts]
    tokenizer = BPETokenizer(reduce(merge_dict, counts, {}), vocab_size, token_initial)
    return tokenizer.train()