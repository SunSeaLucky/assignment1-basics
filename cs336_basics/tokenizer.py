from typing import Iterable, Iterator
import regex as re
import pickle
from .train_bpe import PAT

class Tokenizer:
    PAT = re.compile(PAT)
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.special_pattern = re.compile('('+ "|".join(re.escape(tok) for tok in special_tokens_sorted) +')')
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.encode_cache = {}
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def proccess_single_token(self, token: str) -> list[int]:
        if token in self.encode_cache: return self.encode_cache[token]
        discrete_token = tuple(bytes([b]) for b in token.encode('utf-8'))
        while len(discrete_token) >= 2:
            min_rank = float('inf')
            min_pair = None
            for i in range(len(discrete_token)-1):
                pair = (discrete_token[i], discrete_token[i+1])
                if pair not in self.merges: continue
                rank = self.merges[pair]
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None: break
            discrete_token_new = []
            i = 0
            while i < len(discrete_token):
                if i == len(discrete_token)-1:
                    discrete_token_new.append(discrete_token[i])
                    break
                if (discrete_token[i], discrete_token[i+1]) == min_pair:
                    discrete_token_new.append(min_pair[0]+min_pair[1])
                    i += 2
                else:
                    discrete_token_new.append(discrete_token[i])
                    i += 1
            discrete_token = discrete_token_new
        self.encode_cache[token] = [self.vocab_reverse[i] for i in discrete_token]
        return self.encode_cache[token]
    
    def encode(self, text:str) -> list[int]:
        res = []
        chunks_sub = self.special_pattern.split(text) if self.special_tokens else [text]
        for chunk_sub in chunks_sub:
            if not chunk_sub: continue
            if chunk_sub in self.special_tokens:
                res.append(self.vocab_reverse[bytes(chunk_sub, 'utf-8', errors='replace')])
                continue
            data_iter = self.PAT.finditer(chunk_sub)
            for token in data_iter:
                token = token.group()
                res += self.proccess_single_token(token)
        return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")