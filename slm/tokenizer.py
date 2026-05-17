"""Tokenizer abstraction. Byte-level for smoke tests; trained BPE for real work."""
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    vocab_size: int
    eot_token: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Trivial byte-level tokenizer: 256 byte values + 1 EOT. No downloads, no training."""

    vocab_size: int = 257
    eot_token: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i != self.eot_token]
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Adapter around a HuggingFace `tokenizers.Tokenizer` JSON file."""

    def __init__(self, json_path: str | Path) -> None:
        from tokenizers import Tokenizer as HfTokenizer
        self._tok = HfTokenizer.from_file(str(json_path))
        self.vocab_size = self._tok.get_vocab_size()
        eot = self._tok.token_to_id("<eos>")
        if eot is None:
            raise ValueError(f"{json_path} has no <eos> special token")
        self.eot_token = eot

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def build_tokenizer(name: str, **kwargs) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    if name == "bpe":
        path = kwargs.get("path") or "slm/tokenizers/bpe_smoke.json"
        return BPETokenizer(path)
    if name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        class TikTokenAdapter:
            vocab_size = enc.n_vocab
            eot_token = enc.eot_token

            def encode(self, text: str) -> list[int]:
                return enc.encode_ordinary(text)

            def decode(self, ids: list[int]) -> str:
                return enc.decode(ids)

        return TikTokenAdapter()
    raise ValueError(f"Unknown tokenizer: {name}")