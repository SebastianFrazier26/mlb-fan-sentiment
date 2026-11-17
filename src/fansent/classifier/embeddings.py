"""fastText subword vectors, read straight from the official ``.bin`` (no ``fasttext`` package).

Why not the ``fasttext`` PyPI package: it is a C++/pybind11 sdist, the upstream repo was archived
in 2024, and its numpy-2 / Python 3.12+ / macOS-arm64 story is a build-from-source gamble. The
file format is small and stable, so we memory-map the input matrix and reimplement the lookup:
a word's vector is the mean of its row (if in vocabulary) and the rows of its hashed character
n-grams (FNV-1a over the UTF-8 bytes of ``<word>``, bucket count and n-gram range in the header).
Only the lookup path is implemented: quantised or pruned models are rejected.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

MAGIC = 793712314
SUPPORTED_VERSION = 12
DEFAULT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
DEFAULT_ZIP = "crawl-300d-2M-subword.zip"
DEFAULT_BIN = "crawl-300d-2M-subword.bin"
EMBEDDINGS_DIR = "embeddings"


def fnv1a(data: bytes) -> int:
    """fastText's hash: bytes are sign-extended (int8 -> uint32) before the xor."""
    h = 2166136261
    for byte in data:
        h ^= (byte - 256 if byte >= 128 else byte) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def subword_ids(word: str, nwords: int, bucket: int, minn: int, maxn: int) -> list[int]:
    """Input-matrix row ids of the character n-grams of ``<word>`` (word id not included)."""
    w = f"<{word}>"
    out = []
    for i in range(len(w)):
        for j in range(i + 1, min(len(w), i + maxn) + 1):
            n = j - i
            if n >= minn and not (n == 1 and (i == 0 or j == len(w))):
                out.append(nwords + fnv1a(w[i:j].encode("utf-8")) % bucket)
    return out


class FastTextVectors:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        f = open(self.path, "rb")  # noqa: SIM115 - mmap keeps its own handle
        self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        mm = self._mm
        magic, version = struct.unpack_from("<ii", mm, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a fastText model (magic {magic})")
        if version != SUPPORTED_VERSION:
            raise ValueError(f"{path}: fastText file version {version}, expected 12")
        (self.dim, _ws, _epoch, _mincount, _neg, _wng, _loss, _model, self.bucket, self.minn,
         self.maxn, _lr) = struct.unpack_from("<12i", mm, 8)  # fmt: skip
        pos = 8 + 48 + 8  # skip the trailing double `t`
        size, self.nwords, _nlabels, _ntokens, prune = struct.unpack_from("<iiiqq", mm, pos)
        pos += 28
        if prune > 0:
            raise ValueError(f"{path}: pruned (quantised) models are not supported")
        self.words: dict[str, int] = {}
        for i in range(size):
            end = mm.find(b"\0", pos)
            if i < self.nwords:
                self.words[mm[pos:end].decode("utf-8", errors="replace")] = i
            pos = end + 1 + 8 + 1  # NUL, int64 count, uint8 entry type
        quant = mm[pos]
        if quant:
            raise ValueError(f"{path}: quantised models are not supported")
        rows, cols = struct.unpack_from("<qq", mm, pos + 1)
        if cols != self.dim or rows != self.nwords + self.bucket:
            raise ValueError(f"{path}: input matrix {rows}x{cols} does not match header")
        self._mat = np.frombuffer(mm, dtype="<f4", count=rows * cols, offset=pos + 17).reshape(
            rows, cols
        )

    def vector(self, word: str) -> np.ndarray | None:
        """Mean of the word's own row (if in vocab) and its n-gram rows; None if no rows at all."""
        ids = subword_ids(word, self.nwords, self.bucket, self.minn, self.maxn)
        if word in self.words:
            ids.insert(0, self.words[word])
        if not ids:
            return None
        return self._mat[ids].mean(axis=0)


class TokenVectors:
    """Per-token cache of L2-normalised vectors: one lookup per vocabulary token, ever.

    The cache is what gets saved with a model, so scoring does not need the multi-GB ``.bin``
    for tokens seen in training; ``provider`` (optional) fills in the rest.
    """

    def __init__(self, provider: FastTextVectors | None = None, dim: int | None = None) -> None:
        self.provider = provider
        self.dim = provider.dim if provider else dim
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, token: str) -> np.ndarray | None:
        if token not in self._cache:
            v = self.provider.vector(token) if self.provider else None
            if v is not None:
                norm = np.linalg.norm(v)
                v = (v / norm).astype(np.float32) if norm > 0 else None
            self._cache[token] = v
        return self._cache[token]

    def export(self) -> tuple[list[str], np.ndarray]:
        toks = sorted(t for t, v in self._cache.items() if v is not None)
        mat = np.stack([self._cache[t] for t in toks]) if toks else np.zeros((0, self.dim or 0))
        return toks, mat.astype(np.float32)

    @classmethod
    def from_export(
        cls, tokens: list[str], mat: np.ndarray, provider: FastTextVectors | None = None
    ) -> TokenVectors:
        tv = cls(provider, dim=mat.shape[1])
        tv._cache = {t: mat[i] for i, t in enumerate(tokens)}
        return tv


def mean_vector(tokens: list[str], tv: TokenVectors) -> np.ndarray:
    """L2-normalised mean of the tokens' unit vectors; zeros if none has a vector."""
    vecs = [v for t in tokens if (v := tv.get(t)) is not None]
    out = np.zeros(tv.dim or 0, dtype=np.float32)
    if vecs:
        out = np.mean(vecs, axis=0)
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
    return out


def fetch(dest_dir: Path, url: str = DEFAULT_URL, keep_zip: bool = False, log=print) -> Path:
    """Download (resumable) and extract the official model; returns the ``.bin`` path.

    Facebook publishes no checksum for this file, so integrity = the server's Content-Length
    matched against the bytes on disk, plus the zip CRC-32 verified while extracting.
    """
    import zipfile

    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    zpath, binpath = dest_dir / DEFAULT_ZIP, dest_dir / DEFAULT_BIN
    if binpath.exists():
        log(f"{binpath} already present")
        return binpath
    have = zpath.stat().st_size if zpath.exists() else 0
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    if have < total:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            if have and r.status_code != 206:
                have = 0  # server ignored Range: restart rather than append a second copy
            with open(zpath, "ab" if have else "wb") as f:
                f.writelines(r.iter_content(1 << 20))
    if zpath.stat().st_size != total:
        raise OSError(f"{zpath}: {zpath.stat().st_size} bytes on disk, server says {total}")
    log(f"downloaded {total:,} bytes; extracting {DEFAULT_BIN}")
    with zipfile.ZipFile(zpath) as z:
        z.extract(DEFAULT_BIN, dest_dir)  # raises BadZipFile on CRC mismatch
    if not keep_zip:
        zpath.unlink()
    return binpath
