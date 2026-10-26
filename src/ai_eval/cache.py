"""On-disk cache for LLM judge responses.

Judge calls run at temperature 0, so re-running a suite over unchanged outputs would pay for
the same verdicts again. Entries are keyed by a hash of (provider, model, system, prompt) and
stored as small JSON files under ``.ai-eval/cache/`` in the project root.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ValidationError

DEFAULT_CACHE_SUBDIR = Path(".ai-eval") / "cache"


class CachedResponse(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def cache_key(provider: str, model: str, prompt: str, system: str = "") -> str:
    """Stable sha256 over the inputs that determine a judge verdict."""
    blob = json.dumps([provider, model, system, prompt], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JudgeCache:
    """A directory of ``<key[:2]>/<key>.json`` entries. Safe to share across threads/processes:
    writes go to a temp file and are moved into place atomically; unreadable entries are misses."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @classmethod
    def default(cls, project_root: Path) -> JudgeCache:
        return cls(project_root / DEFAULT_CACHE_SUBDIR)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        try:
            return CachedResponse.model_validate_json(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            return None

    def put(self, key: str, value: CachedResponse) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(value.model_dump_json())
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError:
            pass  # a cache that cannot be written is just a cache miss next time
