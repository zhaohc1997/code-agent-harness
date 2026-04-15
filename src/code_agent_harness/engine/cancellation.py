from __future__ import annotations

from pathlib import Path


def _require_session_id(value: str) -> str:
    candidate = Path(value)
    if value in {"", ".", ".."} or candidate.is_absolute() or len(candidate.parts) != 1:
        raise ValueError(f"session_id must be a simple leaf name, got {value!r}")
    return value


class CancellationToken:
    def __init__(self, signal_root: Path | str | None = None) -> None:
        self._cancelled = False
        self._signal_root = Path(signal_root) if signal_root is not None else None
        if self._signal_root is not None:
            self._signal_root.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None

    def bind(self, session_id: str) -> None:
        self._session_id = _require_session_id(session_id)

    def cancel(self, session_id: str | None = None) -> None:
        target_session_id = session_id or self._session_id
        if session_id is None or target_session_id == self._session_id:
            self._cancelled = True
        path = self._resolve_signal_path(session_id)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("cancel\n", encoding="utf-8")

    def is_cancelled(self) -> bool:
        path = self._resolve_signal_path()
        return self._cancelled or (path is not None and path.exists())

    def acknowledge(self) -> None:
        self._cancelled = False
        path = self._resolve_signal_path()
        if path is not None and path.exists():
            path.unlink()

    def _resolve_signal_path(self, session_id: str | None = None) -> Path | None:
        if self._signal_root is None:
            return None
        active_session_id = session_id or self._session_id
        if active_session_id is None:
            return None
        return self._signal_root / f"{_require_session_id(active_session_id)}.cancel"
