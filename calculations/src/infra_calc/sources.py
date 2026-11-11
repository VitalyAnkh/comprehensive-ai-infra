"""Read pinned official inputs and verify them before using any model fields."""
import hashlib
import json
from pathlib import Path
import tempfile
import re
from .units import positive_int
from urllib.request import Request, urlopen

from .paths import PROJECT

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def parse_lfs_pointer(raw: bytes) -> tuple[str, int] | None:
    """Return (sha256_oid, byte_size) if raw bytes are a valid Git LFS v1 pointer."""
    if not raw.startswith(LFS_POINTER_PREFIX) or len(raw) > 512:
        return None
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
    size_match = re.search(r"^size (\d+)$", text, re.M)
    if oid_match is None or size_match is None:
        return None
    return oid_match.group(1), int(size_match.group(1))


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(513)
    return parse_lfs_pointer(head) is not None


def sha256_or_lfs_oid(path: Path) -> str:
    with path.open("rb") as handle:
        head = handle.read(513)
        parsed = parse_lfs_pointer(head)
        if parsed is not None:
            return parsed[0]
        digest = hashlib.sha256(head)
        while chunk := handle.read(1048576):
            digest.update(chunk)
    return digest.hexdigest()


def size_or_lfs_size(path: Path) -> int:
    with path.open("rb") as handle:
        head = handle.read(513)
        parsed = parse_lfs_pointer(head)
        if parsed is not None:
            return parsed[1]
    return path.stat().st_size


def records() -> list[dict]:
    return json.loads((PROJECT / "configs/sources.lock.json").read_text(encoding="utf-8"))["sources"]


def verify_source_record(relative_path: str) -> None:
    matches = [record for record in records() if record["file"] == relative_path]
    if len(matches) != 1 or matches[0]["status"] != "downloaded":
        raise ValueError(f"No downloaded source locked for {relative_path}")
    expected_bytes = positive_int(matches[0]["bytes"], "locked source bytes", allow_zero=True)
    path = PROJECT / relative_path
    if size_or_lfs_size(path) != expected_bytes:
        raise ValueError(f"Source length mismatch: {relative_path}")
    if sha256_or_lfs_oid(path) != matches[0]["sha256"]:
        raise ValueError(f"Source checksum mismatch: {relative_path}")


def read_source(relative_path: str) -> bytes:
    matches = [record for record in records() if record["file"] == relative_path]
    if len(matches) != 1 or matches[0]["status"] != "downloaded":
        raise ValueError(f"No downloaded source locked for {relative_path}")
    expected_bytes = positive_int(matches[0]["bytes"], "locked source bytes", allow_zero=True)
    data = (PROJECT / relative_path).read_bytes()
    if parse_lfs_pointer(data) is not None:
        raise ValueError(
            f"Source length mismatch: {relative_path} (Git LFS pointer not hydrated; run 'git lfs pull' or 'python3 calculations/calc.py fetch')"
        )
    if len(data) != expected_bytes:
        raise ValueError(f"Source length mismatch: {relative_path}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != matches[0]["sha256"]:
        raise ValueError(f"Source checksum mismatch: {relative_path}")
    return data


def model_config(model: str, *, reference: bool = False) -> dict:
    name = "inference/config.json" if reference else "config.json"
    return json.loads(read_source(f"configs/models/{model}/{name}"))


def provenance(model: str) -> list[dict]:
    return [
        {key: row[key] for key in ("file", "url", "revision", "sha256")}
        for row in records()
        if row["model"] in (model, "qwen3" if model.startswith("qwen3-") else model,
                            'fast-hadamard-transform' if model.startswith('deepseek-v4-') else model,
                            'flash-linear-attention' if model == 'kimi-k3' else model)
        and row["status"] == "downloaded"
    ]


def verify_sources() -> dict:
    checked, unavailable = [], []
    for row in records():
        if row["status"] == "downloaded":
            verify_source_record(row["file"])
            checked.append(row["file"])
        else:
            unavailable.append({"model": row["model"], "file": row["file"], "error": row["error"]})
    return {"verified": len(checked), "unavailable": unavailable}


def fetch_sources(model: str | None = None) -> dict:
    """Re-download exact model source group (not transitive dependencies) or all groups."""
    selected = [row for row in records() if model is None or row["model"] == model]
    if not selected:
        raise ValueError(f"Unknown model/source group: {model}")
    downloaded, unavailable = [], []
    for row in selected:
        if row["status"] != "downloaded":
            unavailable.append(row["file"])
            continue
        expected_bytes = positive_int(row["bytes"], "locked source bytes", allow_zero=True)
        headers = {"User-Agent": "ai-infra-book-calculations/0.1"}
        interval = None
        if row.get("http_range"):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", row["http_range"])
            if match is None:
                raise ValueError("Locked source requires a bounded byte range")
            start, end = map(int, match.groups())
            if end < start or end - start + 1 != expected_bytes:
                raise ValueError("Locked source range and byte count disagree")
            interval = (start, end)
            headers["Range"] = row["http_range"]
        request = Request(row["url"], headers=headers)
        with urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None:
                if re.fullmatch(r"\d+", length) is None or int(length) != expected_bytes:
                    raise ValueError("Response Content-Length disagrees with locked bytes")
            content_range = response.headers.get("Content-Range")
            if interval is not None:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
                if response.status != 206 or match is None:
                    raise ValueError("Server did not honor locked header byte range")
                start, end, total = map(int, match.groups())
                if (start, end) != interval or total <= end:
                    raise ValueError("Response Content-Range disagrees with locked byte range")
            elif response.status != 200 or content_range is not None:
                raise ValueError("Full source request returned a partial or invalid response")
            data = response.read(expected_bytes + 1)
        if len(data) != expected_bytes:
            raise ValueError(f"Remote length mismatch: {row['url']}")
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Remote checksum mismatch: {row['url']}")
        path = PROJECT / row["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".download", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        downloaded.append(row["file"])
    return {"downloaded": downloaded, "unavailable": unavailable}
