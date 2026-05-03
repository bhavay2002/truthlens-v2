from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# =========================================================
# CHECKSUM (STREAMING, MEMORY SAFE)
# =========================================================

def compute_checksum(
    path: Path,
    *,
    algorithm: str = "md5",
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Compute checksum of a file using streaming (low memory).

    Parameters
    ----------
    path : Path
        File path

    algorithm : str
        "md5" | "sha1" | "sha256"

    chunk_size : int
        Read size per iteration

    Returns
    -------
    str
        Hex digest
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    digest = hasher.hexdigest()

    logger.debug(
        "Checksum computed | file=%s | algo=%s | digest=%s",
        path,
        algorithm,
        digest,
    )

    return digest


# =========================================================
# VERIFICATION
# =========================================================

def verify_checksum(
    path: Path,
    expected: str,
    *,
    algorithm: str = "md5",
) -> bool:
    """
    Verify file integrity via checksum comparison.

    Raises error if mismatch.

    Returns
    -------
    bool
    """

    actual = compute_checksum(path, algorithm=algorithm)

    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path}:\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    logger.debug("Checksum verified: %s", path)

    return True


# =========================================================
# FILE INTEGRITY VALIDATION
# =========================================================

def validate_file_integrity(
    path: Path,
    *,
    expected_checksum: Optional[str] = None,
    min_size_bytes: int = 1,
) -> None:
    """
    Validate file integrity using size + optional checksum.

    Parameters
    ----------
    path : Path
    expected_checksum : str | None
    min_size_bytes : int
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    size = path.stat().st_size

    if size < min_size_bytes:
        raise RuntimeError(f"File too small (likely corrupted): {path} | size={size}")

    if expected_checksum is not None:
        verify_checksum(path, expected_checksum)

    logger.info(
        "File integrity OK | path=%s | size=%d",
        path,
        size,
    )


# =========================================================
# METADATA INTEGRATION
# =========================================================

def attach_integrity_metadata(
    path: Path,
    *,
    algorithm: str = "md5",
) -> Dict[str, str]:
    """
    Generate integrity metadata for a file.

    Returns
    -------
    dict
        {
            "checksum": "...",
            "algorithm": "...",
            "size": "..."
        }
    """

    path = Path(path)

    checksum = compute_checksum(path, algorithm=algorithm)
    size = path.stat().st_size

    meta = {
        "checksum": checksum,
        "algorithm": algorithm,
        "size": str(size),
    }

    logger.debug("Integrity metadata generated for %s", path)

    return meta


# =========================================================
# VERIFY FROM METADATA
# =========================================================

def verify_from_metadata(
    path: Path,
    metadata: Dict[str, str],
) -> None:
    """
    Validate file using stored metadata.

    Expected metadata keys:
    - checksum
    - algorithm (optional)
    - size (optional)
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    expected_checksum = metadata.get("checksum")
    algorithm = metadata.get("algorithm", "md5")
    expected_size = metadata.get("size")

    if expected_size is not None:
        actual_size = path.stat().st_size
        if int(expected_size) != actual_size:
            raise RuntimeError(
                f"Size mismatch: {path} | expected={expected_size}, got={actual_size}"
            )

    if expected_checksum is not None:
        verify_checksum(path, expected_checksum, algorithm=algorithm)

    logger.info("Integrity verified via metadata: %s", path)