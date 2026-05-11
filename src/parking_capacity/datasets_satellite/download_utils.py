"""Téléchargements HTTP, archives, git et sommes de contrôle."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import httpx


def project_data_datasets_dir(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[3]
    p = root / "data" / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} PiB"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url_streaming(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    timeout_s: float = 600.0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Télécharge une URL vers ``dest`` (fichier). Vérifie SHA256 si fourni."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    progress(downloaded, total)
    tmp.replace(dest)
    if expected_sha256:
        got = sha256_file(dest)
        if got.lower() != expected_sha256.lower():
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 invalide pour {dest}: attendu {expected_sha256}, obtenu {got}")
    return dest


def download_url_urllib(url: str, dest: Path, timeout_s: float = 600.0) -> Path:
    """Secours sans httpx (gros fichiers, certains miroirs)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url, timeout=timeout_s) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


def extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(dest_dir)


def git_clone(
    repo_url: str,
    dest: Path,
    *,
    depth: int = 1,
    branch: Optional[str] = None,
) -> Tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(dest)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def git_lfs_pull(repo_dir: Path) -> Tuple[int, str]:
    p = subprocess.run(
        ["git", "lfs", "pull"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name or "download.bin"
