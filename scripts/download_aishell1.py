"""Download AISHELL-1 from OpenSLR (Apache 2.0)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


URL = "https://www.openslr.org/resources/33/data_aishell.tgz"


def download_archive(url: str, archive: Path) -> None:
    """Download with resume support so a large archive can survive disconnects."""
    curl = shutil.which("curl")
    if curl:
        command = [
            curl,
            "-fL",
            "--retry", "8",
            "--retry-delay", "3",
            "--continue-at", "-",
            "--output", str(archive),
            url,
        ]
        subprocess.run(command, check=True)
        return

    # urllib is kept as a fallback for systems without curl. It cannot resume
    # a partial file, so remove an incomplete archive before starting over.
    if archive.exists():
        archive.unlink()
    urllib.request.urlretrieve(url, archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/aishell1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "data_aishell.tgz"
    marker = args.output / ".extracted"
    if not marker.exists() and not archive.exists():
        print(f"downloading: {URL}")
        download_archive(URL, archive)
    elif not marker.exists():
        print(f"resuming download if needed: {archive}")
        download_archive(URL, archive)
    if not marker.exists():
        print(f"extracting: {archive}")
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(args.output)
        marker.touch()
    print(f"dataset_root: {args.output}")


if __name__ == "__main__":
    main()
