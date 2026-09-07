"""Download AISHELL-1 from OpenSLR (Apache 2.0)."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


URL = "https://www.openslr.org/resources/33/data_aishell.tgz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/aishell1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "data_aishell.tgz"
    if not archive.exists():
        print(f"downloading: {URL}")
        urllib.request.urlretrieve(URL, archive)
    marker = args.output / ".extracted"
    if not marker.exists():
        print(f"extracting: {archive}")
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(args.output)
        marker.touch()
    print(f"dataset_root: {args.output}")


if __name__ == "__main__":
    main()
