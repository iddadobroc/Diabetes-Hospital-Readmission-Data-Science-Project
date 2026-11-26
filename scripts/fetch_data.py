"""Download the UCI Diabetes 130-US hospitals dataset into data/ and verify it.

The CSV files are not in the repository (CLAUDE.md, .gitignore). The GitHub pipeline
runs this script to get exactly the files the project was built on: the SHA-256 of
each file is checked against the copies used for every number in the decision log.

    python scripts/fetch_data.py

Source: Strack et al. (2014), UCI Machine Learning Repository, dataset 296, licence
CC BY 4.0.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
EXPECTED = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}
DATA = Path(__file__).resolve().parents[1] / "data"


def sha256(content):
    """The SHA-256 hex digest of some bytes."""
    return hashlib.sha256(content).hexdigest()


def main():
    """Download the zip if needed, extract the two files, check their hashes."""
    DATA.mkdir(exist_ok=True)
    present = all(
        (DATA / name).exists() and sha256((DATA / name).read_bytes()) == digest
        for name, digest in EXPECTED.items()
    )
    if present:
        print("data/ already holds the expected files")
        return 0
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name, digest in EXPECTED.items():
        member = next(m for m in archive.namelist() if m.endswith(name))
        content = archive.read(member)
        if sha256(content) != digest:
            print(f"{name}: unexpected SHA-256, the source has changed", file=sys.stderr)
            return 1
        (DATA / name).write_bytes(content)
        print(f"{name}: {len(content):,} bytes, SHA-256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
