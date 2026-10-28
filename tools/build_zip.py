"""Build a submission ZIP from an explicit allowlist of files."""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("model.py", "labeling.py", "README.md", "models.txt")


def build_zip(name: str) -> Path:
    source = ROOT / name
    output = ROOT / "dist" / f"{name}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename in ALLOWED:
            path = source / filename
            if not path.is_file():
                continue
            entry = ZipInfo(filename)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    return output


if __name__ == "__main__":
    path = build_zip(sys.argv[1] if len(sys.argv) > 1 else "submission_e1")
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
