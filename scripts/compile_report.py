"""Compile the consolidated report without replacing a valid PDF on failure.

Prefers local pdflatex or Tectonic. --online explicitly permits uploading only
referenced report sources/figures to the repository's existing LaTeXOnline service.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/report"


def compress_image_for_tex(src_path: Path, max_dim: int = 1400) -> bytes:
    if src_path.suffix.lower() == ".pdf":
        return src_path.read_bytes()
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80, optimize=True)
        return buffer.getvalue()


def report_archive(max_dim: int) -> bytes:
    buffer = io.BytesIO()
    sources = [REPORT / "main.tex", *sorted((REPORT / "sections").glob("*.tex"))]
    used_figures = set()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sources:
            text = path.read_text(encoding="utf-8")
            pattern = r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})"
            def normalize(match):
                name = Path(match[2]).name
                used_figures.add(name)
                return match[1] + name + match[3]
            text = re.sub(pattern, normalize, text)
            raw = text.encode("utf-8")
            info = tarfile.TarInfo(path.relative_to(REPORT).as_posix())
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        for name in sorted(used_figures):
            path = ROOT / "reports/figures" / name
            raw = compress_image_for_tex(path, max_dim)
            info = tarfile.TarInfo(f"reports/figures/{name}")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def online_compile() -> bytes:
    # The service's reverse proxy caps multipart uploads at roughly 1 MB.
    for max_dim in (1400, 1000, 850, 750, 650, 500):
        archive = report_archive(max_dim)
        if len(archive) < 980_000:
            break
    else:
        raise RuntimeError("Report archive exceeds the online upload limit; use a local TeX engine")
    boundary = "PakAgriVisionReportBoundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"report.tar.gz\"\r\nContent-Type: application/gzip\r\n\r\n").encode()
    body += archive + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request("https://latexonline.cc/data?target=main.tex", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print(f"Uploading referenced report sources/figures only ({len(archive):,} bytes) to LaTeXOnline.")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")[:3000]) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Permit the existing online compiler if no local engine exists")
    args = parser.parse_args()
    pdflatex, tectonic = shutil.which("pdflatex"), shutil.which("tectonic")
    with tempfile.TemporaryDirectory(prefix="pakagrivision-tex-") as directory:
        output = Path(directory)
        if pdflatex:
            command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={output}", "main.tex"]
            for _ in range(2):
                subprocess.run(command, cwd=REPORT, check=True, timeout=180)
            pdf = (output / "main.pdf").read_bytes()
        elif tectonic:
            subprocess.run([tectonic, "--outdir", str(output), "main.tex"], cwd=REPORT, check=True, timeout=240)
            pdf = (output / "main.pdf").read_bytes()
        elif args.online:
            pdf = online_compile()
        else:
            raise SystemExit("Install pdflatex/Tectonic, or explicitly permit the existing service with --online. Existing main.pdf was not changed.")
        if not pdf.startswith(b"%PDF-") or len(pdf) < 5000 or b"%%EOF" not in pdf[-1024:]:
            raise RuntimeError("Compiler did not return a valid PDF; existing main.pdf was not changed")
        # Only publish a validated successful build. No failed response overwrites the report.
        destination = REPORT / "main.pdf"
        destination.write_bytes(pdf)
        print(f"Compiled {destination.relative_to(ROOT)} ({len(pdf):,} bytes).")


if __name__ == "__main__":
    main()
