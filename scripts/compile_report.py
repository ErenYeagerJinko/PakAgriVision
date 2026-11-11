"""Compile docs/report/main.tex to main.pdf using curl and online TeX engine."""

import io
import re
import subprocess
import tarfile
from pathlib import Path
from PIL import Image


def compress_image_for_tex(src_path: Path) -> bytes:
    """Return compressed bytes of image for compact LaTeX compilation."""
    if src_path.suffix.lower() == ".pdf":
        return src_path.read_bytes()

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        max_dim = 900
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=70, optimize=True)
        return out.getvalue()


def compile_latex():
    report_dir = Path("docs/report")
    main_tex = report_dir / "main.tex"
    output_pdf = report_dir / "main.pdf"

    temp_tar = Path("temp_report.tar.gz")
    with tarfile.open(temp_tar, "w:gz") as tar:
        # Add main.tex
        tar.add(main_tex, arcname="main.tex")

        # Add sections
        sections_dir = report_dir / "sections"
        tex_sources = [main_tex.read_text(encoding="utf-8")]
        for sec in sections_dir.glob("*.tex"):
            tar.add(sec, arcname=f"sections/{sec.name}")
            tex_sources.append(sec.read_text(encoding="utf-8"))

        # Add compressed figures into reports/figures/ (only those the report includes;
        # the online engine rejects uploads larger than about 1.5 MB)
        used_figures = {
            Path(name).name
            for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", "\n".join(tex_sources))
        }
        figures_dir = Path("reports/figures")
        for fig in figures_dir.glob("*"):
            if fig.name not in used_figures:
                continue
            if fig.is_file() and fig.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
                img_bytes = compress_image_for_tex(fig)
                target_arc = f"reports/figures/{fig.name}"
                tarinfo = tarfile.TarInfo(name=target_arc)
                tarinfo.size = len(img_bytes)
                tar.addfile(tarinfo, io.BytesIO(img_bytes))

    tar_size_kb = temp_tar.stat().st_size / 1024
    print(f"Packaged lightweight report archive: {tar_size_kb:.1f} KB. Compiling via TeX engine...")

    cmd = [
        "curl.exe",
        "-s",
        "--max-time", "180",
        "-F", f"file=@{temp_tar}",
        "https://latexonline.cc/data?target=main.tex",
        "-o", str(output_pdf)
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if output_pdf.exists() and output_pdf.stat().st_size > 5000:
            with open(output_pdf, "rb") as f:
                head = f.read(5)
            if head == b"%PDF-":
                print(f"SUCCESS: Generated {output_pdf} ({output_pdf.stat().st_size:,} bytes, ~{output_pdf.stat().st_size/1024/1024:.2f} MB)")
            else:
                print("Failed: Output is not a valid PDF. Header:", head)
                with open(output_pdf, "r", errors="ignore") as f:
                    print(f.read()[:800])
        else:
            print("Failed: output PDF missing or too small.")
            if output_pdf.exists():
                with open(output_pdf, "r", errors="ignore") as f:
                    print(f.read()[:800])
    finally:
        if temp_tar.exists():
            temp_tar.unlink()


if __name__ == "__main__":
    compile_latex()
