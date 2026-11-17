"""Check report citations, labels, figures and treatment-source registry consistency.

This is a structural audit, not a live URL or agronomic-content verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def audit():
    report = ROOT / "docs/report"
    main = report / "main.tex"
    main_text = main.read_text(encoding="utf-8")
    paths = [main, *(report / name for name in re.findall(r"\\input\{([^}]+)\}", main_text))]
    missing_sources = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    text = "\n".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}", text)
    citations = {key.strip() for group in re.findall(r"\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}", text)
                 for key in group.split(",")}
    bibitems = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", text)
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    missing_figures = [name for name in figures if not any(
        (base / name).is_file() for base in [report, ROOT, ROOT / "reports/figures"])]
    urls = lambda value: set(re.findall(r"https?://[^\s,]+", value))
    kb = json.loads((ROOT / "knowledge_base/treatments.json").read_text(encoding="utf-8"))
    kb_urls = set().union(*(urls(source) for value in kb.values() if isinstance(value, dict)
                            for source in value["sources"]))
    registered_urls = urls((ROOT / "knowledge_base/SOURCES.md").read_text(encoding="utf-8"))
    result = {
        "scope": "Structural citation, cross-reference, figure and KB registry checks; live URLs and agronomic claims not reverified",
        "source_files": len(paths), "citation_keys": sorted(citations), "bibliography_entries": len(bibitems),
        "figure_count": len(figures), "kb_source_urls": len(kb_urls),
        "missing_source_files": missing_sources,
        "undefined_citations": sorted(citations - set(bibitems)),
        "duplicate_bibliography_keys": sorted({key for key in bibitems if bibitems.count(key) > 1}),
        "unused_bibliography_keys": sorted(set(bibitems) - citations),
        "undefined_references": sorted(set(refs) - set(labels)),
        "duplicate_labels": sorted({key for key in labels if labels.count(key) > 1}),
        "missing_figures": missing_figures,
        "unregistered_kb_urls": sorted(kb_urls - registered_urls),
        "unreferenced_registry_urls": sorted(registered_urls - kb_urls),
    }
    failure_keys = ["missing_source_files", "undefined_citations", "duplicate_bibliography_keys",
                    "undefined_references", "duplicate_labels", "missing_figures", "unregistered_kb_urls"]
    result["passed"] = not any(result[key] for key in failure_keys)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
