"""Launcher diagnostics, structural report audit and safe compilation publishing."""

import io
from pathlib import Path
import tarfile

import pytest

from scripts import audit_report, compile_report, run_app


def test_report_cross_references_and_sources_resolve():
    result = audit_report.audit()
    assert result["passed"], result
    assert not result["unused_bibliography_keys"]


def test_report_archive_contains_only_tex_and_referenced_figures():
    archive = compile_report.report_archive(500)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = tar.getnames()
        assert "main.tex" in names
        assert "HANDOFF.md" not in names
        assert all(name.endswith(".tex") or name.startswith("reports/figures/") for name in names)
        assert b"Final Consolidated" in tar.extractfile("main.tex").read()


def test_failed_compile_preserves_existing_pdf(monkeypatch, tmp_path):
    original = b"%PDF-original"
    destination = tmp_path / "main.pdf"
    destination.write_bytes(original)
    monkeypatch.setattr(compile_report, "REPORT", tmp_path)
    monkeypatch.setattr(compile_report.shutil, "which", lambda _: None)
    monkeypatch.setattr(compile_report, "online_compile", lambda: b"not a PDF")
    monkeypatch.setattr("sys.argv", ["compile_report.py", "--online"])
    with pytest.raises(RuntimeError, match="valid PDF"):
        compile_report.main()
    assert destination.read_bytes() == original


def test_online_compilation_requires_explicit_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(compile_report, "REPORT", tmp_path)
    monkeypatch.setattr(compile_report.shutil, "which", lambda _: None)
    monkeypatch.setattr("sys.argv", ["compile_report.py"])
    with pytest.raises(SystemExit, match="--online"):
        compile_report.main()


def test_missing_launcher_dependencies_give_install_command(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_app.py", "--check"])
    monkeypatch.setattr(run_app.importlib.util, "find_spec", lambda _: None)
    assert run_app.main() == 1
    assert "pip install -r" in capsys.readouterr().err


def test_missing_launcher_checkpoint_has_download_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["run_app.py", "--check"])
    monkeypatch.setattr(run_app, "ROOT", tmp_path)
    monkeypatch.setattr(run_app.importlib.util, "find_spec", lambda _: True)
    assert run_app.main() == 1
    assert "download_weights.py" in capsys.readouterr().err


def test_launcher_rejects_invalid_port(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_app.py", "--port", "0"])
    with pytest.raises(SystemExit):
        run_app.main()
