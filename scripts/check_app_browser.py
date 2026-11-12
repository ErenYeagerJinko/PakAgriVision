"""Optional real-browser smoke test and screenshot using an installed Chromium browser.

Requires websocket-client (test utility only). Uses a temporary browser profile,
real checkpoint and real validation image; never changes benchmark result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Path to Chrome, Edge or Chromium executable")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    import websocket
    from werkzeug.serving import make_server
    from app.app import app, ENGINE
    import pandas as pd
    if not ENGINE.ready:
        raise RuntimeError(ENGINE.error)
    browser = args.browser or next((str(p) for p in [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ] if p.is_file()), None) or shutil.which("chromium") or shutil.which("google-chrome")
    if not browser:
        raise RuntimeError("Pass --browser with an installed Chromium browser executable")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory(prefix="pakagrivision-browser-") as profile:
        process = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--no-first-run",
                                    "--no-default-browser-check", "--remote-debugging-port=0",
                                    "--remote-allow-origins=*", f"--user-data-dir={profile}", "about:blank"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        connection = None
        sequence = 0
        errors = []
        try:
            active = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 30
            port = None
            while port is None:
                try:
                    lines = active.read_text().splitlines()
                    port = lines[0] if len(lines) >= 2 else None
                except OSError:
                    # Chromium briefly holds an exclusive write handle on Windows.
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Browser debugger did not start")
                if port is None:
                    time.sleep(.1)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as response:
                targets = json.load(response)
            target = next(t for t in targets if t["type"] == "page")
            connection = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=30)

            def cdp(method, params=None):
                nonlocal sequence
                sequence += 1
                connection.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
                while True:
                    message = json.loads(connection.recv())
                    if message.get("method") == "Runtime.exceptionThrown":
                        errors.append(message["params"]["exceptionDetails"])
                    if message.get("id") == sequence:
                        if "error" in message:
                            raise RuntimeError(message["error"])
                        return message.get("result", {})

            def evaluate(expression):
                result = cdp("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                   "awaitPromise": True})
                if "exceptionDetails" in result:
                    raise RuntimeError(result["exceptionDetails"])
                return result["result"].get("value")

            def wait_for(expression):
                end = time.monotonic() + 40
                while not evaluate(expression):
                    if time.monotonic() > end:
                        raise TimeoutError(expression)
                    time.sleep(.2)

            cdp("Runtime.enable")
            cdp("Page.enable")
            cdp("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100,
                                                        "deviceScaleFactor": 1, "mobile": False})
            cdp("Page.navigate", {"url": f"http://127.0.0.1:{server.server_port}"})
            wait_for("document.readyState === 'complete' && typeof renderTreatmentAdvisory === 'function'")
            sample = pd.read_csv(ROOT / "splits/val.csv")
            relative = sample.loc[sample.label == "Tomato___Late_blight"].iloc[0].path
            image = ROOT / "data/raw/plantvillage/color" / relative
            document = cdp("DOM.getDocument")
            node = cdp("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": "#galleryInput"})
            cdp("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [str(image)]})
            wait_for("!document.getElementById('btnRunDiagnosis').disabled")
            evaluate("document.getElementById('btnRunDiagnosis').click()")
            wait_for("latestDiagnosis !== null || !document.getElementById('predictionError').hidden")
            assert evaluate("latestDiagnosis !== null"), evaluate("document.getElementById('predictionError').textContent")
            assert evaluate("document.getElementById('treatmentSources').children.length > 0")
            assert evaluate("document.getElementById('gradcamBaseImg').src.startsWith('data:image/jpeg;base64,')")
            # Language changes must retain the exact server-gated treatment rather than placeholders.
            original = evaluate("document.getElementById('treatmentChemicalText').textContent")
            evaluate("applyLanguage('ur')")
            assert evaluate("document.getElementById('treatmentChemicalText').textContent") == original
            evaluate("applyLanguage('en')")
            # Exercise an explicitly constructed boundary fixture, then restore the real result.
            evaluate("""(() => {
              const fixture = structuredClone(latestDiagnosis);
              fixture.advisory.low_confidence = true;
              fixture.advisory.status = 'uncertain';
              fixture.advisory.warning = 'Boundary test: verify symptoms';
              fixture.advisory.treatment.chemical_treatment = 'Withheld boundary fixture';
              renderTreatmentAdvisory(fixture, 'en');
            })()""")
            assert evaluate("document.getElementById('lowConfidenceAlert').classList.contains('visible')")
            assert evaluate("document.getElementById('treatmentChemicalText').textContent") == "Withheld boundary fixture"
            evaluate("renderDiagnosisUI(latestDiagnosis)")
            if args.screenshot:
                import base64
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                clip = evaluate("""(() => {
                  const r = document.querySelector('.workbench-grid').getBoundingClientRect();
                  return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1};
                })()""")
                shot = cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True, "clip": clip})
                args.screenshot.write_bytes(base64.b64decode(shot["data"]))
            cdp("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844,
                                                        "deviceScaleFactor": 1, "mobile": True})
            assert evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "Mobile horizontal overflow"
            assert not errors, errors
            print("Browser checks passed: real upload, prediction, heatmap, sources, language toggle, warning and mobile layout.")
        finally:
            if connection:
                try:
                    sequence += 1
                    connection.send(json.dumps({"id": sequence, "method": "Browser.close"}))
                finally:
                    connection.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
