"""
test_api.py
-----------
End-to-end test script for the chest X-ray classifier API.

Usage:
    # Against a locally running server (uvicorn)
    python test_api.py

    # Against the Docker container
    python test_api.py --host http://localhost:8000

    # Skip Docker build/run and just hit an already-running server
    python test_api.py --no-docker
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "The 'requests' library is required for this test script.\n"
        "Install it with:  pip install requests"
    )

BASE_DIR = Path(__file__).resolve().parent

TEST_IMAGES = {
    "NORMAL": [
        BASE_DIR / "normal test 1.jpeg",
        BASE_DIR / "normal test 2.jpeg",
        BASE_DIR / "normal test 3.jpeg",
    ],
    "PNEUMONIA": [
        BASE_DIR / "pneumonia test 1.jpeg",
        BASE_DIR / "pneumonia test 2.jpeg",
        BASE_DIR / "pneumonia test 3.jpeg",
    ],
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def wait_for_server(base_url: str, timeout: int = 60) -> bool:
    print(f"  Waiting for server at {base_url}/health (timeout {timeout}s) …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            print("  Server is up.")
            return True
        except Exception:
            time.sleep(2)
    return False


def test_health(base_url: str) -> None:
    section("GET /health")
    r = requests.get(f"{base_url}/health", timeout=10)
    record("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("status == ok", data.get("status") == "ok", str(data.get("status")))
        record("model_version present", "model_version" in data)
        record("device present", "device" in data)
        record("uptime_seconds present", "uptime_seconds" in data)
        record("total_predictions present", "total_predictions" in data)


def test_model_info(base_url: str) -> None:
    section("GET /model/info")
    r = requests.get(f"{base_url}/model/info", timeout=10)
    record("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("architecture present", "architecture" in data)
        record("metrics present", "metrics" in data)
        record("parameter_count present", "parameter_count" in data)
        record("frozen_layers present", "frozen_layers" in data)
        record("normalization_constants present", "normalization_constants" in data)


def test_metrics(base_url: str) -> None:
    section("GET /metrics")
    r = requests.get(f"{base_url}/metrics", timeout=10)
    record("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("total_predictions present", "total_predictions" in data)
        record("confidence_drift present", "confidence_drift" in data)
        record("label_drift present", "label_drift" in data)


def test_single_predict(base_url: str) -> None:
    section("POST /predict — single image per class")
    for expected_class, paths in TEST_IMAGES.items():
        path = paths[0]
        with open(path, "rb") as fh:
            r = requests.post(
                f"{base_url}/predict",
                files={"file": (path.name, fh, "image/jpeg")},
                timeout=30,
            )
        name = f"{expected_class} ({path.name})"
        record(f"{name} HTTP 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            has_class = "class" in data
            has_conf = "confidence" in data
            has_probs = "probabilities" in data
            has_latency = "latency_ms" in data
            record(f"{name} response keys", all([has_class, has_conf, has_probs, has_latency]))
            predicted = data.get("class", "")
            confidence = data.get("confidence", 0)
            correct = predicted == expected_class
            record(
                f"{name} correct class",
                correct,
                f"predicted={predicted} confidence={confidence:.4f}",
            )
            if not correct:
                print(f"         (full response: {json.dumps(data, indent=6)})")


def test_all_images(base_url: str) -> None:
    section("POST /predict — all 6 images individually")
    correct = 0
    total = 0
    for expected_class, paths in TEST_IMAGES.items():
        for path in paths:
            with open(path, "rb") as fh:
                r = requests.post(
                    f"{base_url}/predict",
                    files={"file": (path.name, fh, "image/jpeg")},
                    timeout=30,
                )
            total += 1
            if r.status_code == 200:
                data = r.json()
                predicted = data.get("class", "")
                confidence = data.get("confidence", 0)
                match = predicted == expected_class
                if match:
                    correct += 1
                status = PASS if match else FAIL
                print(
                    f"  [{status}] {path.name:30s}  expected={expected_class:9s}  "
                    f"predicted={predicted:9s}  conf={confidence:.4f}  "
                    f"latency={data.get('latency_ms', 0):.1f}ms"
                )
            else:
                print(f"  [{FAIL}] {path.name} — HTTP {r.status_code}")

    accuracy = correct / total if total else 0
    record(
        f"Overall accuracy ({correct}/{total})",
        accuracy >= 0.5,
        f"{accuracy:.1%}",
    )


def test_batch_predict(base_url: str) -> None:
    section("POST /predict/batch — all 6 images in one request")
    all_paths = TEST_IMAGES["NORMAL"] + TEST_IMAGES["PNEUMONIA"]
    files = [
        ("files", (p.name, open(p, "rb"), "image/jpeg"))
        for p in all_paths
    ]
    try:
        r = requests.post(f"{base_url}/predict/batch", files=files, timeout=60)
    finally:
        for _, (_, fh, _) in files:
            fh.close()

    record("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("predictions list present", "predictions" in data)
        record("aggregate present", "aggregate" in data)
        preds = data.get("predictions", [])
        record(f"6 predictions returned", len(preds) == 6, f"got {len(preds)}")
        agg = data.get("aggregate", {})
        record("aggregate count == 6", agg.get("count") == 6, str(agg.get("count")))
        print(f"  Class distribution: {agg.get('class_distribution')}")
        print(f"  Average confidence: {agg.get('average_confidence')}")


def test_batch_limit(base_url: str) -> None:
    section("POST /predict/batch — 17 files should return HTTP 400")
    path = TEST_IMAGES["NORMAL"][0]
    files = [("files", (path.name, open(path, "rb"), "image/jpeg")) for _ in range(17)]
    try:
        r = requests.post(f"{base_url}/predict/batch", files=files, timeout=30)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    record("HTTP 400 on oversized batch", r.status_code == 400, f"got {r.status_code}")


def test_invalid_file(base_url: str) -> None:
    section("POST /predict — non-image file should return HTTP 400")
    fake = b"this is not an image"
    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("test.txt", fake, "text/plain")},
        timeout=10,
    )
    record("HTTP 400 on non-image", r.status_code == 400, f"got {r.status_code}")


def test_history(base_url: str) -> None:
    section("GET /predictions/history")
    r = requests.get(f"{base_url}/predictions/history", timeout=10)
    record("HTTP 200 default", r.status_code == 200, f"got {r.status_code}")

    r = requests.get(f"{base_url}/predictions/history?class_filter=PNEUMONIA&limit=5", timeout=10)
    record("HTTP 200 with valid filters", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("predictions key present", "predictions" in data)

    r = requests.get(f"{base_url}/predictions/history?class_filter=INVALID", timeout=10)
    record("HTTP 400 on invalid class_filter", r.status_code == 400, f"got {r.status_code}")

    r = requests.get(f"{base_url}/predictions/history?min_confidence=1.5", timeout=10)
    record("HTTP 400 on out-of-range min_confidence", r.status_code == 400, f"got {r.status_code}")

    r = requests.get(f"{base_url}/predictions/history?limit=201", timeout=10)
    record("HTTP 400 on limit > 200", r.status_code == 400, f"got {r.status_code}")


def print_summary() -> None:
    section("TEST SUMMARY")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    print(f"\n  {passed}/{total} passed, {failed} failed.")
    if failed:
        sys.exit(1)


def run_all_tests(base_url: str) -> None:
    test_health(base_url)
    test_model_info(base_url)
    test_metrics(base_url)
    test_single_predict(base_url)
    test_all_images(base_url)
    test_batch_predict(base_url)
    test_batch_limit(base_url)
    test_invalid_file(base_url)
    test_history(base_url)
    print_summary()


def check_docker_running() -> bool:
    """Return True if the Docker CLI can reach the daemon within 5 seconds.

    'docker version' is spawned in a background thread so it can never
    block the test script — if it hasn't returned in 5 seconds the process
    is killed and we treat Docker as unreachable.
    """
    import threading

    outcome = {"returncode": None}

    def run():
        try:
            proc = subprocess.Popen(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                proc.communicate(timeout=5)
                outcome["returncode"] = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                outcome["returncode"] = -1
        except FileNotFoundError:
            outcome["returncode"] = -2

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=6)
    return outcome["returncode"] == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="API end-to-end tests")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip Docker build/run; test against an already-running server",
    )
    parser.add_argument(
        "--docker-tag",
        default="chest-xray-classifier",
        help="Docker image tag to build and run (default: chest-xray-classifier)",
    )
    args = parser.parse_args()

    if args.no_docker:
        print(f"Skipping Docker — testing against {args.host}")
        if not wait_for_server(args.host, timeout=10):
            sys.exit(f"No server found at {args.host}. Start it with:\n  uvicorn src.api:app --host 0.0.0.0 --port 8000")
        run_all_tests(args.host)
        return

    section("Docker Pre-flight")
    print("  Checking Docker daemon …")
    if not check_docker_running():
        print(
            "\n  Docker Desktop is not running.\n"
            "  Please open Docker Desktop, wait for it to finish starting,\n"
            "  then re-run this script.\n\n"
            "  Alternatively, test without Docker against a local server:\n"
            "      uvicorn src.api:app --host 0.0.0.0 --port 8000\n"
            "      python test_api.py --no-docker"
        )
        sys.exit(1)
    record("Docker daemon reachable", True)

    section("Docker Build")
    print(f"  Building image '{args.docker_tag}' …")
    build = subprocess.run(
        ["docker", "build", "-t", args.docker_tag, "."],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    record("docker build succeeded", build.returncode == 0, f"exit code {build.returncode}")
    if build.returncode != 0:
        print_summary()
        sys.exit(1)

    section("Docker Run")
    container_name = "xray-test-container"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    proc = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "--name", container_name,
            "-p", "8000:8000",
            args.docker_tag,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        alive = wait_for_server(args.host, timeout=60)
        record("Docker container started and serving", alive)
        if not alive:
            print_summary()
            return

        run_all_tests(args.host)
    finally:
        print("\n  Stopping Docker container …")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        proc.wait()


if __name__ == "__main__":
    main()
