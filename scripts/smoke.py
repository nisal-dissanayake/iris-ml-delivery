"""Smoke checks for the Iris prediction service (standard library only).

Usage:  python scripts/smoke.py [BASE_URL]      default BASE_URL = http://127.0.0.1:5000

Checks, in order (any unexpected result -> exit code 1):
  1. Readiness: GET / is retried a bounded number of times while the container starts.
  2. Identity:  GET / -> HTTP 200 and JSON "service" == "iris-prediction".
  3. Valid:     POST /predict [5.1, 3.5, 1.4, 0.2] -> HTTP 200 and "prediction" == "setosa".
  4. Invalid:   POST /predict with only two numbers -> HTTP 400 and an "error" field.
"SMOKE PASS" is printed only after all four checks succeed.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")
TIMEOUT_S = 5          # per-request timeout
READY_ATTEMPTS = 30    # bounded readiness retry ...
READY_DELAY_S = 1      # ... roughly 30 s in total

VALID = {"measurements": [5.1, 3.5, 1.4, 0.2]}
INVALID = {"measurements": [5.1, 3.5]}


def fail(message):
    print(f"SMOKE FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def call(method, path, payload=None):
    """Return (status, parsed JSON or None). HTTP error codes are returned, not raised."""
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:   # 4xx / 5xx still carry a body we want to check
        status, raw = err.code, err.read()
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    return status, body


def wait_until_ready():
    last_error = None
    for attempt in range(1, READY_ATTEMPTS + 1):
        try:
            return call("GET", "/")
        except OSError as err:  # connection refused/reset, timeout, URLError
            last_error = err
            print(f"waiting for {BASE_URL} (attempt {attempt}/{READY_ATTEMPTS}): {err}")
            time.sleep(READY_DELAY_S)
    fail(f"{BASE_URL} not ready after {READY_ATTEMPTS} attempts: {last_error}")


def main():
    status, body = wait_until_ready()
    if status != 200 or not isinstance(body, dict) or body.get("service") != "iris-prediction":
        fail(f"GET / expected 200 with service=iris-prediction, got {status} {body}")
    print(f"ok  GET /          -> {status} {body}")

    status, body = call("POST", "/predict", VALID)
    if status != 200 or not isinstance(body, dict) or body.get("prediction") != "setosa":
        fail(f"valid POST /predict expected 200 setosa, got {status} {body}")
    print(f"ok  POST /predict  -> {status} {body}")

    status, body = call("POST", "/predict", INVALID)
    if status != 400 or not isinstance(body, dict) or "error" not in body:
        fail(f"two-number POST /predict expected 400 with error, got {status} {body}")
    print(f"ok  POST /predict (2 numbers) -> {status} {body}")

    print(f"SMOKE PASS {BASE_URL}")


if __name__ == "__main__":
    try:
        main()
    except OSError as err:  # service died between checks
        fail(f"request error: {err}")
