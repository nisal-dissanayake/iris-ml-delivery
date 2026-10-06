"""Locust load test and release gate for the Iris prediction service.

Workload: each simulated user waits 1-2 s between tasks; predictions are 3x as frequent as home checks.
Response checks (catch_response): a sample FAILS when the status code or the expected JSON content is wrong,
so an HTTP 200 with the wrong body is still counted as a failure.

Gate (predeclared classroom thresholds - do not loosen to get a green run):
  * at least 20 requests
  * failure ratio <= 0.01
  * aggregate p95 present and <= P95_LIMIT_MS (default 1000 ms)
process exit code 0 = pass, 1 = fail (used by the CI job to stop the release path).
"""
import os

from locust import HttpUser, between, events, task

VALID_PAYLOAD = {"measurements": [5.1, 3.5, 1.4, 0.2]}   # sepal L, sepal W, petal L, petal W (cm)
EXPECTED_PREDICTION = "setosa"
EXPECTED_SERVICE = "iris-prediction"
REQUEST_TIMEOUT_S = 10

MIN_REQUESTS = 20
MAX_FAIL_RATIO = 0.01
P95_LIMIT_MS = float(os.getenv("P95_LIMIT_MS", "1000"))


def _json(response):
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


class IrisUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def predict(self):
        with self.client.post("/predict", json=VALID_PAYLOAD, timeout=REQUEST_TIMEOUT_S,
                              name="POST /predict", catch_response=True) as response:
            body = _json(response)
            if response.status_code != 200:
                response.failure(f"expected HTTP 200, got {response.status_code}")
            elif body is None or body.get("prediction") != EXPECTED_PREDICTION:
                response.failure(f"expected prediction={EXPECTED_PREDICTION}, got {body}")
            else:
                response.success()

    @task(1)
    def home(self):
        with self.client.get("/", timeout=REQUEST_TIMEOUT_S, name="GET /",
                             catch_response=True) as response:
            body = _json(response)
            if response.status_code != 200:
                response.failure(f"expected HTTP 200, got {response.status_code}")
            elif body is None or body.get("service") != EXPECTED_SERVICE:
                response.failure(f"expected service={EXPECTED_SERVICE}, got {body}")
            else:
                response.success()


@events.quitting.add_listener
def apply_gate(environment, **_kwargs):
    total = environment.stats.total
    requests = total.num_requests
    fail_ratio = total.fail_ratio
    p95 = total.get_response_time_percentile(0.95) if requests > 0 else None

    reasons = []
    if requests < MIN_REQUESTS:
        reasons.append(f"only {requests} requests (< {MIN_REQUESTS})")
    if fail_ratio > MAX_FAIL_RATIO:
        reasons.append(f"failure ratio {fail_ratio:.4f} > {MAX_FAIL_RATIO}")
    if p95 is None:
        reasons.append("no p95 available (no traffic)")
    elif p95 > P95_LIMIT_MS:
        reasons.append(f"p95 {p95:.0f} ms > {P95_LIMIT_MS:.0f} ms")

    summary = (f"requests={requests} failures={total.num_failures} "
               f"fail_ratio={fail_ratio:.4f} p95_ms={p95} limit_ms={P95_LIMIT_MS:.0f}")
    if reasons:
        print(f"LOAD GATE FAIL: {'; '.join(reasons)} | {summary}")
        environment.process_exit_code = 1
    else:
        print(f"LOAD GATE PASS | {summary}")
        environment.process_exit_code = 0
