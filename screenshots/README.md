# Screenshots

Screenshots include deployment, workflow, API, and monitoring evidence for the report.

Monitoring captures recorded during live verification on 2026-10-11:
- `api_metrics_counters.png` and `api_metrics_latency.png`: selected live `/metrics` series after exercising `/health` and `/predict`.
- `prometheus_targets.png`: Prometheus target page showing the Docker API scrape as `1 / 1 up`.
- `prometheus_request_counts.png` and `prometheus_latency_query.png`: Prometheus query results for request counters and histogram observations.
- `api_request_logs.png`: Docker API logs showing method, path, status, and duration.

The Prometheus scrape shown here targets the single Docker API process published on host port 8001. It does not aggregate metrics from the separate two-replica Kubernetes deployment.
