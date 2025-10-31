# Kalki v2.4 — Monitoring & Observability

## Overview

Comprehensive monitoring stack for Kalki v2.4 including metrics collection, logging, alerting, and visualization.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│   Grafana       │
│     Metrics     │    │   Metrics       │    │   Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │    │   AlertManager  │    │   Loki          │
│     Logs        │───▶│   Alerts        │    │   Log Agg       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Application Metrics

```python
# Core application metrics
METRICS = {
    # Performance metrics
    'kalki_requests_total': Counter('Total requests processed'),
    'kalki_request_duration_seconds': Histogram('Request duration'),
    'kalki_active_connections': Gauge('Active connections'),

    # Business metrics
    'kalki_queries_processed': Counter('Queries processed'),
    'kalki_documents_ingested': Counter('Documents ingested'),
    'kalki_sessions_active': Gauge('Active user sessions'),

    # Resource metrics
    'kalki_memory_usage_bytes': Gauge('Memory usage'),
    'kalki_cpu_usage_percent': Gauge('CPU usage'),
    'kalki_disk_usage_bytes': Gauge('Disk usage'),

    # Error metrics
    'kalki_errors_total': Counter('Total errors', ['error_type']),
    'kalki_llm_errors_total': Counter('LLM errors'),
    'kalki_db_errors_total': Counter('Database errors'),
}
```

### System Metrics

- **Host Metrics:** CPU, memory, disk, network
- **Container Metrics:** Docker stats, resource limits
- **Database Metrics:** Connection pools, query performance
- **External API Metrics:** Response times, error rates

## Logging Configuration

### Structured Logging

```python
# JSON structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Log Levels

- **DEBUG:** Detailed debugging information
- **INFO:** General operational messages
- **WARNING:** Warning conditions
- **ERROR:** Error conditions
- **CRITICAL:** Critical system failures

## Alerting Rules

### Critical Alerts

```yaml
# Prometheus alerting rules
groups:
  - name: kalki_critical
    rules:
      - alert: KalkiDown
        expr: up{job="kalki"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kalki service is down"
          description: "Kalki has been down for more than 5 minutes."

      - alert: HighErrorRate
        expr: rate(kalki_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second."
```

### Warning Alerts

```yaml
  - name: kalki_warnings
    rules:
      - alert: HighMemoryUsage
        expr: kalki_memory_usage_bytes / kalki_memory_limit_bytes > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 80%."

      - alert: SlowQueries
        expr: histogram_quantile(0.95, rate(kalki_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow query performance"
          description: "95th percentile query time is {{ $value }}s."
```

## Grafana Dashboards

### Main Dashboard

- **System Overview:** CPU, memory, disk usage
- **Application Health:** Response times, error rates, throughput
- **Business Metrics:** Queries processed, documents ingested
- **Database Performance:** Connection pools, query latency

### Detailed Dashboards

- **LLM Performance:** Token usage, response times, model errors
- **Vector Database:** Index size, search performance, cache hits
- **User Sessions:** Active sessions, session duration, geographic distribution
- **API Endpoints:** Per-endpoint metrics, rate limiting, authentication failures

## Health Checks

### Application Health

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.4.0",
        "checks": {}
    }

    # Database connectivity
    try:
        await db.ping()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Vector database
    try:
        await vector_db.health_check()
        health_status["checks"]["vector_db"] = "healthy"
    except Exception as e:
        health_status["checks"]["vector_db"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # External APIs
    try:
        await check_external_apis()
        health_status["checks"]["external_apis"] = "healthy"
    except Exception as e:
        health_status["checks"]["external_apis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    return health_status
```

### Dependency Health

- **Redis:** Connection and memory usage
- **PostgreSQL:** Connection pool and query performance
- **External APIs:** Response times and error rates
- **File System:** Disk space and permissions

## Log Aggregation

### Loki Configuration

```yaml
# Loki configuration for log aggregation
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  chunk_target_size: 1048576
  max_chunk_age: 1h

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h
```

### Log Queries

```sql
-- Common Loki queries
{job="kalki"} |= "ERROR"  # Error logs
{job="kalki"} |~ "slow.*query"  # Slow queries
rate({job="kalki"}[5m])  # Log rate over 5 minutes
```

## Tracing

### Distributed Tracing

- **OpenTelemetry** integration for request tracing
- **Jaeger** or **Zipkin** for trace visualization
- **Service mesh** integration (Istio/Linkerd)

### Trace Sampling

```python
# Configure trace sampling
from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler

sampler = TraceIdRatioBasedSampler(ratio=0.1)  # 10% sampling
```

## Performance Monitoring

### Profiling

- **Py-Spy** for Python profiling
- **Memory profiling** with memory_profiler
- **CPU profiling** with cProfile

### Benchmarking

```python
# Performance benchmarks
BENCHMARKS = {
    'query_latency': {
        'p50': '< 100ms',
        'p95': '< 500ms',
        'p99': '< 2s'
    },
    'ingestion_rate': {
        'target': '100 docs/minute'
    },
    'concurrent_users': {
        'target': '50 simultaneous users'
    }
}
```

## Alert Channels

### Notification Methods

- **Email:** Critical alerts to on-call engineers
- **Slack:** Real-time notifications to team channels
- **PagerDuty:** Escalation for critical incidents
- **SMS:** Emergency notifications

### Escalation Policy

1. **Warning:** Slack notification
2. **Critical:** Email + Slack + PagerDuty
3. **Emergency:** SMS + phone call escalation

## Monitoring Deployment

### Docker Compose Monitoring

```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
    command: -config.file=/etc/promtail/config.yml
```

### Kubernetes Monitoring

```yaml
# Prometheus Operator deployment
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: kalki-prometheus
spec:
  replicas: 2
  serviceAccountName: prometheus
  serviceMonitorSelector:
    matchLabels:
      team: kalki
```

## Cost Optimization

### Resource Optimization

- **Auto-scaling** based on metrics
- **Spot instances** for non-critical workloads
- **Resource quotas** to prevent over-provisioning

### Monitoring Costs

- **Data retention** policies for metrics/logs
- **Sampling rates** for high-volume metrics
- **Alert optimization** to reduce noise

---

*This monitoring setup provides comprehensive observability for Kalki v2.4 production deployments.*