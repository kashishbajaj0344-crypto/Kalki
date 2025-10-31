# Kalki v2.4 — Production Deployment Guide

## Overview

This guide covers deploying Kalki v2.4 in production environments with proper security, monitoring, and scalability considerations.

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git
- Required API keys (see .env.example)

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd kalki
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

## Production Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

- **Required:** `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`
- **Security:** Set `KALKI_ENV=production`
- **Monitoring:** Enable metrics and health checks
- **Performance:** Tune worker threads and batch sizes

### Security Hardening

- Use strong API keys with minimal permissions
- Enable PII detection: `ENABLE_PII_DETECTION=true`
- Configure data retention: `DATA_RETENTION_DAYS=90`
- Set up rate limiting and session timeouts

## Deployment Options

### Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n kalki
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run orchestrator
python kalki_orchestrator.py --mode production
```

## Monitoring & Observability

### Health Checks

- **Application:** `GET /health` - Overall system health
- **Dependencies:** `GET /health/deps` - External service status
- **Metrics:** `GET /metrics` - Prometheus metrics

### Logging

- Logs written to `kalki.log`
- Structured JSON logging in production
- Log levels: DEBUG, INFO, WARNING, ERROR

### Metrics

- **Performance:** Response times, throughput
- **Resources:** CPU, memory, disk usage
- **Business:** Query counts, user sessions

## Backup & Recovery

### Automated Backups

- Vector database snapshots every 24 hours
- Session data and configurations backed up
- Retention: 7 days rolling window

### Recovery Procedures

```bash
# Restore from backup
docker-compose exec kalki kalki_orchestrator.py --restore /backups/latest

# Emergency recovery
docker-compose down
docker-compose up -d --force-recreate
```

## Scaling Considerations

### Horizontal Scaling

- Multiple Kalki instances behind load balancer
- Shared Redis for session management
- PostgreSQL for persistent data

### Resource Requirements

- **Minimum:** 2GB RAM, 2 CPU cores
- **Recommended:** 8GB RAM, 4 CPU cores
- **High Load:** 16GB+ RAM, 8+ CPU cores

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify keys in `.env`
   - Check API quotas and billing

2. **Memory Issues**
   - Increase Docker memory limits
   - Reduce batch sizes in config

3. **Database Connection**
   - Check PostgreSQL/Redis connectivity
   - Verify network policies

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python kalki_orchestrator.py --verbose
```

## Security Checklist

- [ ] API keys configured and rotated regularly
- [ ] PII detection enabled
- [ ] Data encryption at rest
- [ ] Network security policies applied
- [ ] Regular security updates
- [ ] Access logging enabled
- [ ] Backup encryption configured

## Performance Optimization

### LLM Optimization

- Use appropriate model sizes for workload
- Implement caching for repeated queries
- Batch requests when possible

### Database Tuning

- Optimize vector similarity searches
- Configure connection pooling
- Monitor query performance

### Resource Management

- Set appropriate memory limits
- Configure CPU affinity
- Monitor resource utilization

## Compliance & Governance

### Data Privacy

- PII detection and masking
- Data retention policies
- Audit trail logging
- GDPR/CCPA compliance

### Regulatory Requirements

- SOC 2 Type II compliance framework
- Regular security assessments
- Incident response procedures

## Support & Maintenance

### Regular Tasks

- Weekly: Review logs and metrics
- Monthly: Security updates and patches
- Quarterly: Performance optimization
- Annually: Architecture review

### Contact Information

- **Issues:** GitHub Issues
- **Security:** security@kalki.ai
- **Support:** support@kalki.ai

---

*This deployment guide is for Kalki v2.4. Check documentation for version-specific changes.*