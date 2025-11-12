# Kalki v2.4 — Testing Strategy

## Overview

Comprehensive testing strategy for Kalki v2.4 covering unit tests, integration tests, end-to-end tests, performance tests, and security tests.

## Testing Pyramid

```
┌─────────────────────────────────┐
│    End-to-End Tests (E2E)       │  ◄─ User journey validation
│    Integration Tests            │  ◄─ Component interaction
│    Unit Tests                   │  ◄─ Individual function testing
└─────────────────────────────────┘
          Performance Tests         ◄─ Load and stress testing
          Security Tests            ◄─ Vulnerability assessment
```

## Unit Testing

### Test Structure

```
tests/
├── unit/
│   ├── test_llm.py           # LLM functionality
│   ├── test_vectordb.py      # Vector database operations
│   ├── test_ingest.py        # Document ingestion
│   ├── test_agents.py        # Agent behavior
│   └── test_utils.py         # Utility functions
├── integration/
│   ├── test_orchestrator.py  # Orchestrator integration
│   ├── test_pipeline.py      # Processing pipeline
│   └── test_api.py          # API endpoints
├── e2e/
│   ├── test_user_workflow.py # Complete user workflows
│   └── test_system_limits.py # Boundary testing
└── performance/
    ├── test_load.py         # Load testing
    ├── test_stress.py       # Stress testing
    └── test_scalability.py  # Scalability testing
```

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch
from kalki.modules.llm import LLMClient

class TestLLMClient:
    """Test cases for LLM client functionality."""

    @pytest.fixture
    def llm_client(self):
        """Fixture for LLM client with mocked dependencies."""
        return LLMClient(api_key="test_key")

    @pytest.mark.asyncio
    async def test_generate_response_success(self, llm_client):
        """Test successful response generation."""
        with patch('openai.ChatCompletion.acreate') as mock_create:
            mock_create.return_value = Mock(choices=[
                Mock(message=Mock(content="Test response"))
            ])

            response = await llm_client.generate_response("Test prompt")

            assert response == "Test response"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_api_error(self, llm_client):
        """Test API error handling."""
        with patch('openai.ChatCompletion.acreate') as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await llm_client.generate_response("Test prompt")

    @pytest.mark.parametrize("model,temperature", [
        ("gpt-4", 0.7),
        ("gpt-3.5-turbo", 0.5),
        ("claude-3", 0.3),
    ])
    @pytest.mark.asyncio
    async def test_generate_response_different_models(self, llm_client, model, temperature):
        """Test response generation with different models and temperatures."""
        llm_client.model = model
        llm_client.temperature = temperature

        with patch('openai.ChatCompletion.acreate') as mock_create:
            mock_create.return_value = Mock(choices=[
                Mock(message=Mock(content=f"Response from {model}"))
            ])

            response = await llm_client.generate_response("Test prompt")

            assert f"Response from {model}" in response
            # Verify correct parameters were passed
            call_args = mock_create.call_args[1]
            assert call_args['model'] == model
            assert call_args['temperature'] == temperature
```

## Integration Testing

### API Integration Tests

```python
import pytest
from httpx import AsyncClient
from kalki.kalki_server import app

@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    async def client(self):
        """Test client fixture."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "checks" in data

    async def test_ingest_document(self, client):
        """Test document ingestion endpoint."""
        test_file = {"file": ("test.pdf", b"test content", "application/pdf")}

        response = await client.post("/ingest", files=test_file)

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "ingested"

    async def test_query_endpoint(self, client):
        """Test query endpoint with ingested document."""
        # First ingest a document
        await self.test_ingest_document(client)

        # Then query
        query_data = {"query": "test query", "top_k": 5}
        response = await client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
```

### Database Integration Tests

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from kalki.modules.vectordb import VectorDB

@pytest.mark.asyncio
class TestVectorDBIntegration:
    """Integration tests for vector database operations."""

    @pytest.fixture
    async def vector_db(self):
        """Vector database fixture with test configuration."""
        db = VectorDB(connection_string="sqlite+aiosqlite:///:memory:")
        await db.initialize()
        yield db
        await db.cleanup()

    async def test_store_and_retrieve_vectors(self, vector_db):
        """Test storing and retrieving vectors."""
        test_vectors = [
            {"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Test Doc 1"}},
            {"id": "doc2", "vector": [0.4, 0.5, 0.6], "metadata": {"title": "Test Doc 2"}},
        ]

        # Store vectors
        await vector_db.store_vectors(test_vectors)

        # Retrieve similar vectors
        query_vector = [0.1, 0.2, 0.3]
        results = await vector_db.search_similar(query_vector, top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"  # Most similar should be first

    async def test_metadata_filtering(self, vector_db):
        """Test metadata-based filtering."""
        test_vectors = [
            {"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"category": "science"}},
            {"id": "doc2", "vector": [0.4, 0.5, 0.6], "metadata": {"category": "math"}},
        ]

        await vector_db.store_vectors(test_vectors)

        # Search with metadata filter
        results = await vector_db.search_similar(
            [0.1, 0.2, 0.3],
            top_k=5,
            filters={"category": "science"}
        )

        assert len(results) == 1
        assert results[0]["metadata"]["category"] == "science"
```

## End-to-End Testing

### User Workflow Tests

```python
import pytest
from playwright.async_api import Page

@pytest.mark.e2e
class TestUserWorkflows:
    """End-to-end tests for complete user workflows."""

    async def test_complete_ingestion_workflow(self, page: Page):
        """Test complete document ingestion workflow."""
        # Navigate to application
        await page.goto("http://localhost:3000")

        # Upload document
        await page.set_input_files('input[type="file"]', 'test_document.pdf')

        # Wait for ingestion to complete
        await page.wait_for_selector('.ingestion-complete')

        # Verify document appears in library
        document_list = page.locator('.document-list')
        await expect(document_list).to_contain_text('test_document.pdf')

    async def test_query_and_response_workflow(self, page: Page):
        """Test query submission and response display."""
        await page.goto("http://localhost:3000")

        # Enter query
        await page.fill('input[placeholder="Ask a question..."]', 'What is machine learning?')

        # Submit query
        await page.click('button[type="submit"]')

        # Wait for response
        response_area = page.locator('.response-area')
        await expect(response_area).to_be_visible()

        # Verify response content
        await expect(response_area).to_contain_text('machine learning')

    async def test_agent_interaction_workflow(self, page: Page):
        """Test agent interaction workflow."""
        await page.goto("http://localhost:3000/agents")

        # Select agent
        await page.click('text=Research Agent')

        # Start conversation
        await page.fill('textarea', 'Research quantum computing')
        await page.click('button:has-text("Send")')

        # Verify agent response
        agent_response = page.locator('.agent-response')
        await expect(agent_response).to_be_visible()
        await expect(agent_response).to_contain_text('quantum')
```

## Performance Testing

### Load Testing

```python
import asyncio
import aiohttp
import time
from locust import HttpUser, task, between

class KalkiUser(HttpUser):
    """Load testing user for Kalki API."""

    wait_time = between(1, 3)

    @task
    def query_endpoint(self):
        """Simulate user queries."""
        self.client.post("/query", json={
            "query": "What is artificial intelligence?",
            "top_k": 5
        })

    @task(3)
    def health_check(self):
        """Health check requests."""
        self.client.get("/health")

# Configuration for load testing
LOAD_TEST_CONFIG = {
    "users": 100,           # Number of concurrent users
    "spawn_rate": 10,       # Users spawned per second
    "run_time": "5m",       # Test duration
    "host": "http://localhost:8000"
}
```

### Stress Testing

```python
import pytest
from concurrent.futures import ThreadPoolExecutor
import requests

def test_concurrent_ingestion_stress():
    """Test concurrent document ingestion under stress."""

    def ingest_document(doc_id):
        """Ingest a single document."""
        files = {'file': (f'document_{doc_id}.pdf', b'test content')}
        response = requests.post('http://localhost:8000/ingest', files=files)
        return response.status_code

    # Test with high concurrency
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(ingest_document, i) for i in range(100)]
        results = [future.result() for future in futures]

    # Verify all requests succeeded
    success_count = sum(1 for status in results if status == 200)
    assert success_count >= 95  # Allow 5% failure rate under stress
```

## Security Testing

### Vulnerability Testing

```python
import pytest
from security_test_utils import sql_injection_payloads, xss_payloads

class TestSecurity:
    """Security vulnerability tests."""

    @pytest.mark.parametrize("payload", sql_injection_payloads)
    def test_sql_injection_protection(self, client, payload):
        """Test protection against SQL injection attacks."""
        response = client.post("/query", json={"query": payload})

        # Should not execute SQL or crash
        assert response.status_code in [200, 400, 422]
        # Response should not contain database errors
        assert "sql" not in response.text.lower()
        assert "sqlite" not in response.text.lower()

    @pytest.mark.parametrize("payload", xss_payloads)
    def test_xss_protection(self, client, payload):
        """Test protection against XSS attacks."""
        response = client.post("/query", json={"query": payload})

        # Response should be sanitized
        assert "<script>" not in response.text
        assert "javascript:" not in response.text
        assert "onload=" not in response.text

    def test_rate_limiting(self, client):
        """Test API rate limiting."""
        # Send many requests quickly
        responses = []
        for _ in range(150):  # Exceed rate limit
            response = client.get("/health")
            responses.append(response.status_code)

        # Should see 429 (Too Many Requests) responses
        rate_limited = sum(1 for status in responses if status == 429)
        assert rate_limited > 0
```

## Test Automation

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=kalki --cov-report=xml

    - name: Run integration tests
      run: pytest tests/integration/ -v

    - name: Run security tests
      run: pytest tests/security/ -v

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Data Management

```python
# Test data fixtures
@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"category": "AI", "author": "Test Author"}
        },
        {
            "id": "doc2",
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"category": "AI", "author": "Test Author"}
        }
    ]

@pytest.fixture
async def populated_vector_db(sample_documents):
    """Vector database populated with test documents."""
    db = VectorDB()
    await db.initialize()

    # Convert documents to vectors (simplified)
    vectors = []
    for doc in sample_documents:
        # In real tests, you'd use actual embeddings
        vector = [0.1 * i for i in range(384)]  # Mock 384-dim vector
        vectors.append({
            "id": doc["id"],
            "vector": vector,
            "metadata": doc["metadata"]
        })

    await db.store_vectors(vectors)
    yield db
    await db.cleanup()
```

## Test Reporting

### Coverage Requirements

```python
# Coverage configuration
COVERAGE_CONFIG = {
    "branch": True,
    "source": ["kalki"],
    "omit": [
        "*/tests/*",
        "*/venv/*",
        "*/__pycache__/*",
        "kalki/migrations/*"
    ],
    "fail_under": 85,  # Minimum 85% coverage required
    "show_missing": True
}
```

### Quality Gates

- **Unit Tests:** 85%+ code coverage
- **Integration Tests:** All critical paths tested
- **E2E Tests:** All user workflows covered
- **Performance Tests:** Meet SLAs under load
- **Security Tests:** Zero critical vulnerabilities

## Test Environments

### Local Development

```bash
# Run all tests locally
pytest tests/ -v --cov=kalki

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest tests/e2e/ -v          # End-to-end tests only

# Run with coverage
pytest tests/ --cov=kalki --cov-report=html
```

### Staging Environment

- **Automated Deployment:** Tests run on every deployment
- **Full Test Suite:** All tests execute in staging
- **Performance Testing:** Load tests against staging
- **Security Testing:** Vulnerability scans in staging

### Production Monitoring

- **Synthetic Tests:** Automated tests against production
- **Canary Testing:** Gradual rollout with monitoring
- **Rollback Testing:** Automated rollback on failures

---

*This testing strategy ensures Kalki v2.4 maintains high quality and reliability across all deployment scenarios.*