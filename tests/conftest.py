import pytest
import httpx


@pytest.fixture
def http_client():
    return httpx.AsyncClient()
