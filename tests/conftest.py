import os
import tempfile

# Configure an isolated database and the offline LLM *before* the app is imported.
_db_dir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_db_dir}/test.db"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["API_KEY"] = "test-key"
os.environ["WEBHOOK_SECRET"] = "test-secret"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}
