"""
Tests for backend.api routes using Flask's test client -- a mocked
session_manager and a throwaway SQLite DB, no live server/hardware needed.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.api import register_routes
from backend.database_manager import DatabaseManager


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    db = DatabaseManager(db_path=path)

    session_manager = MagicMock()
    session_manager.start_session.return_value = 1

    app = Flask(__name__)
    register_routes(app, db, session_manager)
    app.config["TESTING"] = True

    with app.test_client() as c:
        yield c, session_manager

    if os.path.exists(path):
        os.remove(path)


def test_run_start_returns_run_id(client):
    c, session_manager = client
    resp = c.post("/run/start", json={"run_type": "natural"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "started"
    assert body["run_id"] == 1
    session_manager.start_session.assert_called_once()


def test_run_start_defaults_run_type(client):
    c, session_manager = client
    c.post("/run/start", json={})
    _, kwargs = session_manager.start_session.call_args
    assert kwargs["run_type"] == "natural"


def test_runs_endpoint_returns_list(client):
    c, _ = client
    resp = c.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)
