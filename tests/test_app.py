from __future__ import annotations

from pathlib import Path

import pytest

from pf_web import create_app


@pytest.fixture
def app(tmp_path: Path):
    app = create_app(
        {
            'TESTING': True,
            'PF_DEV_REPO': str(tmp_path / 'polyfolds'),
            'PF_MODELS_DIR': str(tmp_path / 'models'),
            'PF_DATA_DIR': str(tmp_path / 'data'),
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_home_page_renders(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Polyfolds' in response.data
    assert b'2026 AIX Protodyne' in response.data
    assert b'Contact Us' in response.data
    assert b'Privacy' in response.data
    assert b'AIX TOC' in response.data


def test_healthz_returns_ok(client):
    response = client.get('/healthz')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['service'] == 'polyfolds'
