"""Flask application factory for the standalone Polyfolds sister app.

Role
----
Assemble the lightweight deployed Polyfolds shell that will eventually serve
trained-model interactions while keeping the heavy one-time geometry, dataset,
and training work in the sibling ``polyfolds/`` workspace.

Cross-Repo Context
------------------
``pf_web`` is the user-facing service app. The adjacent ``polyfolds`` package
is the offline development workspace used for dataset generation, manifests,
and training. AIX routes ``/polyfolds`` to this standalone service in cloud.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

from flask import Flask


def _normalize_base_url(value: str) -> str:
    """Normalize the configured AIX hub base URL for footer/navigation links."""

    raw = str(value or '').strip()
    return raw or '/'


def _aix_page_url(base_url: str, path: str) -> str:
    """Build one AIX-owned page URL from the configured hub base URL."""

    base = _normalize_base_url(base_url)
    if base == '/':
        return path
    return urljoin(base.rstrip('/') + '/', path.lstrip('/'))


def create_app(config: dict | None = None) -> Flask:
    """Create the standalone Polyfolds Flask application.

    Role
    ----
    Configure the deployed PF shell and expose the AIX footer/navigation URLs
    needed to keep the sister app visually connected to the umbrella project.
    """

    app = Flask(__name__, template_folder='templates', static_folder='static')
    root = Path(__file__).resolve().parents[1]
    app.config.from_mapping(
        SECRET_KEY='dev-only-secret-key-change-me',
        APP_DISPLAY_NAME='Polyfolds',
        AIX_HUB_URL=os.getenv('AIX_HUB_URL', '/'),
        PF_DEV_REPO=os.getenv('PF_DEV_REPO', str(root / 'polyfolds')),
        PF_MODELS_DIR=os.getenv('PF_MODELS_DIR', str(root / 'models')),
        PF_DATA_DIR=os.getenv('PF_DATA_DIR', str(root / 'data')),
        PF_BUCKET=os.getenv('PF_BUCKET', ''),
    )
    if config:
        app.config.update(config)

    from pf_web.blueprints.main import main_bp

    app.register_blueprint(main_bp)

    @app.context_processor
    def inject_template_globals() -> dict:
        """Expose AIX navigation URLs and PF display metadata to templates."""

        hub_url = _normalize_base_url(app.config.get('AIX_HUB_URL', '/'))
        return {
            'aix_hub_url': hub_url,
            'aix_contact_url': _aix_page_url(hub_url, '/contact'),
            'aix_privacy_url': _aix_page_url(hub_url, '/privacy'),
            'aix_toc_url': _aix_page_url(hub_url, '/toc'),
            'app_display_name': str(app.config.get('APP_DISPLAY_NAME', 'Polyfolds')),
        }

    return app
