from __future__ import annotations

import os
from pathlib import Path

from flask import Flask


def create_app(config: dict | None = None) -> Flask:
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
        return {
            'aix_hub_url': str(app.config.get('AIX_HUB_URL', '/')).strip() or '/',
            'app_display_name': str(app.config.get('APP_DISPLAY_NAME', 'Polyfolds')),
        }

    return app
