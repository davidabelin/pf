from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template


main_bp = Blueprint('main', __name__)


@main_bp.get('/')
def home() -> str:
    return render_template(
        'pages/home.html',
        title='Polyfolds',
        dev_repo=str(current_app.config.get('PF_DEV_REPO', '')),
        bucket_name=str(current_app.config.get('PF_BUCKET', '')),
    )


@main_bp.get('/healthz')
def healthz():
    return jsonify(
        {
            'status': 'ok',
            'service': 'polyfolds',
            'dev_repo': str(current_app.config.get('PF_DEV_REPO', '')),
            'models_dir': str(current_app.config.get('PF_MODELS_DIR', '')),
            'data_dir': str(current_app.config.get('PF_DATA_DIR', '')),
            'bucket': str(current_app.config.get('PF_BUCKET', '')),
        }
    )
