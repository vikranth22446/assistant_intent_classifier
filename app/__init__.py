import logging
import os

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit, join_room
from app.config import config, Config

db = SQLAlchemy()
logger = logging.getLogger(__name__)
socketio = SocketIO()
global_skills_dic = {}

def create_app(config_name=None, db_ref=None) -> Flask:
    if not config_name:
        config_name = os.getenv("FLASK_ENV", "development")
    app = Flask(__name__)
    app_config = config[config_name]
    app.config.from_object(app_config)
    app.static_folder = config[config_name].STATIC_FOLDER
    if db_ref is None:
        db.init_app(app)
        db.reflect(app=app)
    else:
        db_ref.init_app(app)
        db_ref.reflect(app=app)
    with app.app_context():
        db.create_all()

    configure_blueprints(app, app_config)
    configure_error_handlers(app)
    socketio.init_app(app)
    return app


def configure_blueprints(flask_app: Flask, config: Config):
    from app.main import main

    flask_app.register_blueprint(main)


def configure_error_handlers(app):
    @app.errorhandler(404)
    def page_not_found(error):
        return render_template("errors/404.html"), 404
