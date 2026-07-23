from flask import Flask
import logging

from backend import config
from backend.database_manager import DatabaseManager
from backend.services.session_manager import SessionManager
from backend.api import register_routes

logging.basicConfig(level=config.LOG_LEVEL)

app = Flask(__name__)

db = DatabaseManager()

session_manager = SessionManager(db)

register_routes(

    app,

    db,

    session_manager

)

if __name__ == "__main__":

    app.run(

        host=config.HOST,

        port=config.PORT,

        debug=config.DEBUG

    )