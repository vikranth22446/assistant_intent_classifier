import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    socketio.run(app, port=8000)
