from flask import Flask, render_template, jsonify, request
import threading
import time
import os

from bot import PaperEngine, CONFIG

app = Flask(__name__)
engine = PaperEngine()
lock = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok", "paper_only": True}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def state():
    with lock:
        return jsonify(engine.snapshot())


@app.post("/api/settings")
def settings():
    data = request.get_json(silent=True) or {}
    with lock:
        engine.update_settings(data)
        return jsonify(engine.snapshot())


@app.post("/api/toggle")
def toggle():
    with lock:
        engine.running = not engine.running
        return jsonify(engine.snapshot())


@app.post("/api/reset")
def reset():
    with lock:
        engine.reset()
        return jsonify(engine.snapshot())


@app.post("/api/step")
def step():
    with lock:
        result = engine.step()
        return jsonify(result)


def loop():
    while True:
        try:
            with lock:
                if engine.running:
                    engine.step()
        except Exception as exc:
            engine.last_error = str(exc)

        time.sleep(CONFIG["POLL_SECONDS"])


threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
