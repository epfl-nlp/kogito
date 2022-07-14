from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import infer

app = Flask(__name__)
CORS(app)

@app.route("/")
def heartbeat():
    return "Running"

@app.route("/inference", methods=["POST"])
def inference():
    return jsonify(infer(request.json))