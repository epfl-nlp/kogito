import os
import traceback

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
    try:
        return jsonify(infer(request.json))
    except Exception as e:
        traceback.print_exc(e)
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG', False), host='0.0.0.0', port=port)