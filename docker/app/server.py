import os
from pickle import TRUE
import traceback

from flask import Flask, jsonify, request

from unet_inferrer import UnetInferrer

app = Flask(__name__)


u_net = UnetInferrer()


@app.route('/infer', methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    response = u_net.infer(image) 
    return jsonify(response)

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our ML test page !!</h1>"

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    HOST = "0.0.0.0"
    PORT = int(os.environ.get('PORT', 5000))
    app.run(threaded=True, host = HOST, port= PORT)