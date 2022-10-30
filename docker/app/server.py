import os
import traceback

from flask import Flask, jsonify, request

from unet_inferrer import UnetInferrer

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/infer')
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8080))

u_net = UnetInferrer()


@app.route('/infer/', methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    return u_net.infer(image)

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our ML test page !!</h1>"

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(port=PORT_NUMBER)