from flask import Flask, request, jsonify
from model.TrainingBot import response
from model.CustomException import CustomException
from model.CustomResponse import CustomResponse
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['GET'])
def test():
    data = request.get_json()
    a = data.get('a')
    b = data.get('b')
    return str(int(a) + int(b))


@app.route('/api/response', methods=['POST'])
def chat():
    data = request.get_json()
    result = response(data.get('message'))
    URL = 'http://localhost:5000/api/response'
    requests.post(URL, json={'response': result})
    return CustomResponse(200, 'Successful response', result).__dict__

if __name__ == '__main__':
    app.run(debug=True, port=5001)
