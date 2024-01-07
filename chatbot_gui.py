from flask import Flask, render_template, request, jsonify, send_from_directory
from chatbot import get_response, predict_class
import json
import os

app = Flask(__name__)

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

@app.route('/pyto')
def index():
    return render_template('bot.html')

@app.route('/process', methods=['POST'])
def process():
    content = request.json
    user_message = content['message']
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    return jsonify({'response': res})



if __name__ == '__main__':
    app.run(debug=True)
