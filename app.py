# app.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Render!"

@app.route('/api/data')
def get_data():
    data = {"message": "This is some JSON data", "status": "success"}
    return jsonify(data)

if __name__ == '__main__':
    # This is for local development only.
    # Gunicorn will run the app in production.
    app.run(debug=True, host='0.0.0.0', port=5000)