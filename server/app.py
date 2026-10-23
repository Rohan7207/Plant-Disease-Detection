from flask import Flask
from flask_cors import CORS

# Creates the Flask application that will handle HTTP requests.
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Plant Disease Detection API is running"}

if __name__ == "__main__":
    # Starts the development server on port 5000
    app.run(debug=True, port=5000)