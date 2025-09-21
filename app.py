from flask import Flask
from flask_cors import CORS
from chatbot import chatbot_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(chatbot_bp, url_prefix="/chatbot")

@app.route("/")
def home():
    return {"message": "Hack2Skill Project Backend Running"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
