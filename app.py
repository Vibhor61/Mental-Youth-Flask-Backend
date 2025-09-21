from flask import Flask
from flask_cors import CORS
from chatbot import chatbot_bp

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "https://mindspace001.netlify.app/"}})

app.register_blueprint(chatbot_bp, url_prefix="/chatbot")

@app.route("/")
def home():
    return {"message": "Hack2Skill Project Backend Running"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)


