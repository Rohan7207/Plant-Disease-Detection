
from flask import Flask, jsonify
from flask_cors import CORS

from server.config import Config
from server.extensions import db, migrate, jwt, limiter
import server.models
from server.utils.error_handlers import register_error_handlers

from server.routes.prediction_routes import prediction_bp
from server.routes.disease_routes import disease_bp
from server.routes.auth_routes import auth_bp

def create_app(config_class=Config):
    app = Flask(__name__)

    app.config.from_object(config_class)

    # Initialize database extensions
    db.init_app(app)     # Connects SQLAlchemy to Flask
    migrate.init_app(app, db)  # Connects Flask-Migrate to SQLAlchemy

    # Initialize JWT
    jwt.init_app(app)
    
    # Initialize rate-limiter
    limiter.init_app(app)

    # Register CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": app.config["CORS_ORIGINS"]
            }
        }
    )

    # Register API Blueprints
    app.register_blueprint(
        prediction_bp,
        url_prefix="/api"
    )
    app.register_blueprint(disease_bp, url_prefix="/api")
    app.register_blueprint(
        auth_bp,
        url_prefix="/api/auth"
    )
    
    # Register global error handlers
    register_error_handlers(app)

    @app.route("/")
    def home():
        return jsonify({
            "success": True,
            "message": "Plant Disease Detection API is running"
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=app.config["DEBUG"],
        use_reloader=False
    )
