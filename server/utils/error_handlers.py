
import logging

from flask import jsonify
from werkzeug.exceptions import HTTPException

from server.utils.exceptions import AppError

logger = logging.getLogger(__name__)


def register_error_handlers(app):

    @app.errorhandler(AppError)
    def handle_app_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": error.error_code,
                "message": error.message
            }
        }), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": error.name.upper().replace(" ", "_"),
                "message": error.description
            }
        }), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        logger.exception("Unexpected application error")

        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred."
            }
        }), 500
