
from flask import Blueprint, jsonify, request

from server.services.disease_service import DiseaseService

disease_bp = Blueprint("disease", __name__)

disease_service = DiseaseService()


@disease_bp.route("/diseases/<string:crop>/<string:disease>", methods=["GET"])
def get_disease_info(crop, disease):
    language = request.args.get("language", "en")

    result = disease_service.get_disease_info(
        crop=crop,
        disease=disease,
        language=language
    )

    return jsonify({
        "success": True,
        "data": result
    }), 200
