from flask import jsonify, request
import logging

from backend import config

logger = logging.getLogger("strideiq.api")


def register_routes(app, db, session_manager):

    @app.route(
        config.API_ENDPOINTS["run_start"],
        methods=["POST"]
    )
    def run_start():

        body = request.get_json(force=True) or {}

        run_type = body.get(
            "run_type",
            "natural"
        )

        notes = body.get("notes")

        is_filmed = body.get(
            "is_filmed",
            False
        )

        duration_sec = body.get(
            "duration_sec",
            30
        )

        run_id = session_manager.start_session(

            run_type=run_type,

            duration_sec=duration_sec,

            notes=notes,

            is_filmed=is_filmed

        )

        return jsonify({

            "status": "started",

            "run_id": run_id

        })


    @app.route(
        config.API_ENDPOINTS["run_stop"],
        methods=["POST"]
    )
    def run_stop():

        body = request.get_json(force=True)

        run_id = body["run_id"]

        session_manager.stop_session(run_id)

        return jsonify({

            "status": "stopped",

            "run_id": run_id

        })


    @app.route(
        config.API_ENDPOINTS["runs"],
        methods=["GET"]
    )
    def runs():

        runs = db.get_all_runs()

        return jsonify(

            [dict(r) for r in runs]

        )


    @app.route(
        config.API_ENDPOINTS["metrics_live"],
        methods=["GET"]
    )
    def metrics_live():

        run_id = request.args.get(
            "run_id",
            type=int
        )

        features = db.get_features(run_id)

        if not features:

            return jsonify({"latest": None})

        return jsonify(

            dict(features[-1])

        )


    @app.route(
        config.API_ENDPOINTS["recommendations"],
        methods=["GET"]
    )
    def recommendations():

        run_id = request.args.get(
            "run_id",
            type=int
        )

        recs = db.get_recommendations(run_id)

        return jsonify(

            [dict(r) for r in recs]

        )