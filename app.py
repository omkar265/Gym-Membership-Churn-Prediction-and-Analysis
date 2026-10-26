import io
import logging
import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from src.ml_pipeline import (
    analyze_dataset,
    get_dashboard_data,
    get_prediction_fields,
    load_model_summary,
    predict_churn,
    train_and_evaluate_models,
)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
ROOT_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)


@app.route("/")
@app.route("/dashboard")
def overview():
    dashboard = get_dashboard_data()
    analysis = analyze_dataset()
    summary = load_model_summary()
    return render_template(
        "index.html", dashboard=dashboard, analysis=analysis, summary=summary
    )


@app.route("/predict")
def prediction_page():
    return render_template(
        "predict.html",
        fields=get_prediction_fields(),
        model_ready=(ROOT_DIR / "artifacts" / "models" / "best_model.joblib").exists(),
    )


@app.route("/dataset")
@app.route("/dataset-analysis")
def dataset_page():
    return render_template("dataset.html", analysis=analyze_dataset())


@app.route("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", summary=load_model_summary())


@app.route("/about")
def about_page():
    return render_template("about.html", summary=load_model_summary())


@app.route("/api/train", methods=["POST"])
def train_endpoint():
    if os.environ.get("ENABLE_MODEL_TRAINING", "true").lower() != "true":
        return jsonify(
            {"status": "error", "message": "Model training is disabled on this deployment."}
        ), 403
    summary = train_and_evaluate_models()
    return jsonify({"status": "success", "summary": summary})


@app.route("/api/dashboard")
def dashboard_api():
    return jsonify(get_dashboard_data())


@app.route("/api/dataset")
def dataset_api():
    return jsonify(analyze_dataset())


@app.route("/dataset/report.csv")
def dataset_report():
    analysis = analyze_dataset()
    statistics_by_name = {
        item["name"]: item for item in analysis["numeric_statistics"]
    }
    report = pd.DataFrame(
        [
            {
                **column,
                **{
                    key: value
                    for key, value in statistics_by_name.get(column["name"], {}).items()
                    if key != "name"
                },
            }
            for column in analysis["columns"]
        ]
    )
    buffer = io.BytesIO(report.to_csv(index=False).encode("utf-8-sig"))
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="gym_dataset_analysis.csv",
    )


@app.route("/api/evaluation")
def evaluation_api():
    return jsonify({"status": "success", "summary": load_model_summary()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "Send prediction inputs as a JSON object."}), 400

    try:
        fields = get_prediction_fields()
        expected = {field["name"]: field for field in fields}
        missing = [name for name in expected if name not in payload]
        extra = [name for name in payload if name not in expected]
        if missing or extra:
            details = []
            if missing:
                details.append(f"Missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"Unexpected fields: {', '.join(extra)}")
            return jsonify({"status": "error", "message": ". ".join(details)}), 400

        cleaned = {}
        for name, field in expected.items():
            value = payload[name]
            if field["kind"] == "number":
                if isinstance(value, bool):
                    raise ValueError(f"{field['label']} must be a number.")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"{field['label']} must be a valid number.") from None
                if not math.isfinite(number):
                    raise ValueError(f"{field['label']} must be a finite number.")
                if number < field["minimum"] or number > field["maximum"]:
                    raise ValueError(
                        f"{field['label']} must be within the observed dataset range "
                        f"({field['minimum']:g} to {field['maximum']:g})."
                    )
                if field["integer"] and not number.is_integer():
                    raise ValueError(f"{field['label']} must be a whole number.")
                cleaned[name] = int(number) if field["integer"] else number
            else:
                if not isinstance(value, str) or value not in field["choices"]:
                    raise ValueError(
                        f"Choose a valid option for {field['label']}."
                    )
                cleaned[name] = value

        prediction, probability = predict_churn(cleaned)
        return jsonify(
            {
                "status": "success",
                "prediction": prediction,
                "probability": round(probability, 4) if probability is not None else None,
                "label": "Churn" if prediction == 1 else "No Churn",
                "explanation": "This is a model estimate, not a guaranteed outcome.",
            }
        )
    except FileNotFoundError:
        return jsonify(
            {"status": "error", "message": "No trained model is available. Train the model first."}
        ), 503
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected error while generating a churn prediction")
        return jsonify(
            {"status": "error", "message": "Prediction could not be completed. Check the server log."}
        ), 500


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )
