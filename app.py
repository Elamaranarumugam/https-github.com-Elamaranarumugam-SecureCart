
import csv
import math
import re
import time
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from threading import Lock

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    g
)


app = Flask(__name__)

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ML" / "decision_tree.joblib"
TOCTOU_MODEL_PATH = BASE_DIR / "ML" / "toctou_model.joblib"

HISTORY_PATH = BASE_DIR / "ML" / "prediction_history.csv"
TOCTOU_HISTORY_PATH = BASE_DIR / "ML" / "toctou_prediction_history.csv"
LIVE_HISTORY_PATH = BASE_DIR / "ML" / "live_detection_history.csv"

app.template_folder = str(BASE_DIR / "templates")
app.static_folder = str(BASE_DIR / "static")


# =====================================================
# 2. CLASS LABELS AND CSV COLUMNS
# =====================================================

CLASS_LABELS = {
    "0": "Normal Traffic",
    "1": "SQL Injection-like",
    "2": "XSS-like",
    "3": "Brute-force-like",
    "4": "Race-condition-like",
}

HISTORY_COLUMNS = [
    "timestamp",
    "predicted_class",
    "prediction_label",
    "features"
]

TOCTOU_HISTORY_COLUMNS = [
    "timestamp",
    "predicted_class",
    "prediction_label",
    "features"
]

LIVE_HISTORY_COLUMNS = [
    "timestamp",
    "method",
    "path",
    "predicted_class",
    "prediction_label",
    "features"
]

TOCTOU_FEATURES = [
    "check_duration_ms",
    "check_use_delay_ms",
    "use_duration_ms",
    "concurrent_requests",
    "duplicate_request_count",
]


# =====================================================
# 3. LOAD ORIGINAL FIVE-CLASS MODEL
# =====================================================

model = None
FEATURES = []

try:
    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        FEATURES = bundle.get("features", [])
    else:
        model = bundle
        FEATURES = list(
            getattr(model, "feature_names_in_", [])
        )

    if not FEATURES:
        raise ValueError(
            "Feature names not found. Check ML/train_model.py."
        )

    print("Decision Tree model loaded successfully.")
    print("Model features:", FEATURES)

except Exception as exc:
    print("WARNING: Decision Tree model could not be loaded.")
    print("Reason:", exc)


# =====================================================
# 3A. LOAD SEPARATE TOCTOU MODEL
# =====================================================

toctou_model = None

try:
    toctou_bundle = joblib.load(TOCTOU_MODEL_PATH)

    if isinstance(toctou_bundle, dict) and "model" in toctou_bundle:
        toctou_model = toctou_bundle["model"]
        TOCTOU_FEATURES = toctou_bundle.get(
            "features",
            TOCTOU_FEATURES
        )
    else:
        toctou_model = toctou_bundle

        saved_features = getattr(
            toctou_model,
            "feature_names_in_",
            None
        )

        if saved_features is not None:
            TOCTOU_FEATURES = list(saved_features)

    if not TOCTOU_FEATURES:
        raise ValueError(
            "TOCTOU feature names were not found."
        )

    print("TOCTOU model loaded successfully.")
    print("TOCTOU features:", TOCTOU_FEATURES)

except Exception as exc:
    print("WARNING: TOCTOU model could not be loaded.")
    print("Reason:", exc)


# =====================================================
# 4. CREATE CSV FILES
# =====================================================

def create_csv_file(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )
            writer.writeheader()


def ensure_history_file():
    create_csv_file(
        HISTORY_PATH,
        HISTORY_COLUMNS
    )


def ensure_toctou_history_file():
    create_csv_file(
        TOCTOU_HISTORY_PATH,
        TOCTOU_HISTORY_COLUMNS
    )


def ensure_live_history_file():
    create_csv_file(
        LIVE_HISTORY_PATH,
        LIVE_HISTORY_COLUMNS
    )


# =====================================================
# 5. SAVE ORIGINAL MODEL PREDICTIONS
# =====================================================

def save_prediction(prediction, values):

    ensure_history_file()

    label = CLASS_LABELS.get(
        str(prediction),
        f"Unknown class ({prediction})"
    )

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "predicted_class": str(prediction),
        "prediction_label": label,
        "features": "; ".join(
            f"{feature}={values[feature]}"
            for feature in FEATURES
        ),
    }

    with HISTORY_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_COLUMNS
        )

        writer.writerow(record)


# =====================================================
# 5A. SAVE TOCTOU PREDICTIONS
# =====================================================

def save_toctou_prediction(
    prediction,
    label,
    values
):

    ensure_toctou_history_file()

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "predicted_class": str(prediction),
        "prediction_label": label,
        "features": "; ".join(
            f"{feature}={values[feature]}"
            for feature in TOCTOU_FEATURES
        ),
    }

    with TOCTOU_HISTORY_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=TOCTOU_HISTORY_COLUMNS
        )

        writer.writerow(record)


# =====================================================
# 5B. SAVE AUTOMATIC LIVE PREDICTIONS
# =====================================================

def save_live_prediction(
    method,
    path,
    prediction,
    values
):

    ensure_live_history_file()

    label = CLASS_LABELS.get(
        str(prediction),
        f"Unknown class ({prediction})"
    )

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "method": method,
        "path": path,
        "predicted_class": str(prediction),
        "prediction_label": label,
        "features": "; ".join(
            f"{feature}={values[feature]}"
            for feature in FEATURES
        ),
    }

    with LIVE_HISTORY_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LIVE_HISTORY_COLUMNS
        )

        writer.writerow(record)


# =====================================================
# 6. LIVE HTTP TRAFFIC MONITORING
# =====================================================

traffic_lock = Lock()

request_times = defaultdict(
    lambda: deque(maxlen=100)
)

login_times = defaultdict(
    lambda: deque(maxlen=100)
)

duplicate_requests = defaultdict(
    lambda: deque(maxlen=100)
)

active_requests = 0


# Routes that should not be monitored by the
# automatic live detector.

IGNORED_PATHS = {
    "/api/predict",
    "/api/predict-toctou",
    "/health",
    "/dashboard",
    "/download-history",
    "/download-toctou-history",
    "/live-history",
    "/download-live-history",
}


# =====================================================
# 6A. AUTOMATICALLY CAPTURE INCOMING REQUESTS
# =====================================================

@app.before_request
def capture_live_request():

    global active_requests

    # Ignore internal API, dashboard and static requests.
    if (
        request.path in IGNORED_PATHS
        or request.path.startswith("/static/")
    ):
        return

    g.live_detection = False
    g.live_counted = False

    ip = request.remote_addr or "unknown"
    now = time.time()

    try:

        with traffic_lock:

            active_requests += 1
            g.live_counted = True

            # -----------------------------------------
            # Feature 1: Inter-arrival time
            # -----------------------------------------

            previous = request_times[ip]

            if previous:
                inter_arrival_ms = (
                    now - previous[-1]
                ) * 1000
            else:
                inter_arrival_ms = 1000.0

            previous.append(now)

            # -----------------------------------------
            # Feature 2: Login frequency
            # -----------------------------------------

            if "login" in request.path.lower():
                login_times[ip].append(now)

            login_frequency = sum(
                1
                for timestamp in login_times[ip]
                if now - timestamp <= 60
            )

            # -----------------------------------------
            # Feature 3: Duplicate request count
            # -----------------------------------------

            fingerprint = (
                ip,
                request.method,
                request.path,
                request.query_string
            )

            duplicates = duplicate_requests[fingerprint]

            duplicate_request_count = sum(
                1
                for timestamp in duplicates
                if now - timestamp <= 10
            )

            duplicates.append(now)

            # -----------------------------------------
            # Feature 4: Concurrent requests
            # -----------------------------------------

            concurrent_requests = active_requests

        # ---------------------------------------------
        # Inspect request content for basic patterns.
        # Do not save the original request content.
        # ---------------------------------------------

        request_text_parts = []

        # Query string and form values
        for key, value in request.values.items():
            request_text_parts.append(str(key))
            request_text_parts.append(str(value))

        # JSON request values
        json_data = request.get_json(silent=True)

        if isinstance(json_data, dict):

            for key, value in json_data.items():
                request_text_parts.append(str(key))
                request_text_parts.append(str(value))

        request_text = " ".join(
            request_text_parts
        )[:16000]

        # ---------------------------------------------
        # Feature 5: SQL keyword count
        # ---------------------------------------------

        sql_patterns = (
            r"\b(select|union|insert|update|delete|drop|or)\b"
        )

        sql_keyword_count = len(
            re.findall(
                sql_patterns,
                request_text,
                re.IGNORECASE
            )
        )

        # ---------------------------------------------
        # Feature 6: Script pattern count
        # ---------------------------------------------

        script_patterns = (
            r"(<script\b|javascript:|onerror\s*=|onload\s*=)"
        )

        script_pattern_count = len(
            re.findall(
                script_patterns,
                request_text,
                re.IGNORECASE
            )
        )

        # ---------------------------------------------
        # Feature 7: Session reuse proxy
        # ---------------------------------------------

        session_reuse = int(
            "session" in request.cookies
        )

        # Store extracted features temporarily.
        # Raw request content is not stored.

        g.live_features = {
            "sql_keyword_count": sql_keyword_count,
            "script_pattern_count": script_pattern_count,
            "login_frequency": login_frequency,
            "inter_arrival_ms": inter_arrival_ms,
            "concurrent_requests": concurrent_requests,
            "duplicate_request_count": duplicate_request_count,
            "session_reuse": session_reuse,
        }

        g.live_detection = True

    except Exception:

        app.logger.exception(
            "Live request feature extraction failed."
        )


# =====================================================
# 6B. PREDICT AUTOMATICALLY AFTER REQUEST
# =====================================================

def release_live_request():

    global active_requests

    if getattr(g, "live_counted", False):

        with traffic_lock:

            active_requests = max(
                0,
                active_requests - 1
            )

        g.live_counted = False


@app.after_request
def predict_live_traffic(response):

    if not getattr(g, "live_detection", False):
        release_live_request()
        return response

    try:

        if model is None or not FEATURES:
            return response

        values = g.live_features

        # Ensure the trained model's feature names
        # are available in the extracted feature set.

        missing = [
            feature
            for feature in FEATURES
            if feature not in values
        ]

        if missing:

            app.logger.error(
                "Live prediction missing features: %s",
                missing
            )

            return response

        # Create DataFrame in the exact training order.
        features_df = pd.DataFrame(
            [
                {
                    feature: values[feature]
                    for feature in FEATURES
                }
            ],
            columns=FEATURES
        )

        # Run the original five-class model.
        prediction = str(
            model.predict(features_df)[0]
        )

        label = CLASS_LABELS.get(
            prediction,
            "Unknown"
        )

        # Save automatic live prediction.
        save_live_prediction(
            request.method,
            request.path,
            prediction,
            values
        )

        # Add prediction information to response headers.
        response.headers["X-ML-Prediction"] = label

        print(
            "[LIVE DETECTION]",
            request.method,
            request.path,
            "=>",
            label
        )

    except Exception:

        app.logger.exception(
            "Live prediction failed."
        )

    finally:

        release_live_request()

    return response


# =====================================================
# 6C. CLEANUP IF A REQUEST ENDS BEFORE AFTER_REQUEST
# =====================================================

@app.teardown_request
def cleanup_live_request(error=None):

    release_live_request()


# =====================================================
# 7. HOME PAGE
# =====================================================

@app.route("/")
@app.route("/home")
def home():

    products = [
        {"name": "Laptop", "price": 45000},
        {"name": "Headphones", "price": 1500},
        {"name": "Keyboard", "price": 900},
    ]

    return render_template(
        "index.html",
        products=products
    )


# =====================================================
# 8. DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():

    ensure_history_file()

    with HISTORY_PATH.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        history = list(
            csv.DictReader(file)
        )

    history.reverse()

    total_predictions = len(history)

    normal_count = sum(
        row["predicted_class"] == "0"
        for row in history
    )

    attack_like_count = (
        total_predictions - normal_count
    )

    class_counts = {
        class_id: sum(
            row["predicted_class"] == class_id
            for row in history
        )
        for class_id in CLASS_LABELS
    }

    ensure_toctou_history_file()

    with TOCTOU_HISTORY_PATH.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        toctou_history = list(
            csv.DictReader(file)
        )

    toctou_history.reverse()

    toctou_total = len(toctou_history)

    toctou_count = sum(
        row["predicted_class"] == "1"
        for row in toctou_history
    )

    toctou_normal_count = sum(
        row["predicted_class"] == "0"
        for row in toctou_history
    )

    return render_template(
        "dashboard.html",
        history=history,
        total_predictions=total_predictions,
        normal_count=normal_count,
        attack_like_count=attack_like_count,
        class_labels=list(CLASS_LABELS.values()),
        class_counts=[
            class_counts[class_id]
            for class_id in CLASS_LABELS
        ],
        toctou_history=toctou_history,
        toctou_total=toctou_total,
        toctou_count=toctou_count,
        toctou_normal_count=toctou_normal_count,
    )


# =====================================================
# 9. DOWNLOAD ORIGINAL HISTORY
# =====================================================

@app.route("/download-history")
def download_history():

    ensure_history_file()

    return send_file(
        HISTORY_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )


# =====================================================
# 9A. DOWNLOAD TOCTOU HISTORY
# =====================================================

@app.route("/download-toctou-history")
def download_toctou_history():

    ensure_toctou_history_file()

    return send_file(
        TOCTOU_HISTORY_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="toctou_prediction_history.csv"
    )

# =====================================================
# 9B. VIEW LIVE DETECTION HISTORY
# =====================================================

@app.route("/live-history", methods=["GET"])
def live_history():

    ensure_live_history_file()

    try:
        with LIVE_HISTORY_PATH.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            records = list(csv.DictReader(file))

        records.reverse()

        return jsonify({
            "total_live_predictions": len(records),
            "history": records[:100]
        })

    except Exception as exc:

        app.logger.exception("Unable to read live history")

        return jsonify({
            "error": str(exc)
        }), 500

# =====================================================
# 9C. DOWNLOAD LIVE DETECTION HISTORY
# =====================================================

@app.route("/download-live-history")
def download_live_history():

    ensure_live_history_file()

    return send_file(
        LIVE_HISTORY_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="live_detection_history.csv"
    )


# =====================================================
# 10. HEALTH CHECK
# =====================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "features": FEATURES,
        "toctou_model_loaded": toctou_model is not None,
        "toctou_features": TOCTOU_FEATURES,
        "live_monitoring": True,
    })


# =====================================================
# 11. ORIGINAL FIVE-CLASS PREDICTION API
# =====================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    if model is None or not FEATURES:

        return jsonify({
            "error": "Model is not ready. Check the terminal output."
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    missing = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing:

        return jsonify({
            "error": "Required features are missing.",
            "missing": missing,
        }), 400

    values = {}

    try:

        for feature in FEATURES:

            value = data[feature]

            if isinstance(value, bool):
                raise ValueError(
                    f"{feature} must be numeric."
                )

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    f"{feature} must be finite."
                )

            values[feature] = number

    except (TypeError, ValueError) as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    try:

        features_df = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = str(
            model.predict(features_df)[0]
        )

        label = CLASS_LABELS.get(
            prediction,
            f"Unknown class ({prediction})"
        )

        save_prediction(
            prediction,
            values
        )

        return jsonify({
            "predicted_class": prediction,
            "prediction_label": label,
            "note": (
                "Prototype ML prediction only. This result does not "
                "guarantee that a request is safe."
            ),
        })

    except Exception:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error": (
                "Prediction failed. Check the model "
                "and feature format."
            )
        }), 500


# =====================================================
# 12. SEPARATE TOCTOU PREDICTION API
# =====================================================

@app.route("/api/predict-toctou", methods=["POST"])
def predict_toctou():

    if toctou_model is None or not TOCTOU_FEATURES:

        return jsonify({
            "error": "TOCTOU model is not loaded."
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    missing = [
        feature
        for feature in TOCTOU_FEATURES
        if feature not in data
    ]

    if missing:

        return jsonify({
            "error": "Required features are missing.",
            "missing": missing,
        }), 400

    try:

        values = {}

        for feature in TOCTOU_FEATURES:

            value = data[feature]

            if isinstance(value, bool):
                raise ValueError(
                    f"{feature} must be numeric."
                )

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    f"{feature} must be finite."
                )

            values[feature] = number

        features_df = pd.DataFrame(
            [values],
            columns=TOCTOU_FEATURES
        )

        prediction = int(
            toctou_model.predict(features_df)[0]
        )

        label = (
            "TOCTOU-like"
            if prediction == 1
            else "Normal"
        )

        save_toctou_prediction(
            prediction,
            label,
            values
        )

        return jsonify({
            "predicted_class": prediction,
            "prediction_label": label,
            "note": (
                "Experimental prediction using controlled simulated "
                "data. It has not been validated for real-world attacks."
            ),
        })

    except (TypeError, ValueError) as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:

        app.logger.exception(
            "TOCTOU prediction failed"
        )

        return jsonify({
            "error": "TOCTOU prediction failed."
        }), 500


# =====================================================
# 13. START FLASK SERVER
# =====================================================

if __name__ == "__main__":

    ensure_history_file()
    ensure_toctou_history_file()
    ensure_live_history_file()

    print("--------------------------------------")
    print("SECURECART AI CYBER DETECTION")
    print("--------------------------------------")
    print("Main model:", model is not None)
    print("TOCTOU model:", toctou_model is not None)
    print("Live HTTP monitoring: Enabled")
    print("Live history:", LIVE_HISTORY_PATH)

    # Show all registered Flask routes
    print("\nREGISTERED ROUTES:")
    print(app.url_map)

    print("--------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )