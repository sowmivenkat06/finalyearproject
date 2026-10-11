import io
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import wraps

import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:
    # Deployment installs python-dotenv from requirements; process environment still works without it.
    def load_dotenv(*_args, **_kwargs):
        return False
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
load_dotenv(os.path.join(APP_DIR, ".env"))
from config import CLASS_NAMES, DEVICE
from firebase_service import FirebaseService, FirebaseUnavailable
from report_service import build_prediction_report

app = Flask(__name__, static_folder=os.path.join(APP_DIR, "static"), template_folder=os.path.join(APP_DIR, "templates"))
app.config.update(SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", ""), MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)), MAX_IMAGE_PIXELS=int(os.environ.get("MAX_IMAGE_PIXELS", 25_000_000)), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true")
if not app.config["SECRET_KEY"]:
    raise RuntimeError("FLASK_SECRET_KEY must be set in backend/.env")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
# Central UI policy; the model itself only returns classes and probabilities.
CLASS_RISK_MAP = {"Dyskeratotic": "High", "Koilocytotic": "High", "Metaplastic": "Medium", "Parabasal": "Low", "Superficial-Intermediate": "Low"}
MODEL_NAME = "Feature Fusion (Attention + MLP)"
MODEL_COMPONENTS = ["ResNet50", "ResNet101", "ResNet152", "AlexNet", "VGG16", "GoogleNet", "InceptionV3"]
_ML_STATE = {"ready": False, "error": None, "extractor": None, "fusion": None, "yolo_model": None}
firebase = FirebaseService(APP_DIR)

def _init_ml():
    if _ML_STATE["ready"] or _ML_STATE["error"]: return
    try:
        import torch
        from feature_extractor import extractor
        from fusion_network import FusionClassifier
        checkpoint = torch.load(os.path.join(APP_DIR, "fusion", "fusion_model.pth"), map_location=DEVICE)
        fusion = FusionClassifier().to(DEVICE)
        fusion.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        fusion.eval()
        
        yolo_path = os.path.join(APP_DIR, "yolo_cells.pt")
        yolo_model = None
        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(yolo_path)
            except Exception as e:
                app.logger.warning(f"Failed to load YOLO model from {yolo_path}: {e}")
                
        _ML_STATE.update(ready=True, extractor=extractor, fusion=fusion, yolo_model=yolo_model)
    except Exception as exc:
        _ML_STATE["error"] = f"{type(exc).__name__}: {exc}"
        app.logger.exception("ML initialization failed")

def _allowed_file(name): return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
def _safe_next_url(value): return value if value and value.startswith("/") and not value.startswith("//") else url_for("dashboard_page")
def _user_context():
    user = session["user"]; name = user.get("name") or user.get("email", "User").split("@")[0]
    return {"user_full_name": name, "user_initials": "".join(p[0] for p in name.replace("Dr.", "").split()[:2]).upper() or "U"}
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) if session.get("user") else redirect(url_for("login_page", next=request.path))
    return wrapper
def _performance(): return firebase.load_evaluation_metrics(os.path.join(APP_DIR, "output", "feature_fusion_results.txt"), MODEL_COMPONENTS)
def _decision_fusion_performance(): return firebase.load_evaluation_metrics(os.path.join(APP_DIR, "output", "decision_fusion_results.txt"), MODEL_COMPONENTS)
def _history_filters(records, query, predicted_class, risk_status, date_value):
    query = query.casefold()
    filtered = []
    for record in records:
        searchable = " ".join(str(record.get(key, "")) for key in ("patient_id", "image_filename", "predicted_class")).casefold()
        if query and query not in searchable: continue
        if predicted_class and record.get("predicted_class") != predicted_class: continue
        if risk_status and record.get("risk_status") != risk_status: continue
        if date_value and not str(record.get("created_at", "")).startswith(date_value): continue
        filtered.append(record)
    return filtered

@app.route("/")
def root(): return redirect(url_for("dashboard_page" if session.get("user") else "login_page"))

@app.route("/login")
def login_page():
    if session.get("user"):
        return redirect(url_for("dashboard_page"))
    return render_template("login.html", firebase_web_config=firebase.web_config, performance=_performance(), df_performance=_decision_fusion_performance(), error=request.args.get("error"))

@app.route("/auth/session", methods=["POST"])
def create_session():
    try: user = firebase.verify_id_token(request.form.get("id_token", ""))
    except FirebaseUnavailable as exc: return redirect(url_for("login_page", error=str(exc)))
    except Exception as e:
        app.logger.warning("Rejected Firebase login token: %s", str(e))
        return redirect(url_for("login_page", error=f"Sign-in could not be verified: {str(e)}"))
    try:
        profile = firebase.upsert_user_profile(user)
    except Exception:
        app.logger.exception("Unable to synchronize Firebase user profile")
        return redirect(url_for("login_page", error="Sign-in could not initialize your profile."))
    session.clear(); session["user"] = {**user, "name": profile.get("display_name") or user["name"]}; session["last_login"] = datetime.now(timezone.utc).isoformat()
    return redirect(_safe_next_url(request.form.get("next")))

@app.route("/logout")
def logout_page():
    session.clear()
    if firebase.web_config:
        return render_template("logout.html", firebase_web_config=firebase.web_config)
    return redirect(url_for("login_page"))

@app.route("/dashboard")
@login_required
def dashboard_page(): return render_template("dashboard.html", active_nav="dashboard", stats=firebase.dashboard_stats(session["user"]["uid"], CLASS_RISK_MAP), performance=_performance(), df_performance=_decision_fusion_performance(), **_user_context())

@app.route("/upload")
@login_required
def upload_page(): return render_template("upload.html", active_nav="upload", **_user_context())

@app.route("/prediction")
@login_required
def prediction_page(): return render_template("prediction.html", active_nav="prediction", prediction=firebase.get_prediction(session["user"]["uid"], request.args.get("id")) if request.args.get("id") else None, **_user_context())

@app.route("/history")
@login_required
def history_page():
    page_size = min(max(request.args.get("page_size", 20, type=int), 1), 100)
    page = max(request.args.get("page", 1, type=int), 1)
    filters = {"q": request.args.get("q", "").strip()[:120], "class": request.args.get("class", "").strip()[:80], "risk": request.args.get("risk", "").strip()[:20], "date": request.args.get("date", "").strip()[:10]}
    records = _history_filters(firebase.list_predictions(session["user"]["uid"], limit=1000), filters["q"], filters["class"], filters["risk"], filters["date"])
    total = len(records); pages = max(1, (total + page_size - 1) // page_size); page = min(page, pages)
    return render_template("history.html", active_nav="history", records=records[(page - 1) * page_size:page * page_size], page=page, pages=pages, total=total, filters=filters, page_size=page_size, **_user_context())

@app.route("/performance")
@login_required
def performance_page(): return render_template("performance.html", active_nav="performance", performance=_performance(), df_performance=_decision_fusion_performance(), class_labels=CLASS_NAMES, **_user_context())

@app.route("/about")
@login_required
def about_page(): return render_template("about.html", active_nav="about", performance=_performance(), df_performance=_decision_fusion_performance(), **_user_context())
@app.route("/profile")
@login_required
def profile_page():
    profile = firebase.get_user_profile(session["user"]["uid"])
    stats = firebase.dashboard_stats(session["user"]["uid"], CLASS_RISK_MAP)
    stats.update(firebase.get_user_stats(session["user"]["uid"]))
    return render_template("profile.html", active_nav="profile", profile=profile, stats=stats, **_user_context())
@app.route("/api/profile", methods=["POST"])
@login_required
def update_profile():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("display_name", "")).strip()[:120]
    if not name: return jsonify(error="Display name is required."), 400
    profile = firebase.update_user_profile(session["user"]["uid"], name, str(payload.get("role", "")).strip()[:120], str(payload.get("institution", "")).strip()[:160], str(payload.get("license_number", "")).strip()[:120])
    session["user"] = {**session["user"], "name": profile["display_name"]}
    return jsonify(profile)
@app.route("/settings")
@login_required
def settings_page(): return render_template("settings.html", active_nav="settings", performance=_performance(), df_performance=_decision_fusion_performance(), **_user_context())

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    file = request.files.get("file"); patient_id = secure_filename(request.form.get("patient_id", "").strip())
    if not file or not file.filename or not _allowed_file(file.filename): return jsonify(error="Use a PNG, JPG, JPEG, or BMP image."), 400
    if not patient_id: return jsonify(error="Patient/sample ID is required."), 400
    try:
        raw = file.read()
        if not raw: return jsonify(error="The uploaded image is empty."), 400
        probe = Image.open(io.BytesIO(raw)); probe.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB"); original_size = image.size
        if original_size[0] * original_size[1] > app.config["MAX_IMAGE_PIXELS"]: return jsonify(error="Image dimensions exceed the allowed limit."), 413
    except Exception: return jsonify(error="The uploaded file is not a valid image."), 400
    _init_ml()
    if not _ML_STATE["ready"]: return jsonify(error="The prediction service is unavailable. Please contact the administrator."), 503
    try:
        import torch, dataset
        from tiled_yolo import predict_tiled
        started = time.perf_counter()
        
        yolo_model = _ML_STATE.get("yolo_model")
        detections = []
        if yolo_model:
            try:
                detections = predict_tiled(
                    image,
                    yolo_model,
                    tile_size=640,
                    overlap=0.2,
                    conf=0.20,
                    iou_thresh=0.45
                )
            except Exception as e:
                app.logger.warning(f"Tiled YOLO detection failed: {e}")

        w_img, h_img = image.size
        cell_detections_summary = []
        overall_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        if detections:
            for i, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det['box']
                pad = 15
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w_img, x2 + pad)
                cy2 = min(h_img, y2 + pad)

                crop = image.crop((cx1, cy1, cx2, cy2))
                processed = dataset.preprocess_image(crop, auto_crop=False)

                img224 = dataset.transform_224(processed).unsqueeze(0).to(DEVICE)
                img299 = dataset.transform_299(processed).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    features = _ML_STATE["extractor"].extract(img224, img299)
                    output = _ML_STATE["fusion"](features)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                overall_probs += probs
                c_idx = int(np.argmax(probs))
                cell_class = CLASS_NAMES[c_idx]

                cell_detections_summary.append({
                    "cell_index": i,
                    "box": [x1, y1, x2, y2],
                    "yolo_confidence": float(det['conf']),
                    "predicted_class": cell_class,
                    "confidence": float(probs[c_idx])
                })

            overall_probs /= len(detections)
            index = int(np.argmax(overall_probs))
        else:
            processed = dataset.preprocess_image(image, auto_crop=True)
            img224 = dataset.transform_224(processed).unsqueeze(0).to(DEVICE)
            img299 = dataset.transform_299(processed).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                overall_probs = torch.softmax(_ML_STATE["fusion"](_ML_STATE["extractor"].extract(img224, img299)), dim=1).cpu().numpy()[0]
            index = int(np.argmax(overall_probs))

        prediction_id = uuid.uuid4().hex
        filename = secure_filename(file.filename)
        record = {
            "prediction_id": prediction_id,
            "user_id": session["user"]["uid"],
            "patient_id": patient_id,
            "image_filename": filename,
            "storage_path": f"uploads/{session['user']['uid']}/{prediction_id}/{filename}",
            "predicted_class": CLASS_NAMES[index],
            "confidence": float(overall_probs[index]),
            "class_probabilities": {CLASS_NAMES[i]: float(overall_probs[i]) for i in range(len(CLASS_NAMES))},
            "risk_status": CLASS_RISK_MAP[CLASS_NAMES[index]],
            "cells_detected_count": len(detections),
            "cell_detections": cell_detections_summary,
            "model_name": MODEL_NAME,
            "models_used": MODEL_COMPONENTS,
            "processing_time_ms": int((time.perf_counter() - started) * 1000),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_size": f"{original_size[0]} × {original_size[1]} px"
        }
        record["image_url"] = firebase.upload_image(session["user"]["uid"], prediction_id, record["image_filename"], raw)
        firebase.save_prediction(record)
        return jsonify(record), 201
    except FirebaseUnavailable as exc: return jsonify(error=str(exc)), 503
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb + "\n")
        try:
            with open(os.path.join(APP_DIR, "error.log"), "w") as f:
                f.write(tb)
        except Exception:
            pass
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction could not be completed: {str(e)}"), 500
@app.route("/reports/<prediction_id>.pdf")
@login_required
def download_report(prediction_id):
    record = firebase.get_prediction(session["user"]["uid"], prediction_id)
    if not record: abort(404)
    try:
        image_bytes = firebase.download_image(session["user"]["uid"], prediction_id, record["image_filename"])
        report = build_prediction_report(record, image_bytes)
        firebase.increment_report_downloads(session["user"]["uid"])
        return send_file(report, mimetype="application/pdf", as_attachment=True, download_name=f"cerviscan-{prediction_id}.pdf")
    except FirebaseUnavailable as exc: return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Report generation failed")
        return jsonify(error="The report could not be generated."), 500
@app.route("/api/health")
def api_health(): _init_ml(); return jsonify(ok=True, ml_ready=_ML_STATE["ready"], firebase_ready=firebase.ready, classes=CLASS_NAMES)
@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_): return jsonify(error="Image exceeds the 10 MB upload limit."), 413
@app.errorhandler(FirebaseUnavailable)
def firebase_unavailable_error(exc):
    app.logger.warning("Firebase request unavailable: %s", exc)
    if request.path.startswith("/api/") or request.path.startswith("/reports/"):
        return jsonify(error="A required data service is temporarily unavailable."), 503
    return "A required data service is temporarily unavailable. Please try again later.", 503
if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=False)
