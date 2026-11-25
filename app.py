import json
from hmac import compare_digest
from os import environ
from secrets import token_hex

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from config import MAX_IMAGE_BYTES, WEB_MODEL_DIR, WEB_RESULTS_DIR
from src.predict import PhotoTooLarge, load_classifier, load_presence_classifier, predict_image


ENGLISH_NAMES = {
    "Anh đào": "Cherry", "Bơ": "Avocado", "Bưởi": "Pomelo", "Cam": "Orange",
    "Cà chua": "Tomato", "Chanh dây tím": "Purple passion fruit",
    "Chanh dây ngọt": "Sweet granadilla", "Chanh vàng": "Lemon",
    "Chanh xanh": "Lime", "Chà là": "Date", "Chuối": "Banana",
    "Dâu tây": "Strawberry", "Dứa": "Pineapple", "Dừa": "Coconut",
    "Dưa hấu": "Watermelon", "Dưa pepino": "Pepino melon", "Hồng": "Persimmon", "Khế": "Star fruit",
    "Kiwi": "Kiwifruit", "Lê": "Pear", "Mâm xôi": "Raspberry",
    "Mận tím": "Plum", "Mít": "Jackfruit", "Mơ": "Apricot", "Mộc qua": "Quince",
    "Nho": "Grape", "Nho xanh": "Green grape", "Quýt": "Mandarin", "Quất": "Kumquat",
    "Quả xương rồng": "Prickly pear", "Salak": "Snake fruit", "Sapôchê": "Sapodilla",
    "Tamarillo": "Tamarillo", "Thanh long": "Dragon fruit", "Thơm": "Pineapple",
    "Việt quất": "Blueberry", "Vải": "Lychee", "Xoài": "Mango",
    "Táo": "Apple", "Đào": "Peach", "Ổi": "Guava",
}


def create_app(classifier=None, presence_classifier=None):
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES + 64 * 1024,
        SECRET_KEY=environ.get("FRUIT_SESSION_KEY") or token_hex(32),
        SESSION_COOKIE_SAMESITE="Lax",
    )
    admin_password = environ.get("FRUIT_ADMIN_PASSWORD", "admin").encode("utf-8")
    metrics = None
    if classifier is None:
        try:
            classifier = load_classifier()
            metrics_path = WEB_RESULTS_DIR / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Không tải được bộ mô hình tại {WEB_MODEL_DIR}: {error}") from error
        if presence_classifier is None:
            presence_classifier = load_presence_classifier()
    names = classifier[1]

    @app.before_request
    def require_login():
        if request.endpoint in {None, "login", "static"} or session.get("logged_in"):
            return None
        if request.endpoint == "predict":
            return jsonify(error="Vui lòng đăng nhập để phân loại ảnh."), 401
        return redirect(url_for("login"))

    @app.after_request
    def prevent_session_page_caching(response):
        if request.endpoint not in {None, "static"}:
            response.cache_control.no_store = True
        return response

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("logged_in"):
            return redirect(url_for("home"))
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").encode("utf-8")
            if username == "admin" and compare_digest(password, admin_password):
                session.clear()
                session["logged_in"] = True
                return redirect(url_for("home"), code=303)
            return render_template("login.html", error="Tên đăng nhập hoặc mật khẩu chưa đúng."), 401
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"), code=303)

    @app.get("/", endpoint="home")
    @app.get("/recognition")
    def recognize():
        return render_template("predict.html", names=names, max_bytes=MAX_IMAGE_BYTES)

    @app.get("/recognize")
    def legacy_recognize():
        return redirect(url_for("recognize"))

    @app.get("/about")
    def about():
        return render_template("about.html", names=names, metrics=metrics, english=ENGLISH_NAMES)

    @app.post("/predict")
    def predict():
        upload = request.files.get("photo")
        if upload is None or not upload.filename:
            return jsonify(error="Hãy chọn hoặc chụp một ảnh."), 400
        if not upload.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return jsonify(error="Chỉ hỗ trợ ảnh JPG, JPEG, PNG và WebP."), 400
        try:
            result = predict_image(upload.stream, classifier, presence_classifier)
        except PhotoTooLarge as error:
            return jsonify(error=str(error)), 413
        except ValueError as error:
            return jsonify(error=str(error)), 400
        for guess in result["top3"]:
            guess["english"] = ENGLISH_NAMES.get(guess["label"])
        return jsonify(result)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="Ảnh vượt quá 5 MB."), 413

    return app


if __name__ == "__main__":
    create_app().run()
