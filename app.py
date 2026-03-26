
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

app  = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
lr_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
km_model  = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
db_model  = joblib.load(os.path.join(MODELS_DIR, "dbscan.pkl"))
scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le        = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
metrics   = joblib.load(os.path.join(MODELS_DIR, "metrics.pkl"))

FEATURES = ["emoji_count","hashtag_count","exclaim_count",
            "word_count","char_count","likes","follower_bin"]

CLUSTER_PROFILES = {
    0: {"label":"High-Engagement Comments","color":"#00e5ff","desc":"Longer, more enthusiastic comments with higher likes"},
    1: {"label":"Low-Engagement Comments", "color":"#b06fff","desc":"Shorter, lower-engagement comments with fewer emojis"},
}

def extract_features(data):
    text = data.get("comment_text","")
    import re
    emoji_count   = len(re.findall(r'[\U0001F300-\U0001FFFF]', text))
    hashtag_count = text.count("#")
    exclaim_count = text.count("!")
    word_count    = len(text.split()) if text.strip() else 0
    char_count    = len(text)
    likes         = float(data.get("likes", 0))
    follower_bin  = int(data.get("follower_bin", 0))
    row = [emoji_count, hashtag_count, exclaim_count,
           word_count, char_count, likes, follower_bin]
    return np.array(row).reshape(1, -1), {
        "emoji_count":emoji_count,"hashtag_count":hashtag_count,
        "exclaim_count":exclaim_count,"word_count":word_count,
        "char_count":char_count,"likes":likes,"follower_bin":follower_bin
    }

@app.route("/")

def home():
    return send_file("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics)

@app.route("/api/predict/logistic-regression", methods=["POST"])
def predict_lr():
    data = request.get_json()
    X_raw, feats = extract_features(data)
    X = scaler.transform(X_raw)
    pred      = int(lr_model.predict(X)[0])
    proba     = lr_model.predict_proba(X)[0].tolist()
    sentiment = le.classes_[pred]
    conf      = round(float(max(proba)) * 100, 2)
    return jsonify({
        "model":"Logistic Regression","type":"supervised",
        "predicted_sentiment": sentiment,
        "confidence": conf,
        "class_probabilities": {le.classes_[i]: round(p*100,2) for i,p in enumerate(proba)},
        "extracted_features": feats,
    })

@app.route("/api/predict/kmeans", methods=["POST"])
def predict_km():
    data = request.get_json()
    X_raw, feats = extract_features(data)
    X       = scaler.transform(X_raw)
    cluster = int(km_model.predict(X)[0])
    dists   = np.linalg.norm(km_model.cluster_centers_ - X, axis=1).tolist()
    profile = CLUSTER_PROFILES.get(cluster, {"label":f"Cluster {cluster}","color":"#fff","desc":""})
    conf    = round((1 - dists[cluster] / max(dists)) * 100, 2)
    return jsonify({
        "model":"K-Means Clustering","type":"unsupervised",
        "cluster_id": cluster,
        "cluster_label": profile["label"],
        "cluster_color": profile["color"],
        "cluster_desc":  profile["desc"],
        "confidence": conf,
        "distances": [round(d, 4) for d in dists],
    })

@app.route("/api/predict/dbscan", methods=["POST"])
def predict_db():
    data = request.get_json()
    X_raw, feats = extract_features(data)
    X = scaler.transform(X_raw)
    # DBSCAN doesn't have predict() — use nearest core sample approach
    from sklearn.metrics import pairwise_distances_argmin
    if hasattr(db_model, 'core_sample_indices_') and len(db_model.core_sample_indices_) > 0:
        core_samples = db_model.components_
        nearest_idx  = pairwise_distances_argmin(X, core_samples)[0]
        nearest_label= db_model.labels_[db_model.core_sample_indices_[nearest_idx]]
        min_dist     = float(np.linalg.norm(X - core_samples[nearest_idx]))
        is_outlier   = bool(min_dist > 1.2)   # eps threshold
    else:
        nearest_label = -1
        min_dist      = 999.0
        is_outlier    = True

    risk = "HIGH" if is_outlier else "LOW"
    return jsonify({
        "model":"DBSCAN","type":"unsupervised",
        "is_outlier": is_outlier,
        "outlier_label": "OUTLIER / SPAM" if is_outlier else "NORMAL",
        "nearest_cluster": int(nearest_label),
        "distance_to_core": round(min_dist, 4),
        "risk_level": risk,
        "interpretation": (
            "Unusual comment pattern — possible spam or bot activity."
            if is_outlier else
            "Comment fits normal density cluster."
        ),
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_all():
    data = request.get_json()
    X_raw, feats = extract_features(data)
    X = scaler.transform(X_raw)

    # LR
    pred   = int(lr_model.predict(X)[0])
    proba  = lr_model.predict_proba(X)[0].tolist()
    lr_res = {
        "predicted_sentiment": le.classes_[pred],
        "confidence": round(float(max(proba))*100, 2),
        "class_probabilities": {le.classes_[i]: round(p*100,2) for i,p in enumerate(proba)},
    }
    # KM
    cluster = int(km_model.predict(X)[0])
    dists   = np.linalg.norm(km_model.cluster_centers_ - X, axis=1).tolist()
    profile = CLUSTER_PROFILES.get(cluster, {"label":f"Cluster {cluster}","color":"#fff","desc":""})
    km_res  = {
        "cluster_id": cluster,
        "cluster_label": profile["label"],
        "cluster_color": profile["color"],
        "confidence": round((1 - dists[cluster] / max(dists))*100, 2),
    }
    # DBSCAN
    if hasattr(db_model, 'core_sample_indices_') and len(db_model.core_sample_indices_) > 0:
        from sklearn.metrics import pairwise_distances_argmin
        core_samples = db_model.components_
        nearest_idx  = pairwise_distances_argmin(X, core_samples)[0]
        min_dist     = float(np.linalg.norm(X - core_samples[nearest_idx]))
        is_outlier   = bool(min_dist > 1.2)
    else:
        min_dist, is_outlier = 999.0, True

    db_res = {
        "is_outlier": is_outlier,
        "outlier_label": "OUTLIER" if is_outlier else "NORMAL",
        "distance_to_core": round(min_dist, 4),
        "risk_level": "HIGH" if is_outlier else "LOW",
    }
    return jsonify({
        "input": data, "extracted_features": feats,
        "logistic_regression": lr_res,
        "kmeans": km_res,
        "dbscan": db_res,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
