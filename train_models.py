"""
train_models.py — Run ONCE before starting app.py
Place instagram_comments.csv in the same folder first.
Usage: python train_models.py
"""
import pandas as pd, numpy as np, joblib, os, re, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, confusion_matrix

os.makedirs("models", exist_ok=True)
print("=" * 60)
print("  TRAINING — Instagram Sentiment Analysis")
print("=" * 60)

# ── Load ─────────────────────────────────────────────────────
df = pd.read_csv("instagram_comments.csv")
print(f"\n[+] Loaded {len(df):,} comments")
print(df["sentiment"].value_counts().to_string())

FEATURES = ["emoji_count","hashtag_count","exclaim_count",
            "word_count","char_count","likes","follower_bin"]
X  = df[FEATURES].values
le = LabelEncoder()
y  = le.fit_transform(df["sentiment"])
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(le,    "models/label_encoder.pkl")
print("[+] Scaler + LabelEncoder saved")

# ── Model 1: Logistic Regression ─────────────────────────────
print("\n[1] Training Logistic Regression...")
X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.25,random_state=42,stratify=y)
lr = LogisticRegression(max_iter=1000,random_state=42,C=1.0)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test,y_pred)
lr_rep = classification_report(y_test,y_pred,target_names=le.classes_,output_dict=True)
lr_cm  = confusion_matrix(y_test,y_pred).tolist()
coefs  = dict(zip(FEATURES,np.abs(lr.coef_).mean(axis=0).tolist()))
joblib.dump(lr,"models/logistic_regression.pkl")
print(f"    Accuracy: {lr_acc*100:.2f}%")
print(classification_report(y_test,y_pred,target_names=le.classes_))

# ── Model 2: K-Means ──────────────────────────────────────────
print("\n[2] Training K-Means...")
sil_scores,inertias=[],[]
for k in range(2,9):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil=silhouette_score(X_scaled,km.labels_,sample_size=3000,random_state=42)
    sil_scores.append(sil)
    print(f"    k={k}  Sil={sil:.4f}  Inertia={km.inertia_:,.0f}")
best_k=2+sil_scores.index(max(sil_scores))
km_final=KMeans(n_clusters=best_k,random_state=42,n_init=10)
km_labels=km_final.fit_predict(X_scaled)
df["km_cluster"]=km_labels
joblib.dump(km_final,"models/kmeans.pkl")
print(f"    Best k={best_k}, Silhouette={max(sil_scores):.4f}")

# ── Model 3: DBSCAN ───────────────────────────────────────────
print("\n[3] Training DBSCAN...")
db=DBSCAN(eps=1.2,min_samples=8,n_jobs=-1)
db_labels=db.fit_predict(X_scaled)
n_clusters=len(set(db_labels))-(1 if -1 in db_labels else 0)
n_noise=int((db_labels==-1).sum())
noise_df=df[df.index.isin(np.where(db_labels==-1)[0])]
joblib.dump(db,"models/dbscan.pkl")
print(f"    Clusters: {n_clusters}  Outliers: {n_noise} ({n_noise/len(df)*100:.2f}%)")

# ── Metrics ───────────────────────────────────────────────────
metrics={
    "logistic_regression":{
        "accuracy":round(lr_acc*100,2),
        "classes":le.classes_.tolist(),
        "per_class":{cls:{"precision":round(lr_rep[cls]["precision"]*100,1),
                          "recall":round(lr_rep[cls]["recall"]*100,1),
                          "f1":round(lr_rep[cls]["f1-score"]*100,1),
                          "support":int(lr_rep[cls]["support"])} for cls in le.classes_},
        "coefficients":{k:round(v,4) for k,v in coefs.items()},
        "confusion_matrix":lr_cm,
        "train_samples":int(len(X_train)),"test_samples":int(len(X_test)),
    },
    "kmeans":{
        "best_k":int(best_k),"silhouette":round(max(sil_scores),4),
        "inertia":round(km_final.inertia_,2),
        "k_scores":[{"k":k+2,"sil":round(s,4),"inertia":round(inertias[k],2)} for k,s in enumerate(sil_scores)],
        "cluster_profiles":{str(c):{"size":int(len(df[df["km_cluster"]==c])),
            "avg_likes":round(float(df[df["km_cluster"]==c]["likes"].mean()),1),
            "avg_emoji":round(float(df[df["km_cluster"]==c]["emoji_count"].mean()),2),
            "dominant":df[df["km_cluster"]==c]["sentiment"].value_counts().index[0]} for c in range(best_k)},
    },
    "dbscan":{
        "n_clusters":int(n_clusters),"n_outliers":int(n_noise),
        "outlier_pct":round(n_noise/len(df)*100,2),
        "outliers_by_sentiment":{k:int(v) for k,v in noise_df["sentiment"].value_counts().items()},
        "total":int(len(df)),"eps":1.2,"min_samples":8,
    },
    "dataset":{"total":8000,"sentiment_dist":{k:int(v) for k,v in df["sentiment"].value_counts().items()},
               "categories":df["category"].nunique()},
}
joblib.dump(metrics,"models/metrics.pkl")
print("\n[+] All models saved to ./models/")
print("    Run: python app.py")
print("=" * 60)
