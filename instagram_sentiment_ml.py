# =============================================================================
# INSTAGRAM SENTIMENT ANALYSIS — Machine Learning Approach
# Dataset  : instagram_comments.csv (8,000 Instagram comments)
# Model 1  : Logistic Regression        (Supervised)
# Model 2  : K-Means Clustering         (Unsupervised)
# Model 3  : DBSCAN                     (Unsupervised — Outlier Detection)
# Student  : Pranjal Agarwal | Roll No: 2305952
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, silhouette_score)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  INSTAGRAM SENTIMENT ANALYSIS — ML MODELS COMPARISON")
print("=" * 65)

df = pd.read_csv("instagram_comments.csv")
print(f"\n[+] Total comments loaded : {len(df):,}")
print(f"[+] Columns               : {df.columns.tolist()}")
print(f"\n[+] Sentiment Distribution:\n{df['sentiment'].value_counts().to_string()}")
print(f"\n[+] Category Distribution:\n{df['category'].value_counts().to_string()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

print("-" * 65)
print("  FEATURE ENGINEERING")
print("-" * 65)

FEATURES = [
    "emoji_count",     # number of emojis in comment
    "hashtag_count",   # number of hashtags used
    "exclaim_count",   # number of exclamation marks
    "word_count",      # total words in comment
    "char_count",      # total characters in comment
    "likes",           # likes received on the comment
    "follower_bin",    # poster's follower range (0=<1K, 1=1K-10K, 2=>10K)
]

X  = df[FEATURES].values
le = LabelEncoder()
y  = le.fit_transform(df["sentiment"])   # Negative=0, Neutral=1, Positive=2

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"[+] Features used       : {FEATURES}")
print(f"[+] Classes             : {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"[+] Feature matrix      : {X.shape}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION (Supervised)
# Task: Classify comment sentiment — Positive / Negative / Neutral
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  MODEL 1: LOGISTIC REGRESSION  [Supervised]")
print("  Task: Predict sentiment class from comment features")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(f"\n[+] Training samples    : {len(X_train):,}")
print(f"[+] Testing  samples    : {len(X_test):,}")
print(f"\n[+] ACCURACY            : {lr_accuracy * 100:.2f}%")
print(f"\n[+] Classification Report:\n")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

coefs = dict(zip(FEATURES, np.abs(lr_model.coef_).mean(axis=0)))
coefs_sorted = sorted(coefs.items(), key=lambda x: x[1], reverse=True)
print("[+] Feature Coefficients (most influential → least):")
for feat, val in coefs_sorted:
    print(f"    {feat:<22} : {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — K-MEANS CLUSTERING (Unsupervised)
# Task: Discover natural comment groups without using sentiment labels
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  MODEL 2: K-MEANS CLUSTERING  [Unsupervised]")
print("  Task: Group comments by behavioral similarity")
print("=" * 65)

k_range    = range(2, 9)
inertias   = []
sil_scores = []

print("\n[+] Computing Elbow & Silhouette scores...\n")
for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42)
    sil_scores.append(sil)
    print(f"    k={k}  |  Inertia={km.inertia_:,.0f}  |  Silhouette={sil:.4f}")

best_k   = k_range.start + sil_scores.index(max(sil_scores))
km_model = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
km_labels = km_model.fit_predict(X_scaled)
df["km_cluster"] = km_labels

print(f"\n[+] Optimal k           : {best_k}  (highest silhouette score)")
print(f"[+] Silhouette Score    : {max(sil_scores):.4f}")
print(f"[+] Inertia (WCSS)      : {km_model.inertia_:,.2f}\n")

print("[+] Cluster Profiles:")
for c in range(best_k):
    sub = df[df["km_cluster"] == c]
    top = sub["sentiment"].value_counts().index[0]
    print(f"    Cluster {c} | {len(sub):>5,} comments | "
          f"AvgLikes={sub['likes'].mean():.1f} | "
          f"AvgEmoji={sub['emoji_count'].mean():.2f} | "
          f"AvgExclaim={sub['exclaim_count'].mean():.2f} | "
          f"Dominant={top}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — DBSCAN (Unsupervised — Outlier/Spam Detection)
# Task: Detect comments that don't fit any cluster (spam, bots, edge cases)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  MODEL 3: DBSCAN  [Unsupervised — Outlier Detection]")
print("  Task: Detect spam/bot/outlier comments")
print("=" * 65)

db_model  = DBSCAN(eps=1.2, min_samples=8, n_jobs=-1)
db_labels = db_model.fit_predict(X_scaled)
df["dbscan_label"] = db_labels

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = int((db_labels == -1).sum())
noise_pct     = n_noise / len(df) * 100
noise_df      = df[df["dbscan_label"] == -1]

print(f"\n[+] DBSCAN Clusters Found  : {n_clusters_db}")
print(f"[+] Normal Comments        : {len(df) - n_noise:,}  ({100 - noise_pct:.2f}%)")
print(f"[+] Outlier Comments       : {n_noise:,}  ({noise_pct:.2f}%)")
print(f"\n[+] Outlier Distribution by Sentiment:")
print(noise_df["sentiment"].value_counts().to_string())
print(f"\n[+] Outlier Avg Likes      : {noise_df['likes'].mean():.2f}")
print(f"[+] Outlier Avg Emoji      : {noise_df['emoji_count'].mean():.2f}")
print(f"\n[+] Top Outlier Comments:")
print(noise_df[["comment_text","sentiment","likes","emoji_count"]].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MODEL COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  MODEL COMPARISON SUMMARY")
print("=" * 65)
print(f"\n  {'Model':<25} {'Type':<15} {'Key Metric':<25} {'Result'}")
print(f"  {'-'*72}")
print(f"  {'Logistic Regression':<25} {'Supervised':<15} {'Accuracy':<25} {lr_accuracy*100:.2f}%")
print(f"  {'K-Means Clustering':<25} {'Unsupervised':<15} {'Silhouette Score':<25} {max(sil_scores):.4f}")
print(f"  {'DBSCAN':<25} {'Unsupervised':<15} {'Outliers Detected':<25} {n_noise:,} ({noise_pct:.2f}%)")

print(f"\n  Conclusion:")
print(f"  • Logistic Regression achieved {lr_accuracy*100:.1f}% accuracy classifying sentiment")
print(f"  • K-Means found {best_k} natural comment clusters (Silhouette={max(sil_scores):.3f})")
print(f"  • DBSCAN flagged {n_noise} outlier/spam-like comments across {n_clusters_db} clusters")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VISUALIZATIONS (saved as instagram_sentiment_results.png)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[+] Generating visualizations...")

COLORS = {
    "bg":"#0a0a0f", "panel":"#0f0f1a", "cyan":"#00e5ff",
    "green":"#00ff88", "amber":"#ffb800", "red":"#ff3b5c",
    "purple":"#b06fff", "text":"#c8dff0", "sub":"#4a7a9e",
}

fig = plt.figure(figsize=(22, 16), facecolor=COLORS["bg"])
fig.suptitle("Instagram Sentiment Analysis — ML Model Comparison",
             fontsize=16, fontweight="bold", color=COLORS["text"], y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1A: Confusion Matrix ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(COLORS["panel"])
cm = confusion_matrix(y_test, y_pred_lr)
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(le.classes_, rotation=30, color=COLORS["text"], fontsize=9)
ax.set_yticklabels(le.classes_, color=COLORS["text"], fontsize=9)
ax.set_title("Confusion Matrix\n(Logistic Regression)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("Predicted", color=COLORS["sub"]); ax.set_ylabel("Actual", color=COLORS["sub"])
plt.colorbar(im, ax=ax)

# ── 1B: Feature Coefficients ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(COLORS["panel"])
cs = sorted(coefs.items(), key=lambda x: x[1])
ax.barh([k for k, v in cs], [v for k, v in cs],
        color=plt.cm.plasma(np.linspace(0.2, 0.9, len(cs))), edgecolor="none")
ax.set_title("Feature Coefficients\n(Logistic Regression)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("Mean |Coefficient|", color=COLORS["sub"])
ax.tick_params(colors=COLORS["text"], labelsize=8); ax.spines[:].set_visible(False)

# ── 1C: F1 per Sentiment ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(COLORS["panel"])
lr_report = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True)
f1s = [lr_report[c]["f1-score"] * 100 for c in le.classes_]
cols = [COLORS["green"] if f >= 70 else COLORS["amber"] if f >= 50 else COLORS["red"] for f in f1s]
ax.bar(le.classes_, f1s, color=cols, edgecolor="none", width=0.5)
ax.set_ylim(0, 115)
for i, f in enumerate(f1s):
    ax.text(i, f + 2, f"{f:.1f}%", ha="center", color=COLORS["text"], fontsize=9, fontweight="bold")
ax.set_title("F1-Score per Sentiment\n(Logistic Regression)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_ylabel("F1 (%)", color=COLORS["sub"]); ax.tick_params(colors=COLORS["text"]); ax.spines[:].set_visible(False)

# ── 2A: Elbow Curve ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor(COLORS["panel"])
ax.plot(list(k_range), inertias, marker="o", color=COLORS["cyan"], linewidth=2, markersize=7)
ax.axvline(best_k, color=COLORS["green"], linestyle="--", linewidth=1.5, label=f"Best k={best_k}")
ax.set_title("Elbow Curve\n(K-Means)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("k", color=COLORS["sub"]); ax.set_ylabel("Inertia", color=COLORS["sub"])
ax.tick_params(colors=COLORS["text"]); ax.spines[:].set_visible(False)
ax.legend(fontsize=8, labelcolor=COLORS["text"], facecolor=COLORS["bg"], edgecolor="none")

# ── 2B: PCA Cluster Scatter ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(COLORS["panel"])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pal = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = km_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=pal[c], s=3, alpha=0.4, label=f"C{c}")
ax.set_title("K-Means Clusters (PCA 2D)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("PCA 1", color=COLORS["sub"]); ax.set_ylabel("PCA 2", color=COLORS["sub"])
ax.tick_params(colors=COLORS["text"], labelsize=7); ax.spines[:].set_visible(False)
ax.legend(fontsize=7, labelcolor=COLORS["text"], facecolor=COLORS["bg"], edgecolor="none", markerscale=4)

# ── 2C: Sentiment Mix per Cluster ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(COLORS["panel"])
cs2 = df.groupby(["km_cluster", "sentiment"]).size().unstack(fill_value=0)
cs2.plot(kind="bar", ax=ax, color=[COLORS["red"], COLORS["sub"], COLORS["green"]],
         edgecolor="none", width=0.7)
ax.set_title("Sentiment Mix per Cluster\n(K-Means)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("Cluster", color=COLORS["sub"]); ax.set_ylabel("Comments", color=COLORS["sub"])
ax.tick_params(axis="x", rotation=0, colors=COLORS["text"])
ax.tick_params(axis="y", colors=COLORS["text"]); ax.spines[:].set_visible(False)
ax.legend(fontsize=7, labelcolor=COLORS["text"], facecolor=COLORS["bg"], edgecolor="none")

# ── 3A: DBSCAN PCA Scatter ────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0]); ax.set_facecolor(COLORS["panel"])
nm = db_labels != -1; noisem = db_labels == -1
ax.scatter(X_pca[nm, 0], X_pca[nm, 1], color=COLORS["cyan"], s=2, alpha=0.25,
           label=f"Normal ({nm.sum():,})")
ax.scatter(X_pca[noisem, 0], X_pca[noisem, 1], color=COLORS["red"], s=10,
           alpha=0.8, marker="x", label=f"Outlier ({n_noise:,})")
ax.set_title("DBSCAN Outliers in PCA Space", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_xlabel("PCA 1", color=COLORS["sub"]); ax.set_ylabel("PCA 2", color=COLORS["sub"])
ax.tick_params(colors=COLORS["text"], labelsize=7); ax.spines[:].set_visible(False)
ax.legend(fontsize=8, labelcolor=COLORS["text"], facecolor=COLORS["bg"], edgecolor="none", markerscale=2)

# ── 3B: Outlier Sentiment Distribution ───────────────────────────────────
ax = fig.add_subplot(gs[2, 1]); ax.set_facecolor(COLORS["panel"])
ns = noise_df["sentiment"].value_counts()
bc = [COLORS["green"] if s == "Positive" else COLORS["red"] if s == "Negative" else COLORS["sub"]
      for s in ns.index]
ax.bar(ns.index, ns.values, color=bc, edgecolor="none", width=0.5)
for i, (s, v) in enumerate(zip(ns.index, ns.values)):
    ax.text(i, v + 2, str(v), ha="center", color=COLORS["text"], fontsize=9)
ax.set_title("Outlier Distribution by Sentiment\n(DBSCAN)", color=COLORS["text"], fontweight="bold", fontsize=10)
ax.set_ylabel("Count", color=COLORS["sub"]); ax.tick_params(colors=COLORS["text"]); ax.spines[:].set_visible(False)

# ── 3C: Model Comparison Table ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2]); ax.set_facecolor(COLORS["panel"]); ax.axis("off")
tdata = [
    ["Type",         "Supervised",             "Unsupervised",        "Unsupervised"],
    ["Task",         "Sentiment Classification","Comment Clustering",  "Outlier Detection"],
    ["Key Metric",   f"{lr_accuracy*100:.1f}% Accuracy",
                                               f"Sil={max(sil_scores):.3f}",
                                                                      f"{n_noise} outliers ({noise_pct:.1f}%)"],
    ["Labels Needed","Yes",                    "No",                  "No"],
    ["Best For",     "Known sentiments",       "Traffic grouping",    "Spam/bot detection"],
]
tbl = ax.table(cellText=tdata[1:], colLabels=tdata[0], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c_), cell in tbl.get_celld().items():
    cell.set_edgecolor(COLORS["bg"])
    if r == 0: cell.set_facecolor(COLORS["cyan"]); cell.set_text_props(color="black", fontweight="bold")
    elif c_ == 0: cell.set_facecolor("#1a1a2e"); cell.set_text_props(color=COLORS["text"])
    else: cell.set_facecolor(COLORS["panel"]); cell.set_text_props(color=COLORS["text"])
ax.set_title("Model Comparison Table", color=COLORS["text"], fontweight="bold", fontsize=10, pad=12)

plt.savefig("instagram_sentiment_results.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print("[+] Visualization saved  : instagram_sentiment_results.png\n")
plt.show()

print("=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
