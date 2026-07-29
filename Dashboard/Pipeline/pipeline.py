# =========================
# PIPELINE: MISINFO ANALYSIS
# =========================

import numpy as np
import pandas as pd
import networkx as nx
import itertools
from collections import deque, Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================
# GOOGLE SHEETS LOAD
# =========================

import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    "credentials.json",
    scopes=SCOPES
)

client = gspread.authorize(creds)

# Your Google Sheet ID
SHEET_ID = "12CbXpwuDxVGeipzB_6U80kOqLTE-0Fae4j_wJFZYuPU"

sheet = client.open_by_key(SHEET_ID)

# Read Raw_Data tab
df = get_as_dataframe(
    sheet.worksheet("Raw_Data"),
    evaluate_formulas=True,
    dtype=str
)

# Remove completely empty rows
df = df.dropna(how="all")

# Remove rows where the key column is empty
df = df[df["misinfo_case_id"].notna()]
df = df[df["misinfo_case_id"] != ""]

print("Rows loaded:", len(df))
print(df.head())
print(df.columns.tolist())

# Make it behave exactly like pd.read_excel(..., converters=...)
df["post_id"] = df["post_id"].fillna("").astype(str)
df["parent_id"] = df["parent_id"].fillna("N/A").astype(str)
df["shares"] = pd.to_numeric(
    df["shares"],
    errors="coerce"
).fillna(0).astype(int)

df["timestamp"] = pd.to_datetime(
    df["timestamp"].astype(str).str.strip(),
    errors="coerce"
)

cases_platforms_candidate = (
    df[["misinfo_case_id", "platform", "candidate_category"]]
    .drop_duplicates()
)

print("Cases found:", len(cases_platforms_candidate))

# =================
# GRAPH FUNCTIONS
# =================

def build_graph(df_case):
    G = nx.DiGraph()
    for _, row in df_case.iterrows():
        post_id = str(row["post_id"])
        parent_id = row["parent_id"]

        G.add_node(post_id)

        if parent_id != "N/A":
            G.add_edge(str(parent_id), post_id)
    return G


def get_main_post(df_case):
    main_posts = df_case.loc[
        df_case["parent_id"] == "N/A",
        "post_id"
    ].values
    if len(main_posts) > 0:
        return str(main_posts[0])
    all_posts = set(df_case["post_id"].astype(str))
    all_parents = set(df_case["parent_id"].astype(str)) - {"N/A"}
    roots = list(all_posts - all_parents)
    if roots:
        return roots[0]

    return str(df_case["post_id"].values[0])


def compute_depths(G, main_post):
    depths = {main_post: 0}
    queue = deque([main_post])
    while queue:
        node = queue.popleft()
        for neighbor in G.successors(node):
            if neighbor not in depths:
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
    return depths

# =========================
# 1. AMPLIFICATION METRICS 
# =========================

metric_results = []

for case_id, platform, candidate_cat in cases_platforms_candidate.values:

    df_case = df[
        (df["misinfo_case_id"] == case_id) &
        (df["platform"] == platform) &
        (df["candidate_category"] == candidate_cat)
    ].copy()

    # --------------------
    # Build graph
    # --------------------
    G = build_graph(df_case)

    # --------------------
    # Identify main post
    # --------------------
    main_post = get_main_post(df_case)

    # --------------------
    # Compute depths
    # --------------------
    depths = compute_depths(G, main_post)

    # --------------------
    # SIZE
    # --------------------
    size = G.number_of_nodes()

    # --------------------
    # REACH
    # --------------------
    start_time = df_case["timestamp"].min()
    end_time = df_case["timestamp"].max()

    reach_timedelta = (
        end_time - start_time
        if pd.notna(start_time) and pd.notna(end_time)
        else None
    )

    # --------------------
    # DEPTH
    # --------------------
    depth = max(depths.values()) if depths else 0

    # --------------------
    # WIDTH
    # --------------------
    depth_counts = Counter(depths.values())

    width = max(depth_counts.values()) if depth_counts else 0

    # --------------------
    # SPEED
    # --------------------
    t0 = df_case.loc[
        df_case["post_id"].astype(str) == main_post,
        "timestamp"
    ].min()

    speed_timedelta = None

    if 1 in depth_counts:

        depth1_nodes = [
            n for n, d in depths.items()
            if d == 1
        ]

        t1 = df_case[
            df_case["post_id"].astype(str).isin(depth1_nodes)
        ]["timestamp"].min()

        if pd.notna(t1) and pd.notna(t0):
            speed_timedelta = t1 - t0

    # --------------------
    # Convert timedeltas
    # --------------------
    reach = (
        reach_timedelta.total_seconds()
        if reach_timedelta is not None
        else 0
    )

    speed = (
        speed_timedelta.total_seconds()
        if speed_timedelta is not None
        else 0
    )

    # --------------------
    # Store results
    # --------------------
    metric_results.append({
        "misinfo_case_id": case_id,
        "platform": platform,
        "candidate_category": candidate_cat,
        "size": size,
        "reach": reach,
        "depth": depth,
        "width": width,
        "speed": speed
    })


metrics_df = pd.DataFrame(metric_results).fillna(0)

# =========================
# 2. CROSS PLATFORM 
# =========================

cross = df[df["case_type"] == "cross-platform"]

cross_results = []

for case_id in cross["misinfo_case_id"].unique():

    c = cross[cross["misinfo_case_id"] == case_id]

    fb = c[c["platform"].str.lower() == "fb"]
    x = c[c["platform"].str.lower() == "x"]

    if fb.empty or x.empty:
        continue

    fb_first = fb["timestamp"].min() < x["timestamp"].min()

    if fb_first:
        delta = (x["timestamp"].min() - fb["timestamp"].min()).total_seconds() / 3600
        direction = "FB → X"
    else:
        delta = (fb["timestamp"].min() - x["timestamp"].min()).total_seconds() / 3600
        direction = "X → FB"

    cross_results.append({
        "misinfo_case_id": case_id,
        "direction": direction,
        "time_diff_hours": delta
    })

cross_df = pd.DataFrame(cross_results)

# =========================
# 3. CLUSTERING 
# =========================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import itertools

# -------------------------
# KEEP ORIGINAL FEATURE SET ORDER
# -------------------------
labels_df = metrics_df[['misinfo_case_id', 'platform', 'candidate_category']].reset_index(drop=True)

numeric_df = metrics_df.select_dtypes(include=['number']).reset_index(drop=True)

# -------------------------
# STANDARDIZATION (MATCH NOTEBOOK)
# -------------------------
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(numeric_df)

X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=numeric_df.columns
)

# attach labels (IMPORTANT: same structure as notebook)
X_scaled_with_labels = pd.concat([labels_df, X_scaled], axis=1)

# -------------------------
# WARDS K=3
# -------------------------

k = 3

ward_model = AgglomerativeClustering(
    n_clusters=k,
    linkage='ward'
)

X_ward = X_scaled_with_labels[["size","reach","depth","width","speed"]].values
X_scaled_with_labels['wards_cluster'] = ward_model.fit_predict(X_ward)

# -------------------------
# WARDS K=5
# -------------------------

k_explo = 5

ward_model_explo = AgglomerativeClustering(
    n_clusters=k_explo,
    linkage='ward'
)

X_ward = X_scaled_with_labels[["size","reach","depth","width","speed"]].values
X_scaled_with_labels['wards_cluster (k=5)'] = ward_model_explo.fit_predict(X_ward)

# -------------------------
# PCA VISUALIZATION
# -------------------------

pca = PCA(n_components=2)
X_pca_vis = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca_vis, columns=["PC1", "PC2"])

# -------------------------
# OUTPUT DATAFRAME 
# -------------------------

cluster_df = pd.concat([
    X_scaled_with_labels.reset_index(drop=True),
    pca_df.reset_index(drop=True)
], axis=1)


# =========================
# 4. EXPORT TO SHEETS
# =========================

print("Metrics rows:", len(metrics_df))
print("Cross rows:", len(cross_df))
print("Cluster rows:", len(cluster_df))

def write_sheet(df_out, tab_name):
    try:
        ws = sheet.worksheet(tab_name)
        ws.clear()
    except:
        ws = sheet.add_worksheet(title=tab_name, rows="2000", cols="50")

    print(f"Writing to tab: {tab_name} | shape: {df_out.shape}")

    set_with_dataframe(ws, df_out)

    print(f"Done writing: {tab_name}")

write_sheet(metrics_df, "Metrics")
write_sheet(cross_df, "Cross_Platform")
write_sheet(cluster_df, "Clusters")

print("Written Metrics:", len(metrics_df))
print("Written Cross:", len(cross_df))
print("Written Clusters:", len(cluster_df))
