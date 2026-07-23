"""
StrideIQ — Dashboard
Streamlit app reading directly from SQLite via DatabaseManager.
Built to handle an empty/near-empty database gracefully — real
dashboards need to survive "no data yet", not just the happy path.
"""

import streamlit as st
import pandas as pd

from backend.database_manager import DatabaseManager

st.set_page_config(page_title="StrideIQ", layout="wide")

db = DatabaseManager()

st.title("StrideIQ — Running Form Dashboard")

# ------------------------------------------------------------
# Sidebar — run selection
# ------------------------------------------------------------
runs = db.get_all_runs()

if not runs: 
    st.info("No runs recorded yet. Once a session is logged, it'll show up here.")
    st.stop()

run_options = {f"Run {r['run_id']} — {r['date']} ({r['run_type']})": r["run_id"] for r in runs}
selected_label = st.sidebar.selectbox("Select a run", list(run_options.keys()))
selected_run_id = run_options[selected_label]

run_info = db.get_run(selected_run_id)

st.sidebar.markdown("---")
st.sidebar.write(f"**Type:** {run_info['run_type']}")
st.sidebar.write(f"**Duration:** {run_info['duration_sec']}s" if run_info["duration_sec"] else "**Duration:** in progress")
if run_info["notes"]:
    st.sidebar.write(f"**Notes:** {run_info['notes']}")

# ------------------------------------------------------------
# Packet loss / connection quality
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)
loss_l = db.get_packet_loss_rate(selected_run_id, "foot_left")
loss_r = db.get_packet_loss_rate(selected_run_id, "foot_right")
col1.metric("Left Foot Packet Loss", f"{loss_l:.1%}" if loss_l is not None else "N/A")
col2.metric("Right Foot Packet Loss", f"{loss_r:.1%}" if loss_r is not None else "N/A")

try:
    loss_w = db.get_packet_loss_rate(selected_run_id, "waist")
    col3.metric("Waist Packet Loss", f"{loss_w:.1%}" if loss_w is not None else "N/A")
except ValueError:
    col3.metric("Waist Packet Loss", "No waist data")

st.markdown("---")

# ------------------------------------------------------------
# Feature metrics over the run
# ------------------------------------------------------------
st.subheader("Metrics Over Time")

features = db.get_features(selected_run_id)

if not features:
    st.info("No feature data computed for this run yet — run feature extraction first.")
else:
    df = pd.DataFrame([dict(f) for f in features])

    metric_tabs = st.tabs(["Cadence", "Impact & Symmetry", "Pelvic (waist)"])

    with metric_tabs[0]:
        if df["cadence"].notna().any():
            st.line_chart(df.set_index("window_start")["cadence"])
        else:
            st.info("No cadence data yet.")

    with metric_tabs[1]:
        impact_cols = ["impact_left", "impact_right"]
        if df[impact_cols].notna().any().any():
            st.line_chart(df.set_index("window_start")[impact_cols])
        if df["symmetry_score"].notna().any():
            st.line_chart(df.set_index("window_start")["symmetry_score"])
        if not df[impact_cols].notna().any().any() and not df["symmetry_score"].notna().any():
            st.info("No impact/symmetry data yet.")

    with metric_tabs[2]:
        pelvic_cols = ["pelvic_tilt", "pelvic_rotation"]
        if df[pelvic_cols].notna().any().any():
            st.line_chart(df.set_index("window_start")[pelvic_cols])
        else:
            st.info("No waist data yet — waist node not integrated for this run.")

st.markdown("---")

# ------------------------------------------------------------
# Predictions timeline
# ------------------------------------------------------------
st.subheader("Form Predictions")

predictions = db.get_predictions(selected_run_id)

if not predictions:
    st.info("No predictions yet — model hasn't been run on this session.")
else:
    pred_df = pd.DataFrame([dict(p) for p in predictions])
    st.dataframe(pred_df[["window_start", "predicted_label", "confidence", "model_version"]])

st.markdown("---")

# ------------------------------------------------------------
# Recommendations / coaching
# ------------------------------------------------------------
st.subheader("Coaching")

recs = db.get_recommendations(selected_run_id)

if not recs:
    st.info("No recommendations generated for this run yet.")
else:
    for r in recs:
        with st.container(border=True):
            st.write(f"**{r['related_metric'] or 'General'}**")
            st.write(r["llm_explanation"] or r["recommendation_text"])