"""
QueueLess — Streamlit UI
=========================
One app, two views:
  - Customer View   : "Go to Counter X, ~Y min wait"
  - Manager Dashboard: live counts, service-time predictions, alerts

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Data sources (pick one in the sidebar):
  1. Demo / manual  — move sliders to simulate queue lengths (no model needed
     to try the UI, but predictions need your trained RF model).
  2. Upload RF model (.pkl/.joblib) — the same file your Colab notebook
     produces with joblib.dump(rf_model, "rf_model.pkl").
  3. (Advanced, optional) Upload a video + let YOLO count people live. This
     needs `ultralytics` + `opencv-python` installed and is slow on a CPU-only
     machine — it's meant for a local demo, not a lightweight cloud deploy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="QueueLess", page_icon="🛒", layout="wide")

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None
if "queue_names" not in st.session_state:
    st.session_state.queue_names = ["Queue 1", "Queue 2", "Queue 3"]
if "queue_to_counter" not in st.session_state:
    st.session_state.queue_to_counter = {"Queue 1": 1, "Queue 2": 8, "Queue 3": 8}
if "queue_counts" not in st.session_state:
    st.session_state.queue_counts = {"Queue 1": 3, "Queue 2": 5, "Queue 3": 2}

# ----------------------------------------------------------------------
# Core logic (mirrors the notebook: CV count x RF service time = wait)
# ----------------------------------------------------------------------
def predict_service_times(rf_model, counter_ids, art_num, hour, day, is_weekend):
    """Run the trained RandomForest pipeline for each distinct counter id."""
    preds = {}
    for c in counter_ids:
        row = pd.DataFrame({
            "ArtNum": [art_num],
            "hour": [hour],
            "day_of_week": [day],
            "is_weekend": [is_weekend],
            "WorkstationGroupID": [c],
        })
        preds[c] = float(rf_model.predict(row)[0])
    return preds


def compute_wait_times(queue_counts, queue_to_counter, service_times):
    waits = {}
    for queue, counter_id in queue_to_counter.items():
        people = queue_counts.get(queue, 0)
        service_time = service_times.get(counter_id)
        if service_time is None:
            continue
        waits[queue] = people * service_time
    return waits


def known_counters(rf_model):
    """Introspect the pipeline to see which WorkstationGroupIDs it was trained on."""
    try:
        ohe = rf_model.named_steps["preprocessor"].named_transformers_["categorical"]
        cat_cols = rf_model.named_steps["preprocessor"].transformers_[0][2]
        cats = dict(zip(cat_cols, ohe.categories_))
        return list(cats.get("WorkstationGroupID", []))
    except Exception:
        return None


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 QueueLess")
    st.caption("Smart checkout counter recommendation")

    st.header("1. ML model")
    model_file = st.file_uploader("Upload trained RF model (.pkl / .joblib)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            st.session_state.rf_model = joblib.load(model_file)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Could not load model: {e}")

    rf_model = st.session_state.rf_model
    if rf_model is not None:
        counters_seen = known_counters(rf_model)
        if counters_seen:
            st.caption(f"Counters this model was trained on: {counters_seen}")

    st.header("2. Queues & counters")
    n_queues = st.number_input("Number of queues", min_value=1, max_value=8, value=len(st.session_state.queue_names))
    queue_names = [f"Queue {i+1}" for i in range(n_queues)]
    st.session_state.queue_names = queue_names

    st.caption("Map each queue to the checkout counter ID your model was trained on.")
    queue_to_counter = {}
    for q in queue_names:
        default = st.session_state.queue_to_counter.get(q, 1)
        queue_to_counter[q] = st.number_input(f"{q} → Counter ID", value=int(default), step=1, key=f"map_{q}")
    st.session_state.queue_to_counter = queue_to_counter

    st.header("3. Live queue counts")
    st.caption("From your CV/YOLO pipeline. Enter manually here for a demo, or wire in automatically (see bottom of page).")
    queue_counts = {}
    for q in queue_names:
        default = st.session_state.queue_counts.get(q, 3)
        queue_counts[q] = st.slider(f"{q}: people waiting", 0, 20, int(default), key=f"count_{q}")
    st.session_state.queue_counts = queue_counts

    st.header("4. Context")
    avg_art_num = st.number_input("Avg. items per basket", value=15.5, step=0.5)
    use_now = st.checkbox("Use current date/time", value=True)
    if use_now:
        now = datetime.now()
        hour, day, is_weekend = now.hour, now.strftime("%A"), now.weekday() >= 5
    else:
        hour = st.slider("Hour of day", 0, 23, 18)
        day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        is_weekend = day in ("Saturday", "Sunday")

    alert_threshold = st.slider("Manager alert threshold (people)", 1, 20, 6)

# ----------------------------------------------------------------------
# Run predictions
# ----------------------------------------------------------------------
wait_times, service_times, error = {}, {}, None

if rf_model is None:
    error = "Upload your trained RF model in the sidebar to get real predictions. Showing layout with placeholder numbers."
    # Placeholder so the UI still renders meaningfully before a model is loaded
    service_times = {c: 40.0 for c in set(queue_to_counter.values())}
else:
    try:
        service_times = predict_service_times(
            rf_model, set(queue_to_counter.values()), avg_art_num, hour, day, is_weekend
        )
    except Exception as e:
        error = f"Prediction failed: {e}"

wait_times = compute_wait_times(queue_counts, queue_to_counter, service_times)

recommended_queue = min(wait_times, key=wait_times.get) if wait_times else None

# ----------------------------------------------------------------------
# Main area — two views
# ----------------------------------------------------------------------
st.title("QueueLess")
if error:
    st.warning(error)

tab_customer, tab_manager = st.tabs(["🛍️ Customer View", "📊 Manager Dashboard"])

# ---------------- Customer View ----------------
with tab_customer:
    st.subheader("Where should I check out?")

    if recommended_queue:
        rec_wait_min = wait_times[recommended_queue] / 60
        st.markdown(
            f"""
            <div style="padding:2rem;border-radius:1rem;background:#16a34a;color:white;text-align:center;">
                <div style="font-size:1.1rem;opacity:0.9;">Recommended</div>
                <div style="font-size:3rem;font-weight:800;">{recommended_queue}</div>
                <div style="font-size:1.3rem;">≈ {rec_wait_min:.1f} min wait</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No queue data yet.")

    st.write("")
    cols = st.columns(len(queue_names))
    for col, q in zip(cols, queue_names):
        with col:
            is_best = q == recommended_queue
            wait_min = wait_times.get(q, 0) / 60
            border = "3px solid #16a34a" if is_best else "1px solid #ddd"
            st.markdown(
                f"""
                <div style="padding:1rem;border-radius:0.75rem;border:{border};text-align:center;">
                    <div style="font-size:1rem;color:#555;">{q}</div>
                    <div style="font-size:1.8rem;font-weight:700;">{queue_counts[q]} 🧍</div>
                    <div style="font-size:0.95rem;color:#333;">≈ {wait_min:.1f} min</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------- Manager Dashboard ----------------
with tab_manager:
    st.subheader("Live status")

    rows = []
    for q in queue_names:
        counter_id = queue_to_counter[q]
        rows.append({
            "Queue": q,
            "Counter ID": counter_id,
            "People waiting": queue_counts[q],
            "Predicted service time (s/person)": round(service_times.get(counter_id, float("nan")), 1),
            "Estimated wait (min)": round(wait_times.get(q, 0) / 60, 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("People waiting per queue")
        st.bar_chart(df.set_index("Queue")["People waiting"])
    with c2:
        st.caption("Estimated wait per queue (min)")
        st.bar_chart(df.set_index("Queue")["Estimated wait (min)"])

    busiest = df.loc[df["People waiting"].idxmax()] if not df.empty else None
    if busiest is not None and busiest["People waiting"] >= alert_threshold:
        st.error(
            f"⚠️ MANAGER ALERT: {busiest['Queue']} has {int(busiest['People waiting'])} "
            f"people — consider opening another counter."
        )
    else:
        st.success("No congestion alert.")

    if recommended_queue:
        st.info(f"System would currently route customers to **{recommended_queue}**.")

st.divider()
with st.expander("Advanced: wire in live YOLO counts from a video (optional, needs ultralytics + opencv)"):
    st.markdown(
        """
        This tab only takes queue counts as numbers — it doesn't care where they come from.
        To feed it live counts from your CV pipeline instead of the manual sliders:

        1. Run your notebook's YOLO counting cell (the one that outputs a `queue_counts` dict)
           on a schedule (e.g. every 30 seconds) in a small background script.
        2. Have that script write the latest counts somewhere this app can read — the simplest
           options are a small JSON/CSV file, or a lightweight database/cache (e.g. Redis).
        3. In this file, replace the manual `st.slider` calls in the sidebar with code that
           reads that file/DB instead, e.g.:

        ```python
        import json
        with open("latest_counts.json") as f:
            queue_counts = json.load(f)
        ```

        Streamlit re-runs the whole script on each interaction, so pair this with
        `st.autorefresh` (from the `streamlit-autorefresh` package) or a manual
        "Refresh" button so the manager view updates without a full page reload.
        """
    )
