import os
import requests
import pandas as pd
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATASET_CSV = "taxi_zone_lookup.csv"
TRAINING_DATA = "train_data.csv"


def main(df: pd.DataFrame | None = None):
    # --- Title ---
    st.title("NYC Yellow Taxi Fare Estimator")

    # --- Description ---
    st.write(
        """
        This app monitors a sentiment analysis model by visualizing
        performance, data drift, and feedback accuracy.
        """
    )

    # --- Create Selection Options for Taxi Pickup and Dropoff ---
    @st.cache_data
    def load_taxi_zones(path: str = DATASET_CSV):
        # Read the shapefile and retrieve the zone information
        taxi_zone_df = pd.read_csv(path)
        return taxi_zone_df

    # Load the taxi zones info
    taxi_zones_df = load_taxi_zones()

    # Select a pickup date
    date = st.date_input("Select a pickup date")

    # Select a pickup time
    time = st.time_input("Select a pickup time")

    dt_stamp = dt.datetime.combine(date, time)

    # Create two columns
    col1, col2 = st.columns(2)

    # Selectboxes with column names as options
    with col1:
        pu_loc = st.selectbox(
            "Choose a Pickup Location",
            taxi_zones_df['Zone'],
            key="pickup_location"
        )
    with col2:
        do_loc = st.selectbox(
            "Choose a Pickup Location",
            taxi_zones_df['Zone'],
            key="dropoff_location"
        )

    # Display the selected columns
    st.write(
        "You selected: ", pu_loc, " and", do_loc,
        " for pickup at", dt_stamp
    )

    # Predict the fare
    r = None  # Have the response be accessible to Feedback
    prediction_id = None
    if st.button("Estimate Fare"):
        try:
            r = requests.post(
                f"{API_URL}/predict",
                json={
                    "PULocationID": pu_loc,
                    "DOLocationID": do_loc,
                    "timestamp": dt_stamp
                    }, timeout=10
                )
            if r.status_code != 200:
                st.error(f"Prediction failed: HTTP {r.status_code} - {r.text}")
            else:
                st.success("Prediction complete")
                body = r.json()
                prediction_id = body["id"]
                st.json(body)
        except Exception as e:
            st.exception(e)

    # Provide Feedback for the model
    st.divider()
    st.subheader("Leave Feedback")
    if prediction_id is None:
        st.info("Make a prediction first, then submit feedback.")
    else:
        colf1, colf2 = st.columns([2, 1])
        # Provide to actual fare amount
        with colf1:
            correct_value = st.number_input(
                "Actual fare (exclude tips/tolls/taxes)",
                min_value=0.0, step=0.01
            )
            comment = st.text_area("Comment (optional)")
        with colf2:
            approved = st.radio(
                "Was the prediction reasonable?", options=[True, False]
            )

        # Any additional comments you'd like to provide?
        comment = st.text_area(
            "Any additional comments you'd like to leave?",
            placeholder="Type your feedback here..."
        )

        if st.button("Submit Feedback"):
            try:
                f = requests.post(
                    f"{API_URL}/feedback",
                    json={
                        "correct_value": correct_value,
                        "approved": approved,
                        "comment": comment if comment else "",
                        "id": r.json()["id"]
                    }
                )
                if f.status_code != 200:
                    st.error(
                        f"Feedback failed: HTTP {f.status_code} - {f.text}"
                        )
                else:
                    st.success("Thanks! Feedback recorded.")
            except Exception as e:
                st.exception(e)

    # ------------------- Monitor Performance -----------------------
    st.divider()
    st.subheader("Taxi Model Monitoring")

    # If table isn't specified at main, load the table
    # from AWS RDS
    if df is None:
        # --- Retrieve database credentials ---
        DB_URL = os.getenv("DB_URL")
        if not DB_URL:
            DB_HOST = os.getenv("DB_HOST")
            DB_PORT = os.getenv("DB_PORT", 5432)
            DB_NAME = os.getenv("DB_NAME")
            DB_USER = os.getenv("DB_USER")
            DB_PASSWORD = os.getenv("DB_PASSWORD")
            AWS_REGION = os.getenv("AWS_REGION")
            # Use SQLAlchemy's psycopg driver
            if all(
                [DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, AWS_REGION]
            ):
                DB_URL = (
                    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:"
                    f"{DB_PORT}/{DB_NAME}"
                )
            else:
                st.error("DB_URL or DB_* environment variables are not set.")
                st.stop()

        engine = create_engine(DB_URL, pool_pre_ping=True)

        now_utc = pd.Timestamp.utcnow().floor("s")

        preset = st.sidebar.selectbox(
            "Time range",
            ["Last 1 hour",
                "Last 6 hours",
                "Last 24 hours",
                "Last 7 days",
                "Custom"],
            index=2,
        )
        if preset == "Custom":
            c1, c2 = st.columns(2)

            start = c1.datetime_input(
                "Start (UTC)", value=now_utc - pd.Timedelta(days=1)
            )

            end = c2.datetime_input("End (UTC)", value=now_utc)
        else:
            delta = {
                "Last 1 hour": pd.Timedelta(hours=1),
                "Last 6 hours": pd.Timedelta(hours=6),
                "Last 24 hours": pd.Timedelta(hours=24),
                "Last 7 days": pd.Timedelta(days=7),
            }[preset]
            start, end = now_utc - delta, now_utc

        st.sidebar.write(f"Window: {start} → {end}")
        do_refresh = st.sidebar.button("Refresh")

        # --- SQLAlchemy engine + schema bootstrap ---
        DDL_STMTS = [
            # (optional) put everything in its own schema
            "CREATE SCHEMA IF NOT EXISTS taxi",

            # predictions table
            """
            CREATE TABLE IF NOT EXISTS taxi.predictions (
                id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                pu_location_id INT NOT NULL,
                do_location_id INT NOT NULL,
                hour SMALLINT,
                dow SMALLINT,
                month SMALLINT,
                year SMALLINT,
                haversine_km DOUBLE PRECISION NOT NULL,
                model_name TEXT,
                model_version TEXT,
                model_stage TEXT,
                latency_ms INT,
                cache_hit BOOLEAN DEFAULT FALSE,
                prediction NUMERIC(10,2) NOT NULL
            )
            """,

            # helpful indexes
            (
                "CREATE INDEX IF NOT EXISTS idx_predictions_time "
                "ON taxi.predictions (created_at)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_predictions_keys "
                "ON taxi.predictions "
                "(pu_location_id, do_location_id, hour, dow, month, year)"
            ),
            # feedback table (linked to predictions)
            """
            CREATE TABLE IF NOT EXISTS taxi.feedback (
                prediction_id UUID REFERENCES taxi.predictions(id)
                    ON DELETE CASCADE,
                correct_value NUMERIC(10,2),
                approved BOOLEAN,
                comment TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            (
                "CREATE INDEX IF NOT EXISTS idx_feedback_time "
                "ON taxi.feedback (created_at)"
            )
        ]

        def ensure_schema(engine):
            # Run each DDL separately
            with engine.begin() as conn:
                # Optional: set search_path so your existing INSERT into
                # "predictions" works without schema prefix
                conn.execute(text("SET search_path TO taxi, public"))
                for stmt in DDL_STMTS:
                    conn.execute(text(stmt))

        @st.cache_data(ttl=60, show_spinner=False)
        def load_window(_engine, start_ts=None, end_ts=None):
            # Portable query: don't use dialect-specific "AT TIME ZONE"
            q = text("""
                SELECT
                p.id, p.created_at,
                p.pu_location_id, p.do_location_id,
                p.hour, p.dow, p.month, p.year,
                p.haversine_km,
                p.model_name, p.model_version, p.model_stage,
                p.latency_ms, p.cache_hit, p.prediction,
                f.correct_value, f.approved, f.created_at AS feedback_time
                FROM taxi.predictions p
                LEFT JOIN taxi.feedback f ON f.prediction_id = p.id
                WHERE p.created_at >= :start AND p.created_at < :end
                ORDER BY p.created_at
            """)
            try:
                df = pd.read_sql(
                    q, _engine, params={"start": start_ts, "end": end_ts}
                )
                if df.empty:
                    return df
                # Normalize timestamps
                for col in ["created_at", "feedback_time"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(
                            df[col], utc=True, errors="coerce"
                        )
                # Coerce types that might be returned as Decimal/str
                num_cols = [
                    "latency_ms", "prediction", "correct_value", "haversine_km"
                    ]
                for c in num_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                if "approved" in df.columns:
                    # approved may be boolean, int (0/1) or None
                    df["approved"] = (
                        df["approved"].astype("float").round().astype("Int64")
                    )
                return df
            except Exception as e:
                st.exception(e)
                return None

        if do_refresh:
            load_window.clear()  # bust cache

        df = load_window(engine)

        # If load_window still returns nothing
        if df is None:
            st.info("No predictions found for the selected window.")
            st.stop()

    # ---- KPIs ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", f"{len(df):,}")
    c2.metric(
        "Median Latency (ms)",
        f"{int(df['latency_ms'].median()):,}" if "latency_ms" in df else "—"
        )

    # Feedback-derived metrics
    with_feedback = df.dropna(
        subset=["correct_value"]) if "correct_value" in df else pd.DataFrame()
    if not with_feedback.empty:
        mae = mean_absolute_error(
            with_feedback["correct_value"], with_feedback["prediction"]
        )
        mape = mean_absolute_percentage_error(
            with_feedback["correct_value"], with_feedback["prediction"]
        )
        c3.metric("MAE", f"{mae:,.2f}")
        c4.metric("MAPE", f"{mape*100:,.1f}%")
    else:
        c3.metric("MAE", "—")
        c4.metric("MAPE", "—")

    st.divider()

    # ---- Latency over time ----
    st.subheader("Latency Over Time")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=df, x="created_at",
        y="latency_ms",
        ax=ax1,
        s=12,
        alpha=0.5
    )
    if len(df) > 10:
        df_rolling = (
            df.set_index("created_at")
            .sort_index()["latency_ms"]
            .rolling("15min")
            .mean()
            .reset_index()
        )
        sns.lineplot(data=df_rolling, x="created_at", y="latency_ms", ax=ax1)

    ax1.set_xlabel("Timestamp (UTC)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-request latency (dots) & 15-min rolling avg (line)")
    st.pyplot(fig1)

    # ---- Prediction distribution / drift ----
    st.subheader("Prediction Distribution")
    colh1, colh2 = st.columns([2, 1])
    with colh1:
        fig2, ax2 = plt.subplots()
        sns.histplot(df["prediction"].dropna(), bins=40, ax=ax2, kde=True)
        ax2.set_xlabel("Predicted fare")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of predicted fares in window")
        st.pyplot(fig2)
    with colh2:
        # Optionally bucket predictions to emulate “class distribution” drift
        bins = st.slider(
            "Buckets", min_value=5, max_value=50, value=10, step=1
        )
        binned = pd.cut(df["prediction"], bins=bins)
        counts = binned.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        st.bar_chart(counts)

    # ---- Feedback acceptance & accuracy over time ----
    st.subheader("Feedback & Accuracy")
    if not with_feedback.empty:
        # Acceptance (approved) rate over time
        grp = with_feedback.set_index("created_at").sort_index()
        acc = grp["approved"].astype("float").rolling("1h").mean() * 100.0
        mae_series = (
            grp["prediction"] - grp["correct_value"]
        ).abs().rolling("1h").mean()

        fig3, ax3 = plt.subplots()
        ax3.plot(acc.index, acc.values)
        ax3.set_title("Hourly rolling approval rate")
        ax3.set_ylabel("Approval (%)")
        ax3.set_xlabel("Timestamp (UTC)")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots()
        ax4.plot(mae_series.index, mae_series.values)
        ax4.set_title("Hourly rolling MAE")
        ax4.set_ylabel("MAE")
        ax4.set_xlabel("Timestamp (UTC)")
        st.pyplot(fig4)
    else:
        st.info("No feedback yet in the selected window.")

    # Button to force rerun
    if st.button("Refresh Dashboard"):
        st.rerun()


if __name__ == "__main__":
    main()
