from streamlit.testing.v1 import AppTest


def fake_app():
    """Wrap app.main with fake dataframe so AppTest can run it cleanly."""
    import app
    import pandas as pd
    fake_table = pd.DataFrame({
        "id": [1],
        "created_at": pd.to_datetime(["2025-08-26 12:00:00"]),
        "pu_location_id": [100],
        "do_location_id": [200],
        "hour": [12],
        "dow": [2],
        "month": [8],
        "year": [2025],
        "haversine_km": [1.2],
        "model_name": ["modelA"],
        "model_version": ["v1"],
        "model_stage": ["prod"],
        "latency_ms": [120],
        "cache_hit": [True],
        "prediction": [10.5],
        "correct_value": [11.0],
        "approved": [True],
        "feedback_time": pd.to_datetime(["2025-08-26 12:05:00"]),
    })
    app.main(df=fake_table)


def test_dashboard_launch():
    """Ensure Streamlit dashboard loads and renders KPIs with fake df."""
    at = AppTest.from_function(fake_app)

    # Run the app
    at.run(timeout=15)

    # No exceptions should occur
    assert not list(at.exception), f"App raised exceptions: {at.exception}"

    # Metrics should exist
    assert len(at.metric) > 0
    assert at.metric[0].value == "1"  # Predictions count
