import pytest
from fastapi.testclient import TestClient
import uuid
from sqlalchemy import create_engine, text
import pandas as pd
import main as api


# --- Test-only schema creator for SQLite ---
def ensure_sqlite_schema(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON")
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                pu_location_id INTEGER NOT NULL,
                do_location_id INTEGER NOT NULL,
                hour INTEGER,
                dow INTEGER,
                month INTEGER,
                year INTEGER,
                haversine_km REAL NOT NULL,
                model_name TEXT,
                model_version TEXT,
                model_stage TEXT,
                latency_ms INTEGER,
                cache_hit INTEGER DEFAULT 0,  -- 0/1 for False/True
                prediction REAL NOT NULL
            )
        """)
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS feedback (
                prediction_id TEXT REFERENCES predictions(id)
                    ON DELETE CASCADE,
                correct_value REAL,
                approved INTEGER,             -- 0/1 for False/True
                comment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "api.db"


@pytest.fixture
def engine(db_file):
    eng = create_engine(f"sqlite+pysqlite:///{db_file}", future=True)
    ensure_sqlite_schema(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def client(monkeypatch, db_file):
    # Patch DB URL
    monkeypatch.setattr(
        api, "build_db_url", lambda: f"sqlite+pysqlite:///{db_file}"
    )
    monkeypatch.setattr(api, "ensure_schema", ensure_sqlite_schema)

    # Patch centroids
    centroids = pd.DataFrame({
        "LocationID": [1, 2],
        "lon": [-73.0, -72.0],
        "lat": [40.0, 41.0],
        "zone": ["A", "B"],
        "borough": ["X", "Y"],
    }).set_index("LocationID")[["lon", "lat", "zone", "borough"]]
    monkeypatch.setattr(api, "cent", centroids)

    # --- Patch MLflow model loading ---
    class DummyModel:
        def predict(self, X):
            assert list(X.columns) == [
                "haversine_km",
                "hour",
                "day_of_week",
                "month",
                "year",
                "PULocationID",
                "DOLocationID",
            ]
            return [12.34]

    dummy_meta = {
        "name": "fare_predictor_hgbr",
        "version": 3,
        "stage": "Staging",
        "run_id": "test",
        "loaded_uri": "runs:/test/model",
    }

    monkeypatch.setattr(
        api, "resolve_model_meta", lambda uri: (DummyModel(), dummy_meta)
    )

    # Start the app
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def client_no_model(monkeypatch, db_file):
    monkeypatch.setattr(
        api,
        "build_db_url", lambda: f"sqlite+pysqlite:///{db_file}"
    )
    monkeypatch.setattr(api, "ensure_schema", ensure_sqlite_schema)
    # Simulate a model-load failure
    monkeypatch.setattr(api, "resolve_model_meta", lambda uri: (None, None))
    with TestClient(api.app) as client:
        yield client


# --- Test Health Check ---
def test_health_check(client):
    client.app.state.model_uri = "models:/fare_predictor_hgbr/Staging"
    client.app.state.model_meta = {
        "name": "fare_predictor_hgbr",
        "version": 3,
        "stage": "Staging",
        "run_id": "x",
        "loaded_uri": "runs:/x/model",
    }
    client.app.state.db_url = "sqlite+pysqlite:///:memory:"

    r = client.get("/health")
    assert r.status_code == 200


# --- Test Predictions Insert/Select ---
def test_predictions_insert_select(engine):
    pid = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predictions
            (id, created_at, pu_location_id, do_location_id, hour, dow, month,
             year, haversine_km, model_name, model_version, model_stage,
             latency_ms, cache_hit, prediction)
            VALUES (:id, CURRENT_TIMESTAMP, 100, 200, 9, 1, 8, 2025,
                    3.25, 'fare_predictor_hgbr', '3', 'Staging',
                    11, 0, 24.55)
        """), {"id": pid})

        got = conn.execute(
            text("SELECT prediction FROM predictions WHERE id=:id"),
            {"id": pid}
        ).scalar_one()

    assert float(got) == 24.55


# --- Test /predict Endpoint ---
def test_prediction_insert_row(client, engine, monkeypatch):
    monkeypatch.setattr(api.app.state, "engine", engine)

    r = client.post("/predict", json={
        "PULocationID": 1,
        "DOLocationID": 2,
        "timestamp": "2025-08-25T12:00:00Z",
    })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] == 12.34
    assert body["model_name"] == "fare_predictor_hgbr"
    assert body["model_version"] == 3
    assert body["stage"] == "Staging"
    assert isinstance(body["latency_ms"], int)
    assert isinstance(body["cache_hit"], bool)
    assert isinstance(body["id"], str)

    with engine.connect() as c:
        cnt = c.execute(
            text("SELECT COUNT(*) FROM predictions WHERE id = :id"),
            {"id": r.json()["id"]},
        ).scalar_one()
    assert cnt == 1


# --- Test /feedback Endpoint ---
def test_feedback_insert_row(client, engine, monkeypatch):
    monkeypatch.setattr(api.app.state, "engine", engine)

    # First insert prediction
    r = client.post("/predict", json={
        "PULocationID": 1,
        "DOLocationID": 2,
        "timestamp": "2025-08-25T12:00:00Z",
    })
    body = r.json()

    feedback = {
        "id": body["id"],
        "correct_value": 21.11,
        "approved": True,
        "comment": "Looks good",
    }
    f = client.post("/feedback", json=feedback)
    assert f.status_code == 200

    with engine.connect() as c:
        cnt = c.execute(
            text("SELECT COUNT(*) FROM feedback WHERE prediction_id=:id"),
            {"id": body["id"]}
        ).scalar_one()
    assert cnt == 1


# --- Test Model Not Loaded ---
def test_predict_model_not_loaded(client_no_model):
    payload = {
        "PULocationID": 1,
        "DOLocationID": 2,
        "timestamp": "2025-08-25T12:00:00Z"
    }
    r = client_no_model.post("/predict", json=payload)
    assert r.status_code == 503
    assert r.json()["detail"] == "Model not loaded"
