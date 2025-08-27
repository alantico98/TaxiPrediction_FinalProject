# app/main.py
import os
import time
import uuid
import pandas as pd
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Import MLflow
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
# Create the DB engine
from sqlalchemy import create_engine, text

# --- Config ---
# NOTE: Change "Staging" to "Production" when deploying model to Production
#       phase
CENTROIDS_CSV = os.getenv(
    "CENTROIDS_CSV",
    "./centroids.csv"
)
cent = pd.read_csv(CENTROIDS_CSV).set_index("LocationID")[
    ["zone", "lon", "lat"]
]

# --- Resolve registry info & load model ---
client = MlflowClient()


def resolve_model_meta(model_uri: str):
    if not model_uri.startswith("models:/"):
        # The URI is not a Registry URI. If not, it is some other kind of URI
        m = {
            "name": "local",
            "stage": "n/a",
            "version": None,
            "run_id": None,
            "loaded_uri": model_uri
        }
        # Return the model directly from the URI
        return mlflow.pyfunc.load_model(model_uri), m
    # Schema is "models:/<name>/<Production/Staging/etc.>"
    _, rest = model_uri.split("models:/", 1)
    # name: registered model name (e.g. fare_predictor_hgbr)
    # ver_or_stage: either numeric ("3") or a stage label (e.g. "Staging")
    name, alias = rest.split("/", 1)

    try:
        # If it's a version number, retrieve the exact version of the model
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        # If Production, Stage, Archived, etc., grab the latest version
        mv = client.get_latest_versions(name, stages=[alias])[0]
        # NOTE: This might be overkill, since if you know what stage
        #       you're deploying the model to, then you should only need
        #       mv = client.get_latest_version(name, stages=["Stage"]),
        #       but this setup does keep it dynamic

    # URI that points to the artifact that was registered under that run
    model = None
    try:
        loaded_uri = f"runs:/{mv.run_id}/model"
        model = mlflow.pyfunc.load_model(loaded_uri)
    except MlflowException:
        loaded_uri = f"models:/{name}/{mv.version}"
        model = mlflow.pyfunc.load_model(loaded_uri)

    # Return the model's meta data
    return model, {
        "name": name,
        "stage": alias,
        "version": int(mv.version),
        "run_id": mv.run_id,
        "loaded_uri": loaded_uri
    }


# ------------------------------------------------------


# --- Retrieve database credentials ---
def build_db_url():
    url = os.getenv("DB_URL")
    if url:
        return url
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", 5432)
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    aws_region = os.getenv("AWS_REGION")
    if all([host, port, name, user, password, aws_region]):
        return (
            f"postgresql+psycopg://{user}:{password}@{host}:"
            f"{port}/{name}"
        )
    else:
        raise ValueError("Environment variables for DB connection missing")


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
        prediction_id UUID REFERENCES taxi.predictions(id) ON DELETE CASCADE,
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
    # Run each DDL separately (cleaner than one multi-statement execute)
    with engine.begin() as conn:
        # Optional: set search_path so your existing INSERT into "predictions"
        # works without schema prefix
        conn.execute(text("SET search_path TO taxi, public"))
        for stmt in DDL_STMTS:
            conn.execute(text(stmt))
# ------------------------------------------------------


# --- Request/Response schemas ---
class PredictRequest(BaseModel):
    PULocationID: int
    DOLocationID: int
    timestamp: str = Field(description="ISO8601, server will parse")


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: int
    stage: str
    latency_ms: int
    cache_hit: bool = False
    id: str


class FeedbackRequest(BaseModel):
    id: str
    correct_value: float
    approved: bool
    comment: str

# ------------------------------------------------------


# --- Helpers ---
def haversine_km(PU_lon, PU_lat, DO_lon, DO_lat):
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(
        PU_lon, PU_lat,
        DO_lon, DO_lat
    )

    dist_m = dist_m / 1000.0
    return dist_m


def make_features(req: PredictRequest) -> pd.DataFrame:
    dt = pd.to_datetime(req.timestamp)
    try:
        pu = cent.loc[req.PULocationID]
        do = cent.loc[req.DOLocationID]
    except KeyError:
        raise HTTPException(400, "Unknown LocationID; not in centroid table")

    # vectorized haversine for a single pair
    hav_km = haversine_km(
        pu.lon, pu.lat,  # Retrieve values from Pickup Location Row
        do.lon, do.lat   # Retrieve values from Dropoff Location Row
    )

    return pd.DataFrame([{
        "haversine_km": float(hav_km),
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "year": dt.year,
        "PULocationID": req.PULocationID, "DOLocationID": req.DOLocationID,
    }])
# ------------------------------------------------------


model = None
model_meta = None
db_url = None
engine = None


# --- Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    app.state.db_url = build_db_url()
    engine = create_engine(app.state.db_url, pool_pre_ping=True)
    app.state.engine = engine
    ensure_schema(engine)

    # Load the model
    model_uri = os.getenv("MODEL_URI", "models:/fare_predictor_hgbr/Staging")
    model, model_meta = resolve_model_meta(model_uri)
    app.state.model = model
    app.state.model_meta = model_meta
    app.state.model_uri = model_uri

    try:
        yield
    finally:
        # --- Shutdown ---
        engine.dispose()

app = FastAPI(title="Fare API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    db_url = getattr(app.state, "db_url", None)
    driver = db_url.split("://", 1)[0] if db_url else None
    model_uri = getattr(app.state, "model_uri", None)
    resolved = getattr(app.state, "model_meta", None)
    return {
        "status": "ok",
        "model_uri": model_uri,
        "resolved": resolved,
        "db_url_driver": driver
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Make sure the model is loaded
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Make the prediction
    t0 = time.time()
    X = make_features(req)
    yhat = float(model.predict(X)[0])
    latency = int((time.time() - t0) * 1000)
    pred_id = str(uuid.uuid4())
    engine = app.state.engine
    model_meta = getattr(app.state, "model_meta", None)

    cache_hit = False
    if engine:
        # Write to the table storing each prediction
        with engine.begin() as conn:
            timestamp = datetime.now(timezone.utc).isoformat()
            # Extract the scalar from the single row (suppresses
            # pandas "single element Series" warning)
            row = X.iloc[0]
            hour = int(row["hour"])
            dow = int(row["day_of_week"])
            month = int(row["month"])
            year = int(row["year"])
            km = float(row["haversine_km"])
            conn.execute(text("""
                INSERT INTO predictions
                (id, created_at, pu_location_id, do_location_id, hour, dow,
                 month, year, haversine_km, model_name, model_version,
                 model_stage, latency_ms, cache_hit, prediction)
                VALUES (:id, :now, :pu, :do, :hour, :dow, :month, :year,
                        :km, :mname, :mver, :mstage, :lat, :cache, :pred)
            """), dict(
                id=pred_id, now=timestamp,
                pu=req.PULocationID, do=req.DOLocationID,
                hour=hour, dow=dow, month=month,
                year=year,
                km=km,
                mname=model_meta["name"],
                mver=model_meta["version"],
                mstage=model_meta["stage"],
                lat=latency, cache=False, pred=round(yhat, 2),
            ))
    return PredictResponse(
        prediction=round(yhat, 2),
        model_name=model_meta["name"],
        model_version=model_meta["version"],
        stage=model_meta["stage"],
        latency_ms=latency,
        cache_hit=cache_hit,
        id=pred_id
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    engine = app.state.engine
    if engine:
        with engine.begin() as conn:
            timestamp = datetime.now(timezone.utc).isoformat()
            conn.execute(text("""
                INSERT INTO feedback
                (prediction_id, correct_value, approved, comment,
                 created_at)
                VALUES (:prediction_id, :correct_value, :approved,
                        :comment, :now)
            """), dict(
                prediction_id=req.id, correct_value=req.correct_value,
                approved=req.approved, comment=req.comment, now=timestamp
            ))
