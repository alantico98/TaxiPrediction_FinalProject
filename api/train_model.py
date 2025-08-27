import os
import subprocess
import joblib
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
# Import Numpy and Pandas/Geopandas for transformations
import numpy as np
import pandas as pd
import geopandas as gpd
# Import Sci-kit Learn for Training
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
# from tqdm import tqdm
# Import MLFlow for experiment tracking and model registry
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


# --------------------------
# 1) Utils
# --------------------------
def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def last_full_month(today: date) -> date:
    # Get the first of the month, then subtract a day to land
    # in the previous month
    first_of_month = today.replace(day=1)
    return first_of_month - timedelta(days=1)


# def month_iter(start_ym: date, end_ym: date):
#     # Iterate first-of-month dates inclusive
#     ym = date(start_ym.year, start_ym.month, 1)
#     end_first = date(end_ym.year, end_ym.month, 1)
#     while ym <= end_first:
#         yield ym
#         ym = (ym.replace(day=28) + timedelta(days=4)).replace(day=1)


# --------------------------
# 2) Data download (6 months is default)
# --------------------------
def download_tlc_file(
        taxi_type: str = "yellow",
        months: int = 6
) -> tuple[pd.DataFrame, dict]:
    """
    Download a TLC dataset file for the specified taxi type, year, and month.
    inputs:
        taxi_type (str): Type of taxi (e.g., 'yellow', 'green').
        year (int): Year of the data.
        month (int): Month of the data (1-12).
        out_dir (str): Directory to save the downloaded file.

    returns:
        out (str): Path to the downloaded file.
    """
    # Build the yearlong taxi
    base = "https://d37ci6vzurychx.cloudfront.net/trip-data/"

    # Build the end year and month string
    end_date = last_full_month(date.today())
    # end_date = end_ym.strftime("%Y-%m-%d")

    # Build the start year and month string
    start_date = end_date - relativedelta(months=months - 1)
    # start_date = start_ym.strftime("%Y-%m-%d")

    # Store the data for model version control
    frames = []
    used_months = []
    for ym in pd.date_range(start=start_date, end=end_date, freq="MS"):
        if len(used_months) >= months:
            break

        # Build the url
        fname = f"yellow_tripdata_{ym.year}-{ym.month:02d}.parquet"
        url = base + fname
        print(f"Reading {ym.year}-{ym.month:02d} data...")
        df = pd.read_parquet(url)

        frames.append(df)
        used_months.append(f"{ym.year}-{ym.month:02d}")

    taxi_df = pd.concat(frames, ignore_index=True)
    meta = {
        "taxi_type": taxi_type,
        "months": used_months,
        "start_month": used_months[0],
        "end_year": used_months[-1],
        "n_rows": int(len(taxi_df)),
    }
    return taxi_df, meta


# ------------------
# 2) Feature engineering
# ------------------
def build_centroids_from_shp(shp_path: str) -> pd.DataFrame:
    # Read the shapefile nad retrieve the zone information
    gdf = gpd.read_file(shp_path)

    # Coordinates need to be in a projected CRS in order to compute centroids
    gdf_proj = gdf.to_crs(epsg=2263)  # Ensure it's in EPSG:2263
    centroids_proj = gdf_proj.geometry.representative_point()
    centroids_wgs = centroids_proj.to_crs(epsg=4326)  # Convert to lat/lon
    gdf["lon"] = centroids_wgs.x
    gdf["lat"] = centroids_wgs.y
    centroids_df = gdf[["LocationID", "zone", "lon", "lat"]].dropna()
    # Assert type
    centroids_df["LocationID"] = pd.to_numeric(
        centroids_df["LocationID"], errors="coerce").astype("Int64")

    # Write the file to a common location
    centroids_df.to_csv("./centroids.csv", )
    return centroids_df


# Manipulate Dataframes for training
def manipulate_taxi_info(df: pd.DataFrame, shp_path: str) -> pd.DataFrame:
    # Time features
    ts = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.assign(
        hour=ts.dt.hour,
        day_of_week=ts.dt.dayofweek,
        month=ts.dt.month,
        year=ts.dt.year
    )

    # Centroids
    centroids_df = None
    if os.path.exists("./centroids.csv"):
        centroids_df = pd.read_csv("./centroids.csv")
    else:
        centroids_df = build_centroids_from_shp(shp_path)
    # Rename centroids df for PU and DO locations
    pu_cent = centroids_df.rename(columns={
            "LocationID": "PULocationID",
            "zone": "PU_zone",
            "lon": "PU_lon",
            "lat": "PU_lat"
            })

    do_cent = centroids_df.rename(columns={
            "LocationID": "DOLocationID",
            "zone": "DO_zone",
            "lon": "DO_lon",
            "lat": "DO_lat"
            })

    # Drop features that aren't needed
    trips = df.drop(columns=[
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'store_and_fwd_flag',
        'payment_type',
        'extra',
        'mta_tax',
        'tip_amount',
        'tolls_amount',
        'improvement_surcharge',
        'total_amount',
        'congestion_surcharge',
        'Airport_fee',
        'passenger_count',
        'RatecodeID',
        'trip_distance'])

    # Fresh merges (many-to-one guarantees no dup keys in centroids)
    trips = trips.merge(
        pu_cent[["PULocationID", "PU_lon", "PU_lat", "PU_zone"]],
        on="PULocationID", how="left"
    ).merge(
        do_cent[["DOLocationID", "DO_lon", "DO_lat", "DO_zone"]],
        on="DOLocationID", how="left"
    )

    # Drop all trips that have an unknown pickup or dropoff location
    coord_cols = ["PU_lat", "PU_lon", "DO_lat", "DO_lon"]
    trips = trips.dropna(subset=coord_cols).copy()

    # Haversine distance (km) – Use Geod since this is a very large
    # dataframe. Alternative is to use a custom numpy function to compute
    # the distances manually
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(
        trips["PU_lon"].to_numpy(), trips["PU_lat"].to_numpy(),
        trips["DO_lon"].to_numpy(), trips["DO_lat"].to_numpy()
    )

    trips["haversine_km"] = dist_m / 1000.0  # convert from 'm' to 'km'

    # Assert the target is in the Dataframe
    if "fare_amount" not in trips.columns:
        raise ValueError("Expected 'fare_amount' in TLC dataset")

    # Reorder columns for clarity
    order = ["VendorID", "hour", "day_of_week", "month", "year",
             "PULocationID", "DOLocationID", "PU_lon", "PU_lat",
             "DO_lon", "DO_lat", "haversine_km", "fare_amount"]

    # Return the new dataframe
    return trips[order]


# ------------------
# 3) Model & CV
# ------------------
# Build the pipeline to process the data
def build_pipeline(**kw):
    """
    Build machine learning pipeline for taxi fare predictions.
    """
    cat_cols = ["PULocationID", "DOLocationID"]

    # Need to transform the categorical variable into numerical ones
    pre = ColumnTransformer(
        [("ohe", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int32),
            cat_cols)],
        remainder="passthrough",  # leaves num_cols as-is
        verbose_feature_names_out=False
    )

    # Build the pipeline
    model = Pipeline([
        ("pre", pre),
        ("hgb", HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=kw.get("learning_rate", 0.05),
            max_iter=kw.get("max_iter", 400),
            max_leaf_nodes=kw.get("max_leaf_nodes", 64),
            min_samples_leaf=kw.get("min_samples_leaf", 50),
            max_bins=kw.get("max_bins", 255),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
            ))
        ])
    return model


# Train and return the best performing model
def train_with_mlflow(
        trips: pd.DataFrame,
        experiment_name: str = "fare_predictions",
        register_name: str | None = "fare_predictor_hgbr",
        promote_to: str | None = "Staging"
):
    print("Training new model...")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="HGBR_grid_timeseries") as run:
        # Features/target
        num_cols = ["haversine_km", "hour", "day_of_week", "month", "year"]
        cat_cols = ["PULocationID", "DOLocationID"]
        # Shrink the size of the datatype to help avoid running
        # out of ram
        trips = trips.astype(
            {
                "PULocationID": "int32",
                "DOLocationID": "int32",
                "hour": "int16",
                "day_of_week": "int16",
                "month": "int16",
                "year": "int16",
                "haversine_km": "float32",
                "fare_amount": "float32",
            }
        )
        X = trips[num_cols + cat_cols]
        y = trips["fare_amount"]

        # CV + Grid
        tscv = TimeSeriesSplit(
            n_splits=6,
            max_train_size=600_000
        )  # adjust to your data span
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        param_grid = {
            "hgb__learning_rate": [0.03, 0.05, 0.08],
            "hgb__max_iter": [300, 500, 800],
            "hgb__max_leaf_nodes": [31, 64, 127],
            "hgb__min_samples_leaf": [20, 50, 100],
        }
        # Use RandomizedSearchCV to help avoid RAM issues
        grid = RandomizedSearchCV(
            estimator=build_pipeline(),
            param_distributions=param_grid,
            n_iter=5,  # small, fast
            scoring=mae_scorer,
            cv=tscv,
            # n_jobs=1,  # IMPORTANT NOTE: Avoids many processes duplicatingRAM
            verbose=10,
            refit=True,
            random_state=42
        )

        # Fit
        grid.fit(X, y)

        # ---- MLflow logging ----
        # Params & meta:
        # Writes key/value pairs to the curren run (inside mlflow.start_run())
        # Lives inside of Run Page -> Params tab
        mlflow.log_params({
            "algo": "HistGradientBoostingRegressor",
            "features_num": ",".join(num_cols),
            "features_cat": ",".join(cat_cols),
            "git_commit": git_commit(),
            "rows": len(trips),
        })
        # Log search space:
        # Saves GridSearch space (params) as a JSON artifact (grid.json)
        # Lives inside of Run Page -> Artifacts -> grid.json
        mlflow.log_dict(param_grid, "grid.json")

        # CV metrics for the best set:
        # Log each best hyperparameter under Params
        # A numeric metric mae_cv_mean under Metrics (shows as scalar and on
        # the timeline)
        best = grid.best_estimator_
        best_params = grid.best_params_
        best_mae = -grid.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("mae_cv_mean", best_mae)

        # Also log full cv_results_ as artifact:
        # Saves the full CV table (all param combos, mean/std per fold, ranks)
        # that can be downloaded later
        cv_df = pd.DataFrame(grid.cv_results_)
        cv_path = "cv_results.csv"
        mlflow.log_artifact(cv_path)

        # Fit on ALL data is already done (refit=True).
        # Log model with signature into Artifacts/model/ :
        # 1. MLmodel (metadata describing the flavors and entry points)
        # 2. model.pkl
        # 3. conda.yaml / requirements.txt (environment spec so it can be
        #                                   served/reproduced)
        example = X.head(5)
        sig = infer_signature(example, best.predict(example))
        mlflow.sklearn.log_model(
            sk_model=best,
            name="model",
            signature=sig,
            registered_model_name=register_name  # auto-creates new model ver
        )

        # Save a local copy too
        os.makedirs("artifacts", exist_ok=True)
        cv_df.to_csv("artifacts/" + cv_path)
        joblib.dump(best, "artifacts/fare_model_hgbr.joblib")
        mlflow.log_artifact("artifacts/fare_model_hgbr.joblib")

        run_id = run.info.run_id
        print(f"Best mean CV MAE: {best_mae:.3f}")
        print(f"Best params: {best_params}")

        # 1) Register this run's model as a new version
        mv = mlflow.register_model(f"runs:/{run_id}/model", name=register_name)

        # 1a) Attach tags to the model version for quick lookup (doesn't
        #     require clicking into the run)
        client = MlflowClient()
        client.set_model_version_tag(
            register_name,
            mv.version,
            "git_commit",
            git_commit()
        )
        client.set_model_version_tag(
            register_name,
            mv.version,
            "rows",
            str(len(trips))
        )
        client.set_model_version_tag(
            register_name,
            mv.version,
            "feature_list",
            ",".join(num_cols + cat_cols)
        )
        client.set_model_version_tag(
            register_name,
            mv.version,
            "mae_cv_mean",
            best_mae
        )

        # 2) Find the current model in Staging (if any) and get its metric
        def _current_stage_best(client, model_name, stage, metric_key):
            versions = client.search_model_versions(f"name='{model_name}'")
            # Filter to the given stage
            staged = [v for v in versions if v.current_stage == stage]
            if not staged:
                return None  # No model has been staged yet
            # Pick the one with the best metric (lower is better for MAE)
            best_val, best_mv = None, None
            for v in staged:
                r = client.get_run(v.run_id)
                # Check that the performance metric is in the run
                if metric_key in r.data.metrics:
                    val = r.data.metrics[metric_key]
                    # Check to see which is the best performing model
                    if best_val is None or val < best_val:
                        best_val, best_mv = val, v

            # Return the best performing model
            return (best_val, best_mv)

        # Retrieve the current best model
        METRIC_KEY = "mae_cv_mean"  # Metric that was logged
        LOWER_IS_BETTER = True  # MAE: lower is better
        EPS = 0.0   # Optional margin
        current_best = _current_stage_best(
            client,
            register_name,
            "Staging",
            METRIC_KEY
        )

        should_promote = False
        if current_best is None:
            # Nothing is in Staging -> promote a new one
            should_promote = True
        else:
            current_mae, current_mv = current_best
            if LOWER_IS_BETTER:
                should_promote = (best_mae < current_mae - EPS)
            else:
                should_promote = (best_mae > current_best + EPS)

        if should_promote:
            # 3) Promote the new model to Staging and archive previous
            print("Promoting new model to 'Staging'")
            client.set_registered_model_alias(
                register_name,
                "Staging",
                version=mv.version
            )
            # 4) Save the training data corresponding to current version
            trips.to_csv(f"./training_data_{mv.version}.csv")
            print(
                f"Registered model: {register_name} v{mv.version} -> "
                f"{promote_to or 'None'}"
            )
        else:
            print(
                f"Did not promote: new MAE {best_mae:.4f}"
                f"is not better than current Staging ({current_mae:.4f})."
            )

        return best


if __name__ == "__main__":
    # 1) Download the data
    raw_df, data_meta = download_tlc_file(taxi_type="yellow", months=6)

    # 2) Feature engineering
    shp_path = "./taxi_info/taxi_zones.shp"
    trips_df = manipulate_taxi_info(df=raw_df, shp_path=shp_path)

    # 3) Train + log to MLflow (+ register & stage)
    model = train_with_mlflow(
        trips=trips_df,
        experiment_name="fare_prediction",
        register_name="fare_predictor_hgbr",
        promote_to="Staging",  # or "Production" / None
    )

    # 4) Save for local use
    joblib.dump(model, "artifacts/fare_model_hgbr.pkl")
    print("Saved local model → artifacts/fare_model_hgbr.pkl")
