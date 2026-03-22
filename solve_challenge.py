import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold


def infer_column_types(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.95):
    """Infer numeric columns by successful coercion ratio on non-null values."""
    numeric_cols = []
    categorical_cols = []

    for col in train_df.columns:
        if col == "child_id":
            categorical_cols.append(col)
            continue

        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        non_null = combined.notna()
        if non_null.sum() == 0:
            categorical_cols.append(col)
            continue

        coerced = pd.to_numeric(combined, errors="coerce")
        success_ratio = coerced[non_null].notna().mean()
        if success_ratio >= threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    numeric_cols, categorical_cols = infer_column_types(train_df, test_df)

    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    for col in categorical_cols:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

    return train_df, test_df, categorical_cols


def run_cv(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, cat_features: list[str], folds: int, seed: int):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_x), dtype=np.float64)
    test_preds = np.zeros(len(test_x), dtype=np.float64)
    fold_stats = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_x), start=1):
        x_tr = train_x.iloc[tr_idx]
        y_tr = train_y.iloc[tr_idx]
        x_va = train_x.iloc[va_idx]
        y_va = train_y.iloc[va_idx]

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=5000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=6.0,
            random_strength=0.2,
            bagging_temperature=0.5,
            min_data_in_leaf=20,
            grow_policy="SymmetricTree",
            bootstrap_type="Bayesian",
            random_seed=seed + fold,
            od_type="Iter",
            od_wait=300,
            verbose=False,
        )

        model.fit(
            x_tr,
            y_tr,
            eval_set=(x_va, y_va),
            cat_features=cat_features,
            use_best_model=True,
        )

        va_pred = model.predict(x_va)
        te_pred = model.predict(test_x)

        oof[va_idx] = va_pred
        test_preds += te_pred / folds

        fold_rmse = root_mean_squared_error(y_va, va_pred)
        fold_mae = mean_absolute_error(y_va, va_pred)
        best_iter = model.get_best_iteration()
        fold_stats.append((fold, fold_rmse, fold_mae, best_iter))
        print(f"Fold {fold}: RMSE={fold_rmse:.5f} MAE={fold_mae:.5f} best_iter={best_iter}")

    full_rmse = root_mean_squared_error(train_y, oof)
    full_mae = mean_absolute_error(train_y, oof)
    print("-" * 60)
    print(f"OOF RMSE: {full_rmse:.5f}")
    print(f"OOF MAE : {full_mae:.5f}")

    return oof, test_preds, fold_stats


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost model for AI4EAC practice challenge")
    parser.add_argument("--data-dir", type=str, default="the-ai4eac-education-practice-challenge")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="submission_catboost.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "Train.csv"
    test_path = data_dir / "Test.csv"
    sample_path = data_dir / "SampleSubmission.csv"

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    sample_df = pd.read_csv(sample_path)

    target_col = "target"
    id_col = "child_id"

    y = pd.to_numeric(train_df[target_col], errors="coerce")
    keep_rows = y.notna()
    dropped = int((~keep_rows).sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows with missing target")

    train_df = train_df.loc[keep_rows].reset_index(drop=True)
    y = y.loc[keep_rows].reset_index(drop=True)

    train_x = train_df.drop(columns=[target_col]).copy()
    test_x = test_df.copy()

    train_x, test_x, cat_features = prepare_features(train_x, test_x)

    print(f"Train rows: {len(train_x)} | Test rows: {len(test_x)}")
    print(f"Total features: {train_x.shape[1]}")
    print(f"Categorical features: {len(cat_features)}")

    _, test_preds, _ = run_cv(
        train_x=train_x,
        train_y=y,
        test_x=test_x,
        cat_features=cat_features,
        folds=args.folds,
        seed=args.seed,
    )

    submission = sample_df.copy()
    submission[id_col] = test_df[id_col].values
    submission[target_col] = test_preds

    feature_cols = [c for c in submission.columns if c not in [id_col, target_col]]
    for c in feature_cols:
        submission[c] = "feature"

    output_path = Path(args.output)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
