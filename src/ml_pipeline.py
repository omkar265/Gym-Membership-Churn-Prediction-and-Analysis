import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "models"

DATASET_PATH = DATA_DIR / "gym_churn_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
PIPELINE_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "model_summary.json"


def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset():
    rng = np.random.default_rng(42)
    n = 1200
    ages = rng.integers(18, 65, size=n)
    membership_duration = rng.integers(1, 36, size=n)
    attendance_rate = rng.uniform(30, 100, size=n)
    workout_frequency = rng.integers(1, 7, size=n)
    satisfaction = rng.integers(1, 11, size=n)
    bmi = rng.uniform(18, 35, size=n)
    account_balance = rng.uniform(0, 2000, size=n)
    training_goals = rng.choice(["fat_loss", "muscle_gain", "general_fitness", "strength"], size=n)
    contract_type = rng.choice(["monthly", "quarterly", "annual"], size=n)
    engagement_score = rng.uniform(0, 100, size=n)
    recent_visits = rng.integers(0, 30, size=n)

    churn_prob = (
        0.28
        + 0.014 * ((ages - 35) / 10)
        + 0.02 * (membership_duration > 18)
        + 0.01 * (attendance_rate < 60)
        + 0.02 * (workout_frequency < 2)
        + 0.015 * (satisfaction < 5)
        + 0.03 * (account_balance < 300)
        + 0.02 * (engagement_score < 40)
        + 0.015 * (recent_visits < 5)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    churn_flag = rng.random(n) < churn_prob

    df = pd.DataFrame(
        {
            "age": ages,
            "membership_duration_months": membership_duration,
            "attendance_rate_percent": attendance_rate.round(2),
            "workout_frequency_per_week": workout_frequency,
            "satisfaction_score": satisfaction,
            "bmi": bmi.round(2),
            "account_balance_usd": account_balance.round(2),
            "training_goal": training_goals,
            "contract_type": contract_type,
            "engagement_score": engagement_score.round(2),
            "recent_visits_last_month": recent_visits,
            "churn": churn_flag.astype(int),
        }
    )
    return df


def load_or_create_dataset():
    ensure_directories()
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    df = generate_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    return df


def get_feature_columns(df):
    return [col for col in df.columns if col != "churn"]


def build_model_pipeline(model_name):
    numeric_features = [
        "age",
        "membership_duration_months",
        "attendance_rate_percent",
        "workout_frequency_per_week",
        "satisfaction_score",
        "bmi",
        "account_balance_usd",
        "engagement_score",
        "recent_visits_last_month",
    ]
    categorical_features = ["training_goal", "contract_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=42, max_depth=6)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def produce_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Churn", "Churn"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    image_path = DATA_DIR / "confusion_matrix.png"
    plt.savefig(image_path, dpi=150)
    plt.close(fig)
    return image_path


def train_and_evaluate_models():
    ensure_directories()
    df = load_or_create_dataset()
    X = df[get_feature_columns(df)]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    selection_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name in ["logistic_regression", "decision_tree", "random_forest"]:
        pipeline = build_model_pipeline(name)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=selection_cv,
            scoring={"f1_churn": "f1", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["No Churn", "Churn"],
            output_dict=True, zero_division=0,
        )

        model_result = {
            "model": name,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "cross_validation": {
                "f1_churn_mean": float(cv_scores["test_f1_churn"].mean()),
                "f1_churn_std": float(cv_scores["test_f1_churn"].std()),
                "roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
                "roc_auc_std": float(cv_scores["test_roc_auc"].std()),
            },
            "roc_curve": (
                [
                    {"fpr": float(false_positive), "tpr": float(true_positive)}
                    for false_positive, true_positive in zip(*roc_curve(y_test, y_proba, pos_label=1)[:2])
                ]
                if y_test.nunique() == 2
                else []
            ),
        }
        results.append(model_result)

    selected = max(
        results,
        key=lambda r: (
            r["cross_validation"]["f1_churn_mean"],
            r["cross_validation"]["roc_auc_mean"],
        ),
    )

    best_name = selected["model"]
    best_pipeline = build_model_pipeline(best_name)
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, MODEL_PATH)

    # Save a separate preprocessing pipeline for transparency and recovery.
    joblib.dump(best_pipeline.named_steps["preprocessor"], PIPELINE_PATH)

    summary = {
        "dataset_path": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "target": "churn",
        "positive_class": 1,
        "selected_model": best_name,
        "model_name": type(best_pipeline.named_steps["model"]).__name__,
        "hyperparameters": best_pipeline.named_steps["model"].get_params(deep=False),
        "training_size": int(len(X_train)),
        "testing_size": int(len(X_test)),
        "train_test_split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "selection_method": "Highest mean Churn-class F1 across five-fold cross-validation on the training split; ROC-AUC breaks ties.",
        "test_class_distribution": {
            "no_churn": int((y_test == 0).sum()),
            "churn": int((y_test == 1).sum()),
        },
        "preprocessing": {
            "numeric": "Median imputation followed by standard scaling",
            "categorical": "Most-frequent imputation followed by one-hot encoding",
        },
        "results": results,
        "feature_columns": get_feature_columns(df),
        "class_distribution": {
            "no_churn": int((y == 0).sum()),
            "churn": int((y == 1).sum()),
        },
    }

    transformed_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    fitted_estimator = best_pipeline.named_steps["model"]
    if hasattr(fitted_estimator, "feature_importances_"):
        summary["feature_importance"] = sorted(
            [
                {"feature": str(name), "importance": float(value)}
                for name, value in zip(transformed_feature_names, fitted_estimator.feature_importances_)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:12]
    else:
        summary["feature_importance"] = []

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    selected_evaluation_model = build_model_pipeline(best_name)
    selected_evaluation_model.fit(X_train, y_train)
    produce_confusion_matrix(y_test, selected_evaluation_model.predict(X_test))

    return summary


def load_model_summary():
    if not SUMMARY_PATH.exists():
        return None
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model has been created yet.")
    return joblib.load(MODEL_PATH)


def predict_churn(record):
    model = load_saved_model()
    expected_features = list(model.feature_names_in_)
    missing = [feature for feature in expected_features if feature not in record]
    extra = [feature for feature in record if feature not in expected_features]
    if missing or extra:
        raise ValueError(
            f"Input schema mismatch. Missing fields: {missing or 'none'}; "
            f"unexpected fields: {extra or 'none'}."
        )

    df = pd.DataFrame([{feature: record[feature] for feature in expected_features}])
    prediction = model.predict(df)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(model.predict_proba(df)[0][classes.index(1)])
    return int(prediction), probability


def get_dashboard_data():
    df = load_or_create_dataset()
    summary = load_model_summary()
    churn_rate = df["churn"].mean() * 100
    predicted_churn_count = None
    if MODEL_PATH.exists():
        model = load_saved_model()
        features = df[list(model.feature_names_in_)]
        predicted_churn_count = int((model.predict(features) == 1).sum())

    data = {
        "total_records": int(len(df)),
        "actual_churn_rate": round(churn_rate, 2),
        "actual_churn_count": int(df["churn"].sum()),
        "retained_count": int((df["churn"] == 0).sum()),
        "predicted_high_risk_count": predicted_churn_count,
        "model_ready": MODEL_PATH.exists(),
        "selected_model": summary.get("selected_model") if summary else None,
    }
    return data


FEATURE_DESCRIPTIONS = {
    "age": "Member age, in years.",
    "membership_duration_months": "Length of membership, in months.",
    "attendance_rate_percent": "Attendance rate recorded as a percentage.",
    "workout_frequency_per_week": "Number of workouts per week.",
    "satisfaction_score": "Member satisfaction score.",
    "bmi": "Body mass index.",
    "account_balance_usd": "Account balance in US dollars.",
    "training_goal": "Member's selected training goal.",
    "contract_type": "Membership contract category.",
    "engagement_score": "Member engagement score.",
    "recent_visits_last_month": "Number of visits recorded in the previous month.",
    "churn": "Target label: 1 means churn; 0 means retained.",
}

FEATURE_LABELS = {
    "age": "Age",
    "membership_duration_months": "Membership duration (months)",
    "attendance_rate_percent": "Attendance rate (%)",
    "workout_frequency_per_week": "Workouts per week",
    "satisfaction_score": "Satisfaction score",
    "bmi": "BMI",
    "account_balance_usd": "Account balance (USD)",
    "training_goal": "Training goal",
    "contract_type": "Contract type",
    "engagement_score": "Engagement score",
    "recent_visits_last_month": "Visits in the last month",
}


def get_prediction_fields():
    df = load_or_create_dataset()
    model = load_saved_model() if MODEL_PATH.exists() else None
    columns = list(model.feature_names_in_) if model is not None else get_feature_columns(df)
    fields = []
    for column in columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            median = float(series.median())
            is_integer = pd.api.types.is_integer_dtype(series)
            fields.append({
                "name": column,
                "label": FEATURE_LABELS.get(column, column.replace("_", " ").title()),
                "kind": "number",
                "integer": is_integer,
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "default": int(round(median)) if is_integer else round(median, 3),
            })
        else:
            choices = sorted(series.dropna().astype(str).unique().tolist())
            default = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else ""
            fields.append({
                "name": column,
                "label": FEATURE_LABELS.get(column, column.replace("_", " ").title()),
                "kind": "category",
                "choices": choices,
                "default": default,
            })
    return fields


def analyze_dataset():
    df = load_or_create_dataset()
    target = "churn" if "churn" in df.columns else None
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    columns = []
    for column in df.columns:
        series = df[column]
        columns.append({
            "name": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "description": FEATURE_DESCRIPTIONS.get(column, "Dataset feature."),
        })

    numeric_statistics = []
    for column in numeric_columns:
        series = df[column]
        numeric_statistics.append({
            "name": column,
            "count": int(series.count()),
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
            "min": float(series.min()) if series.notna().any() else None,
            "median": float(series.median()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        })

    categorical_distributions = {}
    for column in categorical_columns:
        categorical_distributions[column] = [
            {"value": str(value), "count": int(count)}
            for value, count in df[column].value_counts(dropna=False).head(10).items()
        ]

    target_distribution = {}
    if target:
        target_distribution = {
            str(value): int(count) for value, count in df[target].value_counts(dropna=False).items()
        }

    numerical_churn_means = []
    if target and target in numeric_columns:
        for column in numeric_columns:
            if column == target:
                continue
            grouped = df.groupby(target, dropna=False)[column].mean()
            numerical_churn_means.append({
                "name": column,
                "retained_mean": float(grouped.get(0)) if pd.notna(grouped.get(0)) else None,
                "churn_mean": float(grouped.get(1)) if pd.notna(grouped.get(1)) else None,
            })

    correlations = []
    if target and target in numeric_columns:
        matrix = df[numeric_columns].corr(numeric_only=True)
        correlations = [
            {"name": column, "correlation": float(matrix.loc[column, target])}
            for column in numeric_columns if column != target and pd.notna(matrix.loc[column, target])
        ]
        correlations.sort(key=lambda item: abs(item["correlation"]), reverse=True)

    categorical_churn = {}
    if target:
        for column in categorical_columns:
            if column == target:
                continue
            cross_tab = pd.crosstab(df[column].fillna("Missing"), df[target], normalize="index")
            categorical_churn[column] = [
                {
                    "value": str(index),
                    "churn_rate": float(row.get(1, 0.0)),
                    "records": int((df[column].fillna("Missing") == index).sum()),
                }
                for index, row in cross_tab.iterrows()
            ]

    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "duplicate_count": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "columns": columns,
        "numeric_statistics": numeric_statistics,
        "categorical_distributions": categorical_distributions,
        "target": target,
        "target_distribution": target_distribution,
        "numerical_churn_means": numerical_churn_means,
        "correlations": correlations,
        "categorical_churn": categorical_churn,
    }
