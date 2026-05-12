import re
import joblib
import pandas as pd

# Lazy load model + scaler + columns
_model = None
_scaler = None
_feature_columns = None


def load_artifacts():
    """Load saved model, scaler, and feature column order."""
    global _model, _scaler, _feature_columns
    if _model is None:
        _model = joblib.load("models/xgboost_heart.pkl")
        _scaler = joblib.load("models/standard_scaler.pkl")
        _feature_columns = joblib.load("models/feature_columns.pkl")


# Columns that were scaled during training
SCALE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]


# 13 features to collect — same as training data
FEATURES = [
    {"key": "age",      "type": "number",
     "question": "What is your age?"},

    {"key": "sex",      "type": "sex",
     "question": "What is your sex? Please say male or female."},

    {"key": "cp",       "type": "category",
     "options": {1: "typical angina", 2: "atypical angina",
                 3: "non-anginal",    4: "asymptomatic"},
     "question": "What type of chest pain do you have? Say 1 for typical angina, "
                 "2 for atypical angina, 3 for non-anginal, or 4 for asymptomatic."},

    {"key": "trestbps", "type": "number",
     "question": "What is your resting blood pressure in mm Hg?"},

    {"key": "chol",     "type": "number",
     "question": "What is your cholesterol level in mg per dl?"},

    {"key": "fbs",      "type": "yes_no",
     "question": "Is your fasting blood sugar above 120 mg per dl? Say yes or no."},

    {"key": "restecg",  "type": "category",
     "options": {1: "normal", 2: "st-t abnormality", 3: "lv hypertrophy"},
     "question": "What is your resting ECG result? Say 1 for normal, "
                 "2 for ST-T abnormality, or 3 for LV hypertrophy."},

    {"key": "thalch",   "type": "number",
     "question": "What is your maximum heart rate achieved?"},

    {"key": "exang",    "type": "yes_no",
     "question": "Do you have exercise-induced angina? Say yes or no."},

    {"key": "oldpeak",  "type": "number",
     "question": "What is your S T depression value induced by exercise?"},

    {"key": "slope",    "type": "category",
     "options": {1: "upsloping", 2: "flat", 3: "downsloping"},
     "question": "What is the slope of your peak exercise S T segment? "
                 "Say 1 for upsloping, 2 for flat, or 3 for downsloping."},

    {"key": "ca",       "type": "number",
     "question": "How many major vessels are colored by fluoroscopy? A number from 0 to 3."},

    {"key": "thal",     "type": "category",
     "options": {1: "normal", 2: "fixed defect", 3: "reversable defect"},
     "question": "What is your thalassemia value? Say 1 for normal, "
                 "2 for fixed defect, or 3 for reversible defect."},
]


def parse_value(user_text, feature):
    """Convert user's spoken text into the right format for the feature."""
    text = user_text.lower().strip()
    ftype = feature["type"]

    if ftype == "number":
        nums = re.findall(r"-?\d+\.?\d*", text)
        return float(nums[0]) if nums else None

    if ftype == "yes_no":
        if any(w in text for w in ["yes", "yeah", "yep", "true", "i do", "i have"]):
            return 1
        if any(w in text for w in ["no", "nope", "false", "don't", "do not"]):
            return 0
        return None

    if ftype == "sex":
        if "female" in text or "woman" in text:
            return 0
        if "male" in text or "man" in text:
            return 1
        return None

    if ftype == "category":
        # Try to find a number choice (1, 2, 3, 4)
        nums = re.findall(r"\d+", text)
        if nums:
            choice = int(nums[0])
            if choice in feature["options"]:
                return feature["options"][choice]
        # Try to match category name directly
        for value in feature["options"].values():
            if value in text:
                return value
        return None

    return None


def get_next_question(collected):
    """Return the next question to ask, or None if done."""
    idx = len(collected)
    if idx >= len(FEATURES):
        return None
    return FEATURES[idx]["question"]


def is_complete(collected):
    return len(collected) >= len(FEATURES)


def predict_heart_disease(collected):
    """Apply same preprocessing as training, then predict."""
    load_artifacts()

    # Step 1: Build single-row DataFrame
    row = pd.DataFrame([collected])

    # Step 2: One-hot encode categoricals (NO drop_first — reindex handles it)
    row = pd.get_dummies(row, columns=["cp", "restecg", "slope", "thal"])

    # Step 3: Align columns exactly with training (missing → 0, extra → drop)
    row = row.reindex(columns=_feature_columns, fill_value=0)

    # Step 4: Scale the same 6 numeric columns
    row[SCALE_COLS] = _scaler.transform(row[SCALE_COLS])

    # Step 5: Predict — 0 = no disease, 1 = disease
    pred = _model.predict(row)[0]
    return int(pred)