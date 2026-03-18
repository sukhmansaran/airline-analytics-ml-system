import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from training.feature_engineering import engineer_features


def preprocess(raw_df: pd.DataFrame):
    df = raw_df.drop_duplicates().reset_index(drop=True).copy()
    df = engineer_features(df)

    # Fit and save a LabelEncoder per string column so inference can reuse them
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_cols = df.drop("booking_complete", axis=1).columns.tolist()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[feature_cols])
    df = pd.DataFrame(df_scaled, columns=feature_cols)
    df["booking_complete"] = raw_df.drop_duplicates().reset_index(drop=True)["booking_complete"]

    df = shuffle(df).reset_index(drop=True)
    x = df.drop("booking_complete", axis=1)
    y = df["booking_complete"]
    return x, y, scaler, feature_cols, label_encoders
