"""Feature engineering applied to raw booking data before encoding/scaling."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the booking DataFrame.

    All features are computed from existing columns and are applied
    identically during training and inference.
    """
    df = df.copy()

    # Binary flags
    df["is_weekend_flight"] = df["flight_day"].isin(["Sat", "Sun"]).astype(int)
    df["is_long_haul"] = (df["flight_duration"] > 6).astype(int)
    df["is_last_minute"] = (df["purchase_lead"] < 7).astype(int)
    df["is_group"] = (df["num_passengers"] > 1).astype(int)

    # Count of optional extras requested
    df["extras_count"] = (
        df["wants_extra_baggage"] + df["wants_preferred_seat"] + df["wants_in_flight_meals"]
    )

    # Ratio features — how far ahead booked relative to flight/stay length
    df["lead_per_duration"] = df["purchase_lead"] / (df["flight_duration"] + 1)
    df["stay_per_duration"] = df["length_of_stay"] / (df["flight_duration"] + 1)

    return df
