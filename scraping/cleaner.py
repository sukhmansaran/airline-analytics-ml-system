import random
import numpy as np
import pandas as pd

TEXT_TO_REPLACE = [
    "trip verified |", "not verified |", "unverified |", "verified review |",
    "\xa0", "…", "❌", "✅", "❎", "<", ">", ";", ":", "!", "#", "&",
    "£", "$", "€", "+", "(", ")", "'", '"', "\u201c", "\u201d", "\u2018", "\u2019",
    '"', "%", "\\'", "/", "-", ".", ",", "!", "?",
]


def clean_data(array: list) -> list:
    clean_array = []
    for ar in array:
        ar = str(ar).lower()
        for txt in TEXT_TO_REPLACE:
            ar = ar.replace(txt, "")
        clean_array.append(ar.strip())
    return clean_array


def create_df(
    reviewer_name, review_sub, review_text, date_uploaded,
    airport_name, seat_type, ratings_given, recommendations, id_range=(1000, 5000)
) -> pd.DataFrame:
    ids = random.sample(range(*id_range), len(reviewer_name))
    dataset = []
    for i in range(len(review_text)):
        def safe(arr, idx):
            try:
                return arr[idx]
            except Exception:
                return np.nan

        dataset.append([
            safe(ids, i), safe(reviewer_name, i), safe(review_sub, i),
            safe(review_text, i), safe(date_uploaded, i), safe(airport_name, i),
            safe(seat_type, i), safe(ratings_given, i), safe(recommendations, i),
        ])

    cols = ["Id", "Customer Name", "Review Head", "Review", "Date",
            "Airport Name", "Seat Type", "Ratings(10)", "Recommended"]
    return pd.DataFrame(dataset, columns=cols)
