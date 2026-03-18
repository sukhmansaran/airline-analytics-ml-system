"""
Run this script to scrape British Airways reviews and save CSVs to data/
"""
import os
from scraping.scraper import get_urls, scrape_data
from scraping.cleaner import clean_data, create_df

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

REVIEW_TYPES = [
    ("airline", 40, (1000, 5000), "Airline_Reviews_BA.csv"),
    ("seat",    2,  (5000, 6000), "Seat_Reviews_BA.csv"),
    ("lounge",  5,  (6000, 7000), "Lounge_Reviews_BA.csv"),
]

for review_type, pages, id_range, filename in REVIEW_TYPES:
    print(f"Scraping {review_type} reviews ({pages} pages)...")
    urls = get_urls(review_type, pages)
    rn, rs, rt, du, an, st, rg, rc = scrape_data(urls)

    rn = clean_data(rn)
    rs = clean_data(rs)
    rt = clean_data(rt)

    df = create_df(rn, rs, rt, du, an, st, rg, rc, id_range=id_range)
    out = os.path.join(DATA_DIR, filename)
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows -> {out}")

print("Done.")
