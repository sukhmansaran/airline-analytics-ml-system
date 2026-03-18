import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import warnings
warnings.filterwarnings("ignore")


def get_urls(review_type: str = "airline", pages: int = 40) -> list:
    base = "https://www.airlinequality.com"
    paths = {
        "airline": "airline-reviews/british-airways",
        "seat": "seat-reviews/british-airways",
        "lounge": "lounge-reviews/british-airways",
    }
    path = paths.get(review_type, paths["airline"])
    urls = []
    for i in range(1, pages + 1):
        if i == 1:
            urls.append(f"{base}/{path}/?sortby=post_date%3ADesc&pagesize=100")
        else:
            urls.append(f"{base}/{path}/page/{i}/?sortby=post_date%3ADesc&pagesize=100")
    return urls


def _get_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    return webdriver.Chrome(service=Service(), options=opts)


def scrape_data(base_url: list) -> tuple:
    reviewer_name, review_sub, review_text = [], [], []
    date_uploaded, airport_name, seat_type = [], [], []
    ratings_given, recommendations = [], []

    driver = _get_driver()

    for url in base_url:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html5lib")

        try:
            for dt in soup.find_all("span", attrs={"itemprop": "name"}):
                reviewer_name.append(dt.get_text())
        except Exception:
            reviewer_name.append("Not Available")

        try:
            for dt in soup.find_all("h2", attrs={"class": "text_header"}):
                review_sub.append(dt.get_text())
        except Exception:
            review_sub.append("Not Available")

        try:
            for dt in soup.find_all("div", attrs={"class": "text_content"}):
                review_text.append(dt.get_text())
        except Exception:
            review_text.append("Not Available")

        try:
            for dt in soup.find_all("time", attrs={"itemprop": "datePublished"}):
                date_uploaded.append(dt.get_text())
        except Exception:
            date_uploaded.append(np.nan)

        try:
            for data in soup.find_all("div", attrs={"class": "review-stats"}):
                temp_dt = data.find("table", attrs={"class": "review-ratings"})
                df = pd.read_html(str(temp_dt))[0].T
                cols = df.iloc[0]
                df = df[1:]
                df.columns = cols
                df.reset_index(drop=True, inplace=True)
                airport_name.append(df["Airport Name"][0] if "Airport Name" in df.columns else np.nan)
                seat_type.append(df["Seat Type"][0] if "Seat Type" in df.columns else np.nan)
                recommendations.append(df["Recommended"][0] if "Recommended" in df.columns else np.nan)
        except Exception:
            pass

        try:
            for data in soup.find_all("div", attrs={"itemprop": "reviewRating", "class": "rating-10"}):
                dt = data.find("span", attrs={"itemprop": "ratingValue"})
                ratings_given.append(dt.get_text())
        except Exception:
            ratings_given.append(np.nan)

    driver.quit()
    return reviewer_name, review_sub, review_text, date_uploaded, airport_name, seat_type, ratings_given, recommendations
