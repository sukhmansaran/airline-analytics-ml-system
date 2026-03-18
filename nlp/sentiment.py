import os
import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Suppress HuggingFace symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

_sia = None
_tokenizer = None
_model = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


def _get_roberta():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
        _model.eval()
    return _tokenizer, _model


def warm_up():
    """Call at startup to download/cache models before any request arrives."""
    _get_sia()
    _get_roberta()
    print("NLP models ready.")


def results_vader(dataset: pd.DataFrame) -> dict:
    sia = _get_sia()
    results = {}
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="VADER"):
        results[row["Id"]] = sia.polarity_scores(str(row["Review"]))
    return results


def roberta_scores(text: str) -> dict:
    import torch
    tokenizer, model = _get_roberta()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded)
    scores = softmax(output[0][0].detach().numpy())
    return {"roberta_neg": float(scores[0]), "roberta_neu": float(scores[1]), "roberta_pos": float(scores[2])}


def results_roberta(dataset: pd.DataFrame) -> dict:
    results = {}
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="RoBERTa"):
        results[row["Id"]] = roberta_scores(str(row["Review"]))
    return results


def merge_results(dict1: dict, dict2: dict) -> dict:
    return {k: {**dict1[k], **dict2[k]} for k in dict1 if k in dict2}


def create_results_df(dictionary: dict) -> pd.DataFrame:
    df = pd.DataFrame(dictionary).T.reset_index()
    df = df.rename(columns={
        "index": "Id", "neg": "vader_neg", "neu": "vader_neu",
        "pos": "vader_pos", "compound": "vader_comp",
    })
    df["avg_neg"] = (df["vader_neg"] + df["roberta_neg"]) / 2
    df["avg_neu"] = (df["vader_neu"] + df["roberta_neu"]) / 2
    df["avg_pos"] = (df["vader_pos"] + df["roberta_pos"]) / 2

    df.loc[(df["avg_neg"] > df["avg_neu"]) & (df["avg_neg"] > df["avg_pos"]), "pred_result"] = "Negative"
    df.loc[(df["avg_neu"] > df["avg_neg"]) & (df["avg_neu"] > df["avg_pos"]), "pred_result"] = "Neutral"
    df.loc[(df["avg_pos"] > df["avg_neg"]) & (df["avg_pos"] > df["avg_neu"]), "pred_result"] = "Positive"
    return df


def analyse_reviews(df: pd.DataFrame) -> pd.DataFrame:
    vader = results_vader(df)
    roberta = results_roberta(df)
    merged = merge_results(vader, roberta)
    return create_results_df(merged)
