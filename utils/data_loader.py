from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "results.csv"


def load_matches():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["total_goals"] = df["home_score"] + df["away_score"]

    df["winner"] = df.apply(
        lambda row:
        row["home_team"]
        if row["home_score"] > row["away_score"]
        else row["away_team"]
        if row["away_score"] > row["home_score"]
        else "Draw",
        axis=1
    )

    return df