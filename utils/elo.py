def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, score_a, k=20):
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - expected_b)

    return new_a, new_b


def calculate_elo_history(df, initial_rating=1500):
    data = df.sort_values("date").copy()

    ratings = {}
    elo_rows = []

    for _, match in data.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        home_rating = ratings.get(home, initial_rating)
        away_rating = ratings.get(away, initial_rating)

        elo_rows.append(
            {
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_difference": home_rating - away_rating,
            }
        )

        if match["home_score"] > match["away_score"]:
            home_score = 1.0
        elif match["home_score"] < match["away_score"]:
            home_score = 0.0
        else:
            home_score = 0.5

        new_home, new_away = update_elo(
            home_rating,
            away_rating,
            home_score,
        )

        ratings[home] = new_home
        ratings[away] = new_away

    return elo_rows, ratings