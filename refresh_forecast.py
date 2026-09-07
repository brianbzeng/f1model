"""Reproducible in-season constructor forecast; see README.md for methodology."""
import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["points_share", "current_rank", "avg_grid", "avg_finish", "dnf_rate", "win_rate", "podium_rate", "top10_rate"]
ALIASES = {"red_bull": "Red Bull", "rb": "RB", "racing_bulls": "RB", "toro_rosso": "RB", "alpine": "Alpine", "renault": "Alpine", "lotus_f1": "Alpine", "force_india": "Aston Martin", "racing_point": "Aston Martin", "aston_martin": "Aston Martin", "sauber": "Sauber", "alfa": "Sauber", "audi": "Sauber", "haas": "Haas", "manor": "Marussia", "marussia": "Marussia", "virgin": "Marussia", "lotus_racing": "Lotus", "caterham": "Caterham", "hrt": "HRT"}
DISPLAY = {"Red Bull": "Red Bull Racing", "RB": "Racing Bulls", "Sauber": "Audi", "Haas": "Haas F1 Team"}
ALIASES["alphatauri"] = "RB"
ALIASES["cadillac"] = "Cadillac"


def constructor(item):
    return ALIASES.get(item["constructorId"], item["name"])


def request_json(path):
    url = "https://api.jolpi.ca/ergast/f1/" + path
    with urlopen(Request(url, headers={"User-Agent": "BrianZeng-F1Forecast/1.0"}), timeout=45) as response:
        return json.load(response)["MRData"]


def races(season, kind):
    """Pagination is mandatory: the service caps each page at 100 results."""
    offset, merged = 0, {}
    key = "SprintResults" if kind == "sprint" else "Results"
    while True:
        data = request_json(f"{season}/{kind}/?limit=100&offset={offset}")
        for race in data["RaceTable"]["Races"]:
            entry = merged.setdefault(int(race["round"]), {**race, key: []})
            entry[key].extend(race[key])
        offset += int(data["limit"])
        if offset >= int(data["total"]):
            break
    return list(sorted(merged.values(), key=lambda race: int(race["round"])))


def load_inputs(season, refresh):
    cache = ROOT / "data" / f"inputs-{season}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    history = pd.read_csv(ROOT / "f1_race_results_clean.csv")
    years = sorted(int(y) for y in history.season.unique() if y < season)
    final = {}
    for year in years:
        print(f"Fetching official constructor ranks: {year}", flush=True)
        table = request_json(f"{year}/constructorStandings/?limit=100")["StandingsTable"]["StandingsLists"]
        final[str(year)] = table[0]["ConstructorStandings"]
    inputs = {
        "races": races(season, "results"), "sprints": races(season, "sprint"),
        "calendar": request_json(f"{season}/?limit=100")["RaceTable"]["Races"],
        "standings": request_json(f"{season}/constructorStandings/?limit=100")["StandingsTable"]["StandingsLists"][0],
        "drivers": request_json(f"{season}/driverStandings/?limit=100")["StandingsTable"]["StandingsLists"][0],
        "finalStandings": final,
    }
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(inputs, indent=2) + "\n")
    return inputs


def aggregate(frame):
    frame = frame.copy()
    # Lapped classified finishes are not retirements. DNS/DSQ/retirements are failures.
    frame["dnf"] = ~frame.status.astype(str).str.match(r"^(Finished|\+\d+ Laps?)$")
    frame["grid_position"] = frame.grid_position.replace(0, np.nan)
    grouped = frame.groupby("constructor")
    out = grouped.agg(points=("points", "sum"), avg_grid=("grid_position", "mean"), avg_finish=("finish_position", "mean"), dnf_rate=("dnf", "mean"))
    out["points_share"] = out.points / max(1, out.points.sum())
    out["current_rank"] = out.points.rank(method="min", ascending=False)
    for name, threshold in [("win_rate", 1), ("podium_rate", 3), ("top10_rate", 10)]:
        out[name] = grouped.finish_position.apply(lambda positions: (positions <= threshold).mean())
    return out


def pipeline():
    # Fixed regularization avoids selecting hyperparameters using evaluation seasons.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def generate(inputs, season, as_of):
    completed = len(inputs["races"])
    scheduled = len(inputs["calendar"])
    if completed < 2 or scheduled < completed:
        raise ValueError("Incomplete current-season race/calendar data")
    if int(inputs["standings"]["round"]) != int(inputs["races"][-1]["round"]):
        raise ValueError("Race results and standings are not from the same round")
    if inputs["races"][-1]["date"] > as_of:
        raise ValueError("As-of date is earlier than the latest result")
    historical = pd.read_csv(ROOT / "f1_race_results_clean.csv")
    frames = []
    for year, frame in historical.groupby("season"):
        # The legacy CSV merges the two distinct 2018 Force India entries.
        # Exclude that season rather than assigning an ambiguous final target.
        if int(year) >= season or int(year) == 2018:
            continue
        rounds = sorted(frame["round"].unique())
        cutoff = max(1, math.floor(len(rounds) * completed / scheduled))
        snapshot = aggregate(frame[frame["round"].isin(rounds[:cutoff])])
        final_ranks = {constructor(row["Constructor"]): int(row["position"]) for row in inputs["finalStandings"][str(year)]}
        snapshot["target"] = snapshot.index.map(final_ranks)
        if snapshot.target.isna().any():
            raise ValueError(f"Unmapped constructor in {year}: {snapshot[snapshot.target.isna()].index.tolist()}")
        snapshot["season"] = year
        frames.append(snapshot)
    train = pd.concat(frames)
    residuals, baseline_errors, folds = [], [], []
    years = sorted(train.season.unique())
    for year in years[5:]:
        fit = train[train.season < year]
        held = train[train.season == year]
        model = pipeline().fit(fit[FEATURES], fit.target)
        error = held.target.to_numpy() - model.predict(held[FEATURES])
        residuals.extend(error)
        baseline_errors.extend(held.target - held.current_rank)
        folds.append({"season": int(year), "trainingThrough": int(year - 1), "teams": len(held), "rmse": float(np.sqrt(np.mean(error ** 2)))})
    rows = []
    for race in inputs["races"]:
        if len({r["Driver"]["driverId"] for r in race["Results"]}) != len(race["Results"]):
            raise ValueError("Duplicate race result after pagination")
        for result in race["Results"]:
            rows.append({"constructor": constructor(result["Constructor"]), "points": float(result["points"]), "grid_position": float(result["grid"]), "finish_position": float(result["position"]), "status": result["status"]})
    current = aggregate(pd.DataFrame(rows))
    official = {constructor(row["Constructor"]): row for row in inputs["standings"]["ConstructorStandings"]}
    if set(current.index) != set(official):
        raise ValueError("Current features and official standings have different teams")
    prediction = pipeline().fit(train[FEATURES], train.target).predict(current[FEATURES])
    noise = np.random.default_rng(42).choice(np.array(residuals), size=(10000, len(current)))
    scores = prediction + noise
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1) + 1
    teams = []
    for index, name in enumerate(current.index):
        row = official[name]
        teams.append({"name": DISPLAY.get(name, name), "currentRank": int(row["position"]), "currentPoints": float(row["points"]), "meanRank": round(float(ranks[:, index].mean()), 4), "title": round(float((ranks[:, index] == 1).mean() * 100), 2), "top3": round(float((ranks[:, index] <= 3).mean() * 100), 2), "top5": round(float((ranks[:, index] <= 5).mean() * 100), 2)})
    result = {"season": season, "asOf": as_of, "through": inputs["races"][-1]["raceName"], "completedRaces": completed, "totalRaces": scheduled, "completedSprints": len(inputs["sprints"]), "simulations": 10000, "validationRmse": round(float(np.sqrt(np.mean(np.square(residuals)))), 4), "validationMae": round(float(np.mean(np.abs(residuals))), 4), "baselineRmse": round(float(np.sqrt(np.mean(np.square(baseline_errors)))), 4), "methodVersion": "in-season-v2", "features": FEATURES, "validation": {"method": "expanding-window season holdout", "folds": folds}, "teams": sorted(teams, key=lambda team: team["meanRank"])}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = generate(load_inputs(args.season, args.refresh), args.season, args.as_of)
    target = ROOT / "data" / "f1-forecast.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"asOf": result["asOf"], "through": result["through"], "rmse": result["validationRmse"], "baselineRmse": result["baselineRmse"], "teams": result["teams"]}, indent=2))


if __name__ == "__main__":
    main()
