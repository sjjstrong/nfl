import nflreadpy as nr
import polars as pl
from math import sqrt
import requests
from datetime import datetime

# ===============================
# CONFIG
# ===============================

API_KEY = "cfab50385a861a5ffb83c9b7d48e937b"
SPORT = "americanfootball_nfl"
REGIONS = "us"
ODDS_FORMAT = "american"
MAIN_MARKETS = "h2h"

SPORT_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

# ===============================
# LOAD DATA ONCE (CRITICAL)
# ===============================

print("Loading NFL datasets...")
ROSTERS = nr.load_rosters_weekly()
PLAYER_STATS = nr.load_player_stats(summary_level="reg")
TEAM_STATS = nr.load_team_stats(summary_level="reg")
print("Datasets loaded.\n")

# ===============================
# SAFE HELPERS
# ===============================

def safe_sqrt(x):
    """Prevents sqrt crashes"""
    if x is None or x <= 0:
        return 0.0
    return sqrt(x)

def safe(value, default=0.0):
    """Replaces None with default"""
    return default if value is None else value

# ===============================
# ROSTER EXTRACTION
# ===============================

def get_skill_players(team_abbr):
    """
    Returns active QB, RB, WR, TE lists.
    Always returns lists (never None).
    """
    if ROSTERS.is_empty():
        return [], [], [], []

    latest_week = ROSTERS.select(pl.col("week").max()).item()

    team = ROSTERS.filter(
        (pl.col("team") == team_abbr) &
        (pl.col("week") == latest_week) &
        (pl.col("game_type") == "REG") &
        (pl.col("status") == "ACT")
    )

    if team.is_empty():
        return [], [], [], []

    skill = team.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))

    return (
        skill.filter(pl.col("position") == "QB")["full_name"].to_list(),
        skill.filter(pl.col("position") == "RB")["full_name"].to_list(),
        skill.filter(pl.col("position") == "WR")["full_name"].to_list(),
        skill.filter(pl.col("position") == "TE")["full_name"].to_list(),
    )

# ===============================
# STAT ACCESS (NO RELOADS)
# ===============================

def get_player_stat(player, column):
    if column not in PLAYER_STATS.columns:
        return 0.0

    row = PLAYER_STATS.filter(pl.col("player_display_name") == player)

    if row.is_empty():
        return 0.0

    return safe(row[column].item())

def get_last3_total(player, column):
    if column not in PLAYER_STATS.columns:
        return 0.0

    df = PLAYER_STATS.filter(pl.col("player_display_name") == player)

    if df.is_empty():
        return 0.0

    max_week = df["week"].max()
    if max_week is None:
        return 0.0

    last3 = df.filter(pl.col("week") >= (max_week - 2))

    return safe(last3[column].sum())

# ===============================
# DEFENSIVE ADJUSTMENT
# ===============================

def get_def_adjustment(team_abbr):
    """
    Uses opponent defensive EPA per play allowed.
    Positive EPA allowed → weak defense → boost offense.
    """
    if TEAM_STATS.is_empty():
        return 1.0

    row = TEAM_STATS.filter(pl.col("team") == team_abbr)

    if row.is_empty():
        return 1.0

    if "def_epa_per_play" not in TEAM_STATS.columns:
        return 1.0

    def_epa = row["def_epa_per_play"].item()

    # Scale adjustment modestly
    return 1 + (safe(def_epa) * 0.5)

# ===============================
# PACE ADJUSTMENT
# ===============================

def get_pace_adjustment(team_abbr):
    """
    Faster teams run more plays → higher volume.
    """
    if "seconds_per_play" not in TEAM_STATS.columns:
        return 1.0

    row = TEAM_STATS.filter(pl.col("team") == team_abbr)

    if row.is_empty():
        return 1.0

    seconds = row["seconds_per_play"].item()

    if seconds == 0 or seconds is None:
        return 1.0

    league_avg = TEAM_STATS["seconds_per_play"].mean()

    return league_avg / seconds  # faster = >1

# ===============================
# POSITION SCORING
# ===============================

def qb_score(player):
    attempts = get_player_stat(player, "attempts")
    if attempts < 20:
        return None

    completions = get_player_stat(player, "completions")
    yards = get_player_stat(player, "passing_yards")
    epa = get_player_stat(player, "passing_epa")
    pacr = get_player_stat(player, "pacr")

    pass_pct = completions / attempts if attempts else 0
    ypa = yards / attempts if attempts else 0

    base = (
        pass_pct * 10 +
        ypa * 2 +
        epa * 0.4 +
        safe_sqrt(pacr) * 5 +
        get_last3_total(player, "passing_yards") * 0.01
    )

    return base

def rb_score(player):
    carries = get_player_stat(player, "carries")
    if carries < 10:
        return None

    yards = get_player_stat(player, "rushing_yards")
    epa = get_player_stat(player, "rushing_epa")

    ypc = yards / carries if carries else 0

    base = (
        carries * 0.5 +
        ypc * 2 +
        epa * 0.4 +
        get_last3_total(player, "rushing_yards") * 0.01
    )

    return base

def rec_score(player):
    receptions = get_player_stat(player, "receptions")
    if receptions < 10:
        return None

    yards = get_player_stat(player, "receiving_yards")
    epa = get_player_stat(player, "receiving_epa")
    wopr = get_player_stat(player, "wopr")

    ypr = yards / receptions if receptions else 0

    base = (
        ypr * 2 +
        epa * 0.4 +
        wopr * 10 +
        get_last3_total(player, "receiving_yards") * 0.01
    )

    return base

# ===============================
# TEAM MODEL BUILDER
# ===============================

def build_team_model(team_abbr, opponent_abbr):
    """
    Builds full team offensive rating
    including defensive and pace adjustments.
    """

    qbs, rbs, wrs, tes = get_skill_players(team_abbr)

    def_adj = get_def_adjustment(opponent_abbr)
    pace_adj = get_pace_adjustment(team_abbr)

    total_score = 0

    # QB weighted heavier
    for p in qbs:
        s = qb_score(p)
        if s:
            total_score += s * 1.5

    for p in rbs:
        s = rb_score(p)
        if s:
            total_score += s

    for p in wrs:
        s = rec_score(p)
        if s:
            total_score += s

    for p in tes:
        s = rec_score(p)
        if s:
            total_score += s * 0.8

    # Apply adjustments
    total_score *= def_adj
    total_score *= pace_adj

    return total_score

# ===============================
# IMPLIED PROBABILITY
# ===============================

def american_to_prob(odds):
    """
    Converts American odds to implied probability.
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

# ===============================
# SIMPLE EDGE DETECTOR
# ===============================

def compare_moneyline(team1, team2):
    games = requests.get(SPORT_ODDS_URL, params={
        "api_key": API_KEY,
        "regions": REGIONS,
        "markets": MAIN_MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }).json()

    for game in games:
        if team1 in (game["home_team"], game["away_team"]):

            bookmaker = game["bookmakers"][0]
            market = bookmaker["markets"][0]

            for outcome in market["outcomes"]:
                prob = american_to_prob(outcome["price"])
                print(f"{outcome['name']} ML {outcome['price']} → Implied {round(prob,3)}")

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":

    home = "SEA"
    away = "NE"

    home_score = build_team_model(home, away)
    away_score = build_team_model(away, home)

    print("\nMODEL RATINGS")
    print(home, ":", round(home_score, 2))
    print(away, ":", round(away_score, 2))

    diff = home_score - away_score
    print("\nModel Edge (raw rating difference):", round(diff, 2))

    print("\nSportsbook Moneyline:")
    compare_moneyline("Seattle Seahawks", "New England Patriots")
