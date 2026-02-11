import re
import streamlit as st
import pandas as pd
import requests
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# -------------------- Constants --------------------
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ELEMENT_SUMMARY = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_ORDER = ["GK", "DEF", "MID", "FWD"]

# -------------------- Data loaders --------------------
@st.cache_data(ttl=60 * 60)
def load_bootstrap() -> pd.DataFrame:
    data = requests.get(FPL_BOOTSTRAP, timeout=30).json()
    elements = pd.DataFrame(data["elements"])
    teams = (
        pd.DataFrame(data["teams"])[["id", "name"]]
        .rename(columns={"id": "team_id", "name": "team_name"})
    )

    keep = [
        "id", "first_name", "second_name", "web_name", "team", "element_type",
        "now_cost", "status", "chance_of_playing_next_round",
        "total_points", "form", "points_per_game", "selected_by_percent",
        "minutes", "transfers_in_event", "transfers_out_event",
    ]
    elements = elements[keep].copy()
    elements = elements.rename(
        columns={"team": "team_id", "element_type": "pos_id", "id": "player_id"}
    )
    elements["position"] = elements["pos_id"].map(POS_MAP)
    elements = elements.merge(teams, on="team_id", how="left")
    elements["name"] = elements["web_name"].fillna(elements["second_name"])

    for col in ["form", "points_per_game", "selected_by_percent"]:
        elements[col] = pd.to_numeric(elements[col], errors="coerce")

    return elements


@st.cache_data(ttl=60 * 60)
def load_fixtures() -> pd.DataFrame:
    return pd.DataFrame(requests.get(FPL_FIXTURES, timeout=30).json())


@st.cache_data(ttl=60 * 30)
def get_gw_stats_and_fixtures() -> tuple[int, int]:
    """
    gw_stats: last FINISHED gameweek id (our 'latest completed reference point')
    gw_fixtures: gw_stats + 1 (the upcoming GW we should optimize for)
    """
    data = requests.get(FPL_BOOTSTRAP, timeout=30).json()
    events = data.get("events", [])

    finished_ids = [int(e["id"]) for e in events if e.get("finished")]
    gw_stats = max(finished_ids) if finished_ids else 0
    gw_fixtures = gw_stats + 1 if gw_stats > 0 else 1  # early season safety

    return gw_stats, gw_fixtures


@st.cache_data(ttl=60 * 15)
def load_entry_picks_with_fallback(
    entry_id: int,
    gw_try_1: int,
    gw_try_2: int | None = None
) -> tuple[list[int], int]:
    """
    Tries to load a manager's picks for gw_try_1 first, then optionally gw_try_2.
    Returns (player_ids, gw_used).

    This avoids the common FPL API behavior where the "upcoming" GW endpoint
    may 404 before picks are available.
    """
    def _fetch(gw: int) -> list[int]:
        url = f"https://fantasy.premierleague.com/api/entry/{int(entry_id)}/event/{int(gw)}/picks/"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")
        data = resp.json()
        picks = data.get("picks", [])
        if not picks:
            raise ValueError("No picks found")
        return [int(p["element"]) for p in picks]

    try:
        return _fetch(int(gw_try_1)), int(gw_try_1)
    except Exception:
        if gw_try_2 is None:
            raise ValueError(
                f"Could not load team for GW{gw_try_1}. "
                "Double-check Entry ID and that the team is public."
            )
        try:
            return _fetch(int(gw_try_2)), int(gw_try_2)
        except Exception as e:
            raise ValueError(
                f"Could not load team. Tried GW{gw_try_1} and GW{gw_try_2}. "
                "Double-check Entry ID and that the team is public."
            ) from e


# -------------------- Fixture difficulty (horizon) --------------------
def fixture_multiplier(avg_difficulty: float) -> float:
    """
    Difficulty is 1 (easy) to 5 (hard).

    We keep a linear base around 3, but add a gentle nonlinear kicker so
    extremes (1 and 5) matter more without destabilizing the mid-range.

    Linear base (same as before): 0.06 per difficulty step.
    Nonlinear kicker: pushes 1/5 further than 2/4, capped for stability.
    """
    if avg_difficulty is None or pd.isna(avg_difficulty):
        return 1.0

    d = float(avg_difficulty)
    x = (3.0 - d)  # + for easier, - for harder

    # Base linear behavior (previous model)
    mult = 1.0 + 0.06 * x

    # Gentle nonlinear kicker: grows with distance from 3
    # Keeps 2/4 modest, makes 1/5 more pronounced
    mult += 0.015 * (abs(x) ** 2) * (1 if x >= 0 else -1)

    # Safety clamp to preserve stability
    return float(min(1.18, max(0.82, mult)))



def team_fixture_context_horizon(
    fixtures_df: pd.DataFrame, gw_start: int, horizon: int
) -> tuple[dict, dict, dict]:
    """
    Returns:
      - avg difficulty per team across gw_start..gw_start+horizon-1 (DGWs averaged per GW first)
      - home_ratio per team across the horizon (0..1)
      - fixture_count per team across the horizon
    """
    if fixtures_df.empty:
        return {}, {}, {}

    gws = list(range(int(gw_start), int(gw_start) + int(horizon)))
    f = fixtures_df[fixtures_df["event"].isin(gws)].copy()
    if f.empty:
        return {}, {}, {}

    home = f[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team_id", "team_h_difficulty": "difficulty"}
    )
    home["is_home"] = 1
    away = f[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team_id", "team_a_difficulty": "difficulty"}
    )
    away["is_home"] = 0

    all_rows = pd.concat([home, away], ignore_index=True)

    # Difficulty: average within each GW first (DGW-safe)
    per_gw = all_rows.groupby(["team_id", "event"])["difficulty"].mean().reset_index()
    horizon_avg = per_gw.groupby("team_id")["difficulty"].mean().to_dict()

    # Home/away ratio and fixture counts (DGWs count as multiple fixtures)
    home_ratio = all_rows.groupby("team_id")["is_home"].mean().to_dict()
    fixture_count = all_rows.groupby("team_id")["is_home"].count().to_dict()

    return horizon_avg, home_ratio, fixture_count


# -------------------- Expected points (mean) --------------------
def expected_points_v1(row, risk_mode: str) -> float:
    """
    Simple heuristic:
      - Base: points_per_game
      - + small form boost
      - × minutes reliability
      - × availability probability
      - penalties for injured/suspended
    """
    ppg = row["points_per_game"] if pd.notna(row["points_per_game"]) else 0.0
    form = row["form"] if pd.notna(row["form"]) else 0.0

    mins = row["minutes"] if pd.notna(row["minutes"]) else 0
    minutes_factor = min(1.0, max(0.3, mins / 1800))  # 1800 mins ~ 20 full matches

    chance = row["chance_of_playing_next_round"]
    if pd.isna(chance):
        chance = 100
    chance = float(chance) / 100.0

    if risk_mode == "Safe":
        chance_weight = 1.2
        minutes_weight = 1.2
        form_weight = 0.25
    elif risk_mode == "Aggro":
        chance_weight = 0.8
        minutes_weight = 0.9
        form_weight = 0.40
    else:
        chance_weight = 1.0
        minutes_weight = 1.0
        form_weight = 0.32

    # Blend long-run PPG with recent form (form is last ~5)
    base = (1.0 - form_weight) * ppg + form_weight * form
    exp = base
    exp *= (chance ** chance_weight)
    exp *= (minutes_factor ** minutes_weight)

    status = str(row["status"])
    if status in ["i", "s", "u"]:
        exp *= 0.25
    elif status == "d":
        exp *= 0.75

    return max(0.0, float(exp))


def add_fixture_adjusted_mean_only(
    df: pd.DataFrame,
    team_diff_map: dict,
    team_home_ratio: dict,
    risk_mode: str,
) -> pd.DataFrame:
    """
    Fast path: computes ONLY mean expected points (fixture-adjusted).
    No element-summary calls (safe to run over all players).
    """
    out = df.copy()
    out["base_xPts"] = out.apply(lambda r: expected_points_v1(r, risk_mode), axis=1)
    out["avg_fixture_difficulty"] = out["team_id"].map(team_diff_map)
    out["fixture_mult"] = out["avg_fixture_difficulty"].apply(fixture_multiplier)
    # Small home/away balance adjustment (conservative)
    out["home_ratio"] = out["team_id"].map(team_home_ratio)
    out["home_ratio"] = out["home_ratio"].fillna(0.5)
    out["home_mult"] = 1.0 + 0.04 * ((out["home_ratio"] - 0.5) * 2.0)
    out["home_mult"] = out["home_mult"].clip(0.96, 1.04)

    out["exp_points"] = out["base_xPts"] * out["fixture_mult"] * out["home_mult"]
    return out


# -------------------- Floor / Ceiling via recent history volatility --------------------
@st.cache_data(ttl=60 * 60)
def load_element_summary(player_id: int) -> dict:
    url = FPL_ELEMENT_SUMMARY.format(player_id=int(player_id))
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def recent_points_std(player_id: int, n_matches: int = 6) -> float:
    """
    Std dev of FPL total_points across last n matches with minutes > 0.
    Used as a volatility proxy.
    """
    try:
        js = load_element_summary(int(player_id))
        hist = js.get("history", [])
        if not hist:
            return 2.5
        df = pd.DataFrame(hist)
        df = df[df["minutes"] > 0].tail(n_matches)
        if len(df) < 3:
            return 2.5
        std = float(df["total_points"].std(ddof=0))
        return max(1.0, min(std, 8.0))
    except Exception:
        return 2.5


def floor_ceiling_from_mean(mean_xpts: float, std_points: float) -> tuple[float, float]:
    floor = max(0.0, float(mean_xpts) - 0.7 * float(std_points))
    ceil = max(0.0, float(mean_xpts) + 1.2 * float(std_points))
    return floor, ceil


def add_fixture_adjusted_xpts(
    df: pd.DataFrame,
    team_diff_map: dict,
    team_home_ratio: dict,
    risk_mode: str,
    n_matches_std: int = 6
) -> pd.DataFrame:
    """
    Adds columns:
      - exp_points (mean fixture-adjusted)
      - floor_points / ceiling_points using volatility proxy from recent match points
    """
    out = add_fixture_adjusted_mean_only(df, team_diff_map, team_home_ratio, risk_mode)
    out["recent_std_pts"] = out["player_id"].apply(lambda pid: recent_points_std(pid, n_matches=n_matches_std))
    fc = out.apply(lambda r: floor_ceiling_from_mean(r["exp_points"], r["recent_std_pts"]), axis=1)
    out["floor_points"] = [x[0] for x in fc]
    out["ceiling_points"] = [x[1] for x in fc]
    return out


def resolve_points_col(optimize_target: str, risk_mode: str) -> str:
    """Maps UI selection to the points column used by optimizer."""
    if optimize_target == "Floor (safe)":
        return "floor_points"
    if optimize_target == "Mean":
        return "exp_points"
    if optimize_target == "Ceiling (upside)":
        return "ceiling_points"

    # Auto
    if risk_mode == "Safe":
        return "floor_points"
    if risk_mode == "Aggro":
        return "ceiling_points"
    return "exp_points"


def friendly_lens(points_col: str) -> str:
    return {
        "floor_points": "Floor (safe)",
        "exp_points": "Mean",
        "ceiling_points": "Ceiling (upside)",
    }.get(points_col, points_col)


# -------------------- Optimizer --------------------
def optimize_lineup(team_df: pd.DataFrame, bench_weight: float = 0.10, points_col: str = "exp_points"):
    df = team_df.reset_index(drop=True).copy()

    x = [LpVariable(f"start_{i}", cat=LpBinary) for i in range(len(df))]
    b = [LpVariable(f"bench_{i}", cat=LpBinary) for i in range(len(df))]
    c = [LpVariable(f"capt_{i}", cat=LpBinary) for i in range(len(df))]

    prob = LpProblem("FPL_Lineup_Optimizer", LpMaximize)

    exp = df[points_col].tolist()
    prob += (
        lpSum(x[i] * exp[i] for i in range(len(df)))
        + bench_weight * lpSum(b[i] * exp[i] for i in range(len(df)))
        + lpSum(c[i] * exp[i] for i in range(len(df)))  # captain bonus
    )

    prob += lpSum(x) == 11
    prob += lpSum(b) == 4

    for i in range(len(df)):
        prob += x[i] + b[i] == 1

    prob += lpSum(c) == 1
    for i in range(len(df)):
        prob += c[i] <= x[i]

    def idxs(pos):
        return [i for i in range(len(df)) if df.loc[i, "position"] == pos]

    gk = idxs("GK")
    de = idxs("DEF")
    mi = idxs("MID")
    fw = idxs("FWD")

    prob += lpSum(x[i] for i in gk) == 1
    prob += lpSum(x[i] for i in de) >= 3
    prob += lpSum(x[i] for i in de) <= 5
    prob += lpSum(x[i] for i in mi) >= 2
    prob += lpSum(x[i] for i in mi) <= 5
    prob += lpSum(x[i] for i in fw) >= 1
    prob += lpSum(x[i] for i in fw) <= 3

    prob.solve(PULP_CBC_CMD(msg=False))

    df["is_start"] = [int(v.value()) for v in x]
    df["is_bench"] = [int(v.value()) for v in b]
    df["is_captain"] = [int(v.value()) for v in c]

    starters = df[df["is_start"] == 1].copy().sort_values(["position", points_col], ascending=[True, False])
    bench = df[df["is_bench"] == 1].copy().sort_values(points_col, ascending=False)

    captain = df[df["is_captain"] == 1].iloc[0]
    starters_no_c = starters[starters["player_id"] != captain["player_id"]].sort_values(points_col, ascending=False)
    vice = starters_no_c.iloc[0] if len(starters_no_c) else captain

    return starters, bench, captain, vice


def lineup_objective_score(starters: pd.DataFrame, bench: pd.DataFrame, captain: pd.Series, bench_weight: float, points_col: str) -> float:
    starter_points = float(starters[points_col].sum())
    bench_points = float(bench[points_col].sum())
    return starter_points + bench_weight * bench_points + float(captain[points_col])


# -------------------- Transfers (lineup-aware) --------------------
def suggest_best_transfer_by_lineup(
    squad: pd.DataFrame,
    all_players_with_xp: pd.DataFrame,
    bank_cost_tenths: int,
    bench_weight: float,
    points_col: str,
    max_ins_per_out: int = 20,
    top_n_results: int = 10
) -> pd.DataFrame:
    """
    Returns top transfers by incremental expected points under the selected lens.
    incremental_xPts = (optimized team after transfer) - (current optimized team)
    """
    squad = squad.copy().reset_index(drop=True)
    squad_ids = set(squad["player_id"].tolist())

    base_starters, base_bench, base_capt, _ = optimize_lineup(
        squad, bench_weight=bench_weight, points_col=points_col
    )
    base_score = lineup_objective_score(
        base_starters, base_bench, base_capt, bench_weight, points_col=points_col
    )

    team_counts = squad["team_id"].value_counts().to_dict()
    candidates = all_players_with_xp[~all_players_with_xp["player_id"].isin(squad_ids)].copy()

    results = []

    for out_idx, outp in squad.iterrows():
        out_pos = outp["position"]
        out_cost = int(outp["now_cost"])
        out_team = int(outp["team_id"])
        max_buy_cost = out_cost + int(bank_cost_tenths)

        counts_after_sell = dict(team_counts)
        counts_after_sell[out_team] = counts_after_sell.get(out_team, 0) - 1

        ins = candidates[
            (candidates["position"] == out_pos) &
            (candidates["now_cost"] <= max_buy_cost)
        ].copy()

        # Apply 3-per-team constraint
        ins = ins[ins["team_id"].apply(lambda tid: counts_after_sell.get(int(tid), 0) + 1 <= 3)]
        if ins.empty:
            continue

        ins = ins.sort_values(points_col, ascending=False).head(max_ins_per_out)

        for _, inp in ins.iterrows():
            new_squad = squad.copy()
            new_squad.loc[out_idx] = inp[new_squad.columns].values
            new_squad = new_squad.reset_index(drop=True)

            starters, bench, captain, vice = optimize_lineup(
                new_squad, bench_weight=bench_weight, points_col=points_col
            )
            new_score = lineup_objective_score(
                starters, bench, captain, bench_weight, points_col=points_col
            )

            inc = new_score - base_score

            results.append({
                "sell_player": outp["name"],
                "sell_team": outp["team_name"],
                "sell_pos": out_pos,
                "sell_cost_£m": out_cost / 10.0,
                "buy_player": inp["name"],
                "buy_team": inp["team_name"],
                "buy_cost_£m": int(inp["now_cost"]) / 10.0,
                "bank_used_£m": max(0, (int(inp["now_cost"]) - out_cost) / 10.0),
                "incremental_xPts": inc,
                "lens": friendly_lens(points_col),
                "new_captain": captain["name"],
                "new_vice": vice["name"],
            })

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("incremental_xPts", ascending=False)
        .head(top_n_results)
    )


def style_transfer_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def color_delta(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #0f766e; font-weight: 700;"
        if val < 0:
            return "color: #b91c1c; font-weight: 700;"
        return "color: #334155;"

    styler = df.style.format({
        "sell_cost_£m": "{:.1f}",
        "buy_cost_£m": "{:.1f}",
        "bank_used_£m": "{:.1f}",
        "incremental_xPts": "{:+.2f}",
    })
    styler = styler.applymap(color_delta, subset=["incremental_xPts"])
    return styler


# -------------------- UI helpers --------------------
def settings_summary(risk_mode: str, fixture_horizon: int, bench_weight: float, n_matches_std: int):
    st.info(
        f"**Settings:** {risk_mode} • Horizon **{fixture_horizon} GW** • "
        f"Bench **{bench_weight:.2f}** • Volatility **{n_matches_std} matches**\n\n"
        "👉 Change these in the **sidebar** (they apply everywhere)."
    )


# -------------------- UI --------------------
st.set_page_config(page_title="FPL Lineup Optimizer", page_icon="⚽", layout="wide")

# Minimal, clean styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="st-"] {
  font-family: "Manrope", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
  font-weight: 700;
  letter-spacing: -0.01em;
}

[data-testid="stMetricValue"] {
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

.section-card {
  padding: 1.25rem 1.25rem 1rem 1.25rem;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #ffffff;
}

.section-title {
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}

.muted {
  color: #64748b;
  font-size: 0.9rem;
}

.tight hr {
  margin: 0.75rem 0;
}

/* Streamlit input polish */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] > label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiselect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label {
  position: static !important;
  display: block;
  margin-bottom: 0.4rem;
  font-weight: 600;
  color: #0f172a;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stMultiselect"] div[role="combobox"] {
  min-height: 42px;
  padding: 0.5rem 0.75rem;
  border-radius: 10px;
}

div[data-testid="stTextInput"] input::placeholder {
  color: #94a3b8;
  opacity: 1;
}

/* Dark theme adjustments */
.stApp[data-theme="dark"] div[data-testid="stTextInput"] label,
.stApp[data-theme="dark"] div[data-testid="stNumberInput"] label,
.stApp[data-theme="dark"] div[data-testid="stSelectbox"] label,
.stApp[data-theme="dark"] div[data-testid="stMultiselect"] label,
.stApp[data-theme="dark"] div[data-testid="stSlider"] label,
.stApp[data-theme="dark"] div[data-testid="stTextArea"] label {
  color: #e2e8f0;
}

.stApp[data-theme="dark"] div[data-testid="stTextInput"] input,
.stApp[data-theme="dark"] div[data-testid="stNumberInput"] input,
.stApp[data-theme="dark"] div[data-testid="stSelectbox"] div[role="combobox"],
.stApp[data-theme="dark"] div[data-testid="stMultiselect"] div[role="combobox"] {
  background: #0b1220;
  color: #e2e8f0;
  border-color: #334155;
}

.stApp[data-theme="dark"] div[data-testid="stTextInput"] input::placeholder {
  color: #94a3b8;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("FPL Optimizer")
st.caption("A guided, minimal workflow for squad loading, transfers, and optimization.")

# Shared state
st.session_state.setdefault("squad_ids", None)
st.session_state.setdefault("squad_df", None)
st.session_state.setdefault("transfer_df", None)
st.session_state.setdefault("entry_error", None)
st.session_state.setdefault("gw_used_for_picks", None)

# Global settings state
st.session_state.setdefault("risk_mode", "Balanced")
st.session_state.setdefault("bench_weight", 0.10)
st.session_state.setdefault("fixture_horizon", 3)
st.session_state.setdefault("n_matches_std", 6)

elements = load_bootstrap()

# Sidebar: Global settings
with st.sidebar:
    st.header("Model settings")
    st.caption("These apply everywhere.")

    st.selectbox("Risk mode", ["Safe", "Balanced", "Aggro"], index=1, key="risk_mode")
    st.slider("Bench importance", 0.0, 0.3, 0.10, 0.01, key="bench_weight")
    st.slider("Fixture horizon (GWs)", 1, 6, 3, 1, key="fixture_horizon")
    st.slider("Volatility lookback (matches)", 4, 10, 6, 1, key="n_matches_std")

# Pull global settings
risk_mode = st.session_state.risk_mode
bench_weight = float(st.session_state.bench_weight)
fixture_horizon = int(st.session_state.fixture_horizon)
n_matches_std = int(st.session_state.n_matches_std)

# ---- Separate GW for stats vs fixtures ----
gw_stats, gw_fixtures = get_gw_stats_and_fixtures()  # fixtures start at gw_stats + 1

fixtures = load_fixtures()
team_diff, team_home_ratio, team_fixture_count = team_fixture_context_horizon(
    fixtures, gw_start=gw_fixtures, horizon=fixture_horizon
)

# Global caption (visible above stepper)
st.caption(
    f"Stats reference: **GW{gw_stats}** • "
    f"Fixture window: **GW{gw_fixtures}–GW{gw_fixtures + fixture_horizon - 1}**"
)

step = st.radio(
    "Workflow",
    ["1. Load team", "2. Transfers", "3. Optimize", "4. Top players", "5. Assumptions"],
    horizontal=True,
    label_visibility="collapsed",
)

# -------------------- STEP 1: Load team --------------------
if step == "1. Load team":
    settings_summary(risk_mode, fixture_horizon, bench_weight, n_matches_std)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Load your squad</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Paste your Entry ID or URL to auto-load your 15 players.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Team ID (Entry ID)**")

        with st.expander("How to find your Team ID", expanded=False):
            st.markdown(
                """
**Method 1: copy it from the URL**
1. Open the Fantasy Premier League website and log in.
2. Go to **Points** (or **Pick Team**) for your squad.
3. Look at your browser address bar — your URL will include `/entry/<NUMBER>/`.

**Generic formats**
- `https://fantasy.premierleague.com/entry/<ENTRY_ID>/event/<GW>/points`
- `https://fantasy.premierleague.com/entry/<ENTRY_ID>/`

**Example**
- If you see: `.../entry/1234567/event/22/points`
- Then your Team ID is: **1234567**
                """
            )

        entry_id = st.text_input("FPL Team ID", value="", placeholder="e.g., 1234567")

        url_paste = st.text_input(
            "Or paste your FPL URL here (optional)",
            value="",
            placeholder="https://fantasy.premierleague.com/entry/1234567/event/22/points"
        )
        if url_paste.strip():
            m = re.search(r"/entry/(\d+)", url_paste)
            if m:
                entry_id = m.group(1)
                st.success(f"Found Team ID: {entry_id}")
            else:
                st.warning("Couldn’t find `/entry/<number>/` in that URL. Try copying the Points page URL.")

        load_team = st.button("Load my squad", type="primary")
        if load_team:
            try:
                if not entry_id.strip().isdigit():
                    raise ValueError("Please enter a numeric Team ID (Entry ID).")

                # Try last finished GW first (usually exists), then upcoming anchor GW
                prefill_ids, gw_used = load_entry_picks_with_fallback(
                    int(entry_id.strip()),
                    gw_try_1=gw_stats,
                    gw_try_2=gw_fixtures
                )
                st.session_state.gw_used_for_picks = gw_used
                st.session_state.squad_ids = prefill_ids
                st.session_state.entry_error = None
                st.session_state.transfer_df = None
            except Exception as e:
                st.session_state.entry_error = str(e)

        if st.session_state.entry_error:
            st.error(st.session_state.entry_error)

    with right:
        all_labels = elements["name"] + " — " + elements["team_name"] + " (" + elements["position"] + ")"
        id_to_label = dict(zip(elements["player_id"], all_labels))
        label_to_id = dict(zip(all_labels, elements["player_id"]))

        default_selected_labels = []
        if st.session_state.squad_ids:
            default_selected_labels = [id_to_label[i] for i in st.session_state.squad_ids if i in id_to_label]

        selected = st.multiselect(
            "Squad (auto-filled after loading)",
            options=all_labels.tolist(),
            default=default_selected_labels,
            help="If your Team ID loads, your 15 players will appear here automatically.",
        )

        selected_ids = [label_to_id[s] for s in selected] if selected else (st.session_state.squad_ids or [])

        if len(selected_ids) == 15:
            st.session_state.squad_ids = selected_ids
            squad_raw = elements[elements["player_id"].isin(selected_ids)].copy()
            squad = add_fixture_adjusted_xpts(
                squad_raw, team_diff, team_home_ratio, risk_mode, n_matches_std=n_matches_std
            )
            st.session_state.squad_df = squad
            st.success("Squad loaded. Move to Transfers or Optimize.")
            st.dataframe(
                squad[[
                    "name", "team_name", "position", "now_cost", "status",
                    "avg_fixture_difficulty", "fixture_mult",
                    "floor_points", "exp_points", "ceiling_points"
                ]].sort_values(["position", "exp_points"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.session_state.squad_df = None
            if st.session_state.squad_ids and len(st.session_state.squad_ids) != 15:
                st.info("Loaded squad is incomplete. Try re-loading via Team ID.")
            elif selected and len(selected_ids) != 15:
                st.info(f"Squad must contain exactly 15 players. Currently: {len(selected_ids)}")
            else:
                st.info("Enter your Team ID and click Load my squad to populate your 15 players.")

# -------------------- STEP 2: Transfers --------------------
if step == "2. Transfers":
    settings_summary(risk_mode, fixture_horizon, bench_weight, n_matches_std)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Transfer recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Lineup-aware 1-transfer moves, re-optimized after each swap.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    squad = st.session_state.squad_df
    if squad is None or len(squad) == 0:
        st.info("Load your squad in Step 1 first.")
    else:
        colA, colB, colC = st.columns([1, 1, 2], gap="large")

        with colA:
            bank_m = st.slider("Money in the bank (£m)", 0.0, 10.0, 1.0, 0.1, key="bank_m")
            st.caption("Budget available in addition to the sell price.")

        with colB:
            optimize_target = st.selectbox(
                "Optimize for",
                ["Auto (from risk mode)", "Floor (safe)", "Mean", "Ceiling (upside)"],
                index=0,
                key="opt_target"
            )
            st.caption("Floor = safer returns; Ceiling = haul-chasing; Mean = average expectation.")

        with colC:
            candidate_pool_size = st.slider("Transfer search breadth", 100, 450, 250, 25, key="cand_pool")
            st.caption("Limits how many players we consider (keeps it fast).")

        points_col = resolve_points_col(optimize_target, risk_mode)
        lens_name = friendly_lens(points_col)

        st.caption(
            f"**Incremental expected points** = (optimized team after transfer) − (current optimized team), "
            f"using the **{lens_name}** lens."
        )

        run = st.button("Find best transfer", type="primary")
        if run:
            candidate_pool = elements.sort_values("points_per_game", ascending=False).head(candidate_pool_size).copy()
            all_with_xp = add_fixture_adjusted_xpts(
                candidate_pool, team_diff, team_home_ratio, risk_mode, n_matches_std=n_matches_std
            )

            bank_cost_tenths = int(round(bank_m * 10))
            transfer_df = suggest_best_transfer_by_lineup(
                squad=squad,
                all_players_with_xp=all_with_xp,
                bank_cost_tenths=bank_cost_tenths,
                bench_weight=bench_weight,
                points_col=points_col,
                max_ins_per_out=20,
                top_n_results=10
            )
            st.session_state.transfer_df = transfer_df

        transfer_df = st.session_state.transfer_df
        if transfer_df is not None:
            if transfer_df.empty:
                st.info("No positive-value transfers found within the constraints. Try increasing bank or changing the lens.")
            else:
                best = transfer_df.iloc[0]
                st.markdown(
                    f"**Best move:** Sell **{best['sell_player']}** → Buy **{best['buy_player']}** "
                    f"(**{best['incremental_xPts']:+.2f} incremental expected points**)"
                )
                st.caption(
                    f"Lens: **{best['lens']}** • Bank used £{best['bank_used_£m']:.1f}m • "
                    f"New captain: {best['new_captain']} • New vice: {best['new_vice']}"
                )

                display_cols = [
                    "sell_player", "sell_team", "sell_pos", "sell_cost_£m",
                    "buy_player", "buy_team", "buy_cost_£m",
                    "bank_used_£m",
                    "incremental_xPts",
                    "new_captain", "new_vice"
                ]
                shown = transfer_df[display_cols].copy()
                shown["incremental_xPts"] = shown["incremental_xPts"].round(2)

                st.dataframe(
                    style_transfer_df(shown),
                    use_container_width=True,
                    hide_index=True
                )

# -------------------- STEP 3: Optimize & captaincy --------------------
if step == "3. Optimize":
    settings_summary(risk_mode, fixture_horizon, bench_weight, n_matches_std)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Optimize XI + captaincy</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Best legal XI, bench, and captain for your chosen lens.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    squad = st.session_state.squad_df
    if squad is None or len(squad) == 0:
        st.info("Load your squad in Step 1 first.")
    else:
        left, right = st.columns([1, 2], gap="large")

        with left:
            optimize_target_opt = st.selectbox(
                "Optimize for",
                ["Auto (from risk mode)", "Floor (safe)", "Mean", "Ceiling (upside)"],
                index=0,
                key="opt_target_opt"
            )
            st.caption("Auto maps Safe→Floor, Balanced→Mean, Aggro→Ceiling.")
            points_col_opt = resolve_points_col(optimize_target_opt, risk_mode)
            st.caption(f"Optimizer is using **{friendly_lens(points_col_opt)}** (Risk mode: **{risk_mode}**).")

            do_opt = st.button("Optimize lineup", type="primary")

        if do_opt:
            starters, bench, captain, vice = optimize_lineup(
                squad, bench_weight=bench_weight, points_col=points_col_opt
            )

            starter_points = float(starters[points_col_opt].sum())
            bench_points = float(bench[points_col_opt].sum())
            total = starter_points + bench_weight * bench_points + float(captain[points_col_opt])

            with right:
                st.markdown("### Starting XI")
                st.dataframe(
                    starters[[
                        "name", "team_name", "position", "status",
                        "floor_points", "exp_points", "ceiling_points"
                    ]].sort_values(["position", points_col_opt], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Bench (best first)")
                st.dataframe(
                    bench[[
                        "name", "team_name", "position", "status",
                        "floor_points", "exp_points", "ceiling_points"
                    ]].sort_values(points_col_opt, ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Captaincy")
                st.metric("Captain", f"{captain['name']} ({captain['team_name']})", f"{float(captain[points_col_opt]):.2f}")
                st.metric("Vice", f"{vice['name']} ({vice['team_name']})", f"{float(vice[points_col_opt]):.2f}")

                st.markdown("### Objective score")
                st.write(f"Starters: **{starter_points:.2f}**")
                st.write(f"Bench (weighted): **{bench_weight * bench_points:.2f}**")
                st.write(f"Captain bonus: **{float(captain[points_col_opt]):.2f}**")
                st.write(f"**Total:** **{total:.2f}**")

        st.divider()
        st.markdown("**Squad overview**")
        st.caption("Sanity-check safety vs upside and fixture impact.")
        st.dataframe(
            squad[[
                "name", "team_name", "position", "now_cost", "status",
                "avg_fixture_difficulty", "fixture_mult",
                "floor_points", "exp_points", "ceiling_points"
            ]].sort_values(["position", "exp_points"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )

# -------------------- STEP 4: Top Players (by position) --------------------
if step == "4. Top players":
    settings_summary(risk_mode, fixture_horizon, bench_weight, n_matches_std)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top players by position</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Mean expected points over your fixture horizon (fast view).</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Keeping this fast: mean-only over full player pool
    all_mean = add_fixture_adjusted_mean_only(elements, team_diff, team_home_ratio, risk_mode)

    col1, col2, col3 = st.columns([1, 1, 2], gap="large")
    with col1:
        min_minutes = st.slider("Min minutes played", 0, 2000, 0, 100)
        st.caption("Filter out players with tiny samples (optional).")
    with col2:
        only_available = st.checkbox("Only likely available", value=True)
        st.caption("Removes i/s/u and doubtful players.")
    with col3:
        per_pos_n = st.slider("How many per position", 5, 10, 5, 1)
        st.caption("Top N for each of GK/DEF/MID/FWD.")

    df = all_mean.copy()
    if min_minutes > 0:
        df = df[df["minutes"].fillna(0) >= min_minutes]

    if only_available:
        df = df[~df["status"].isin(["i", "s", "u", "d"])]

    df["cost_£m"] = (df["now_cost"] / 10.0).round(1)
    df["exp_points"] = df["exp_points"].round(2)
    df["avg_fixture_difficulty"] = df["avg_fixture_difficulty"].round(2)
    df["fixture_mult"] = df["fixture_mult"].round(3)

    # Render per-position tables
    for pos in POS_ORDER:
        sub = df[df["position"] == pos].sort_values("exp_points", ascending=False).head(per_pos_n).copy()
        st.markdown(f"### {pos}")
        if sub.empty:
            st.info("No players matched your filters.")
            continue

        st.dataframe(
            sub[[
                "name", "team_name", "position", "cost_£m", "status",
                "avg_fixture_difficulty", "fixture_mult",
                "exp_points"
            ]],
            use_container_width=True,
            hide_index=True
        )

# -------------------- STEP 5: Assumptions --------------------
if step == "5. Assumptions":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Projection assumptions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">A plain-language summary of how expected points are computed.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Base expectation (per player)")
    st.markdown(
        """
- **Base points** = blend of long-run PPG and recent form (last ~5 in FPL data).
- **Risk blend (Safe)**: 75% PPG + 25% Form
- **Risk blend (Balanced)**: 68% PPG + 32% Form
- **Risk blend (Aggro)**: 60% PPG + 40% Form
- **Minutes reliability**: `minutes_factor = clamp(minutes / 1800, 0.3..1.0)`
- **Availability**: uses FPL chance-of-playing; Safe weights it more, Aggro less.
- **Status penalty**: i/s/u → ×0.25, d → ×0.75
        """
    )

    st.markdown("### Fixture adjustments")
    st.markdown(
        f"""
- **Fixture horizon**: next **{fixture_horizon} GWs** (GW{gw_fixtures}–GW{gw_fixtures + fixture_horizon - 1})
- **Difficulty multiplier**: linear base around difficulty=3 with a gentle nonlinear kicker (capped `0.82` to `1.18`).
- **Home/away balance**: small conservative adjustment based on % of home fixtures in the horizon (range `0.96` to `1.04`).
        """
    )

    st.markdown("### Default settings")
    st.markdown(
        f"""
- Risk mode: **{risk_mode}**
- Bench importance: **{bench_weight:.2f}**
- Volatility lookback: **{n_matches_std} matches**
        """
    )
