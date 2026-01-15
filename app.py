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
    elements = elements.rename(columns={"team": "team_id", "element_type": "pos_id", "id": "player_id"})
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
def get_current_gw() -> int:
    data = requests.get(FPL_BOOTSTRAP, timeout=30).json()
    events = data["events"]
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return int(current["id"])
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"])
    unfinished = next((e for e in events if not e.get("finished")), None)
    return int(unfinished["id"]) if unfinished else 1


@st.cache_data(ttl=60 * 15)
def load_entry_picks(entry_id: int, gw: int) -> list[int]:
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"Could not load team. HTTP {resp.status_code}. Is the team ID correct and public?")
    data = resp.json()
    picks = data.get("picks", [])
    if not picks:
        raise ValueError("No picks found. This can happen if the season hasn't started or your team is private.")
    return [int(p["element"]) for p in picks]


# -------------------- Fixture difficulty (horizon) --------------------
def fixture_multiplier(avg_difficulty: float) -> float:
    """Difficulty 1..5 -> multiplier about 1.12..0.88."""
    if avg_difficulty is None or pd.isna(avg_difficulty):
        return 1.0
    return 1.00 + (3.0 - float(avg_difficulty)) * 0.06


def team_fixture_difficulty_map_horizon(fixtures_df: pd.DataFrame, gw_start: int, horizon: int) -> dict:
    """
    team_id -> avg difficulty across gw_start..gw_start+horizon-1.
    Handles DGWs by averaging within each GW first.
    """
    if fixtures_df.empty:
        return {}

    gws = list(range(int(gw_start), int(gw_start) + int(horizon)))
    f = fixtures_df[fixtures_df["event"].isin(gws)].copy()
    if f.empty:
        return {}

    home = f[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team_id", "team_h_difficulty": "difficulty"}
    )
    away = f[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team_id", "team_a_difficulty": "difficulty"}
    )

    all_rows = pd.concat([home, away], ignore_index=True)

    per_gw = all_rows.groupby(["team_id", "event"])["difficulty"].mean().reset_index()
    horizon_avg = per_gw.groupby("team_id")["difficulty"].mean().to_dict()
    return horizon_avg


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
        form_weight = 0.10
    elif risk_mode == "Aggro":
        chance_weight = 0.8
        minutes_weight = 0.9
        form_weight = 0.20
    else:
        chance_weight = 1.0
        minutes_weight = 1.0
        form_weight = 0.15

    exp = ppg + form_weight * form
    exp *= (chance ** chance_weight)
    exp *= (minutes_factor ** minutes_weight)

    status = str(row["status"])
    if status in ["i", "s", "u"]:
        exp *= 0.25
    elif status == "d":
        exp *= 0.75

    return max(0.0, float(exp))


def add_fixture_adjusted_mean_only(df: pd.DataFrame, team_diff_map: dict, risk_mode: str) -> pd.DataFrame:
    """
    Fast path: computes ONLY mean expected points (fixture-adjusted).
    No element-summary calls (so safe to run over all players).
    """
    out = df.copy()
    out["base_xPts"] = out.apply(lambda r: expected_points_v1(r, risk_mode), axis=1)
    out["avg_fixture_difficulty"] = out["team_id"].map(team_diff_map)
    out["fixture_mult"] = out["avg_fixture_difficulty"].apply(fixture_multiplier)
    out["exp_points"] = out["base_xPts"] * out["fixture_mult"]
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
    risk_mode: str,
    n_matches_std: int = 6
) -> pd.DataFrame:
    """
    Adds columns:
      - exp_points (mean fixture-adjusted)
      - floor_points / ceiling_points using volatility proxy from recent match points
    """
    out = add_fixture_adjusted_mean_only(df, team_diff_map, risk_mode)

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

    base_starters, base_bench, base_capt, base_vice = optimize_lineup(
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

    df = pd.DataFrame(results).sort_values("incremental_xPts", ascending=False).head(top_n_results)
    return df


def style_transfer_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Green for positive, red for negative
    def color_delta(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #0f766e; font-weight: 700;"  # teal-ish
        if val < 0:
            return "color: #b91c1c; font-weight: 700;"  # red-ish
        return "color: #334155;"

    styler = df.style.format({
        "sell_cost_£m": "{:.1f}",
        "buy_cost_£m": "{:.1f}",
        "bank_used_£m": "{:.1f}",
        "incremental_xPts": "{:+.2f}",
    })
    styler = styler.applymap(color_delta, subset=["incremental_xPts"])
    return styler


# -------------------- UI --------------------
st.set_page_config(page_title="FPL Lineup Optimizer", page_icon="⚽", layout="wide")
st.title("⚽ FPL Assistant")

# Shared state
if "squad_ids" not in st.session_state:
    st.session_state.squad_ids = None
if "squad_df" not in st.session_state:
    st.session_state.squad_df = None
if "transfer_df" not in st.session_state:
    st.session_state.transfer_df = None
if "entry_error" not in st.session_state:
    st.session_state.entry_error = None

elements = load_bootstrap()

with st.sidebar:
    st.header("Global settings")
    st.caption("These affect projections everywhere (transfers + optimize + top players).")

    risk_mode = st.selectbox("Risk mode", ["Safe", "Balanced", "Aggro"], index=1)
    st.caption("Safe favors reliability; Aggro favors upside.")

    bench_weight = st.slider("Bench importance", 0.0, 0.3, 0.10, 0.01)
    st.caption("How much bench points matter in the objective score (useful for Bench Boost).")

    fixture_horizon = st.slider("Fixture horizon (GWs)", 1, 6, 3, 1)
    st.caption("How many upcoming gameweeks to average fixture difficulty across.")

    n_matches_std = st.slider("Volatility lookback (matches)", 4, 10, 6, 1)
    st.caption("Used for floor/ceiling. Larger = steadier estimate.")

gw = get_current_gw()
fixtures = load_fixtures()
team_diff = team_fixture_difficulty_map_horizon(fixtures, gw_start=gw, horizon=fixture_horizon)
st.caption(
    f"Anchor gameweek: **GW{gw}** • "
    f"Fixture window: **GW{gw}–GW{gw + fixture_horizon - 1}**"
)


tab_load, tab_transfers, tab_opt, tab_top = st.tabs(
    ["🧩 Load team", "🔁 Transfers", "🚀 Optimize & captaincy", "📈 Top 10 players"]
)

# -------------------- TAB 1: Load team --------------------
with tab_load:
    st.subheader("Load or select your 15-man squad")
    st.caption("Load your current squad using your FPL Team ID (Entry ID), or manually pick 15 players.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Load via Team ID (Entry ID)")
        st.caption("The Team ID is a number in the FPL website URL.")

        with st.expander("How to find your Team ID (Entry ID)", expanded=True):
            st.markdown(
                """
**Method 1 (fastest): copy it from the URL**
1. Open the Fantasy Premier League website and log in.
2. Go to **Points** (or **Pick Team**) for your squad.
3. Look at your browser address bar — your URL will include `/entry/<NUMBER>/`.

**Generic formats**
- `https://fantasy.premierleague.com/entry/<ENTRY_ID>/event/<GW>/points`
- `https://fantasy.premierleague.com/entry/<ENTRY_ID>/`

**Example**
- If you see: `.../entry/1234567/event/22/points`
- Then your Team ID is: **1234567**

**Troubleshooting**
- Mobile app: open the web version in a browser (or use “share/open in browser”), then copy the URL.
- Team ID is **numbers only**.
                """
            )

        # Entry ID input (can be overwritten by URL extraction below)
        entry_id = st.text_input("FPL Team ID (Entry ID)", value="", placeholder="e.g., 1234567")

        # URL auto-extract
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

        load_team = st.button("⬇️ Load my squad", type="primary")
        if load_team:
            try:
                if not entry_id.strip().isdigit():
                    raise ValueError("Please enter a numeric Team ID (Entry ID).")
                prefill_ids = load_entry_picks(int(entry_id.strip()), gw)
                st.session_state.squad_ids = prefill_ids
                st.session_state.entry_error = None
                st.session_state.transfer_df = None
            except Exception as e:
                st.session_state.entry_error = str(e)

        if st.session_state.entry_error:
            st.error(st.session_state.entry_error)

        st.markdown("---")
        st.markdown("**Manual mode**")
        st.caption("Use the selector on the right to pick exactly 15.")

    with right:
        all_labels = elements["name"] + " — " + elements["team_name"] + " (" + elements["position"] + ")"
        label_to_id = dict(zip(all_labels, elements["player_id"]))
        id_to_label = dict(zip(elements["player_id"], all_labels))

        default_selected_labels = []
        if st.session_state.squad_ids:
            default_selected_labels = [id_to_label[i] for i in st.session_state.squad_ids if i in id_to_label]

        selected = st.multiselect(
            "Search and select players (choose exactly 15)",
            options=all_labels.tolist(),
            default=default_selected_labels,
        )

        if len(selected) != 15:
            st.info(f"Select exactly 15 players. Currently selected: {len(selected)}")
            st.session_state.squad_df = None
        else:
            selected_ids = [label_to_id[s] for s in selected]
            st.session_state.squad_ids = selected_ids

            squad_raw = elements[elements["player_id"].isin(selected_ids)].copy()
            squad = add_fixture_adjusted_xpts(squad_raw, team_diff, risk_mode, n_matches_std=n_matches_std)
            st.session_state.squad_df = squad

            st.success("Squad saved. Head to the Transfers or Optimize tabs.")

            st.markdown("**Squad preview (Floor / Mean / Ceiling)**")
            st.dataframe(
                squad[[
                    "name", "team_name", "position", "now_cost", "status",
                    "avg_fixture_difficulty", "fixture_mult",
                    "floor_points", "exp_points", "ceiling_points"
                ]].sort_values(["position", "exp_points"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True
            )

# -------------------- TAB 2: Transfers --------------------
with tab_transfers:
    st.subheader("Transfer recommendations (lineup-aware)")
    st.caption("Evaluates each 1-transfer move by re-optimizing your XI and captaincy after the transfer.")

    squad = st.session_state.squad_df
    if squad is None or len(squad) == 0:
        st.info("Load or select a 15-man squad in the **Load team** tab first.")
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
            st.caption("Limits how many players we consider for transfers (keeps it fast).")

        points_col = resolve_points_col(optimize_target, risk_mode)
        lens_name = friendly_lens(points_col)

        st.caption(
            f"**Incremental expected points** = (optimized team after transfer) − (current optimized team), "
            f"using the **{lens_name}** lens."
        )

        run = st.button("🔎 Find best transfer", type="primary")
        if run:
            candidate_pool = elements.sort_values("points_per_game", ascending=False).head(candidate_pool_size).copy()
            all_with_xp = add_fixture_adjusted_xpts(candidate_pool, team_diff, risk_mode, n_matches_std=n_matches_std)

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

# -------------------- TAB 3: Optimize & captaincy --------------------
with tab_opt:
    st.subheader("Optimize XI, bench, and captaincy")
    st.caption("Picks the best legal XI + bench + captain based on your chosen target (floor/mean/ceiling).")

    squad = st.session_state.squad_df
    if squad is None or len(squad) == 0:
        st.info("Load or select a 15-man squad in the **Load team** tab first.")
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

            do_opt = st.button("🚀 Optimize lineup", type="primary")

        if do_opt:
            starters, bench, captain, vice = optimize_lineup(squad, bench_weight=bench_weight, points_col=points_col_opt)

            starter_points = float(starters[points_col_opt].sum())
            bench_points = float(bench[points_col_opt].sum())
            total = starter_points + bench_weight * bench_points + float(captain[points_col_opt])

            with right:
                st.markdown("### ✅ Starting XI")
                st.dataframe(
                    starters[[
                        "name", "team_name", "position", "status",
                        "floor_points", "exp_points", "ceiling_points"
                    ]].sort_values(["position", points_col_opt], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### 🪑 Bench (best first)")
                st.dataframe(
                    bench[[
                        "name", "team_name", "position", "status",
                        "floor_points", "exp_points", "ceiling_points"
                    ]].sort_values(points_col_opt, ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### 🧢 Captaincy")
                st.metric("Captain", f"{captain['name']} ({captain['team_name']})", f"{float(captain[points_col_opt]):.2f}")
                st.metric("Vice", f"{vice['name']} ({vice['team_name']})", f"{float(vice[points_col_opt]):.2f}")

                st.markdown("### 🧮 Objective score")
                st.write(f"Starters: **{starter_points:.2f}**")
                st.write(f"Bench (weighted): **{bench_weight * bench_points:.2f}**")
                st.write(f"Captain bonus: **{float(captain[points_col_opt]):.2f}**")
                st.write(f"**Total:** **{total:.2f}**")

        st.markdown("---")
        st.markdown("**Squad overview**")
        st.caption("Sanity-check who is ‘safe’ vs ‘spiky’ and how fixtures are affecting projections.")
        st.dataframe(
            squad[[
                "name", "team_name", "position", "now_cost", "status",
                "avg_fixture_difficulty", "fixture_mult",
                "floor_points", "exp_points", "ceiling_points"
            ]].sort_values(["position", "exp_points"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )

# -------------------- TAB 4: Top 10 players --------------------
with tab_top:
    st.subheader("Top 10 players by expected points")
    st.caption("Shows the 10 players with the highest **mean** expected points (fixture-adjusted) over your selected fixture horizon.")

    # Fast computation over all players (mean only)
    all_mean = add_fixture_adjusted_mean_only(elements, team_diff, risk_mode)

    col1, col2, col3 = st.columns([1, 1, 2], gap="large")
    with col1:
        min_minutes = st.slider("Min minutes played", 0, 2000, 0, 100)
        st.caption("Filter out players with tiny samples (optional).")
    with col2:
        only_available = st.checkbox("Only likely available", value=True)
        st.caption("Removes i/s/u and doubtful players.")
    with col3:
        st.caption("Tip: this tab is intentionally fast (no floor/ceiling). Floor/ceiling requires extra match-history fetches.")

    df = all_mean.copy()
    if min_minutes > 0:
        df = df[df["minutes"].fillna(0) >= min_minutes]

    if only_available:
        df = df[~df["status"].isin(["i", "s", "u", "d"])]

    top10 = (
        df.sort_values("exp_points", ascending=False)
          .head(10)
          .copy()
    )

    top10["cost_£m"] = (top10["now_cost"] / 10.0).round(1)
    top10["exp_points"] = top10["exp_points"].round(2)
    top10["avg_fixture_difficulty"] = top10["avg_fixture_difficulty"].round(2)
    top10["fixture_mult"] = top10["fixture_mult"].round(3)

    st.dataframe(
        top10[[
            "name", "team_name", "position", "cost_£m", "status",
            "avg_fixture_difficulty", "fixture_mult",
            "exp_points"
        ]],
        use_container_width=True,
        hide_index=True
    )
