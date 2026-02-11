import os
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import requests
import streamlit as st


# ----------------------------
# Keys
# ----------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w342"


# ----------------------------
# Catalogs (mainstream)
# ----------------------------
PROVIDER_NAME_TO_ID = {
    "Netflix": 8,
    "Hulu": 15,
    "Prime Video": 9,
    "Disney+": 337,
    "Max": 1899,
    "Apple TV+": 350,
    "Paramount+": 531,
    "Peacock": 386,
}

GENRES = {
    "Action": 28,
    "Comedy": 35,
    "Drama": 18,
    "Horror": 27,
    "Romance": 10749,
    "Sci-Fi": 878,
    "Thriller": 53,
    "Animation": 16,
    "Documentary": 99,
}

REGIONS = ["US", "CA", "GB", "AU"]


# ----------------------------
# TMDB / OMDb helpers
# ----------------------------
def tmdb_get(path: str, params: Optional[dict] = None) -> dict:
    if not TMDB_API_KEY:
        raise RuntimeError("Missing TMDB_API_KEY env var.")
    params = params or {}
    params["api_key"] = TMDB_API_KEY
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60 * 60)
def tmdb_search_movie(title: str) -> Optional[dict]:
    data = tmdb_get("/search/movie", {"query": title, "include_adult": "false"})
    results = data.get("results", [])
    return results[0] if results else None


@st.cache_data(ttl=60 * 60)
def tmdb_movie_external_ids(movie_id: int) -> dict:
    return tmdb_get(f"/movie/{movie_id}/external_ids")


@st.cache_data(ttl=60 * 60)
def tmdb_movie_watch_providers(movie_id: int) -> dict:
    return tmdb_get(f"/movie/{movie_id}/watch/providers")


@st.cache_data(ttl=60 * 30)
def tmdb_recommendations(movie_id: int, pages: int = 2) -> List[dict]:
    out: List[dict] = []
    for p in range(1, pages + 1):
        data = tmdb_get(f"/movie/{movie_id}/recommendations", {"page": p})
        out.extend(data.get("results", []))
    return out


@st.cache_data(ttl=60 * 30)
def tmdb_discover_movies(
    region: str,
    genre_id: Optional[int],
    provider_ids: List[int],
    pages: int = 3,
) -> List[dict]:
    out: List[dict] = []
    for p in range(1, pages + 1):
        params = {
            "sort_by": "popularity.desc",
            "watch_region": region,
            "with_watch_providers": "|".join(str(x) for x in provider_ids) if provider_ids else None,
            "with_genres": str(genre_id) if genre_id else None,
            "include_adult": "false",
            "vote_count.gte": 150,
            "page": p,
        }
        params = {k: v for k, v in params.items() if v is not None}
        data = tmdb_get("/discover/movie", params)
        out.extend(data.get("results", []))
    return out


def safe_float(x: str) -> Optional[float]:
    try:
        x = (x or "").strip()
        if not x or x == "N/A":
            return None
        return float(x)
    except Exception:
        return None


@st.cache_data(ttl=60 * 60)
def omdb_lookup(imdb_id: str) -> dict:
    if not OMDB_API_KEY:
        return {}
    r = requests.get(
        "https://www.omdbapi.com/",
        params={"apikey": OMDB_API_KEY, "i": imdb_id},
        timeout=20,
    )
    if r.status_code != 200:
        return {}
    data = r.json()
    if data.get("Response") != "True":
        return {}
    return data


def google_link(query: str) -> str:
    return "https://www.google.com/search?q=" + urllib.parse.quote(query)


def extract_provider_ids_for_region(watch_data: dict, region: str) -> Set[int]:
    region_data = (watch_data.get("results") or {}).get(region) or {}
    flatrate = region_data.get("flatrate") or []
    return {p.get("provider_id") for p in flatrate if p.get("provider_id") is not None}


def movie_available_on_selected_services(movie_id: int, region: str, selected_provider_ids: Set[int]) -> bool:
    if not selected_provider_ids:
        return True
    watch = tmdb_movie_watch_providers(movie_id)
    ids_here = extract_provider_ids_for_region(watch, region)
    return len(ids_here & selected_provider_ids) > 0


# ----------------------------
# UI components: service pills + tag chips
# ----------------------------
def inject_styles() -> None:
    st.markdown(
        """
<style>
/* --- Service pills (toggle buttons) --- */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
  border: 1px solid rgba(49, 51, 63, 0.2);
}

/* --- Chips for liked movies --- */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.18);
  margin: 0.2rem 0.25rem 0.2rem 0;
  font-size: 0.9rem;
  background: rgba(49, 51, 63, 0.04);
}
.chip button {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 1rem;
  line-height: 1;
  padding: 0;
}
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.9rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def service_pills(all_services: List[str], selected: Set[str]) -> Set[str]:
    """
    Render services as pill-style toggle buttons. Returns updated selection.
    """
    st.write("Streaming services")
    cols = st.columns(4)
    updated = set(selected)

    for i, s in enumerate(all_services):
        with cols[i % 4]:
            is_on = s in updated
            label = f"✅ {s}" if is_on else s
            if st.button(label, key=f"svc_{s}", type="secondary"):
                if is_on:
                    updated.remove(s)
                else:
                    updated.add(s)
    return updated


def render_like_chips(titles: List[str]) -> None:
    """
    Show liked titles as chips with remove buttons.
    """
    if not titles:
        st.markdown('<div class="small-muted">No “like” movies added yet.</div>', unsafe_allow_html=True)
        return

    # Render chips in rows (simple approach)
    for t in titles:
        c1, c2 = st.columns([10, 1])
        with c1:
            st.markdown(f'<span class="chip">{t}</span>', unsafe_allow_html=True)
        with c2:
            if st.button("✕", key=f"rm_{t}"):
                st.session_state.like_titles = [x for x in st.session_state.like_titles if x != t]
                st.rerun()


# ----------------------------
# Recommendation model
# ----------------------------
@dataclass
class Rec:
    tmdb_id: int
    title: str
    year: Optional[int]
    overview: str
    poster_url: Optional[str]
    tmdb_vote: Optional[float]
    imdb_rating: Optional[float]
    score: float


def build_rec(movie: dict, like_bonus: float = 0.0) -> Optional[Rec]:
    tmdb_id = movie.get("id")
    title = movie.get("title") or ""
    if not tmdb_id or not title:
        return None

    release_date = movie.get("release_date") or ""
    year = int(release_date[:4]) if release_date[:4].isdigit() else None
    overview = movie.get("overview") or ""
    poster_path = movie.get("poster_path")
    poster_url = f"{TMDB_IMG}{poster_path}" if poster_path else None
    tmdb_vote = movie.get("vote_average")

    imdb_rating = None
    if OMDB_API_KEY:
        ext = tmdb_movie_external_ids(tmdb_id)
        imdb_id = ext.get("imdb_id")
        if imdb_id:
            om = omdb_lookup(imdb_id)
            imdb_rating = safe_float(om.get("imdbRating"))

    base = float(tmdb_vote or 0.0)
    quality = 0.25 if imdb_rating is not None else 0.0
    score = base + like_bonus + quality

    return Rec(
        tmdb_id=tmdb_id,
        title=title,
        year=year,
        overview=overview,
        poster_url=poster_url,
        tmdb_vote=tmdb_vote,
        imdb_rating=imdb_rating,
        score=score,
    )


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="What should I watch?", page_icon="🎬", layout="wide")
inject_styles()

st.title("🎬 What should I watch?")
st.caption("Choose services + genre + scores, add a few movies you like, and get ranked picks.")


# Session state setup
if "like_titles" not in st.session_state:
    st.session_state.like_titles = []
if "selected_services" not in st.session_state:
    st.session_state.selected_services = {"Netflix", "Prime Video"}


with st.sidebar:
    st.header("Filters")
    region = st.selectbox("Region", REGIONS, index=0)

    # Services as pills
    st.session_state.selected_services = service_pills(
        list(PROVIDER_NAME_TO_ID.keys()),
        st.session_state.selected_services,
    )

    genre = st.selectbox("Genre", options=["(Any)"] + list(GENRES.keys()), index=0)
    genre_id = None if genre == "(Any)" else GENRES[genre]

    st.subheader("Score sliders")
    imdb_min = st.slider("IMDb minimum", 0.0, 10.0, 6.5, 0.1)

    # UI-only by default (until you add a data source)
    rt_min = st.slider("Rotten Tomatoes minimum (optional)", 0, 100, 60, 1)
    lb_min = st.slider("Letterboxd minimum (optional)", 0.0, 5.0, 3.5, 0.1)

    st.divider()
    n_results = st.slider("How many recommendations?", 5, 30, 12, 1)

# Like movies tag UI
st.subheader("Movies you like (tags)")
cA, cB = st.columns([3, 1])
with cA:
    new_like = st.text_input("Add a movie", placeholder="e.g., Heat", label_visibility="collapsed")
with cB:
    if st.button("Add", type="secondary"):
        t = (new_like or "").strip()
        if t and t not in st.session_state.like_titles:
            st.session_state.like_titles.append(t)
            st.rerun()

render_like_chips(st.session_state.like_titles)

st.divider()

go = st.button("Recommend 🍿", type="primary")


if go:
    selected_services = st.session_state.selected_services
    selected_provider_ids = {PROVIDER_NAME_TO_ID[s] for s in selected_services}

    rec_pool: Dict[int, Rec] = {}

    with st.spinner("Building recommendations..."):
        # 1) Personalized from likes
        liked_tmdb_ids: List[int] = []
        for t in st.session_state.like_titles:
            hit = tmdb_search_movie(t)
            if hit:
                liked_tmdb_ids.append(hit["id"])

        for mid in liked_tmdb_ids:
            for m in tmdb_recommendations(mid, pages=2):
                r = build_rec(m, like_bonus=1.25)
                if not r:
                    continue
                if r.tmdb_id not in rec_pool or r.score > rec_pool[r.tmdb_id].score:
                    rec_pool[r.tmdb_id] = r

        # 2) Broad discover (fills gaps)
        for m in tmdb_discover_movies(region=region, genre_id=genre_id, provider_ids=list(selected_provider_ids), pages=3):
            r = build_rec(m, like_bonus=0.0)
            if not r:
                continue
            if r.tmdb_id not in rec_pool or r.score > rec_pool[r.tmdb_id].score:
                rec_pool[r.tmdb_id] = r

        # 3) Filter
        filtered: List[Rec] = []
        for r in rec_pool.values():
            # Best-effort: enforce service availability using TMDB providers
            if not movie_available_on_selected_services(r.tmdb_id, region, selected_provider_ids):
                continue

            # IMDb min: enforce only if we have an IMDb rating
            if OMDB_API_KEY and r.imdb_rating is not None and r.imdb_rating < imdb_min:
                continue

            filtered.append(r)

        filtered.sort(key=lambda x: x.score, reverse=True)
        top = filtered[:n_results]

    if not top:
        st.warning("No matches found. Try selecting fewer services or lowering the IMDb minimum.")
    else:
        st.success(f"Top {len(top)} picks based on your filters.")

        for r in top:
            cols = st.columns([1, 3, 2])

            with cols[0]:
                if r.poster_url:
                    st.image(r.poster_url)
                else:
                    st.write("🖼️ No poster")

            with cols[1]:
                title_line = f"**{r.title}**" + (f" ({r.year})" if r.year else "")
                st.markdown(title_line)
                if r.overview:
                    st.caption(r.overview[:220] + ("…" if len(r.overview) > 220 else ""))

                tmdb_txt = f"TMDB: {r.tmdb_vote:.1f}/10" if r.tmdb_vote is not None else "TMDB: n/a"
                imdb_txt = f"IMDb: {r.imdb_rating:.1f}/10" if r.imdb_rating is not None else "IMDb: n/a"
                st.write(f"{tmdb_txt} • {imdb_txt}")

            with cols[2]:
                st.markdown("**Links**")
                st.link_button("IMDb (Google)", google_link(f"{r.title} imdb"))
                if selected_services:
                    # pick one (first) to keep it simple
                    svc = sorted(list(selected_services))[0]
                    st.link_button("Where to watch (Google)", google_link(f"{r.title} watch on {svc}"))
                else:
                    st.link_button("Where to watch (Google)", google_link(f"{r.title} where to watch"))

        st.caption(
            "RT/Letterboxd sliders are UI placeholders until you connect a data source for those scores. "
            "IMDb filtering is applied when OMDb is configured."
        )
else:
    st.info("Pick your services + genre, add a couple movies you like, then hit **Recommend 🍿**.")
