# app.py
from groq import Groq

import os
import shutil
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer

import re
import json
import calendar
import feedparser
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from dateutil.relativedelta import relativedelta
from time import perf_counter

st.set_page_config("Release-Notes Chat", "💬")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: #0b1020;
  color: #f3f6ff;
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  max-width: 980px;
  padding-top: 1.2rem;
}

[data-testid="stChatInput"] {
  position: sticky;
  bottom: 10px;
}

[data-testid="stChatInput"] > div {
  background: rgba(18, 24, 40, 0.95);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
}

.query-card {
  background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 10px;
  font-weight: 600;
}

.live-chip {
  color: #b9c2d9;
  font-size: 0.92rem;
  margin: 4px 0 10px 2px;
}

.answer-card {
  background: transparent;
  border-radius: 14px;
  padding: 0;
}

.answer-card p, .answer-card li {
  line-height: 1.7;
}

.answer-card ul {
  margin-top: 0.4rem;
}

a {
  color: #8fb2ff !important;
  text-decoration: none !important;
}

a:hover {
  text-decoration: underline !important;
}

[data-testid="stCaptionContainer"] {
  color: #b9c2d9;
}
</style>
""", unsafe_allow_html=True)

# ── creds ────────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = ""

if not GROQ_API_KEY:
    st.error("Set GROQ_API_KEY in your .env or Streamlit secrets")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── data / embedding config ─────────────────────────────────────────────────
CSV_PATH = "SoftwareUpdateSurvey.csv"
OS_API = "https://releasetrain.io/api/component?q=os"
REDDIT_API = "https://releasetrain.io/api/reddit"
MAX_OS, MAX_RED = 50, 50
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
DATA_DIR = "release_notes_store"
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")

CACHE_DIR = Path(".live_cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------------- one-time status chips ---------------------------
if "live_marks" not in st.session_state:
    st.session_state.live_marks = set()


def mark_live(name: str):
    if name not in st.session_state.live_marks:
        st.session_state.live_marks.add(name)
        st.caption(f"✓ {name}: live")


# -------------------------- utilities / fetch -------------------------------
def _normalize_results(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return []


def clean_html(text: str):
    text = text or ""
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_json(url: str, name: str, headers: dict | None = None):
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{url}".lower()).strip("_")
    cache_path = CACHE_DIR / f"{safe}.json"

    sess = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.7,
        status_forcelist=(502, 503, 504),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    try:
        base_headers = {"User-Agent": "ReleaseNotesRec/1.0"}
        if headers:
            base_headers.update(headers)

        r = sess.get(url, timeout=12, headers=base_headers)

        if r.status_code == 200:
            data = r.json()
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            return data

    except Exception:
        pass

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    return None


# ---------------- natural-language time & filters ---------------------------
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_WEEK_REX = re.compile(r"\bweek\s+(\d{1,2})\s+of\s+(\d{4})\b", re.I)


def _as_utc(dt):
    if not dt.tzinfo:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_isoish(s: str | None):
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00").split(".")[0]
        return _as_utc(datetime.fromisoformat(s))
    except Exception:
        return None


def parse_time_window(q: str, now=None):
    q = (q or "").strip()
    now = _as_utc(now or datetime.now(timezone.utc))
    rel = q.lower()

    if "yesterday" in rel:
        d = now.date() - timedelta(days=1)
        return (
            datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc),
        )

    if "last week" in rel:
        monday = (now - timedelta(days=now.weekday() + 7)).date()
        sunday = monday + timedelta(days=6)
        return (
            datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc),
        )

    if "this week" in rel:
        monday = (now - timedelta(days=now.weekday())).date()
        sunday = monday + timedelta(days=6)
        return (
            datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc),
        )

    if "last month" in rel:
        first = (now.replace(day=1) - relativedelta(months=1)).date()
        last = (now.replace(day=1) - timedelta(days=1)).date()
        return (
            datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(last, datetime.max.time(), tzinfo=timezone.utc),
        )

    if "this month" in rel:
        first = now.replace(day=1).date()
        last_day = calendar.monthrange(now.year, now.month)[1]
        last = datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc), last

    if "last year" in rel:
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year - 1, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    if "this year" in rel:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    if "last quarter" in rel:
        current_quarter = (now.month - 1) // 3 + 1
        year = now.year
        last_quarter = current_quarter - 1
        if last_quarter == 0:
            last_quarter = 4
            year -= 1
        start_month = (last_quarter - 1) * 3 + 1
        end_month = start_month + 2
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        last_day = calendar.monthrange(year, end_month)[1]
        end = datetime(year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    m = _WEEK_REX.search(q)
    if m:
        wk = int(m.group(1))
        yr = int(m.group(2))
        monday = datetime.fromisocalendar(yr, wk, 1).replace(tzinfo=timezone.utc)
        sunday = datetime.fromisocalendar(yr, wk, 7).replace(
            tzinfo=timezone.utc, hour=23, minute=59, second=59
        )
        return monday, sunday

    for name, idx in _MONTHS.items():
        m2 = re.search(rf"\b{name}\b\s+(\d{{4}})", q, re.I)
        if m2:
            yr = int(m2.group(1))
            first = datetime(yr, idx, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(yr, idx)[1]
            last = datetime(yr, idx, last_day, 23, 59, 59, tzinfo=timezone.utc)
            return first, last

    rng = re.search(
        r"(between|from)\s+([A-Za-z0-9,\-\s/]+)\s+(and|to)\s+([A-Za-z0-9,\-\s/]+)",
        q,
        re.I,
    )
    if rng:
        def _try_dt(t):
            for fmt in ("%Y-%m-%d", "%b %d, %Y", "%Y/%m/%d"):
                try:
                    return _as_utc(datetime.strptime(t.strip(), fmt))
                except Exception:
                    pass
            return None

        s = _try_dt(rng.group(2))
        e = _try_dt(rng.group(4))
        if s and e and s <= e:
            return s, e + timedelta(hours=23, minutes=59, seconds=59)

    mdate = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", q)
    if mdate:
        d = datetime.strptime(mdate.group(1), "%Y-%m-%d").date()
        return (
            datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc),
        )

    y = re.search(r"\b(20\d{2}|19\d{2})\b", q)
    if y:
        yr = int(y.group(1))
        return (
            datetime(yr, 1, 1, tzinfo=timezone.utc),
            datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

    return None


def is_release_query(q: str):
    ql = (q or "").lower()
    release_terms = {
        "release", "releases", "version", "versions", "update",
        "updates", "patch", "patches", "announcement", "history"
    }
    return any(term in ql for term in release_terms)


def is_rc_query(q: str):
    ql = (q or "").lower()
    return " rc " in f" {ql} " or "release candidate" in ql or "rc builds" in ql


def extract_vendors(q: str):
    if not q:
        return []

    ql = q.lower()
    alias_map = {
        "postgresql": "postgresql",
        "postgres": "postgres",
        "postgre": "postgres",
        "k8s": "kubernetes",
        "kubernetes": "kubernetes",
        "docker": "docker",
        "grafana": "grafana",
        "redis": "redis",
        "cpython": "python",
        "python": "python",
        "golang": "golang",
        "go": "golang",
        "mongodb": "mongodb",
        "mongo": "mongodb",
        "mysql": "mysql",
        "nginx": "nginx",
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "pytorch": "pytorch",
        "torch": "pytorch",
        "ubuntu": "ubuntu",
        "linux": "linux",
        "kernel": "kernel",
        "windows": "windows",
        "debian": "debian",
        "android": "android",
        "ios": "ios",
        "macos": "macos",
        "nvidia": "nvidia",
        "node.js": "node",
        "nodejs": "node",
        "node": "node",
        "openssl": "openssl",
        "microsoft": "windows",
    }

    found = []
    for alias, canonical in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", ql):
            found.append(canonical)

    if not found:
        if "python" in ql:
            found.append("python")
        elif "grafana" in ql:
            found.append("grafana")
        elif "kubernetes" in ql or "k8s" in ql:
            found.append("kubernetes")
        elif "redis" in ql:
            found.append("redis")
        elif "docker" in ql:
            found.append("docker")
        elif "node" in ql:
            found.append("node")
        elif "tensorflow" in ql:
            found.append("tensorflow")

    seen = set()
    out = []
    for v in found:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def filter_by_time_and_vendor(items, start_end, vendors):
    def _dt(it):
        return _parse_isoish(
            it.get("updatedAt")
            or it.get("createdAt")
            or it.get("date")
            or it.get("published")
            or it.get("published_at")
            or it.get("created_utc")
        )

    out = []
    for it in items:
        dt = _dt(it)
        if start_end:
            s, e = start_end
            if not dt or not (s <= dt <= e):
                continue
        if vendors:
            hay = " ".join(
                str(it.get(k, "")) for k in (
                    "title", "name", "versionProductName", "versionReleaseNotes",
                    "summary", "description", "repo"
                )
            ).lower()
            if not any(v in hay for v in vendors):
                continue
        out.append(it)
    return out


def filter_rc_only(items):
    out = []
    for it in items:
        title = (it.get("title") or "").lower()
        summary = (it.get("summary") or "").lower()
        if "rc" in title or "release candidate" in summary:
            out.append(it)
    return out


def build_grounded_answer(title, items, limit=5):
    if not items:
        return ""

    lines = []
    for it in items[:limit]:
        t = clean_html(it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled")
        url = it.get("url") or it.get("link") or ""

        t = re.sub(r"Download page.*", "", t)
        t = re.sub(r"What's new.*", "", t)

        dt = _parse_isoish(
            it.get("updatedAt")
            or it.get("createdAt")
            or it.get("date")
            or it.get("published")
            or it.get("published_at")
            or it.get("created_utc")
        )
        ds = dt.date().isoformat() if dt else ""

        if url:
            lines.append(f"- **{t.strip()}** ({ds}) [source]({url})")
        else:
            lines.append(f"- **{t.strip()}** ({ds})")

    return "\n".join(lines)


# ------------------------------ ingestion (RAG) -----------------------------
def load_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    return [
        {"text": "\n".join(f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]))}
        for _, row in df.iterrows()
    ]


def fetch(url, max_items, mapping, name):
    raw = _get_json(url, name=name)
    if raw is None:
        return []

    data = _normalize_results(raw) or (raw if isinstance(raw, list) else [])
    return [
        {"text": "\n".join(f"{k}: {item.get(v, '')}" for k, v in mapping.items())}
        for item in data[:max_items]
    ]


def build_store():
    docs = load_csv(CSV_PATH)
    docs += fetch(
        OS_API,
        MAX_OS,
        {
            "OS_ID": "_id",
            "OS_Name": "versionProductName",
            "OS_ReleaseNotes": "versionReleaseNotes",
        },
        name="os",
    )
    docs += fetch(
        REDDIT_API,
        MAX_RED,
        {
            "REDDIT_ID": "_id",
            "Subreddit": "subreddit",
            "Title": "title",
            "URL": "url",
        },
        name="reddit",
    )

    model = SentenceTransformer(EMB_MODEL)
    ds = DatasetDict({"train": Dataset.from_dict({"text": [d["text"] for d in docs]})})
    ds = ds.map(
        lambda b: {"embeddings": model.encode(b["text"], batch_size=16, show_progress_bar=False)},
        batched=True,
        batch_size=16,
    )

    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    ds.save_to_disk(DATA_DIR)
    ds["train"].add_faiss_index("embeddings")
    ds["train"].save_faiss_index("embeddings", FAISS_PATH)


def load_store():
    ds = load_from_disk(DATA_DIR)
    ds["train"].load_faiss_index("embeddings", FAISS_PATH)
    return SentenceTransformer(EMB_MODEL), ds


@st.cache_resource(show_spinner="Loading vector store…")
def get_store():
    if not os.path.exists(DATA_DIR) or not os.path.exists(FAISS_PATH):
        build_store()
        return load_store()
    try:
        return load_store()
    except Exception:
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        build_store()
        return load_store()


embedder, datastore = get_store()


def retrieve(query, k):
    emb = embedder.encode(query, show_progress_bar=False)
    _, ex = datastore["train"].get_nearest_examples("embeddings", emb, k=k)
    return ex["text"]


SYSTEM_PROMPT = (
    "Answer using the provided context. Prefer live vendor-routed results as the primary source of truth. "
    "Use RAG context to add background or clarification, not to override live facts. "
    "Do not invent facts."
)


def call_llm(msgs):
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Groq call failed: {e}")
        return ""


def make_msgs(user_q, ctx_docs):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "\n\n".join(
                f"Document {i+1}:\n{d[:500]}" for i, d in enumerate(ctx_docs[:3])
            ),
        },
        {"role": "user", "content": user_q},
    ]


def combine_live_and_rag(user_q, live_answer, rag_answer):
    prompt = f"""
You are answering a software update question.

User query:
{user_q}

Live vendor-routed results:
{live_answer}

RAG context:
{rag_answer}

Instructions:
- Prefer live vendor-routed results as the primary source of truth.
- Use RAG only to add helpful context.
- If live results exist, summarize them confidently.
- If exact matches are unavailable but close vendor matches exist, present them as the closest relevant results.
- Do not say the information is unavailable unless both live and RAG are empty.
- Write one polished final answer in a single section.
- Use 1 short intro sentence and then 2 to 5 bullet points when appropriate.
- Preserve inline markdown hyperlinks like [source](...) when useful.
- Do not create a separate Sources section.
- Keep the tone concise and presentation-ready.
"""
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt[:4000]}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Groq summary failed: {e}")
        return live_answer or rag_answer or "No answer available."


@st.cache_data(ttl=600, show_spinner=False)
def fetch_github_releases(repo: str, limit: int = 30):
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {"Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return [
                {
                    "repo": repo,
                    "title": rel.get("name") or rel.get("tag_name") or "Untitled",
                    "summary": (rel.get("body") or "")[:500],
                    "url": rel.get("html_url") or "",
                    "published": rel.get("published_at") or rel.get("created_at"),
                }
                for rel in data[:limit]
            ]
        return []
    except Exception:
        return []


def fetch_json_generic(url: str, list_path: list[str], field_map: dict, name: str):
    data = _get_json(url, name=name) or {}
    lst = data
    try:
        for key in list_path:
            lst = lst.get(key, [])
    except Exception:
        lst = []

    out = []
    for it in lst:
        out.append(
            {
                "title": next((it.get(p) for p in field_map.get("title", []) if it.get(p)), it.get("title")) or "Untitled",
                "summary": next((it.get(p) for p in field_map.get("summary", []) if it.get(p)), it.get("summary")) or "",
                "url": next((it.get(p) for p in field_map.get("url", []) if it.get(p)), it.get("url")) or "",
                "published": next((it.get(p) for p in field_map.get("published", []) if it.get(p)), it.get("published")) or it.get("date"),
            }
        )
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_atom_rss(url: str, name: str):
    try:
        fp = feedparser.parse(url)
    except Exception:
        return []

    out = []
    for e in fp.entries[:100]:
        out.append(
            {
                "title": e.get("title", "Untitled"),
                "summary": e.get("summary", "") or (
                    e.get("content", [{}])[0].get("value", "") if e.get("content") else ""
                ),
                "url": e.get("link", ""),
                "published": e.get("published") or e.get("updated"),
            }
        )
    return out


# ── Extra live feeds ────────────────────────────────────────────────────────
SOURCES = {
    "cisa_kev": {
        "kind": "json",
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "json_path": ["vulnerabilities"],
        "map": {
            "title": ["cveID"],
            "summary": ["shortDescription"],
            "url": ["cisaAction"],
            "published": ["dateAdded"],
        },
    },
    "github_linux": {"kind": "atom", "url": "https://github.com/torvalds/linux/releases.atom"},
    "github_kubernetes": {"kind": "atom", "url": "https://github.com/kubernetes/kubernetes/releases.atom"},
    "github_docker": {"kind": "atom", "url": "https://github.com/docker/cli/releases.atom"},
}

GITHUB_VENDOR_REPOS = {
    "kubernetes": "kubernetes/kubernetes",
    "docker": "docker/cli",
    "python": "python/cpython",
    "grafana": "grafana/grafana",
    "redis": "redis/redis",
    "node": "nodejs/node",
    "nginx": "nginx/nginx",
    "tensorflow": "tensorflow/tensorflow",
    "pytorch": "pytorch/pytorch",
    "postgres": "postgres/postgres",
    "postgresql": "postgres/postgres",
    "mysql": "mysql/mysql-server",
    "mongodb": "mongodb/mongo",
    "golang": "golang/go",
    "go": "golang/go",
    "linux": "torvalds/linux",
}

ATOM_VENDOR_SOURCES = {
    "linux": ["github_linux"],
    "kernel": ["github_linux"],
    "kubernetes": ["github_kubernetes"],
    "docker": ["github_docker"],
}

DISCUSSION_HINTS = {
    "reddit", "discussion", "discussions", "user", "users", "complaint",
    "complaints", "report", "reports", "issue", "issues", "bug", "bugs", "feedback",
}

SECURITY_HINTS = {
    "cve", "cves", "vulnerability", "vulnerabilities", "security", "exploit",
    "exploited", "kev", "severity",
}


def determine_allowed_sources(query: str, vendors: list[str]):
    ql = (query or "").lower()
    vendor_set = set(vendors)

    wants_discussion = any(term in ql for term in DISCUSSION_HINTS)
    wants_security = any(term in ql for term in SECURITY_HINTS)
    wants_release = is_release_query(query)

    use_os = False
    use_reddit = False
    use_cisa = False
    atom_keys = []
    gh_release_repos = []

    if not vendors:
        return {
            "use_os": wants_release,
            "use_reddit": wants_discussion,
            "use_cisa": wants_security,
            "atom_keys": [],
            "gh_release_repos": [],
        }

    for v in vendors:
        if v in GITHUB_VENDOR_REPOS:
            gh_release_repos.append(GITHUB_VENDOR_REPOS[v])

        if v in ATOM_VENDOR_SOURCES:
            atom_keys.extend(ATOM_VENDOR_SOURCES[v])

    if vendor_set & {"windows", "ubuntu", "debian", "linux", "kernel", "macos", "ios", "android"}:
        use_os = True

    if wants_discussion:
        use_reddit = True

    if wants_security:
        use_cisa = True
        use_os = False
        atom_keys = []
        gh_release_repos = []

    return {
        "use_os": use_os,
        "use_reddit": use_reddit,
        "use_cisa": use_cisa,
        "atom_keys": sorted(set(atom_keys)),
        "gh_release_repos": sorted(set(gh_release_repos)),
    }


# ── UI ──────────────────────────────────────────────────────────────────────
st.sidebar.button("🔄 Rebuild vector store from API", on_click=lambda: (build_store(), st.cache_resource.clear()))
st.title("Release Notes Chat")

top_k = st.slider("Retrieval depth", 1, 15, 5)
use_live_api = True

if "hist" not in st.session_state:
    st.session_state.hist = []

for role, msg in st.session_state.hist:
    st.chat_message(role).write(msg)

user_q = st.chat_input(
    'Ask anything (e.g., “Windows driver issues last month”, “NVIDIA updates in March 2024”).'
)

if user_q:
    t0 = perf_counter()
    t_live = 0.0
    t_rag = 0.0

    st.chat_message("user").write(user_q)
    st.session_state.hist.append(("user", user_q))

    vendors = extract_vendors(user_q)
    route = determine_allowed_sources(user_q, vendors)

    live_answer = ""
    os_f, rd_f, cisa_f, atom_f, gh_rel_f = [], [], [], [], []

    if use_live_api:
        try:
            t_live0 = perf_counter()
            win = parse_time_window(user_q)

            os_items = []
            rd_items = []
            cisa_items = []
            atom_items = []
            gh_rel = []

            if route["use_os"]:
                os_raw = _get_json(OS_API, "os") or []
                if os_raw:
                    mark_live("os")
                os_items = _normalize_results(os_raw) if isinstance(os_raw, (list, dict)) else []

            if route["use_reddit"]:
                rd_raw = _get_json(REDDIT_API, "reddit") or []
                if rd_raw:
                    mark_live("reddit")
                rd_items = _normalize_results(rd_raw) if isinstance(rd_raw, (list, dict)) else []

            os_f = filter_by_time_and_vendor(os_items, win, vendors)
            rd_f = filter_by_time_and_vendor(rd_items, win, vendors)

            if route["use_cisa"]:
                cfg = SOURCES["cisa_kev"]
                kev_hits = fetch_json_generic(cfg["url"], cfg["json_path"], cfg["map"], name="cisa_kev")
                if kev_hits:
                    cisa_items += kev_hits
                    mark_live("cisa_kev")

            any_gh_atom = False
            for key in route["atom_keys"]:
                cfg = SOURCES.get(key)
                if cfg and cfg.get("kind") == "atom":
                    gh = fetch_atom_rss(cfg["url"], name=key)
                    if gh:
                        atom_items += gh
                        any_gh_atom = True
            if any_gh_atom:
                mark_live("github_atom")

            cisa_f = filter_by_time_and_vendor(cisa_items, win, vendors)
            atom_f = filter_by_time_and_vendor(atom_items, win, [])

            for repo in route["gh_release_repos"]:
                gh_rel.extend(fetch_github_releases(repo, limit=60))

            gh_rel_f = filter_by_time_and_vendor(gh_rel, win, [])

            if is_rc_query(user_q):
                gh_rel_f = filter_rc_only(gh_rel_f)

            if not gh_rel_f and gh_rel:
                gh_rel_f = gh_rel[:top_k]

            if is_rc_query(user_q) and not gh_rel_f and gh_rel:
                gh_rel_f = filter_rc_only(gh_rel)
                gh_rel_f = gh_rel_f[:top_k] if gh_rel_f else gh_rel[:top_k]

            if gh_rel_f:
                mark_live("github_releases")

            sections = []
            if route["use_os"] and os_f:
                sections.append(build_grounded_answer("OS Updates & Vulnerabilities", os_f, limit=top_k))
            if route["use_reddit"] and rd_f:
                sections.append(build_grounded_answer("Reddit Discussions & Announcements", rd_f, limit=top_k))
            if route["use_cisa"] and cisa_f:
                sections.append(build_grounded_answer("CISA Vulnerability Feed", cisa_f, limit=top_k))
            if route["atom_keys"] and atom_f:
                sections.append(build_grounded_answer("GitHub Atom Feed", atom_f, limit=top_k))
            if route["gh_release_repos"] and gh_rel_f:
                sections.append(build_grounded_answer("GitHub Releases", gh_rel_f, limit=top_k))

            if sections:
                live_answer = "\n\n".join(sections)
            elif gh_rel:
                fallback_items = gh_rel[:top_k]
                live_answer = build_grounded_answer("Closest Matching Releases", fallback_items, limit=top_k)
            else:
                live_answer = ""

            t_live = perf_counter() - t_live0

        except Exception as e:
            st.warning(f"Live path failed; continuing with RAG. {e}")

    t_rag0 = perf_counter()
    ctx = retrieve(user_q, top_k)
    rag_answer = call_llm(make_msgs(user_q, ctx)) if ctx else ""
    t_rag = perf_counter() - t_rag0

    answer = combine_live_and_rag(user_q, live_answer, rag_answer)

    st.markdown(f'<div class="query-card">{user_q}</div>', unsafe_allow_html=True)

    live_labels = []
    if gh_rel_f:
        live_labels.append("github_releases")
    if atom_f:
        live_labels.append("github_atom")
    if cisa_f:
        live_labels.append("cisa_kev")
    if os_f:
        live_labels.append("os")
    if rd_f:
        live_labels.append("reddit")

    if live_labels:
        st.markdown(
            f'<div class="live-chip">✓ {", ".join(live_labels)}: live</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

    elapsed = perf_counter() - t0
    st.caption(f"⏱️ Total: **{elapsed:.2f}s** | Live: **{t_live:.2f}s** | RAG: **{t_rag:.2f}s**")

    st.session_state.hist.append(("assistant", answer))