# app.py
from langchain_google_genai import ChatGoogleGenerativeAI
ChatGoogleGenerativeAI.model_rebuild()

import os, shutil, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer

# NEW ↓
import re, json, calendar, feedparser
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from dateutil.relativedelta import relativedelta
from urllib.parse import urlencode
from time import perf_counter
# NEW ↑

st.set_page_config("Release-Notes Chat", "💬")

# ── creds ────────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("Set GOOGLE_API_KEY in your .env"); st.stop()

_gem = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ── data / embedding config ─────────────────────────────────────────────────
CSV_PATH   = "SoftwareUpdateSurvey.csv"
OS_API     = "https://releasetrain.io/api/component?q=os"
REDDIT_API = "https://releasetrain.io/api/reddit"
MAX_OS, MAX_RED = 50, 50
EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
DATA_DIR   = "release_notes_store"
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")

# Cache (no seeds)
CACHE_DIR = Path(".live_cache"); CACHE_DIR.mkdir(exist_ok=True)

# -------------------------- one-time status chips ---------------------------
if "live_marks" not in st.session_state:
    st.session_state.live_marks = set()

def mark_live(name: str):
    if name not in st.session_state.live_marks:
        st.session_state.live_marks.add(name)
        st.caption(f"✓ {name}: live")

# -------------------------- utilities / fetch --------------------------------
def _normalize_results(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return []

def _get_json(url: str, name: str, headers: dict | None = None):
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{url}".lower()).strip("_")
    cache_path = CACHE_DIR / f"{safe}.json"

    sess = requests.Session()
    retry = Retry(total=2, backoff_factor=0.7, status_forcelist=(502,503,504),
                  allowed_methods=frozenset(["GET"]))
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))

    try:
        base_headers = {"User-Agent":"ReleaseNotesRec/1.0"}
        if headers:
            base_headers.update(headers)
        r = sess.get(url, timeout=12, headers=base_headers)
        if r.status_code == 200 and "json" in r.headers.get("content-type",""):
            data = r.json()
            try: cache_path.write_text(json.dumps(data), encoding="utf-8")
            except Exception: pass
            return data
        raise RuntimeError(f"HTTP {r.status_code} {r.headers.get('content-type','')}")
    except Exception as e:
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return data
            except Exception:
                pass
        st.warning(f"{name} fetch error from {url}: {e}")
        return None

# ---------------- natural-language time & filters --------------
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_WEEK_REX = re.compile(r"\bweek\s+(\d{1,2})\s+of\s+(\d{4})\b", re.I)

def _as_utc(dt):
    if not dt.tzinfo:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_isoish(s: str | None):
    if not s: return None
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
        return (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc))
    if "last week" in rel:
        monday = (now - timedelta(days=now.weekday()+7)).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "this week" in rel:
        monday = (now - timedelta(days=now.weekday())).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "last month" in rel:
        first = (now.replace(day=1) - relativedelta(months=1)).date()
        last  = (now.replace(day=1) - timedelta(days=1)).date()
        return (datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(last, datetime.max.time(), tzinfo=timezone.utc))
    if "this month" in rel:
        first = now.replace(day=1).date()
        last_day = calendar.monthrange(now.year, now.month)[1]
        last = datetime(now.year, now.month, last_day, 23,59,59, tzinfo=timezone.utc)
        return (datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc), last)
    if "last year" in rel:
        start = datetime(now.year-1, 1, 1, tzinfo=timezone.utc)
        end   = datetime(now.year-1, 12, 31, 23,59,59, tzinfo=timezone.utc)
        return start, end
    if "this year" in rel:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        end   = datetime(now.year, 12, 31, 23,59,59, tzinfo=timezone.utc)
        return start, end

    m = _WEEK_REX.search(q)
    if m:
        wk = int(m.group(1)); yr = int(m.group(2))
        monday = datetime.fromisocalendar(yr, wk, 1).replace(tzinfo=timezone.utc)
        sunday = datetime.fromisocalendar(yr, wk, 7).replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)
        return monday, sunday

    for name, idx in _MONTHS.items():
        m2 = re.search(rf"\b{name}\b\s+(\d{{4}})", q, re.I)
        if m2:
            yr = int(m2.group(1))
            first = datetime(yr, idx, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(yr, idx)[1]
            last = datetime(yr, idx, last_day, 23,59,59, tzinfo=timezone.utc)
            return first, last

    rng = re.search(r"(between|from)\s+([A-Za-z0-9,\-\s/]+)\s+(and|to)\s+([A-Za-z0-9,\-\s/]+)", q, re.I)
    if rng:
        def _try_dt(t):
            for fmt in ("%Y-%m-%d", "%b %d, %Y", "%Y/%m/%d"):
                try: return _as_utc(datetime.strptime(t.strip(), fmt))
                except Exception: pass
            return None
        s = _try_dt(rng.group(2)); e = _try_dt(rng.group(4))
        if s and e and s <= e: return s, e + timedelta(hours=23, minutes=59, seconds=59)

    mdate = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", q)
    if mdate:
        d = datetime.strptime(mdate.group(1), "%Y-%m-%d").date()
        return (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc))

    y = re.search(r"\bin\s+(20\d{2}|19\d{2})\b", q)
    if y:
        yr = int(y.group(1))
        return (datetime(yr,1,1, tzinfo=timezone.utc), datetime(yr,12,31,23,59,59, tzinfo=timezone.utc))

    return None

_STOP = {
    "the","a","an","and","or","to","for","of","on","in","at","by","with","from",
    "is","are","was","were","be","been","am","as","about","this","that","these",
    "those","any","latest","new","update","updates","driver","drivers","patch","patches",
    "version","versions","issues","issue","problem","problems","bug","bugs","os"
}
def extract_vendors(q: str):
    if not q: return []
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", q)
    toks = []
    for t in raw:
        t2 = t.lower()
        if len(t2) < 3: continue
        if t2 in _STOP: continue
        toks.append(t2)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def filter_by_time_and_vendor(items, start_end, vendors):
    def _dt(it):
        return (_parse_isoish(it.get("updatedAt") or it.get("createdAt") or
                              it.get("date") or it.get("published") or it.get("published_at") or it.get("created_utc")))
    out = []
    for it in items:
        dt = _dt(it)
        if start_end:
            s,e = start_end
            if not dt or not (s <= dt <= e): continue
        if vendors:
            hay = " ".join(str(it.get(k,"")) for k in ("title","name","versionProductName","versionReleaseNotes","summary","description","repo")).lower()
            if not any(v in hay for v in vendors): continue
        out.append(it)
    return out

def build_grounded_answer(title, items, limit=5):
    if not items:
        return f"**{title}**\n\n_No matching items in the selected time window and filters._"
    lines = [f"**{title}**"]
    for it in items[:limit]:
        t = it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled"
        url = it.get("url") or it.get("link") or ""
        dt  = (_parse_isoish(it.get('updatedAt') or it.get('createdAt') or it.get('date') or it.get('published') or it.get("published_at") or it.get('created_utc')))
        ds  = dt.date().isoformat() if dt else ""
        notes = (it.get("versionReleaseNotes") or it.get("summary") or it.get("description") or it.get("content") or "")
        blurb = (notes[:220] + "…") if notes and len(notes) > 220 else notes
        if url: lines.append(f"- **{t}** — {blurb}  _(date: {ds})_  • [source]({url})")
        else:   lines.append(f"- **{t}** — {blurb}  _(date: {ds})_")
    return "\n\n".join(lines)

# ------------------------------ ingestion (RAG) ------------------------------------
def load_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return [{"text": "\n".join(f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]))}
            for _, row in df.iterrows()]

def fetch(url, max_items, mapping, name):
    raw = _get_json(url, name=name)
    if raw is None:
        return []
    data = _normalize_results(raw) or (raw if isinstance(raw, list) else [])
    return [{"text": "\n".join(f"{k}: {item.get(v, '')}" for k, v in mapping.items())}
            for item in data[:max_items]]

def build_store():
    docs  = load_csv(CSV_PATH)
    docs += fetch(OS_API, MAX_OS, {"OS_ID":"_id","OS_Name":"versionProductName",
                                   "OS_ReleaseNotes":"versionReleaseNotes"}, name="os")
    docs += fetch(REDDIT_API, MAX_RED, {"REDDIT_ID":"_id","Subreddit":"subreddit",
                                        "Title":"title","URL":"url"}, name="reddit")
    model = SentenceTransformer(EMB_MODEL)
    ds = DatasetDict({"train": Dataset.from_dict({"text":[d["text"] for d in docs]})})
    ds = ds.map(lambda b: {"embeddings": model.encode(b["text"], batch_size=16, show_progress_bar=False)},
                batched=True, batch_size=16)
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

SYSTEM_PROMPT = "Answer using only the provided context. If unsure, say you don’t know."

def call_llm(msgs):
    prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in msgs)
    resp = _gem.invoke(prompt)
    return getattr(resp, "content", str(resp))

def make_msgs(user_q, ctx_docs):
    return [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"system","content":"\n\n".join(f"Document {i+1}:\n{d[:1000]}" for i,d in enumerate(ctx_docs))},
        {"role":"user","content":user_q},
    ]

# ── Extra live feeds ─────────────────────────────────────────────────────
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
    "github_linux":      {"kind":"atom","url":"https://github.com/torvalds/linux/releases.atom"},
    "github_kubernetes": {"kind":"atom","url":"https://github.com/kubernetes/kubernetes/releases.atom"},
    "github_docker":     {"kind":"atom","url":"https://github.com/docker/cli/releases.atom"},
}

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
        out.append({
            "title":     (next((it.get(p) for p in field_map.get("title", []) if it.get(p)), it.get("title")) or "Untitled"),
            "summary":   (next((it.get(p) for p in field_map.get("summary", []) if it.get(p)), it.get("summary")) or ""),
            "url":       (next((it.get(p) for p in field_map.get("url", []) if it.get(p)), it.get("url")) or ""),
            "published": (next((it.get(p) for p in field_map.get("published", []) if it.get(p)), it.get("published")) or it.get("date")),
        })
    return out

@st.cache_data(ttl=600, show_spinner=False)
def fetch_atom_rss(url: str, name: str):
    try:
        fp = feedparser.parse(url)
    except Exception:
        return []
    out = []
    for e in fp.entries[:100]:
        out.append({
            "title": e.get("title", "Untitled"),
            "summary": e.get("summary", "") or (e.get("content", [{}])[0].get("value", "") if e.get("content") else ""),
            "url": e.get("link", ""),
            "published": e.get("published") or e.get("updated"),
        })
    return out

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
    "golang": "golang/go",
}

ATOM_VENDOR_SOURCES = {
    "linux": ["github_linux"],
    "kernel": ["github_linux"],
    "kubernetes": ["github_kubernetes"],
    "docker": ["github_docker"],
}

DISCUSSION_HINTS = {
    "reddit", "discussion", "discussions", "user", "users", "complaint", "complaints",
    "report", "reports", "issue", "issues", "bug", "bugs", "feedback"
}

SECURITY_HINTS = {
    "cve", "cves", "vulnerability", "vulnerabilities", "security", "exploit", "exploited", "kev"
}

def determine_allowed_sources(query: str, vendors: list[str]):
    ql = (query or "").lower()
    vendor_set = set(vendors)

    wants_discussion = any(term in ql for term in DISCUSSION_HINTS)
    wants_security = any(term in ql for term in SECURITY_HINTS)

    use_os = False
    use_reddit = False
    use_cisa = False
    atom_keys = []
    gh_release_repos = []

    if not vendors:
        return {
            "use_os": True,
            "use_reddit": True if wants_discussion else False,
            "use_cisa": True if wants_security else False,
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

    return {
        "use_os": use_os,
        "use_reddit": use_reddit,
        "use_cisa": use_cisa,
        "atom_keys": sorted(set(atom_keys)),
        "gh_release_repos": sorted(set(gh_release_repos)),
    }

@st.cache_data(ttl=600, show_spinner=False)
def fetch_github_releases(repo: str, limit: int = 5):
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

# ── UI ──────────────────────────────────────────────────────────────────────
st.sidebar.button("🔄 Rebuild vector store from API", on_click=lambda: (build_store(), st.cache_resource.clear()))
st.title("💬 Release-Notes Chat — Live API + RAG")

top_k = st.slider("Top-K (RAG & live merge)", 1, 15, 5)
use_live_api = True

if "hist" not in st.session_state: st.session_state.hist = []
for role, msg in st.session_state.hist: st.chat_message(role).write(msg)

user_q = st.chat_input("Ask anything (e.g., “Windows driver issues last month”, “NVIDIA updates in March 2024”).")

if user_q:
    t0 = perf_counter()
    t_live = 0.0
    t_rag = 0.0

    st.chat_message("user").write(user_q)
    st.session_state.hist.append(("user", user_q))

    live_answer = None
    live_has_matches = False

    if use_live_api:
        try:
            t_live0 = perf_counter()

            win = parse_time_window(user_q)
            vendors = extract_vendors(user_q)
            route = determine_allowed_sources(user_q, vendors)

            os_items = []
            rd_items = []
            extra = []
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
                    extra += kev_hits
                    mark_live("cisa_kev")

            any_gh_atom = False
            for key in route["atom_keys"]:
                cfg = SOURCES.get(key)
                if cfg and cfg.get("kind") == "atom":
                    gh = fetch_atom_rss(cfg["url"], name=key)
                    if gh:
                        extra += gh
                        any_gh_atom = True
            if any_gh_atom:
                mark_live("github_atom")

            extra_f = filter_by_time_and_vendor(extra, win, vendors)

            for repo in route["gh_release_repos"]:
                gh_rel.extend(fetch_github_releases(repo, limit=5))
            gh_rel_f = filter_by_time_and_vendor(gh_rel, win, vendors)
            if gh_rel_f:
                mark_live("github_releases")

            sections = []
            if route["use_os"]:
                sections.append(build_grounded_answer("OS Updates & Vulnerabilities", os_f, limit=top_k))
            if route["use_reddit"]:
                sections.append(build_grounded_answer("Reddit Discussions & Announcements", rd_f, limit=top_k))
            if route["use_cisa"] or route["atom_keys"]:
                sections.append(build_grounded_answer("Other Vendor Feeds (CISA/GitHub Atom)", extra_f, limit=top_k))
            if route["gh_release_repos"]:
                sections.append(build_grounded_answer("GitHub Releases", gh_rel_f, limit=top_k))

            live_has_matches = any([os_f, rd_f, extra_f, gh_rel_f])

            if not sections:
                live_answer = "_No strict vendor sources matched this query. Falling back to RAG._"
            else:
                live_answer = "\n\n---\n\n".join(sections)

            t_live = perf_counter() - t_live0

        except Exception as e:
            st.warning(f"Live path failed; will still try RAG. {e}")

    rag_answer = ""
    if not live_has_matches:
        t_rag0 = perf_counter()
        ctx = retrieve(user_q, top_k)
        rag_answer = call_llm(make_msgs(user_q, ctx)) if ctx else ""
        t_rag = perf_counter() - t_rag0

    if live_answer and rag_answer:
        answer = f"{live_answer}\n\n---\n\n**RAG Summary**\n\n{rag_answer}"
    else:
        answer = live_answer or rag_answer or "_No matching information found._"

    st.chat_message("assistant").write(answer)
    elapsed = perf_counter() - t0
    st.caption(f"⏱️ Total: **{elapsed:.2f}s** | Live: **{t_live:.2f}s** | RAG: **{t_rag:.2f}s**")
    st.session_state.hist.append(("assistant", answer))