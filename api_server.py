# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any
import requests, re, json, os
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import calendar
from dateutil.relativedelta import relativedelta
from pathlib import Path
import feedparser
from urllib.parse import urlencode
import concurrent.futures
import logging

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
OS_API     = "https://releasetrain.io/api/component?q=os"
REDDIT_API = "https://releasetrain.io/api/reddit"
CACHE_DIR  = Path(".live_cache"); CACHE_DIR.mkdir(exist_ok=True)
NVD_API_KEY = os.getenv("NVD_API_KEY", "")  # optional
MAX_WORKERS = int(os.getenv("MAX_WORKERS","8"))

# ---------- Helpers ----------
def _normalize_results(payload):
    if isinstance(payload, list): return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return []

def _get_json(url: str, name: str, headers: dict | None = None, timeout: int = 12):
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{url}".lower()).strip("_")
    cache_path = CACHE_DIR / f"{safe}.json"
    sess = requests.Session()
    retry = Retry(total=2, backoff_factor=0.7, status_forcelist=(502,503,504),
                  allowed_methods=frozenset(["GET"]))
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    base_headers = {"User-Agent":"ReleaseNotesRec/1.0"}
    if headers: base_headers.update(headers)
    try:
        r = sess.get(url, timeout=timeout, headers=base_headers)
        if r.status_code == 200 and "json" in r.headers.get("content-type",""):
            data = r.json()
            try: cache_path.write_text(json.dumps(data), encoding="utf-8")
            except Exception: pass
            return {"ok": True, "data": data}
        # if non-JSON or non-200, return informative error
        return {"ok": False, "error": f"HTTP {r.status_code} {r.headers.get('content-type','')}", "status_code": r.status_code}
    except Exception as e:
        # fallback to cached file
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return {"ok": True, "data": cached, "cached": True}
            except Exception:
                pass
        return {"ok": False, "error": str(e)}

def _as_utc(dt):
    if not dt.tzinfo: return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_isoish(s: Optional[str]):
    if not s: return None
    try:
        s = s.replace("Z","+00:00").split(".")[0]
        return _as_utc(datetime.fromisoformat(s))
    except Exception:
        return None

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_WEEK_REX = re.compile(r"\bweek\s+(\d{1,2})\s+of\s+(19|20)\d{2}\b", re.I)

def parse_time_window(q: str, now=None) -> Optional[Tuple[datetime, datetime]]:
    q = (q or "").strip()
    now = _as_utc(now or datetime.now(timezone.utc))
    s = q.lower()

    if "yesterday" in s:
        d = now.date() - timedelta(days=1)
        return (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc))
    if "last week" in s:
        monday = (now - timedelta(days=now.weekday()+7)).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "this week" in s:
        monday = (now - timedelta(days=now.weekday())).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "last month" in s:
        first = (now.replace(day=1) - relativedelta(months=1)).date()
        last  = (now.replace(day=1) - timedelta(days=1)).date()
        return (datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(last, datetime.max.time(), tzinfo=timezone.utc))
    if "this month" in s:
        first = now.replace(day=1).date()
        last_day = calendar.monthrange(now.year, now.month)[1]
        last = datetime(now.year, now.month, last_day, 23,59,59, tzinfo=timezone.utc)
        return (datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc), last)
    if "last year" in s:
        return (datetime(now.year-1,1,1,tzinfo=timezone.utc),
                datetime(now.year-1,12,31,23,59,59,tzinfo=timezone.utc))
    if "this year" in s:
        return (datetime(now.year,1,1,tzinfo=timezone.utc),
                datetime(now.year,12,31,23,59,59,tzinfo=timezone.utc))

    m = _WEEK_REX.search(q)
    if m:
        wk = int(m.group(1)); yr = int(q[m.start(2):m.start(2)+4])
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
        S = _try_dt(rng.group(2)); E = _try_dt(rng.group(4))
        if S and E and S <= E: return S, E + timedelta(hours=23,minutes=59,seconds=59)

    mdate = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", q)
    if mdate:
        d = datetime.strptime(mdate.group(1), "%Y-%m-%d").date()
        return (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc))

    y = re.search(r"\bin\s+(20\d{2}|19\d{2})\b", q)
    if y:
        yr = int(y.group(1))
        return (datetime(yr,1,1,tzinfo=timezone.utc), datetime(yr,12,31,23,59,59,tzinfo=timezone.utc))

    return None

_STOP = {
    "the","a","an","and","or","to","for","of","on","in","at","by","with","from",
    "is","are","was","were","be","been","am","as","about","this","that","these",
    "those","any","latest","new","update","updates","driver","drivers","patch","patches",
    "version","versions","issues","issue","problem","problems","bug","bugs"
}
def extract_vendors(q: str) -> List[str]:
    if not q: return []
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", q)
    tokens = []
    for t in raw:
        t = t.lower()
        if len(t) < 3 or t in _STOP:
            continue
        tokens.append(t)
    out, seen = [], set()
    for t in tokens:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def filter_by_time_and_vendor(items, start_end, vendors):
    def _dt(it):
        return (_parse_isoish(it.get("updatedAt") or it.get("createdAt") or
                              it.get("date") or it.get("published") or it.get("created_utc")))
    out = []
    for it in items:
        dt = _dt(it)
        if start_end:
            s,e = start_end
            if not dt or not (s <= dt <= e): continue
        if vendors:
            hay = " ".join(str(it.get(k,"")) for k in (
                "title","name","versionProductName","versionReleaseNotes","summary","description"
            )).lower()
            if not any(v in hay for v in vendors): continue
        out.append(it)
    return out

def build_grounded_answer(title, items, limit=8):
    if not items:
        return f"{title}: no matching items."
    lines = [title + ":"]
    for it in items[:limit]:
        t = it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled"
        url = it.get("url") or it.get("link") or ""
        dt  = (_parse_isoish(it.get('updatedAt') or it.get('createdAt') or it.get('date') or it.get('published') or it.get('created_utc')))
        ds  = dt.date().isoformat() if dt else ""
        notes = (it.get("versionReleaseNotes") or it.get("summary") or it.get("description") or it.get("content") or "")
        blurb = (notes[:220] + "…") if notes and len(notes) > 220 else notes
        part = f"- {t} — {blurb} (date: {ds})"
        if url: part += f" • {url}"
        lines.append(part)
    return "\n".join(lines)

# ---------- Extra vendor feeds ----------
SOURCES = {
    "cisa_kev": {
        "kind": "json",
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "json_path": ["vulnerabilities"],
        "map": {"title":["cveID"],"summary":["shortDescription"],"url":["cisaAction"],"published":["dateAdded"]},
    },
    "github_chromium":   {"kind":"atom", "url":"https://github.com/chromium/chromium/releases.atom"},
    "github_kubernetes": {"kind":"atom", "url":"https://github.com/kubernetes/kubernetes/releases.atom"},
    "github_openssl":    {"kind":"atom", "url":"https://github.com/openssl/openssl/releases.atom"},
    "github_node":       {"kind":"atom", "url":"https://github.com/nodejs/node/releases.atom"},
    "github_python":     {"kind":"atom", "url":"https://github.com/python/cpython/releases.atom"},
    "github_postgres":   {"kind":"atom", "url":"https://github.com/postgres/postgres/releases.atom"},
    "github_nginx":      {"kind":"atom", "url":"https://github.com/nginx/nginx/releases.atom"},
    "github_redis":      {"kind":"atom", "url":"https://github.com/redis/redis/releases.atom"},
    "github_linux":      {"kind":"atom", "url":"https://github.com/torvalds/linux/releases.atom"},
    "github_docker":     {"kind":"atom", "url":"https://github.com/docker/cli/releases.atom"},
    "github_containerd": {"kind":"atom", "url":"https://github.com/containerd/containerd/releases.atom"},
    "github_istio":      {"kind":"atom", "url":"https://github.com/istio/istio/releases.atom"},
    "github_grafana":    {"kind":"atom", "url":"https://github.com/grafana/grafana/releases.atom"},
    "github_prometheus": {"kind":"atom", "url":"https://github.com/prometheus/prometheus/releases.atom"},
    "github_openvpn":    {"kind":"atom", "url":"https://github.com/OpenVPN/openvpn/releases.atom"},
    "github_vscode":     {"kind":"atom", "url":"https://github.com/microsoft/vscode/releases.atom"},
    "github_tensorflow": {"kind":"atom", "url":"https://github.com/tensorflow/tensorflow/releases.atom"},
    "github_pytorch":    {"kind":"atom", "url":"https://github.com/pytorch/pytorch/releases.atom"},
    "github_mariadb":    {"kind":"atom", "url":"https://github.com/MariaDB/server/releases.atom"},
    "github_mongodb":    {"kind":"atom", "url":"https://github.com/mongodb/mongo/releases.atom"},
    "github_elasticsearch":{"kind":"atom","url":"https://github.com/elastic/elasticsearch/releases.atom"},
    "github_kafka":      {"kind":"atom","url":"https://github.com/apache/kafka/releases.atom"},
    "github_spark":      {"kind":"atom","url":"https://github.com/apache/spark/releases.atom"},
    "github_airflow":    {"kind":"atom","url":"https://github.com/apache/airflow/releases.atom"},
    "github_flask":      {"kind":"atom","url":"https://github.com/pallets/flask/releases.atom"},
    "github_django":     {"kind":"atom","url":"https://github.com/django/django/releases.atom"},
    "github_fastapi":    {"kind":"atom","url":"https://github.com/tiangolo/fastapi/releases.atom"},
    "github_rust":       {"kind":"atom","url":"https://github.com/rust-lang/rust/releases.atom"},
    "github_go":         {"kind":"atom","url":"https://github.com/golang/go/releases.atom"},
}

def _take(d, path_list):
    for p in path_list:
        v = d.get(p)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def fetch_json_generic(url: str, list_path: list[str], field_map: dict, name: str, headers: dict | None = None):
    res = _get_json(url, name=name, headers=headers)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error")}
    data = res["data"] or {}
    lst = data
    try:
        for key in list_path:
            lst = lst.get(key, [])
    except Exception:
        lst = []
    out = []
    for it in lst:
        out.append({
            "title":     _take(it, field_map.get("title", [])) or it.get("title") or "Untitled",
            "summary":   _take(it, field_map.get("summary", [])) or it.get("summary", ""),
            "url":       _take(it, field_map.get("url", [])) or it.get("url", ""),
            "published": _take(it, field_map.get("published", [])) or it.get("published") or it.get("date"),
        })
    return {"ok": True, "data": out}

def fetch_atom_rss(url: str, name: str):
    try:
        fp = feedparser.parse(url)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    out = []
    for e in fp.entries[:200]:
        out.append({
            "title": e.get("title", "Untitled"),
            "summary": e.get("summary", "") or (e.get("content", [{}])[0].get("value", "") if e.get("content") else ""),
            "url": e.get("link", ""),
            "published": e.get("published") or e.get("updated"),
        })
    return {"ok": True, "data": out}

# NVD (unchanged except we return structured dict)
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def _fmt_nvd(dt: datetime) -> str:
    return _as_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def _chunks(start: datetime, end: datetime, days: int = 120):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt + timedelta(seconds=1)

def expand_vendor_tokens(vendors: list[str]) -> list[str]:
    synonyms = {
        "macos": ["macos","os x","apple"], "ios":["ios","apple","iphone","ipad"],
        "ipados":["ipados","apple"], "watchos":["watchos","apple"], "tvos":["tvos","apple"],
        "windows":["windows","microsoft"], "edge":["edge","microsoft"],
        "chrome":["chrome","google","chromium"], "android":["android","google"],
        "safari":["safari","apple"], "debian":["debian"], "ubuntu":["ubuntu","canonical"],
        "redhat":["red hat","rhel","redhat"], "openssh":["openssh"], "openssl":["openssl"],
        "nginx":["nginx","f5"], "apache":["apache","httpd"], "kubernetes":["kubernetes","k8s"],
        "firefox":["firefox","mozilla"], "postgres":["postgres","postgresql"], "mysql":["mysql","oracle"],
    }
    out = set(vendors)
    for v in list(vendors):
        out.update(synonyms.get(v, []))
    return list(out)

def fetch_nvd(vendors: list[str], win, limit: int = 20):
    if not vendors:
        return []
    vendors = expand_vendor_tokens([v.lower() for v in vendors])
    noise = {"update","updates","driver","drivers","issue","issues","bug","bugs","chip","chips","latest"}
    terms = [t for t in vendors if t not in noise] or vendors[:1]

    if not win:
        end = datetime.now(timezone.utc); start = end - timedelta(days=365)
    else:
        start, end = win

    results = []
    per_page = min(200, max(50, limit * 10))
    hdr = {"apiKey": NVD_API_KEY} if NVD_API_KEY else None

    for s, e in _chunks(start, end, 120):
        base = {"keywordSearch": " ".join(sorted(set(terms))), "resultsPerPage": per_page}

        p = base | {"pubStartDate": _fmt_nvd(s), "pubEndDate": _fmt_nvd(e)}
        url = f"{NVD_API}?{urlencode(p)}"
        r = _get_json(url, name="nvd", headers=hdr)
        if not r.get("ok"):
            # try lastMod fallback
            p2 = base | {"lastModStartDate": _fmt_nvd(s), "lastModEndDate": _fmt_nvd(e)}
            url2 = f"{NVD_API}?{urlencode(p2)}"
            r = _get_json(url2, name="nvd", headers=hdr) or {"ok": False}

        data = r.get("data", {}) if r.get("ok") else {}
        for v in (data or {}).get("vulnerabilities", []):
            c = v.get("cve", {})
            descs = c.get("descriptions", []) or []
            en = next((d.get("value") for d in descs if d.get("lang") == "en"), "") or (descs[0].get("value") if descs else "")
            ref_url = next((r.get("url") for r in (c.get("references") or []) if r.get("url")), "")
            results.append({"title": c.get("id",""), "url": ref_url, "published": c.get("published"), "summary": en})

        if len(results) >= limit * 10:
            break

    return results[: limit * 10]

# ---------- FastAPI ----------
app = FastAPI(title="Release Notes RAG API")

class AskRequest(BaseModel):
    query: str
    top_k: int = 8
    strict: bool = False

@app.get("/")
def health():
    return {"status": "ok"}

def _is_source_relevant_to_vendors(source_key: str, vendors: List[str]) -> bool:
    if not vendors: 
        return True
    sk = source_key.lower()
    for v in vendors:
        if v in sk:
            return True
    # allow a couple of special cases: nvd and cisa_kev are useful for any vendor
    if sk in ("nvd", "cisa_kev"):
        return True
    return False

@app.post("/ask")
def ask(req: AskRequest):
    q = req.query
    k = req.top_k
    strict_mode = bool(req.strict)

    # track unique live sources to avoid duplicate chips
    live_sources: set[str] = set()

    # parse filters
    win = parse_time_window(q)
    vendors = extract_vendors(q)

    # build list of source keys to fetch. If strict_mode and vendors present, prefer only vendor-relevant sources
    source_keys = list(SOURCES.keys())
    # always include built-in os and reddit and nvd/cisa
    base_always = ["cisa_kev"]
    # if strict and vendors are detected, filter down
    if strict_mode and vendors:
        filtered = []
        for key in source_keys:
            if _is_source_relevant_to_vendors(key, vendors):
                filtered.append(key)
        # ensure we still include cisa/nvd if present
        for b in base_always:
            if b in source_keys and b not in filtered:
                filtered.append(b)
        source_keys = filtered

    # We'll fetch os and reddit (these are separate endpoints), plus the selected SOURCES in parallel
    feed_status: Dict[str, Any] = {}
    fetched_items = {"os": [], "reddit": [], "extra": []}

    # helper fetch functions for parallelism
    def fetch_os():
        res = _get_json(OS_API, "os")
        if res.get("ok"):
            data = res.get("data") or []
            return {"ok": True, "data": _normalize_results(data) if data else []}
        return {"ok": False, "error": res.get("error")}

    def fetch_reddit():
        res = _get_json(REDDIT_API, "reddit")
        if res.get("ok"):
            data = res.get("data") or []
            return {"ok": True, "data": _normalize_results(data) if data else []}
        return {"ok": False, "error": res.get("error")}

    # parallel fetch for os, reddit, and SOURCES keys (atom/json)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, (len(source_keys)+2 or 2))) as exe:
        futures = {}
        futures['os'] = exe.submit(fetch_os)
        futures['reddit'] = exe.submit(fetch_reddit)

        for key in source_keys:
            cfg = SOURCES.get(key, {})
            if cfg.get("kind") == "json":
                futures[key] = exe.submit(fetch_json_generic, cfg["url"], cfg.get("json_path", []), cfg.get("map", {}), key)
            elif cfg.get("kind") == "atom":
                futures[key] = exe.submit(fetch_atom_rss, cfg["url"], key)
            else:
                # unknown kind -> skip
                feed_status[key] = {"ok": False, "error": "unknown kind"}
        # gather
        for name, fut in futures.items():
            try:
                r = fut.result(timeout=30)
            except Exception as e:
                r = {"ok": False, "error": str(e)}
            feed_status[name] = r

    # collect os/reddit
    if feed_status.get("os", {}).get("ok"):
        os_items = feed_status["os"]["data"] or []
        if os_items: live_sources.add("✓ os: live")
        fetched_items["os"] = os_items
    else:
        fetched_items["os"] = []

    if feed_status.get("reddit", {}).get("ok"):
        rd_items = feed_status["reddit"]["data"] or []
        if rd_items: live_sources.add("✓ reddit: live")
        fetched_items["reddit"] = rd_items
    else:
        fetched_items["reddit"] = []

    # collect extra items
    extra = []
    for key in source_keys:
        if key in ("os","reddit"): continue
        r = feed_status.get(key)
        if not r:
            continue
        if r.get("ok"):
            data = r.get("data") or []
            extra.extend(data)
            if data:
                live_sources.add(f"✓ {key}: live")
        else:
            # record feed error into feed_status details (already there)
            pass

    # apply filtering by time and vendor to each group
    os_f = filter_by_time_and_vendor(fetched_items["os"], win, vendors)
    rd_f = filter_by_time_and_vendor(fetched_items["reddit"], win, vendors)
    extra_f = filter_by_time_and_vendor(extra, win, vendors)

    # NVD fetch (still synchronous; small improvement: run it in executor if needed)
    nvd_results = []
    try:
        nvd_results = fetch_nvd(vendors, win, limit=k) if vendors else []
        if nvd_results:
            live_sources.add("✓ nvd: live")
            extra_f = extra_f + nvd_results
    except Exception as e:
        logger.exception("NVD fetch failed: %s", e)

    # build answers
    answer = {
        "os":     build_grounded_answer("OS Updates", os_f, limit=k),
        "reddit": build_grounded_answer("Reddit Discussions", rd_f, limit=k),
        "feeds":  build_grounded_answer("Other Vendor Feeds", extra_f, limit=k),
    }

    # Compose final debug info
    feed_errors = {k:v for k,v in feed_status.items() if not v.get("ok")}
    return {
        "query": q,
        "time_window": [win[0].isoformat(), win[1].isoformat()] if win else None,
        "vendors_detected": vendors,
        "sources_live": sorted(live_sources),
        "feed_status": feed_status,
        "feed_errors": feed_errors,
        "strict": strict_mode,
        "answer": answer,
    }