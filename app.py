from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
import numpy as np
import os
import requests
import unicodedata
import hashlib
import threading
import time
import re
import json
import traceback

CATALOG_PATH = Path(__file__).parent / "hagstofa_catalog.json"
REQUIRED_KEYS = {"table_id", "title_en", "api_url", "keywords", "dimensions"}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Embedding backend, chosen with the EMBEDDING_BACKEND environment variable:
#   "ollama" (default): bge-m3 served by Ollama (works on CPU, e.g. a laptop)
#   "local": BAAI/bge-m3 loaded in Python with sentence-transformers (uses the GPU, e.g. the DGX Spark)
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "ollama")
EMBEDDING_MODEL = {"ollama": "bge-m3", "local": "BAAI/bge-m3"}[EMBEDDING_BACKEND]
# One cache file per backend (their vectors differ slightly); computing them takes ~20 min on CPU
EMBEDDINGS_PATH = Path(__file__).parent / f"catalog_embeddings_{EMBEDDING_BACKEND}.npz"
EMBEDDING_BATCH = 32

# Filled once at startup, then only read by the requests.
CATALOG = []             # list of tables (dicts from the JSON)
TABLE_EMBEDDINGS = None  # numpy array (n_tables, 1024), row i = CATALOG[i], rows normalised to length 1
LOCAL_MODEL = None       # SentenceTransformer, only with EMBEDDING_BACKEND=local
LOCAL_MODEL_LOCK = threading.Lock()  # FastAPI runs requests in threads: one GPU encode at a time


def load_catalog():
    """Read hagstofa_catalog.json into RAM, validate each entry and precompute its search fields.
    Some table_ids appear twice (same table published under two paths): the first one is kept."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    CATALOG.clear()
    seen_ids = set()
    for table in raw:
        if not REQUIRED_KEYS.issubset(table) or table["table_id"] in seen_ids:
            continue
        index_table(table)
        CATALOG.append(table)
        seen_ids.add(table["table_id"])
    print(f"[catalog] {len(CATALOG)} tables loaded into RAM ({len(raw) - len(CATALOG)} skipped: duplicates or invalid)")


def load_embedding_model():
    """With the local backend, load BAAI/bge-m3 once (downloaded from Hugging Face on the first run)."""
    global LOCAL_MODEL
    if EMBEDDING_BACKEND == "local":
        from sentence_transformers import SentenceTransformer  # only installed where the local backend is used
        LOCAL_MODEL = SentenceTransformer(EMBEDDING_MODEL)  # picks the GPU (cuda) when available
        print(f"[embeddings] {EMBEDDING_MODEL} loaded on {LOCAL_MODEL.device}")


def embed(texts):
    """Embed a list of texts; returns an array of unit-length vectors."""
    if EMBEDDING_BACKEND == "local":
        with LOCAL_MODEL_LOCK:
            return LOCAL_MODEL.encode(texts, batch_size=EMBEDDING_BATCH, normalize_embeddings=True,
                                      convert_to_numpy=True).astype(np.float32)
    res = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBEDDING_MODEL, "input": texts}, timeout=600)
    res.raise_for_status()
    vectors = np.array(res.json()["embeddings"], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def table_text(table):
    """Text embedded for a table: title, then each dimension with up to 12 value labels (time values skipped)."""
    parts = [table["title_en"]]
    for d in table["dimensions"].values():
        labels = [label for code, label in d["values"].items()
                  if not re.fullmatch(YEAR_LIKE, code) and not re.fullmatch(YEAR_LIKE, label)]
        parts.append(f"{d['label']}: {', '.join(labels[:12])}" if labels else d["label"])
    return ". ".join(parts)


def load_embeddings():
    """Load the table embeddings from the cache file, or compute and save them if the catalog changed."""
    global TABLE_EMBEDDINGS
    texts = [table_text(t) for t in CATALOG]
    digest = hashlib.sha256("\n".join([EMBEDDING_MODEL, *texts]).encode()).hexdigest()
    if EMBEDDINGS_PATH.exists():
        cache = np.load(EMBEDDINGS_PATH)
        if str(cache["digest"]) == digest:
            TABLE_EMBEDDINGS = cache["vectors"]
            print(f"[embeddings] {len(TABLE_EMBEDDINGS)} table embeddings loaded from {EMBEDDINGS_PATH.name}")
            return
    print(f"[embeddings] computing {len(texts)} table embeddings with {EMBEDDING_MODEL} "
          f"(one-time; ~20 min on CPU, much faster on GPU)...")
    start = time.time()
    TABLE_EMBEDDINGS = np.vstack([embed(texts[i:i + EMBEDDING_BATCH]) for i in range(0, len(texts), EMBEDDING_BATCH)])
    np.savez(EMBEDDINGS_PATH, vectors=TABLE_EMBEDDINGS, digest=digest)
    print(f"[embeddings] done in {time.time() - start:.0f}s, saved to {EMBEDDINGS_PATH.name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_catalog()     # runs once, before the server accepts requests
    load_embedding_model()
    load_embeddings()
    yield              # the server runs here
    # code after yield would run at shutdown (nothing to clean up)


app = FastAPI(title="Statistics Iceland Smart API Server", lifespan=lifespan)

# Allow any web frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Region name (as typed by the user, lowercase) -> PX-API region code.
# 1 = Capital area, 2 = Southwest, 3 = West, 4 = Westfjords,
# 5 = Northwest, 6 = Northeast, 7 = East, 8 = South
REGION_MAPPING = {
    "capital area": "1", "capital region": "1", "capital": "1", "reykjavik": "1",
    "southwest": "2", "southern peninsula": "2", "west": "3", "west iceland": "3",
    "westfjords": "4", "west fjords": "4", "northwest": "5", "northeast": "6",
    "east": "7", "east iceland": "7", "south": "8", "south iceland": "8"
}

# Month names (lowercase) -> two-digit month number.
# Matched as WHOLE WORDS, so "mar" does not fire inside words like "market".
MONTH_MAPPING = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

TOTAL_WORDS = ["total", "all"]  # value labels meaning "total" in the metadata


def clean_text(text):
    """Normalise text: NFKD, strip accents, lowercase, then map Icelandic letters
    to ASCII (þ->th, æ->ae, ö->o, ð->d) so place names like "Reykjavík" still match."""
    nfkd_form = unicodedata.normalize('NFKD', str(text))
    flat_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
    for k, v in {'þ': 'th', 'æ': 'ae', 'ö': 'o', 'ð': 'd'}.items():
        flat_text = flat_text.replace(k, v)
    return flat_text


def has_word(text, words):
    """True if any of `words` appears in `text` as a whole word
    (so that 'men' does not match inside 'women')."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


# ---------------------------------------------------------------------------
# Table search over the in-RAM catalog: keyword score + embedding similarity
# ---------------------------------------------------------------------------

# Words ignored when matching a question against table titles, keywords and value labels
STOPWORDS = set("""a an the of in on at to for from by and or with without is was were are be been what which who whom
how many much did do does has have had there their this that these those per between into over under than as it its
me show give tell number amount iceland icelandic statistics data value values total year years month months
latest last current people""".split())

# Question word -> words used in the value labels of the tables (e.g. Sex: "Females", Item: "Annual change, %").
# Synonyms of table topics are not needed here: the embeddings handle them.
SYNONYMS = {
    "inflation": ["annual", "change"],
    "women": ["females"], "woman": ["females"], "men": ["males"], "man": ["males"],
}

YEAR_LIKE = r"(1[789]|20)\d\d([MQK]\d{1,2})?"  # "2024", "2024M03", "2024Q1", "2024K1"
YEAR_IN_QUESTION = r"\b(1[5-9]\d\d|20\d\d)\b"   # years 1500-2099 typed by the user
EMBEDDING_WEIGHT = 20  # final score = keyword score + EMBEDDING_WEIGHT * cosine similarity
MIN_KEYWORD_SCORE = 2.5  # a table is relevant if its keyword score
MIN_SIMILARITY = 0.5     # or its cosine similarity reaches these thresholds
MIGRATION_TABLE_ID = "MAN01002"  # has its own from/to region rules below


def words(text):
    """Significant lowercase words of a text (accents stripped, no digits, no stopwords)."""
    return [w for w in re.findall(r"[a-z]+", clean_text(text)) if w not in STOPWORDS and len(w) > 1]


def query_tokens(question):
    """Question words plus their synonyms, in order, without duplicates."""
    out = []
    for w in words(question):
        out += [w] + SYNONYMS.get(w, [])
    return list(dict.fromkeys(out))


def same_word(a, b):
    """Equal, or same stem for words of 5+ letters ("migrated" ~ "migration", "hotel" ~ "hotels")."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 5 and long.startswith(short[:max(5, len(short) - 2)])


def hits(tokens, vocab):
    return sum(any(same_word(w, x) for x in vocab) for w in tokens)


def index_table(table):
    """Precompute the search fields of a catalog table (called once at startup)."""
    dims = table["dimensions"].values()
    table["_title_list"] = words(table["title_en"])
    table["_title"] = set(table["_title_list"])
    table["_kw"] = set(words(" ".join(table["keywords"] + [d["label"] for d in dims]))) | table["_title"]
    table["_vals"] = set(words(" ".join(label for d in dims for label in d["values"].values())))
    # Years covered by the table, read from time codes ("2024M03") or labels ("1992")
    years = [int(x[:4]) for d in dims for code, label in d["values"].items() for x in (code, label)
             if re.fullmatch(YEAR_LIKE, x)]
    table["_years"] = (min(years), max(years)) if years else None


def keyword_score(tokens, table, years=()):
    # Title words count most, then keywords/dimension names, then value labels (e.g. "Westfjords")
    s = 3 * hits(tokens, table["_title"]) + hits(tokens, table["_kw"]) + hits(tokens, table["_vals"])
    # Bonus when two consecutive question words appear together in the title ("consumer price")
    tl = table["_title_list"]
    s += 2 * sum(any(same_word(a, tl[i]) and same_word(b, tl[i + 1]) for i in range(len(tl) - 1))
                 for a, b in zip(tokens, tokens[1:]))
    # Prefer short titles and simple tables (general tables over very detailed ones)
    s -= 0.1 * len(table["_title"]) + 0.2 * len(table["dimensions"])
    if table["_years"]:
        lo, hi = table["_years"]
        if any(not lo <= y <= hi for y in years):
            s -= 5  # the table does not cover the requested year
        if hi >= 2024:
            s += 1  # still updated
    elif years:
        s -= 5  # a year was asked but the table has no time dimension
    return s


def find_tables(question, k=3):
    """Top-k tables for a question, best first, as dicts with the table and its scores.
    Returns [] if no table is relevant enough."""
    tokens = query_tokens(question)
    years = [int(y) for y in re.findall(YEAR_IN_QUESTION, question)]
    similarity = TABLE_EMBEDDINGS @ embed([question])[0]  # cosine similarity (all vectors have length 1)

    def rank(use_years):
        keyword = np.array([keyword_score(tokens, t, years if use_years else []) for t in CATALOG])
        final = keyword + EMBEDDING_WEIGHT * similarity
        best = [i for i in np.argsort(-final)[:k] if keyword[i] >= MIN_KEYWORD_SCORE or similarity[i] >= MIN_SIMILARITY]
        return [{"table": CATALOG[i], "score": round(float(final[i]), 1), "keyword_score": round(float(keyword[i]), 1),
                 "similarity": round(float(similarity[i]), 3)} for i in best]

    # If no table covers the requested year, rank by topic only: the query step will report the available years
    return rank(use_years=True) or rank(use_years=False)
class QuestionRequest(BaseModel):
    # Accepts {"question": "..."} or {"prompt": "..."} (the README's curl example)
    question: str = Field(validation_alias=AliasChoices("question", "prompt"))


# "/spyrja_px_api" is the path given in the hackathon spec; "/ask" is the same endpoint.
@app.post("/ask")
@app.post("/spyrja_px_api")
def ask_px_api(req: QuestionRequest):
    try:
        return _answer_question(req)
    except Exception as e:
        traceback.print_exc()  # full details in the uvicorn terminal
        return {"answer": f"Sorry, I could not process this question ({type(e).__name__}: {e})."}


MAX_CELLS = 200       # cap on the number of values requested from PX-API
MAX_ANSWER_ROWS = 20  # rows listed in the answer text


def is_time_variable(var):
    values, texts = var.get("values", []), var.get("valueTexts", [])
    return bool(var.get("time")) or any(re.fullmatch(YEAR_LIKE, x) for x in values[:1] + texts[:1])


def matched_words(tokens, label):
    return [w for w in words(label) if any(same_word(w, t) for t in tokens)]


def label_score(tokens, label):
    """How well a value label matches the question: +1 per label word found, -0.1 per word not found.
    0 if no word matches ("Unemployment rate, %" beats "Activity rate, %" for "unemployment rate")."""
    found = len(matched_words(tokens, label))
    return found - 0.1 * (len(words(label)) - found) if found else 0


def build_generic_query(question_clean, variables, years, month, table):
    """Build a PX-API filter for any table from its metadata. Returns (query_list, error_message)."""
    tokens = query_tokens(question_clean)
    query_list = []
    for var in variables:
        code, values, texts = var.get("code"), var.get("values", []), var.get("valueTexts", [])
        if not values:
            continue
        if is_time_variable(var):
            if years:
                selected = [v for v, t in zip(values, texts) if any(v.startswith(y) or t.startswith(y) for y in years)]
                if not selected:
                    return [], (f"This table has no data for {', '.join(years)} "
                                f"(available: {texts[0]} to {texts[-1]}).")
                if month:
                    selected = [v for v in selected if v.endswith(f"M{month}")] or selected
            else:
                selected = [values[-1]]  # latest period
        else:
            scores = [label_score(tokens, t) for t in texts]
            best = max(scores)
            selected = [v for v, sc in zip(values, scores) if best > 0 and sc == best]
            totals = [v for v, t in zip(values, texts) if has_word(clean_text(t), TOTAL_WORDS)]
            # Matched only through the table topic ("births" in a births table): that is not a filter, use the total
            in_title = lambda w: any(same_word(w, y) for y in table["_title"])
            topic_only = all(in_title(w) for v, t in zip(values, texts) if v in selected for w in matched_words(tokens, t))
            if selected and totals and topic_only:
                selected = totals[:1]
            if not selected:
                if var.get("elimination"):
                    continue  # PX-API returns the aggregate when the variable is left out
                selected = totals[:1] or values[:1]
        query_list.append({"code": code, "selection": {"filter": "item", "values": selected}})

    # Size limit: halve the largest selection (keeping the most recent values) until small enough
    def cells():
        n = 1
        for q in query_list:
            n *= len(q["selection"]["values"])
        return n
    while cells() > MAX_CELLS:
        biggest = max(query_list, key=lambda q: len(q["selection"]["values"]))
        vals = biggest["selection"]["values"]
        biggest["selection"]["values"] = vals[len(vals) // 2:]
    return query_list, None


def build_migration_query(question_clean, variables, years):
    """Rules specific to internal migration (MAN01002): sex, age total, region from / region to."""
    # Default variable codes; overwritten below if found in the metadata
    sex_code, year_code, departure_code, arrival_code, age_code, age_total = "Kyn", "Ár", "Brottflutningssvæði", "Aðflutningssvæði", None, None
    sex_total_value = None

    for var in variables:
        v_code, v_text = var.get("code"), clean_text(var.get("text", ""))
        v_vals, v_vtexts = var.get("values", []), var.get("valueTexts", [])
        if v_text == "sex":
            sex_code = v_code
            for idx, vt in enumerate(v_vtexts):
                if any(w in clean_text(vt) for w in TOTAL_WORDS):
                    sex_total_value = v_vals[idx]
                    break
        elif v_text == "year":
            year_code = v_code
        elif "migrated from" in v_text:  # "Region migrated from"
            departure_code = v_code
        elif "migrated to" in v_text:  # "Region migrated to"
            arrival_code = v_code
        elif v_text == "age":
            age_code = v_code
            for idx, vt in enumerate(v_vtexts):
                if any(w in clean_text(vt) for w in TOTAL_WORDS):
                    age_total = v_vals[idx]
                    break
            if not age_total and v_vals:
                age_total = v_vals[0]

    mentions_men = has_word(question_clean, ["men", "man", "male", "males"])
    mentions_women = has_word(question_clean, ["women", "woman", "female", "females"])
    asks_total = has_word(question_clean, ["total", "overall", "altogether", "in all"])

    if asks_total and not (mentions_men or mentions_women):
        sex_val = [sex_total_value] if sex_total_value else ["1", "2"]
    else:
        if mentions_men and mentions_women:
            sex_val = ["1", "2"]
        elif mentions_women:
            sex_val = ["2"]
        elif mentions_men:
            sex_val = ["1"]
        else:
            sex_val = [sex_total_value] if sex_total_value else ["1", "2"]

    query_list = [{"code": sex_code, "selection": {"filter": "item", "values": sex_val}}]
    if age_code and age_total:
        query_list.append({"code": age_code, "selection": {"filter": "item", "values": [age_total]}})
    if years:
        query_list.append({"code": year_code, "selection": {"filter": "item", "values": years}})

    # Find regions in the question, ordered by where they appear in the sentence
    found_positions = sorted(
        (m.start(), code)
        for s, code in REGION_MAPPING.items()
        if (m := re.search(rf"\b{re.escape(s)}\b", question_clean))
    )
    regions = []
    for _, k in found_positions:
        if not regions or regions[-1] != k:
            regions.append(k)

    all_regions = [str(i) for i in range(1, 9)]
    if len(regions) >= 2:
        # Two regions: first = origin, second = destination
        origin, destination = [regions[0]], [regions[1]]
    elif len(regions) == 1:
        # One region: "from X" -> X to every other region, otherwise every other region -> X
        others = [r for r in all_regions if r != regions[0]]
        if has_word(question_clean, ["from"]):
            origin, destination = [regions[0]], others
        else:
            origin, destination = others, [regions[0]]
    else:
        origin, destination = ["7"], ["1"]  # no region given: hardcoded default (East -> Capital)

    query_list.append({"code": departure_code, "selection": {"filter": "item", "values": origin}})
    query_list.append({"code": arrival_code, "selection": {"filter": "item", "values": destination}})
    return query_list, sex_val, sex_total_value


def generic_answer_text(table_title, variables, api_data):
    """List the returned values with readable labels, e.g. "2024M03, Annual change, %: 6.8"."""
    labels = {v.get("code"): dict(zip(v.get("values", []), v.get("valueTexts", []))) for v in variables}
    dim_codes = [c["code"] for c in api_data.get("columns", []) if c.get("type") != "c"]
    rows = api_data.get("data", [])
    lines = []
    for row in rows[:MAX_ANSWER_ROWS]:
        keys = [labels.get(code, {}).get(key, key) for code, key in zip(dim_codes, row.get("key", []))]
        lines.append(f"- {', '.join(keys)}: {', '.join(row.get('values', []))}")
    if len(rows) > MAX_ANSWER_ROWS:
        lines.append(f"... and {len(rows) - MAX_ANSWER_ROWS} more rows.")
    return f"According to Statistics Iceland ({table_title}):\n" + "\n".join(lines)


def _answer_question(req: QuestionRequest):
    question_clean = clean_text(req.question)
    try:
        found = find_tables(req.question)
    except requests.RequestException as e:
        return {"answer": f"The embedding model is not reachable (is Ollama running with {EMBEDDING_MODEL}?): {e}"}
    candidates = [{"table_id": c["table"]["table_id"], "title": c["table"]["title_en"],
                   **{k: v for k, v in c.items() if k != "table"}} for c in found]
    table = found[0]["table"] if found else None
    if table is None:
        return {"answer": "I could not find a Statistics Iceland table matching this question. "
                          "Try naming the topic more precisely (e.g. population, consumer price index, migration).",
                "candidates": candidates}

    url = table["api_url"]
    is_migration = table["table_id"] == MIGRATION_TABLE_ID
    headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

    # ---- Step 1: GET the table metadata (variables and their allowed values) ----
    try:
        meta_res = requests.get(url, headers=headers, timeout=15)
        if meta_res.status_code != 200:
            return {"answer": "Could not fetch metadata from Statistics Iceland."}
        metadata = meta_res.json()
        variables = metadata.get("variables", [])
    except Exception as e:
        return {"answer": f"Connection error: {e}"}

    years = re.findall(YEAR_IN_QUESTION, question_clean)
    month = next((m_value for m_key, m_value in MONTH_MAPPING.items() if has_word(question_clean, [m_key])), None)
    result = {"table_id": table["table_id"], "table_title": table["title_en"], "table_url": url, "candidates": candidates}

    # ---- Step 2: build the filter ("query") ----
    # Variable codes stay Icelandic (e.g. "Kyn", "Ár") even in the English API; only their labels are English.
    if is_migration:
        query_list, sex_val, sex_total_value = build_migration_query(question_clean, variables, years)
    else:
        query_list, error = build_generic_query(question_clean, variables, years, month, table)
        if error:
            return {**result, "answer": error, "filters": []}

    # ---- Step 3: POST the filter to get the data ----
    try:
        data_res = requests.post(url, json={"query": query_list, "response": {"format": "json"}}, headers=headers, timeout=15)
    except Exception as e:
        return {**result, "answer": f"Data retrieval error: {e}"}
    if data_res.status_code != 200:
        return {**result, "answer": f"Statistics Iceland rejected the query (status {data_res.status_code}).", "filters": query_list}

    api_data = data_res.json()
    data_rows = api_data.get("data", [])
    if not data_rows:
        return {**result, "answer": "No data was found for this selection at Statistics Iceland.", "filters": query_list}

    # ---- Step 4: build the answer text (no LLM yet) ----
    if is_migration:
        # Sum over the selected regions (only one sex value is selected unless both were asked for)
        total_count = sum(int(row["values"][0]) for row in data_rows if row.get("values") and row["values"][0].isdigit())
        year_text = years[0] if years else "the year in question"

        if len(sex_val) > 1 or sex_total_value in sex_val:
            who = "people"
        elif sex_val == ["1"]:
            who = "men"
        else:
            who = "women"
        result_text = f"According to Statistics Iceland, {total_count} {who} moved between the regions in {year_text}."
    else:
        result_text = generic_answer_text(table["title_en"], variables, api_data)

    return {**result, "answer": result_text, "filters": query_list}
