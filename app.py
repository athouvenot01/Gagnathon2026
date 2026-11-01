from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
import requests
import unicodedata
import re
import traceback

# The only tables the chatbot can answer from.
# "keywords": words used to pick the table, written by hand (lowercase English words or phrases).
# "keyword_columns": columns (variable codes) whose values also become keywords, read from the metadata
# at startup (e.g. every species name of the catch table). Leave out general columns (months, units, regions).
TABLES = {
    "MAN05201": {
        "title": "Fertility 1853-2025",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Ibuar/Faeddirdanir/faeddir/faedingar/MAN05201.px",
        "keywords": ["fertility", "birth", "children"],
        "keyword_columns": [],
    },
    "LIF03215": {
        "title": "Households: Tenancy status by type of household 2004-2022",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/lifskjor/4_husnaedismal/0_stadaahusnaedismarkadi/LIF03215.px",
        "keywords": ["households", "tenancy status", "type of household", "renting"],
        "keyword_columns": ["Staða á húsnæðismarkaði"],  # Tenancy status: Owners, Tenants
    },
    "MEN05205": {
        "title": "Concerts in concert halls 1999-2019",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/menning/1_listir/3_tonlist/MEN05205.px",
        "keywords": ["concerts", "concert halls", "guests", "orchestra", "symphony", "choir", "opera"],
        "keyword_columns": ["Tónleikahald"],  # Concerts: Harpa, Hof, Salurinn, Ýmir...
    },
    "SJA09001": {
        "title": "Catch and value by species and fishing areas 1993-2024",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Atvinnuvegir/sjavarutvegur/aflatolur/fiskveidisvaedi/SJA09001.px",
        "keywords": ["catch", "value", "species", "fishing areas", "fish"],
        "keyword_columns": ["Fisktegund", "Veiðisvæði"],  # Species (42 names), Fishing areas
    },
    "HEI04103": {
        "title": "Hospital beds 2007-2020",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/heilbrigdismal/heilbrigdisthjonusta/HEI04103.px",
        "keywords": ["hospital beds", "hospitals", "healthcare", "health service", "patients"],
        "keyword_columns": ["Tegund"],  # Type of beds: Curative, Psychiatric...
    },
    "MEN07002": {
        "title": "Museum visits by type of museum and region 1995-2023",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/menning/2_sofn/1_sofn/MEN07002.px",
        "keywords": ["museum", "visits", "type of museum", "region", "visitors", "culture"],
        "keyword_columns": ["Tegund safns"],  # Type of museum: Art, History, Nature, Zoological...
    },
}

HEADERS = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}


def clean_text(text):
    """Normalise text: NFKD, strip accents, lowercase, then map Icelandic letters
    to ASCII (þ->th, æ->ae, ö->o, ð->d) so place names like "Reykjavík" still match."""
    nfkd_form = unicodedata.normalize('NFKD', str(text))
    flat_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
    for k, v in {'þ': 'th', 'æ': 'ae', 'ö': 'o', 'ð': 'd'}.items():
        flat_text = flat_text.replace(k, v)
    return flat_text


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

# Words ignored when matching a question against keywords and value labels
STOPWORDS = set("""a an the of in on at to for from by and or with without is was were are be been what which who whom
how many much did do does has have had there their this that these those per between into over under than as it its
me show give tell number amount iceland icelandic statistics data total year years month months
latest last current people""".split())

# Question word -> words used in the value labels of the tables (e.g. Tenancy status: "Owners", "Tenants")
SYNONYMS = {
    "renters": ["tenants"], "renting": ["tenants"], "rent": ["tenants"],
    "owning": ["owners"], "own": ["owners"],
    "reykjavik": ["capital"],
    "visits": ["guests"], "visitors": ["guests"], "visited": ["guests"],
}

YEAR_LIKE = r"(1[789]|20)\d\d([MQK]\d{1,2})?"  # "2024", "2024M03", "2024Q1", "2024K1"
YEAR_IN_QUESTION = r"\b(1[5-9]\d\d|20\d\d)\b"   # years 1500-2099 typed by the user
NOT_KEYWORDS = {"all", "other", "general"}  # words of keyword_columns values that say nothing about the topic
KEYWORD_WEIGHT = 3  # a table keyword counts 3 times more than a word found in the table's value labels


def words(text):
    """Significant lowercase words of a text (accents stripped, no stopwords).
    Number ranges are kept as one word ("20-24") to match age groups like "20-24 years"."""
    return [w for w in re.findall(r"[a-z]+|\d+-\d+", clean_text(text)) if w not in STOPWORDS and len(w) > 1]


def query_tokens(question):
    """Question words plus their synonyms, in order, without duplicates."""
    out = []
    for w in words(question):
        out += [w] + SYNONYMS.get(w, [])
    return list(dict.fromkeys(out))


def same_word(a, b):
    """Equal, or same stem for words of 5+ letters ("concert" ~ "concerts", "museum" ~ "museums")."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 5 and long.startswith(short[:max(5, len(short) - 2)])


def hits(tokens, vocab):
    """Number of question tokens found in `vocab`."""
    return sum(any(same_word(w, x) for x in vocab) for w in tokens)


# ---------------------------------------------------------------------------
# Startup: fetch the metadata of each table once
# ---------------------------------------------------------------------------

def load_metadata(table):
    """GET the table metadata (its variables and their allowed values) and index its value labels."""
    res = requests.get(table["url"], headers=HEADERS, timeout=15)
    res.raise_for_status()
    table["variables"] = res.json().get("variables", [])
    table["_vals"] = set(words(" ".join(t for v in table["variables"] for t in v.get("valueTexts", []))))
    column_words = words(" ".join(t for v in table["variables"] if v.get("code") in table["keyword_columns"]
                                  for t in v.get("valueTexts", [])))
    table["_kw"] |= set(column_words) - NOT_KEYWORDS


def load_tables():
    for table_id, table in TABLES.items():
        table["table_id"] = table_id
        table["_kw"] = set(words(" ".join(table["keywords"])))
        table["_title"] = set(words(table["title"]))
        table["_vals"] = set()
        try:
            load_metadata(table)
        except requests.RequestException as e:
            print(f"[tables] {table_id}: metadata not loaded ({e}), will retry on the first question")
    print(f"[tables] {sum('variables' in t for t in TABLES.values())}/{len(TABLES)} tables ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_tables()  # runs once, before the server accepts requests
    yield          # the server runs here


app = FastAPI(title="Statistics Iceland Smart API Server", lifespan=lifespan)

# Allow any web frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Step 1: keywords -> table
# ---------------------------------------------------------------------------

def find_table(question):
    """Score every table on the question; returns (best table or None, scores of all tables, best first).
    A table needs at least one keyword hit; words found in its value labels ("cod", "Harpa") add to the score.
    (Value labels alone are not enough: "March" is a value of the fish catch table, not a topic.)"""
    tokens = query_tokens(question)
    scores = sorted(({"table_id": t["table_id"], "title": t["title"], "keyword_hits": hits(tokens, t["_kw"]),
                      "score": KEYWORD_WEIGHT * hits(tokens, t["_kw"]) + hits(tokens, t["_vals"])}
                     for t in TABLES.values()), key=lambda c: -c["score"])
    best = scores[0]
    return (TABLES[best["table_id"]] if best["keyword_hits"] > 0 else None), scores


# ---------------------------------------------------------------------------
# Step 2: keywords -> values inside the table (the PX-API filter)
# ---------------------------------------------------------------------------

MAX_CELLS = 200       # cap on the number of values requested from PX-API
MAX_ANSWER_ROWS = 20  # rows listed in the answer text



def is_time_variable(var):
    values, texts = var.get("values", []), var.get("valueTexts", [])
    return bool(var.get("time")) or any(re.fullmatch(YEAR_LIKE, x) for x in values[:1] + texts[:1])


def is_total(label):
    """"Total", "All species", "Total hospital beds", "Years total" -- but not "Percentage of total guests"."""
    return bool(re.match(r"(total|all)\b", clean_text(label))) or not words(label)


def matched_words(tokens, label):
    return [w for w in set(words(label)) if any(same_word(w, t) for t in tokens)]


def label_score(tokens, label):
    """How well a value label matches the question: +1 per label word found, -0.1 per word not found.
    0 if no word matches ("Harpa, total concerts" beats "Harpa, total concerts, attendance" for "concerts in Harpa")."""
    # Each word counted once: "Psychiatric care beds of ... long-term care beds" repeats "care beds"
    found = len(matched_words(tokens, label))
    return found - 0.1 * (len(set(words(label))) - found) if found else 0


def build_query(question, table, years):
    """Build the PX-API filter from the table metadata. Returns (query_list, error_message)."""
    tokens = query_tokens(question)
    query_list = []
    for var in table["variables"]:
        code, values, texts = var.get("code"), var.get("values", []), var.get("valueTexts", [])
        if not values:
            continue
        if is_time_variable(var):
            if years:
                selected = [v for v, t in zip(values, texts) if any(v.startswith(y) or t.startswith(y) for y in years)]
                if not selected:
                    return [], (f"This table has no data for {', '.join(years)} "
                                f"(available: {texts[0]} to {texts[-1]}).")
            else:
                selected = [values[-1]]  # latest period
        else:
            scores = [label_score(tokens, t) for t in texts]
            best = max(scores)
            selected = [v for v, sc in zip(values, scores) if best > 0 and sc == best]
            # Purest total first: "Total" before "Total without children"
            totals = [v for v, t in sorted(zip(values, texts), key=lambda vt: len(words(vt[1]))) if is_total(t)]
            # Matched only through the table topic ("museums" in the museum table): that is not a filter, use the total.
            # Exact comparison (plural aside), so that "tenants" is not taken for the title word "tenancy".
            in_title = lambda w: w.rstrip("s") in {x.rstrip("s") for x in table["_title"]}
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


def answer_text(table, api_data):
    """List the returned values with readable labels, e.g. "2019, Harpa, total concerts: 1234"."""
    labels = {v.get("code"): dict(zip(v.get("values", []), v.get("valueTexts", []))) for v in table["variables"]}
    dim_codes = [c["code"] for c in api_data.get("columns", []) if c.get("type") != "c"]
    rows = api_data.get("data", [])
    lines = []
    for row in rows[:MAX_ANSWER_ROWS]:
        keys = [labels.get(code, {}).get(key, key) for code, key in zip(dim_codes, row.get("key", []))]
        # PX-API writes missing values as "." or ".." (e.g. Harpa concerts before Harpa opened in 2011)
        values = [v if v.strip(".") else "no data" for v in row.get("values", [])]
        lines.append(f"- {', '.join(keys)}: {', '.join(values)}")
    if len(rows) > MAX_ANSWER_ROWS:
        lines.append(f"... and {len(rows) - MAX_ANSWER_ROWS} more rows.")
    return f"According to Statistics Iceland ({table['title']}):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    # Accepts {"question": "..."} or {"prompt": "..."} (the README's curl example)
    question: str = Field(validation_alias=AliasChoices("question", "prompt"))


# "/spyrja_px_api" is the path given in the hackathon spec; "/ask" is the same endpoint.
@app.post("/ask")
@app.post("/spyrja_px_api")
def ask_px_api(req: QuestionRequest):
    try:
        return _answer_question(req.question)
    except Exception as e:
        traceback.print_exc()  # full details in the uvicorn terminal
        return {"answer": f"Sorry, I could not process this question ({type(e).__name__}: {e})."}


def _answer_question(question):
    # ---- Step 1: find the table ----
    table, candidates = find_table(question)
    if table is None:
        topics = "; ".join(t["title"] for t in TABLES.values())
        return {"answer": f"I can only answer questions about these Statistics Iceland tables: {topics}.",
                "candidates": candidates}
    if "variables" not in table:  # metadata could not be fetched at startup
        try:
            load_metadata(table)
        except requests.RequestException as e:
            return {"answer": f"Could not fetch metadata from Statistics Iceland: {e}", "candidates": candidates}

    result = {"table_id": table["table_id"], "table_title": table["title"], "table_url": table["url"],
              "candidates": candidates}

    # ---- Step 2: build the filter ("query") ----
    # Variable codes stay Icelandic (e.g. "Ár") even in the English API; only their labels are English.
    years = re.findall(YEAR_IN_QUESTION, question)
    query_list, error = build_query(question, table, years)
    if error:
        return {**result, "answer": error, "filters": []}

    # ---- Step 3: POST the filter to get the data ----
    try:
        data_res = requests.post(table["url"], json={"query": query_list, "response": {"format": "json"}},
                                 headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        return {**result, "answer": f"Data retrieval error: {e}"}
    if data_res.status_code != 200:
        return {**result, "answer": f"Statistics Iceland rejected the query (status {data_res.status_code}).",
                "filters": query_list}

    api_data = data_res.json()
    if not api_data.get("data"):
        return {**result, "answer": "No data was found for this selection at Statistics Iceland.", "filters": query_list}

    # ---- Step 4: build the answer text (no LLM yet) ----
    return {**result, "answer": answer_text(table, api_data), "filters": query_list}
