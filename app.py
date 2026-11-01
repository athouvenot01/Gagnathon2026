from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
import requests
import datetime
import unicodedata
import math
import re
import os
import threading
import time
import traceback

# The only tables the chatbot can answer from.
# "keywords": words used to pick the table, written by hand (lowercase English words or phrases).
# "keyword_columns": columns (variable codes) whose values also become keywords, read from the metadata
# at startup (e.g. every species name of the catch table). Leave out general columns (months, units, regions).
TABLES = {
    "MAN05201": {
        "title": "Fertility 1853-2025",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Ibuar/Faeddirdanir/faeddir/faedingar/MAN05201.px",
        "keywords": ["fertility", "birth", "children", "births", "born", "baby", "babies", "newborns",
                     "childbirth", "mothers", "maternity", "birth rate"],
        "keyword_columns": [],
    },
    "LIF03215": {
        "title": "Households: Tenancy status by type of household 2004-2022",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/lifskjor/4_husnaedismal/0_stadaahusnaedismarkadi/LIF03215.px",
        "keywords": ["households", "tenancy status", "type of household", "renting", "housing", "home",
                     "homes", "homeownership", "rental", "rented", "dwellings", "accommodation", "landlord"],
        "keyword_columns": ["Staða á húsnæðismarkaði"],  # Tenancy status: Owners, Tenants
    },
    "MEN05205": {
        "title": "Concerts in concert halls 1999-2019",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/menning/1_listir/3_tonlist/MEN05205.px",
        "keywords": ["concerts", "concert halls", "guests", "orchestra", "symphony", "choir", "opera", "concert",
                     "music", "musical", "gigs", "performances", "audience", "jazz", "classical", "recital"],
        "keyword_columns": ["Tónleikahald"],  # Concerts: Harpa, Hof, Salurinn, Ýmir...
    },
    "SJA09001": {
        "title": "Catch and value by species and fishing areas 1993-2024",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Atvinnuvegir/sjavarutvegur/aflatolur/fiskveidisvaedi/SJA09001.px",
        "keywords": ["catch", "value", "species", "fishing areas", "fish", "fishing", "fisheries", "fishery",
                     "fishermen", "caught", "landings", "landed", "seafood", "trawl", "tonnes"],
        "keyword_columns": ["Fisktegund", "Veiðisvæði"],  # Species (42 names), Fishing areas
    },
    "HEI04103": {
        "title": "Hospital beds 2007-2020",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/heilbrigdismal/heilbrigdisthjonusta/HEI04103.px",
        "keywords": ["hospital beds", "hospitals", "healthcare", "health service", "patients", "hospital", "bed",
                     "inpatient", "clinics", "medical", "wards", "nursing", "infirmary", "health care"],
        "keyword_columns": ["Tegund"],  # Type of beds: Curative, Psychiatric...
    },
    "MEN07002": {
        "title": "Museum visits by type of museum and region 1995-2023",
        "url": "https://px.hagstofa.is:443/pxen/api/v1/en/Samfelag/menning/2_sofn/1_sofn/MEN07002.px",
        "keywords": ["museum", "visits", "type of museum", "region", "visitors", "culture", "museums", "zoo", "zoos",
                     "exhibitions", "gallery", "galleries", "botanical gardens", "aquarium", "heritage", "attractions"],
        "keyword_columns": ["Tegund safns"],  # Type of museum: Art, History, Nature, Zoological...
    },
}

HEADERS = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

# LLM (Ollama) that turns the values into a sentence. USE_LLM=0 disables it (raw list of values instead).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
USE_LLM = os.environ.get("USE_LLM", "1") != "0"
LLM_TIMEOUT = 60  # seconds; on timeout or error the raw list of values is returned
# FastAPI runs each request in its own thread, so questions are answered concurrently; the lock lets
# only one of them use the LLM at a time (the GPU), the PX-API calls still run in parallel.
ollama_lock = threading.Lock()


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
    "zoo": ["zoological", "museums"], "zoos": ["zoological", "museums"],  # "Zoological museums"
    "gallery": ["art", "museums"], "galleries": ["art", "museums"],      # "Art museums"
    "bed": ["beds"],  # too short for same_word() to match "beds"
}

# Weight of a word when scoring the tables, written by hand. A word not listed weighs DEFAULT_WEIGHT.
# HIGH: words that name one topic (a fish species, "museum", "fertility"): they decide the table.
# LOW: words that many tables can have (regions, "value", "rate", "guests"): they only help to choose
# between tables, and are not enough to add a second table to the answer.
# (rarity() below computes weights automatically from the tables; not used for now: USE_RARITY.)
HIGH, LOW, DEFAULT_WEIGHT = 2.0, 0.3, 1.0
WORD_WEIGHTS = {
    **dict.fromkeys([
        # Fertility
        "fertility", "birth", "births", "born", "baby", "babies", "newborns", "childbirth", "maternity",
        # Households
        "households", "household", "tenants", "owners", "tenancy", "homeownership", "landlord",
        "renting", "rental", "rented",
        # Concerts
        "concerts", "concert", "harpa", "hof", "salurinn", "ymir", "orchestra", "symphony", "choir", "opera",
        "jazz", "recital", "gigs",
        # Fish catch, and the species
        "fish", "fishing", "fisheries", "fishery", "fishermen", "catch", "caught", "landings", "landed",
        "seafood", "trawl", "cod", "haddock", "herring", "capelin", "mackerel", "saithe", "redfish", "halibut",
        "plaice", "ling", "tusk", "monkfish", "wolffish", "catfish", "shrimp", "lobster", "scallop", "whiting",
        "megrim", "dab", "skate", "ray", "shark", "dogfish", "grenadier", "smelt", "sole", "flatfish",
        "shellfish", "pelagic", "demersal", "cyprine", "whitch",
        # Hospital beds
        "hospital", "hospitals", "beds", "bed", "inpatient", "psychiatric", "curative", "rehabilitative",
        "rehabil", "infirmary", "wards",
        # Museums
        "museum", "museums", "exhibitions", "zoo", "zoos", "zoological", "botanical", "aquarium",
        "gallery", "galleries",
    ], HIGH),
    **dict.fromkeys([
        # Regions and places
        "region", "regions", "capital", "reykjavik", "north", "south", "east", "west", "northern", "southern",
        "eastern", "western", "northeast", "northwest", "southwest", "westfjords", "areas", "zone", "grounds",
        "sea", "ocean", "atlantic", "greenland", "norway", "norwegian", "russian", "barents", "flemish",
        # Measures and units
        "value", "rate", "type", "status", "tonnes", "isk", "percentage", "inhabitants", "estimated",
        # Words that several topics use
        "home", "homes", "housing", "dwellings", "accommodation", "children", "mothers", "guests", "visits",
        "visitors", "attendance", "audience", "general", "student", "music", "musical", "performances",
        "classical", "culture", "heritage", "history", "nature", "art", "collection", "attractions",
        "health", "healthcare", "medical", "nursing", "patients", "service", "clinics", "care", "long", "term",
        "species",
        # Parts of fish names ("Long rough dab", "Atlantic spring-spawning herring")
        "blue", "silver", "spring", "spawning", "rough", "spotted", "starry", "spiny", "lemon", "roe",
    ], LOW),
}
USE_RARITY = False  # True: weights computed by rarity() instead of WORD_WEIGHTS

# Relative years: phrase -> number of years before the current year ("last year" = 2025 in 2026)
RELATIVE_YEARS = {
    "this year": 0, "current year": 0, "present year": 0, "so far this year": 0,
    "last year": 1, "previous year": 1, "past year": 1, "prior year": 1, "a year ago": 1,
    "one year ago": 1, "1 year ago": 1, "1 years ago": 1,
    "year before last": 2, "the year before last": 2, "two years ago": 2, "2 years ago": 2,
    "three years ago": 3, "3 years ago": 3,
    "four years ago": 4, "4 years ago": 4,
    "five years ago": 5, "5 years ago": 5,
    "six years ago": 6, "6 years ago": 6,
    "seven years ago": 7, "7 years ago": 7,
    "eight years ago": 8, "8 years ago": 8,
    "nine years ago": 9, "9 years ago": 9,
    "ten years ago": 10, "10 years ago": 10, "a decade ago": 10,
}

YEAR_LIKE = r"(1[789]|20)\d\d([MQK]\d{1,2})?"  # "2024", "2024M03", "2024Q1", "2024K1"
YEAR_IN_QUESTION = r"\b(1[5-9]\d\d|20\d\d)\b"   # years 1500-2099 typed by the user
NOT_KEYWORDS = {"all", "other", "general"}  # words of keyword_columns values that say nothing about the topic
KEYWORD_WEIGHT = 3  # a table keyword counts 3 times more than a word found in the table's value labels
MIN_EXTRA_SCORE = 1.5  # score needed on the words left for a second (third...) table to be added:
                       # one LOW keyword (3 x 0.3) is not enough, one other keyword (3 x 1 or more) is


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


def warm_up_llm():
    """Load the model into GPU memory now, so that the first question does not wait for it."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "keep_alive": -1}, timeout=300)
        print(f"[llm] {OLLAMA_MODEL} ready")
    except requests.RequestException as e:
        print(f"[llm] {OLLAMA_MODEL} not available ({e}), answers will be raw values")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_tables()  # runs once, before the server accepts requests
    if USE_LLM:
        threading.Thread(target=warm_up_llm, daemon=True).start()
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

def in_table(word, table):
    """The word is a keyword of the table or a word of its value labels."""
    return any(same_word(word, x) for x in table["_kw"] | table["_vals"])


def rarity(word):
    """Weight of a word by how few tables use it (BM25 idf): "fertility" (1 table of 6) weighs 1.5,
    "guests" (2 tables) 1.0, a word in every table almost 0. With many tables, common words count for nothing."""
    n = len(TABLES)
    df = sum(in_table(word, t) for t in TABLES.values())
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def word_weight(word):
    """Weight of a word: from WORD_WEIGHTS (the word, or the same word in plural/singular), else DEFAULT_WEIGHT."""
    if USE_RARITY:
        return rarity(word)
    if word in WORD_WEIGHTS:
        return WORD_WEIGHTS[word]
    return next((wt for w, wt in WORD_WEIGHTS.items() if same_word(word, w)), DEFAULT_WEIGHT)


def score_tables(tokens):
    """Score every table on the tokens, best first. A keyword counts KEYWORD_WEIGHT times a value-label word,
    and each word is weighted by word_weight(). A table needs at least one keyword hit to be chosen;
    words found in its value labels ("cod", "Harpa") add to the score.
    (Value labels alone are not enough: "March" is a value of the fish catch table, not a topic.)"""
    weights = {w: word_weight(w) for w in tokens}
    scores = []
    for t in TABLES.values():
        kw = [w for w in tokens if any(same_word(w, x) for x in t["_kw"])]
        vals = [w for w in tokens if w not in kw and any(same_word(w, x) for x in t["_vals"])]
        score = KEYWORD_WEIGHT * sum(weights[w] for w in kw) + sum(weights[w] for w in vals)
        scores.append({"table_id": t["table_id"], "title": t["title"], "keyword_hits": len(kw),
                       "words": kw + vals, "score": round(score, 2)})
    return sorted(scores, key=lambda c: -c["score"])


def find_table(question):
    """Best table for the question, or None; returns (table, scores of all tables, best first)."""
    scores = score_tables(query_tokens(question))
    best = scores[0]
    return (TABLES[best["table_id"]] if best["keyword_hits"] > 0 else None), scores


def find_tables(question):
    """Tables for the question, best first, so that one question can ask about several topics
    ("hospital beds in 2020, and museum guests that year"). Returns (tables, scores of all tables).
    Tables are chosen one by one: once a table is chosen, the words it explains (its keywords and value labels)
    are removed, and the next table must score on the words left. So "guests" in "guests visited museums"
    is explained by the museum table and does not also bring in the concert table."""
    tokens = query_tokens(question)
    scores = score_tables(tokens)
    chosen = []
    while True:
        left = [w for w in tokens if not any(in_table(w, t) for t in chosen)]
        best = next((c for c in score_tables(left) if TABLES[c["table_id"]] not in chosen), None)
        if not best or best["keyword_hits"] == 0 or (chosen and best["score"] < MIN_EXTRA_SCORE):
            return chosen, scores
        chosen.append(TABLES[best["table_id"]])
        for c in scores:  # shown in the answer: why each extra table was added
            if c["table_id"] == best["table_id"]:
                c["chosen_on"] = best["words"]


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
    several_parts = bool(re.search(r"\b(and|og)\b", question, re.IGNORECASE))  # the question asks several things
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
            elif several_parts and totals and totals[0] not in selected:
                # "museums in the Westfjords, and art museums in all of Iceland": one selection per variable
                # would only give art museums in the Westfjords. With the totals too, the rows include
                # Westfjords x all museums and all of Iceland x art museums; the LLM reads the right ones.
                selected = selected + totals[:1]
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


def table_rows(table, api_data):
    """Rows of the PX-API answer as (labels, value), e.g. (["2019", "Harpa, total concerts"], "539")."""
    labels = {v.get("code"): dict(zip(v.get("values", []), v.get("valueTexts", []))) for v in table["variables"]}
    dim_codes = [c["code"] for c in api_data.get("columns", []) if c.get("type") != "c"]
    rows = []
    for row in api_data.get("data", []):
        keys = [labels.get(code, {}).get(key, key) for code, key in zip(dim_codes, row.get("key", []))]
        # PX-API writes missing values as "." or ".." (e.g. Harpa concerts before Harpa opened in 2011)
        values = [v if v.strip(".") else "no data" for v in row.get("values", [])]
        rows.append((keys, ", ".join(values)))
    return rows


def answer_text(table, rows, calcs=()):
    """List the values with readable labels, e.g. "2019, Harpa, total concerts: 1234", then the calculations."""
    lines = [f"- {', '.join(keys)}: {value}" for keys, value in rows[:MAX_ANSWER_ROWS]]
    if len(rows) > MAX_ANSWER_ROWS:
        lines.append(f"... and {len(rows) - MAX_ANSWER_ROWS} more rows.")
    if calcs:
        lines += ["Calculations (done by the program, not by the LLM):"] + [f"- {c}" for c in calcs]
    return f"According to Statistics Iceland ({table['title']}):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Calculations: done here, never by the LLM (it only copies the results)
# ---------------------------------------------------------------------------

# Words of the question -> calculation. Matched on clean_text(question) (accents stripped).
CALCULATION_WORDS = {
    "difference": r"\b(change[sd]?|differen\w*|more|less|fewer|higher|lower|increase[sd]?|decrease[sd]?|"
                  r"gr[eo]w\w*|drop\w*|ris[ei]n?g?|rose|f[ae]ll\w*|gap|breyt\w*|munur\w*|fleiri|faerri|haerr\w*)\b",
    "sum": r"\b(together|combined|sum|altogether|samtals|samanlagt)\b",
    "ratio": r"\b(average|per|each|share|percentage|percent|proportion|ratio|medaltal\w*|hlutfall\w*)\b",
}
MAX_CALCULATIONS = 15


def decimals(value):
    return len(value.split(".")[1]) if "." in value else 0


def calculations(question, rows, unit_columns=()):
    """Differences, sums and ratios between rows that differ by one label only (2023 vs 2024 for the same
    species), when the question asks for one. Returns lines "labels: result" like the value lines, e.g.
    "Cod, All fishing areas, Tonnes, Years total, 2024 minus 2023: 1396".
    Rows that differ by their unit (column numbers in unit_columns) are only divided (value / tonnes),
    never subtracted or added (guests minus museums means nothing)."""
    q = clean_text(question)
    asked = [kind for kind, pattern in CALCULATION_WORDS.items() if re.search(pattern, q)]
    rows = [(keys, value) for keys, value in rows if re.fullmatch(r"-?\d+(\.\d+)?", value)]  # not "no data"
    if not asked or len(rows) < 2:
        return []
    calcs = []
    for i in range(len(rows[0][0])):  # rows that differ only by their label number i
        groups = {}
        for keys, value in rows:
            groups.setdefault(tuple(keys[:i] + keys[i + 1:]), []).append((keys[i], value))
        for common, members in groups.items():
            common = ", ".join(common)
            parts = [(label, value) for label, value in members if not is_total(label)]
            for a, (la, va) in enumerate(members):
                for lb, vb in members[a + 1:]:
                    if "difference" in asked and i not in unit_columns:
                        d = max(decimals(va), decimals(vb))
                        calcs.append(f"{common}, {lb} minus {la}: {float(vb) - float(va):.{d}f}")
                    if "ratio" in asked and float(va) and float(vb):
                        calcs.append(f"{common}, {la} divided by {lb}: {float(va) / float(vb):.2f}")
                        calcs.append(f"{common}, {lb} divided by {la}: {float(vb) / float(va):.2f}")
                        if is_total(lb):  # share of the total
                            calcs.append(f"{common}, {la} as a percentage of {lb}: {100 * float(va) / float(vb):.1f}")
            if "sum" in asked and len(parts) >= 2 and i not in unit_columns:
                d = max(decimals(v) for _, v in parts)
                calcs.append(f"{common}, {' plus '.join(l for l, _ in parts)}: {sum(float(v) for _, v in parts):.{d}f}")
    return calcs[:MAX_CALCULATIONS]


LLM_PROMPT = """You answer questions about official statistics of Iceland.
Write a short answer (1 to 4 sentences) to the question, using ONLY the values below.
If the question asks several things, answer each of them.
Rules:
- Never compute anything yourself. If the question asks for a difference, a change, a sum, an average,
  a ratio or a share, use the matching line under "Calculations" (computed by the program) and copy its result.
  A negative difference is a decrease: "2020 minus 2007: -244" -> "decreased by 244".
  If no line of the values or calculations answers the question, say which values are missing.
- Copy the numbers exactly as written; do not round, convert or invent any number.
  You may only add thousands separators (277511 -> 277,511).
- A unit in thousands stays in thousands: "Value (1000 ISK): 18595190" -> "18,595,190 thousand ISK".
- Give the unit when the labels give one (e.g. tonnes, ISK, %, number of guests) and the year.
- Labels such as "Total", "All ...", "Years total" mean "all categories together" or "the whole year":
  they are not a separate quantity; leave them out of the sentence or say "in total".
- Write natural sentences: do not copy the labels as a list.
- If the values do not answer the question exactly, say what they show instead.
- Only a value written "no data" is missing: then say so. Never mention data that is not listed below.
- Answer in the language of the question. No introduction, no source, no markdown.

Question: {question}

{tables}"""

LLM_TABLE = """Table: {title}
{year_note}
Values (labels: value):
{values}"""


def llm_answer(question, parts, values_text):
    """Turn the values of all the tables into one answer with the LLM. Returns None if the LLM is off or fails."""
    if not USE_LLM:
        return None
    tables = "\n\n".join(LLM_TABLE.format(title=p["table_title"], year_note=p["year_note"],
                                          values=p["values"].split("\n", 1)[1])  # drop the "According to..." header
                         for p in parts)
    prompt = LLM_PROMPT.format(question=question, tables=tables)
    try:
        with ollama_lock:
            res = requests.post(f"{OLLAMA_URL}/api/chat", timeout=LLM_TIMEOUT, json={
                "model": OLLAMA_MODEL, "stream": False, "keep_alive": -1,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.2, "num_predict": 300},
            })
        res.raise_for_status()
        sentence = res.json()["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[llm] failed ({e}), returning the raw values")
        return None
    return sentence if numbers_are_safe(sentence, question, values_text) else None


def numbers_in(text):
    """Numbers of a text, as floats ("171,180,664" -> 171180664.0, "20-24" -> 20.0 and 24.0).
    "1.032" is read as 1032: Icelandic writes thousands with a dot."""
    found = re.findall(r"\d[\d,]*(?:\.\d+)*", text)
    return {float(n.replace(".", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+", n) else n.replace(",", ""))
            for n in found}


def numbers_are_safe(sentence, question, values_text):
    """Reject the sentence if it has a number that is neither in the values nor in the question
    (a changed or invented value), or if it gives none of the values (when there are some)."""
    # not the headers: their titles have year ranges ("1993-2024")
    values_text = re.sub(r"^According to .*\n", "", values_text, flags=re.M)
    allowed = numbers_in(values_text) | numbers_in(question)
    invented = numbers_in(sentence) - allowed
    # The values, and the results of the calculations ("2024 minus 2023: -244" may be written "decreased by 244")
    values = {float(v.replace(",", "")) for v in re.findall(r": -?([\d.,]+)$", values_text, re.M)}
    if invented or (values and not numbers_in(sentence) & values):
        print(f"[llm] sentence rejected (invented {sorted(invented)}): {sentence}")
        return False
    # "Value (1000 ISK): 18595190" written "18,595,190 ISK" would be 1000 times too small
    if re.search(r"\(1000 ", values_text) and not re.search(r"thousand|þúsund|1[,.]?000 ISK", sentence, re.I):
        print(f"[llm] sentence rejected (unit in thousands lost): {sentence}")
        return False
    return True


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


def with_years(question):
    """Replace each relative phrase of RELATIVE_YEARS by its year: "catch 3 years ago" -> "catch in 2023",
    "in the previous year" -> "in 2025". The LLM then reads the year instead of computing it
    (it does not know today's date, and still computed 2022 for "two years ago (2024)")."""
    now = datetime.date.today().year
    # One regex, longest phrases first: "so far this year" is not matched again as "this year".
    # \b: "11 years ago" does not contain "1 years ago".
    phrases = sorted(RELATIVE_YEARS, key=len, reverse=True)
    pattern = r"\b(?:(?:in|during|for) )?(?:the )?(" + "|".join(re.escape(p) for p in phrases) + r")\b"
    return re.sub(pattern, lambda m: f"in {now - RELATIVE_YEARS[m.group(1).lower()]}", question, flags=re.IGNORECASE)


def requested_years(question):
    """Years typed in the question, plus relative ones from RELATIVE_YEARS: "last year" (2025 in 2026), "3 years ago"."""
    return list(dict.fromkeys(re.findall(YEAR_IN_QUESTION, with_years(question))))


def latest_year(table):
    """Year of the last period of the table ("2020", or "2024" for "2024M12"), None if it has no time variable."""
    for var in table["variables"]:
        if is_time_variable(var) and var.get("valueTexts"):
            return var["valueTexts"][-1][:4]
    return None


def table_values(question, table, years):
    """Steps 2 and 3 for one table: filter, POST, values. Returns a dict with "values", or with "error"."""
    result = {"table_id": table["table_id"], "table_title": table["title"], "table_url": table["url"]}
    if "variables" not in table:  # metadata could not be fetched at startup
        try:
            load_metadata(table)
        except requests.RequestException as e:
            return {**result, "error": f"Could not fetch metadata from Statistics Iceland: {e}"}

    # ---- Step 2: build the filter ("query") ----
    # Variable codes stay Icelandic (e.g. "Ár") even in the English API; only their labels are English.
    # Told to the LLM, which does not know today's date ("last year", "10 years ago")
    year_note = (f"(Year asked: {', '.join(years)}.)" if years else
                 "(No year was asked: the values below are the latest available; say which year.)")
    # A year after the end of the table ("last year" = 2025 in a table ending in 2020): give the latest year instead
    note = ""
    latest = latest_year(table)
    if years and latest and all(y > latest for y in years):
        note = f"Note: {table['title']} has no data for {', '.join(years)} yet; the latest year is {latest}.\n\n"
        year_note = (f"(The user has already been told there is no data for {', '.join(years)}: "
                     f"just answer with the values below, which are for the single year {latest}, "
                     f"and do not mention {', '.join(years)}.)")
        years = []
    query_list, error = build_query(question, table, years)
    if error:
        return {**result, "error": error, "filters": []}

    # ---- Step 3: POST the filter to get the data ----
    try:
        data_res = requests.post(table["url"], json={"query": query_list, "response": {"format": "json"}},
                                 headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        return {**result, "error": f"Data retrieval error: {e}"}
    if data_res.status_code != 200:
        return {**result, "error": f"Statistics Iceland rejected the query (status {data_res.status_code}).",
                "filters": query_list}
    api_data = data_res.json()
    if not api_data.get("data"):
        return {**result, "error": "No data was found for this selection at Statistics Iceland.", "filters": query_list}
    rows = table_rows(table, api_data)
    dims = [c.get("text", "") for c in api_data.get("columns", []) if c.get("type") != "c"]
    unit_columns = {i for i, text in enumerate(dims) if clean_text(text) in ("unit", "eining")}
    calcs = calculations(question, rows, unit_columns)
    if not calcs and any(re.search(p, clean_text(question)) for p in CALCULATION_WORDS.values()):
        # Without this, "average value of one kilogram of cod" was answered with the total value of the catch
        year_note += (" (The program could not compute anything from these values. If the question asks for "
                      "a difference, sum, average or ratio, do not answer it with one of the values: "
                      "say that the values needed for the calculation are missing.)")
    return {**result, "filters": query_list, "values": answer_text(table, rows, calcs),
            "note": note, "year_note": year_note}


def _answer_question(question):
    # ---- Step 1: find the tables (one question can ask about several topics) ----
    tables, candidates = find_tables(question)
    if not tables:
        return {"answer": "I can't answer that question because I don't have the information needed.",
                "candidates": candidates}

    # Steps 2 and 3 for each table; a table that gives no values is left out (a keyword can match by chance),
    # its error is only shown when no table gives values.
    years = requested_years(question)
    results = [table_values(question, table, years) for table in tables]
    parts = [r for r in results if "values" in r] or results[:1]
    main = parts[0]
    result = {"table_id": main["table_id"], "table_title": main["table_title"], "table_url": main["table_url"],
              "filters": main.get("filters", []), "candidates": candidates,
              "tables": [{k: p.get(k) for k in ("table_id", "table_title", "table_url", "filters")} for p in parts]}
    if "error" in main:
        return {**result, "answer": main["error"]}

    # ---- Step 4: the LLM writes one answer from all the values; the sources and filters are added by the code ----
    note = "".join(p["note"] for p in parts)
    values_text = "\n\n".join(p["values"] for p in parts)
    started = time.perf_counter()
    sentence = llm_answer(with_years(question), parts, values_text)
    llm_seconds = round(time.perf_counter() - started, 2)
    if sentence is None:
        return {**result, "answer": note + values_text, "values": values_text, "llm": None}
    sources = "; ".join(p["table_title"] for p in parts)
    return {**result, "answer": f"{note}{sentence}\n\nSource: Statistics Iceland, {sources}.",
            "values": values_text, "llm": {"model": OLLAMA_MODEL, "seconds": llm_seconds}}
