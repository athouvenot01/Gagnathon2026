from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
import requests
import unicodedata
import re
import traceback

app = FastAPI(title="Statistics Iceland Smart API Server")

# Allow any web frontend to call this API (fine for a hackathon, too open for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mini "catalog" of the three PX-API tables supported by this starter
# (original name: HELSTU_TAFLUR = "main tables").
MAIN_TABLES = {
    "cpi": {
        "text": "Consumer price index",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Efnahagur/visitolur/1_vnv/1_vnv/VIS01002.px"
    },
    "population": {
        "text": "Population by sex and age",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00000.px"
    },
    "migration": {
        "text": "Domestic migration between regions",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Ibuar/buferlaflutningar/buferlaflinnanlands/buferlaflinnanlands/MAN01002.px"
    }
    # Here you can easily add more tables when you find their URL.
}

# Region name (as typed by the user, accent-stripped) -> PX-API region code.
# 1 = Capital region, 2 = Southern Peninsula, 3 = West, 4 = Westfjords,
# 5 = Northwest, 6 = Northeast, 7 = East, 8 = South
REGION_MAPPING = {
    "hofudborgarsvad": "1", "reykjavik": "1", "sudurnes": "2",
    "vesturland": "3", "vestfirdir": "4", "nordurland vestra": "5",
    "nordurland eystra": "6", "austurland": "7", "sudurland": "8",
    # English names
    "capital region": "1", "southern peninsula": "2", "west iceland": "3",
    "westfjords": "4", "west fjords": "4", "northwest": "5", "northeast": "6",
    "east iceland": "7", "south iceland": "8"
}

# Month names (Icelandic + English, accent-stripped) -> two-digit month number.
# Matched as WHOLE WORDS, so "mar" no longer fires inside words like "market".
MONTH_MAPPING = {
    "january": "01", "januar": "01", "jan": "01",
    "february": "02", "februar": "02", "feb": "02", "febr": "02",
    "march": "03", "mars": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05", "mai": "05",
    "june": "06", "juni": "06", "jun": "06",
    "july": "07", "juli": "07", "jul": "07",
    "august": "08", "agust": "08", "agv": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oktober": "10", "okt": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "desember": "12", "des": "12", "dec": "12",
}


def clean_text(text):
    """Normalise text (original name: hreinsa_texta): NFKD, strip accents, lowercase,
    then map Icelandic letters to ASCII (þ->th, æ->ae, ö->o, ð->d)."""
    nfkd_form = unicodedata.normalize('NFKD', str(text))
    flat_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
    for k, v in {'þ': 'th', 'æ': 'ae', 'ö': 'o', 'ð': 'd'}.items():
        flat_text = flat_text.replace(k, v)
    return flat_text


def has_word(text, words):
    """True if any of `words` appears in `text` as a whole word
    (so that 'men' does not match inside 'women')."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def to_float(value):
    """float(value), or None for non-numeric entries such as '..' (missing data in PX-API)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_table(question):
    """Pick a table by keyword matching (original name: finna_taflu).
    Keywords are Icelandic because users ask in Icelandic."""
    q = clean_text(question)
    # moved / lived / migration / regions / "landshluta" (parts of the country) / men / women
    if any(k in q for k in ["flutti", "flutt", "buid", "buferla", "fjords", "landshluta", "karlar", "konur",
                          "moved", "moving", "migrat", "relocat"]):
        return MAIN_TABLES["migration"]
    # price index / consumer price / price / inflation
    elif any(k in q for k in ["visitala", "neysluverds", "verd", "verdbolga",
                          "price", "inflation", "cpi", "cost of living"]):
        return MAIN_TABLES["cpi"]
    # population / inhabitants
    elif any(k in q for k in ["mannfjoldi", "mannfjolda", "ibuar",
                          "population", "inhabitants", "residents"]):
        return MAIN_TABLES["population"]
    return MAIN_TABLES["cpi"]  # default table


class QuestionRequest(BaseModel):
    # Accepts {"spurning": "..."} (original), {"prompt": "..."} or {"question": "..."}
    spurning: str = Field(validation_alias=AliasChoices("spurning", "prompt", "question"))


# Endpoint path kept as in the original ("spyrja" = "ask")
@app.post("/spyrja_px_api")
def ask_px_api(req: QuestionRequest):
    try:
        return _answer_question(req)
    except Exception as e:
        traceback.print_exc()  # full details in the uvicorn terminal
        return {"svar": f"Sorry, I could not process this question ({type(e).__name__}: {e})."}


def _answer_question(req: QuestionRequest):
    selected_table = find_table(req.spurning)
    url = selected_table["url"]
    is_migration = (url == MAIN_TABLES["migration"]["url"])
    is_cpi = (url == MAIN_TABLES["cpi"]["url"])
    is_population = (url == MAIN_TABLES["population"]["url"])

    headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

    # ---- Step 1: GET the table metadata (variables and their allowed values) ----
    try:
        meta_res = requests.get(url, headers=headers, timeout=15)
        if meta_res.status_code != 200:
            return {"svar": "Could not fetch metadata from Statistics Iceland."}
        metadata = meta_res.json()
        variables = metadata.get("variables", [])
    except Exception as e:
        return {"svar": f"Connection error: {e}"}

    question_clean = clean_text(req.spurning)
    query_list = []
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', question_clean)  # 4-digit years 1900-2099
    is_inflation = any(w in question_clean for w in ["verdbolga", "verdbolgun", "inflation"])

    # ---- Step 2: build the filter ("query") depending on the table ----
    if is_migration:
        # Default variable codes; overwritten below if found in the metadata
        sex_code, year_code, departure_code, arrival_code, age_code, age_total = "Kyn", "Ár", "Brottfararsvæði", "Aðflutningssvæði", None, None
        sex_total_value = None

        for var in variables:
            v_code, v_text = var.get("code"), clean_text(var.get("text", ""))
            v_vals, v_vtexts = var.get("values", []), var.get("valueTexts", [])
            if "kyn" in v_text:  # "kyn" = sex
                sex_code = v_code
                for idx, vt in enumerate(v_vtexts):
                    if any(w in clean_text(vt) for w in ["alls", "samtals", "allir"]):  # "total"
                        sex_total_value = v_vals[idx]
                        break
            elif "ar" in v_text:  # "ár" = year
                year_code = v_code
            elif "brott" in v_text:  # "brottfarar..." = departure
                departure_code = v_code
            elif "adflutnings" in v_text or "til" in v_text:  # "aðflutnings..." / "til" = arrival / to
                if "brott" not in v_text:
                    arrival_code = v_code
            elif "aldur" in v_text:  # "aldur" = age
                age_code = v_code
                for idx, vt in enumerate(v_vtexts):
                    if any(w in clean_text(vt) for w in ["alls", "samtals", "allir"]):
                        age_total = v_vals[idx]
                        break
                if not age_total and v_vals:
                    age_total = v_vals[0]

        # "karl(ar)" = man/men, "kona/konur" = woman/women
        mentions_men = has_word(question_clean, ["karlar", "karl", "karla", "karlmenn", "men", "male", "males"])
        mentions_women = has_word(question_clean, ["konur", "kona", "kvenna", "women", "woman", "female", "females"])
        # "heildina / alls / samtals" = "overall / in total"
        asks_total = has_word(question_clean, ["heildina", "alls", "samtals", "total", "overall", "altogether"])

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

        query_list.append({"code": sex_code, "selection": {"filter": "item", "values": sex_val}})
        if age_code and age_total:
            query_list.append({"code": age_code, "selection": {"filter": "item", "values": [age_total]}})
        if years:
            query_list.append({"code": year_code, "selection": {"filter": "item", "values": years}})

        # Find regions in the question, ordered by where they appear in the sentence
        found_positions = sorted(
            (m.start(), code)
            for s, code in REGION_MAPPING.items()
            if (m := re.search(rf"\b{re.escape(s)}", question_clean))
        )
        regions = []
        for _, k in found_positions:
            if not regions or regions[-1] != k:
                regions.append(k)

        # Two regions: first = origin, second = destination.
        # One region: assume a default for the other side. None: hardcoded default (East -> Capital).
        origin, destination = (regions[0], regions[1]) if len(regions) >= 2 else (regions[0] if regions else "7", "1")
        # "fra" = "from". If the user did NOT say "from" and gave one region, treat it as the destination
        if not has_word(question_clean, ["fra", "from"]) and len(regions) == 1:
            origin, destination = "7", regions[0]

        query_list.append({"code": departure_code, "selection": {"filter": "item", "values": [origin]}})
        query_list.append({"code": arrival_code, "selection": {"filter": "item", "values": [destination]}})

    elif is_cpi:
        # "manudur" = month
        month_found = next((m_value for m_key, m_value in MONTH_MAPPING.items() if has_word(question_clean, [m_key])), None)

        # For inflation we need both the requested month AND the same month one year earlier
        month_list = []
        if month_found and years:
            chosen_year = int(years[0])
            month_list = [f"{chosen_year}M{month_found}", f"{chosen_year - 1}M{month_found}"]
        elif years:
            month_list = years

        for var in variables:
            v_code, v_text, v_vals = var.get("code"), clean_text(var.get("text", "")), var.get("values", [])
            if "ar" in v_text:  # year
                if years and not month_found:
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": years}})
            elif "manudur" in v_text:  # month
                if month_list:
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": month_list}})
            elif "grunnur" in v_text:  # "grunnur" = base period of the index
                if v_vals:
                    # Skip two old base periods, otherwise take the last available one
                    base_val = next((g for g in v_vals if g not in ["B1939", "B1988"]), v_vals[-1])
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": [base_val]}})
            else:
                # Any other variable: just take its first value
                if v_vals:
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": [v_vals[0]]}})

    elif is_population:
        year_code = next((var.get("code") for var in variables if "ar" in clean_text(var.get("text", ""))), "Ár")
        # Requested year(s), or the latest year available if none given
        target_years = years if years else [variables[0].get("values", [""])[-1]]
        query_list.append({"code": year_code, "selection": {"filter": "item", "values": target_years}})

    # ---- Step 3: POST the filter to get the data ----
    try:
        data_res = requests.post(url, json={"query": query_list, "response": {"format": "json"}}, headers=headers, timeout=15)
        if data_res.status_code != 200:
            # Fallback: retry with no filter at all (can return a very large result)
            data_res = requests.post(url, json={"query": [], "response": {"format": "json"}}, headers=headers, timeout=15)
            query_list = []
    except Exception as e:
        return {"svar": f"Data retrieval error: {e}"}

    if data_res.status_code != 200:
        return {"svar": f"An error occurred while retrieving data (status {data_res.status_code})."}

    api_data = data_res.json()
    data_rows = api_data.get("data", [])

    # ---- Step 4: build the answer text with hardcoded templates (no LLM yet) ----
    if is_migration:
        total_count = 0
        men, women = 0, 0
        for row in data_rows:
            keys, vals = row.get("key", []), row.get("values", ["0"])
            try:
                number = int(vals[0])
            except Exception:
                continue

            total_count += number
            if "1" in keys and "2" not in keys:
                men += number
            elif "2" in keys:
                women += number

        year_text = years[0] if years else "the year in question"

        if len(sex_val) > 1 or sex_total_value in sex_val or "0" in sex_val:
            total_number = total_count if total_count > 0 else int(data_rows[0].get('values', [0])[0])
            result_text = f"According to Statistics Iceland, a total of {total_number} people moved between the regions in {year_text}."
        elif sex_val == ["1"]:
            result_text = f"According to Statistics Iceland, {men or int(data_rows[0].get('values', [0])[0])} men moved between the regions in {year_text}."
        else:
            result_text = f"According to Statistics Iceland, {women or int(data_rows[0].get('values', [0])[0])} women moved between the regions in {year_text}."

    elif is_cpi:
        if not data_rows:
            return {"svar": "No data was found for this period at Statistics Iceland."}

        if is_inflation and month_found and years:
            # Pick out the index value for both months from the response
            value_now = None
            value_previous = None
            target_now = f"{years[0]}M{month_found}"
            target_previous = f"{int(years[0]) - 1}M{month_found}"

            for row in data_rows:
                keys = row.get("key", [])
                vals = row.get("values", [])
                if not vals:
                    continue
                try:
                    val_float = float(vals[0])
                except Exception:
                    continue

                if target_now in keys:
                    value_now = val_float
                elif target_previous in keys:
                    value_previous = val_float

            if value_now and value_previous:
                # 12-month inflation = % change of the index versus the same month a year earlier
                inflation = ((value_now - value_previous) / value_previous) * 100
                result_text = f"According to Statistics Iceland, the 12-month inflation in month {month_found} of {years[0]} was about {inflation:.2f}% (the index went from {value_previous} to {value_now})."
            else:
                result_text = "Could not find data for both periods to calculate inflation."
        else:
            # Plain index value: first value above 200, otherwise the first row
            cpi_value = next((row.get("values")[0] for row in data_rows if row.get("values") and (to_float(row.get("values")[0]) or 0) > 200), "Unknown")
            if cpi_value == "Unknown" and data_rows:
                cpi_value = data_rows[0].get("values", ["Unknown"])[0]

            year_text = f" in {years[0]}" if years else ""
            month_text = f" in month {month_found}" if month_found else ""
            result_text = f"According to Statistics Iceland, the consumer price index was {cpi_value}{month_text}{year_text}."

    elif is_population:
        population_value = data_rows[0].get("values", ["Unknown"])[0] if data_rows else "Unknown"
        result_text = f"According to Statistics Iceland, the population according to the latest data is: {population_value} people."
    else:
        result_text = "Data found."

    # Response keys kept as in the original:
    # "svar" = answer, "töflu_slóð" = table URL, "notud_sia" = filter that was used
    return {"svar": result_text, "töflu_slóð": url, "notud_sia": query_list}
