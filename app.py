from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import unicodedata
import re

app = FastAPI(title="Hagstofa Snjallt API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HELSTU_TAFLUR = {
    "visitala": {
        "text": "Vísitala neysluverðs",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Efnahagur/visitolur/1_vnv/1_vnv/VIS01002.px"
    },
    "mannfjoldi": {
        "text": "Mannfjöldi eftir kyni og aldri",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Ibuar/mannfjoldi/1_yfirlit/Yfirlit_mannfjolda/MAN00000.px"
    },
    "buferlaflutningar": {
        "text": "Búferlaflutningar innanlands milli landsfjórðunga",
        "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Ibuar/buferlaflutningar/buferlaflinnanlands/buferlaflinnanlands/MAN01002.px"
    }
    # Here you can easily add more tables when you find their URL:
    # "mannfjoldi": {
    #     "text": "Population by sex and age",
    #     "url": "https://px.hagstofa.is:443/pxis/api/v1/is/Ibuar/mannfjoldi/yfirlit/MAN00000.px"
    # }
}

SVÆÐA_MAPPING = {
    "hofudborgarsvad": "1", "reykjavik": "1", "sudurnes": "2",
    "vesturland": "3", "vestfirdir": "4", "nordurland vestra": "5",
    "nordurland eystra": "6", "austurland": "7", "sudurland": "8"
}

MANUDIR_MAPPING = {
    "jan": "01", "janar": "01", "feb": "02", "febr": "02", "febrúar": "02", "februar": "02",
    "mar": "03", "mars": "03", "apr": "04", "apríl": "04", "april": "04", "mai": "05",
    "maí": "05", "jun": "06", "júní": "06", "juni": "06", "jul": "07", "júlí": "07", "juli": "07",
    "agv": "08", "ágúst": "08", "agust": "08", "august": "08", "sep": "09", "september": "09", 
    "okt": "10", "október": "10", "oktober": "10", "nov": "11", "nóvember": "11", "november": "11", 
    "des": "12", "desember": "12"
}

def hreinsa_texta(texti):
    nfkd_form = unicodedata.normalize('NFKD', str(texti))
    flat_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
    for k, v in {'þ': 'th', 'æ': 'ae', 'ö': 'o', 'ð': 'd'}.items():
        flat_text = flat_text.replace(k, v)
    return flat_text

def finna_taflu(spurning):
    spurning_hreinsuð = hreinsa_texta(spurning)
    if any(k in spurning_hreinsuð for k in ["flutti", "flutt", "buid", "buferla", "fjords", "landshluta", "karlar", "konur"]):
        return HELSTU_TAFLUR["buferlaflutningar"]
    elif any(k in spurning_hreinsuð for k in ["visitala", "neysluverds", "verd", "verdbolga"]):
        return HELSTU_TAFLUR["visitala"]
    elif any(k in spurning_hreinsuð for k in ["mannfjoldi", "mannfjolda", "ibuar"]):
        return HELSTU_TAFLUR["mannfjoldi"]
    return HELSTU_TAFLUR["visitala"]

class SpurningRequest(BaseModel):
    spurning: str

@app.post("/spyrja_px_api")
def spyrja_px_api(req: SpurningRequest):
    valin_tafla = finna_taflu(req.spurning)
    url = valin_tafla["url"]
    is_buferla = (url == HELSTU_TAFLUR["buferlaflutningar"]["url"])
    is_visitala = (url == HELSTU_TAFLUR["visitala"]["url"])
    is_mannfjoldi = (url == HELSTU_TAFLUR["mannfjoldi"]["url"])
    
    headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

    try:
        meta_res = requests.get(url, headers=headers, timeout=15)
        if meta_res.status_code != 200:
            return {"svar": "Tókst ekki að sækja lýsigögn frá Hagstofu."}
        metagogn = meta_res.json()
        variables = metagogn.get("variables", [])
    except Exception as e:
        return {"svar": f"Tengingarvilla: {e}"}

    spurning_hreinsuð = hreinsa_texta(req.spurning)
    query_list = []
    arid = re.findall(r'\b(20\d{2}|19\d{2})\b', spurning_hreinsuð)
    er_verdbolga = any(w in spurning_hreinsuð for w in ["verdbolga", "verdbolgun", "verðbólga", "verðbólgan"])

    if is_buferla:
        kyn_code, ar_code, brott_code, ad_code, aldur_code, aldur_samtals = "Kyn", "Ár", "Brottfararsvæði", "Aðflutningssvæði", None, None
        kyn_samtals_gildi = None

        for var in variables:
            v_code, v_text = var.get("code"), hreinsa_texta(var.get("text", ""))
            v_vals, v_vtexts = var.get("values", []), var.get("valueTexts", [])
            if "kyn" in v_text:
                kyn_code = v_code
                for idx, vt in enumerate(v_vtexts):
                    if any(w in hreinsa_texta(vt) for w in ["alls", "samtals", "allir"]):
                        kyn_samtals_gildi = v_vals[idx]
                        break
            elif "ar" in v_text: ar_code = v_code
            elif "brott" in v_text: brott_code = v_code
            elif "adflutnings" in v_text or "til" in v_text:
                if "brott" not in v_text: ad_code = v_code
            elif "aldur" in v_text:
                aldur_code = v_code
                for idx, vt in enumerate(v_vtexts):
                    if any(w in hreinsa_texta(vt) for w in ["alls", "samtals", "allir"]):
                        aldur_samtals = v_vals[idx]
                        break
                if not aldur_samtals and v_vals: aldur_samtals = v_vals[0]

        finnum_karl = "karlar" in spurning_hreinsuð or "karl" in spurning_hreinsuð
        finnum_konu = "konur" in spurning_hreinsuð or "kona" in spurning_hreinsuð
        heildar_spurning = any(w in spurning_hreinsuð for w in ["heildina", "alls", "samtals"])

        if heildar_spurning and not (finnum_karl or finnum_konu):
            kyn_val = [kyn_samtals_gildi] if kyn_samtals_gildi else ["1", "2"]
        else:
            if finnum_karl and finnum_konu: kyn_val = ["1", "2"]
            elif finnum_konu: kyn_val = ["2"]
            elif finnum_karl: kyn_val = ["1"]
            else: kyn_val = [kyn_samtals_gildi] if kyn_samtals_gildi else ["1", "2"]

        query_list.append({"code": kyn_code, "selection": {"filter": "item", "values": kyn_val}})
        if aldur_code and aldur_samtals:
            query_list.append({"code": aldur_code, "selection": {"filter": "item", "values": [aldur_samtals]}})
        if arid:
            query_list.append({"code": ar_code, "selection": {"filter": "item", "values": arid}})

        leita_vid = sorted([(spurning_hreinsuð.find(s), kode) for s, kode in SVÆÐA_MAPPING.items() if spurning_hreinsuð.find(s) != -1])
        svæði = []
        for _, k in leita_vid:
            if not svæði or svæði[-1] != k: svæði.append(k)

        brott, adfl = (svæði[0], svæði[1]) if len(svæði) >= 2 else (svæði[0] if svæði else "7", "1")
        if "fra" not in spurning_hreinsuð and len(svæði) == 1:
            brott, adfl = "7", svæði[0]

        query_list.append({"code": brott_code, "selection": {"filter": "item", "values": [brott]}})
        query_list.append({"code": ad_code, "selection": {"filter": "item", "values": [adfl]}})

    elif is_visitala:
        found_month = next((m_gildi for m_lykil, m_gildi in MANUDIR_MAPPING.items() if m_lykil in spurning_hreinsuð), None)
        
        # Ef spurt er um verðbólgu þurfum við bæði árinu sem spurt er um og árið á undan (12 mánuði fyrr)
        manudir_listi = []
        if found_month and arid:
            valin_ar = int(arid[0])
            manudir_listi = [f"{valin_ar}M{found_month}", f"{valin_ar - 1}M{found_month}"]
        elif arid:
            manudir_listi = arid

        for var in variables:
            v_code, v_text, v_vals = var.get("code"), hreinsa_texta(var.get("text", "")), var.get("values", [])
            if "ar" in v_text:
                if arid and not found_month: 
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": arid}})
            elif "manudur" in v_text:
                if manudir_listi: 
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": manudir_listi}})
            elif "grunnur" in v_text:
                if v_vals:
                    g_val = next((g for g in v_vals if g not in ["B1939", "B1988"]), v_vals[-1])
                    query_list.append({"code": v_code, "selection": {"filter": "item", "values": [g_val]}})
            else:
                if v_vals: query_list.append({"code": v_code, "selection": {"filter": "item", "values": [v_vals[0]]}})

    elif is_mannfjoldi:
        ar_code = next((var.get("code") for var in variables if "ar" in hreinsa_texta(var.get("text", ""))), "Ár")
        target_ar = arid if arid else [variables[0].get("values", [""])[-1]]
        query_list.append({"code": ar_code, "selection": {"filter": "item", "values": target_ar}})

    try:
        data_res = requests.post(url, json={"query": query_list, "response": {"format": "json"}}, headers=headers, timeout=15)
        if data_res.status_code != 200:
            data_res = requests.post(url, json={"query": [], "response": {"format": "json"}}, headers=headers, timeout=15)
            query_list = []
    except Exception as e:
        return {"svar": f"Gagnaöflunarvilla: {e}"}

    if data_res.status_code != 200:
        return {"svar": f"Villa kom upp við gagnaöflun (status {data_res.status_code})."}

    api_gögn = data_res.json()
    data_rows = api_gögn.get("data", [])

    if is_buferla:
        samtals_fjoldi = 0
        karlar, konur = 0, 0
        for row in data_rows:
            keys, vals = row.get("key", []), row.get("values", ["0"])
            try: tala = int(vals[0])
            except: continue
            
            samtals_fjoldi += tala
            if "1" in keys and "2" not in keys: karlar += tala
            elif "2" in keys: konur += tala
        
        ar_textarid = arid[0] if arid else 'umræddu ári'
        
        if len(kyn_val) > 1 or kyn_samtals_gildi in kyn_val or "0" in kyn_val:
            heildar_tala = samtals_fjoldi if samtals_fjoldi > 0 else int(data_rows[0].get('values', [0])[0])
            utkomu_texti = f"Samkvæmt Hagstofu Íslands fluttu samtals {heildar_tala} manns milli svæðanna árið {ar_textarid}."
        elif kyn_val == ["1"]:
            utkomu_texti = f"Samkvæmt Hagstofu Íslands fluttu {karlar or int(data_rows[0].get('values', [0])[0])} karlar milli svæðanna árið {ar_textarid}."
        else:
            utkomu_texti = f"Samkvæmt Hagstofu Íslands fluttu {konur or int(data_rows[0].get('values', [0])[0])} konur milli svæðanna árið {ar_textarid}."

    elif is_visitala:
        if not data_rows:
            return {"svar": "Engin gögn fundust fyrir þetta tímabil hjá Hagstofunni."}
        
        if er_verdbolga and found_month and arid:
            # Finnum gildi fyrir bæði árin úr svarinu
            gildi_nu = None
            gildi_fyrra = None
            target_nu = f"{arid[0]}M{found_month}"
            target_fyrra = f"{int(arid[0]) - 1}M{found_month}"
            
            for row in data_rows:
                keys = row.get("key", [])
                vals = row.get("values", [])
                if not vals: continue
                try: val_float = float(vals[0])
                except: continue
                
                if target_nu in keys:
                    gildi_nu = val_float
                elif target_fyrra in keys:
                    gildi_fyrra = val_float
            
            if gildi_nu and gildi_fyrra:
                verdbolga = ((gildi_nu - gildi_fyrra) / gildi_fyrra) * 100
                utkomu_texti = f"Samkvæmt Hagstofu Íslands var 12 mánaða verðbólga í mánuði {found_month} árið {arid[0]} um {verdbolga:.2f}% (vísitalan fór úr {gildi_fyrra} í {gildi_nu})."
            else:
                utkomu_texti = "Tókst ekki að finna gögn fyrir bæði tímabilin til að reikna verðbólgu."
        else:
            visitala_gildi = next((row.get("values")[0] for row in data_rows if row.get("values") and float(row.get("values")[0]) > 200), "Óþekkt")
            if visitala_gildi == "Óþekkt" and data_rows:
                visitala_gildi = data_rows[0].get("values", ["Óþekkt"])[0]
            
            ar_texti = arid[0] if arid else ""
            manudur_texti = f"í mánuði {found_month}" if found_month else ""
            utkomu_texti = f"Samkvæmt Hagstofu Íslands var vísitala neysluverðs {visitala_gildi} {manudur_texti} árið {ar_texti}."

    elif is_mannfjoldi:
        mannfjoldi_gildi = data_rows[0].get("values", ["Óþekkt"])[0] if data_rows else "Óþekkt"
        utkomu_texti = f"Samkvæmt Hagstofu Íslands er mannfjöldi samkvæmt nýjustu gögnum: {mannfjoldi_gildi} manns."
    else:
        utkomu_texti = "Gögn fundust."

    return {"svar": utkomu_texti, "töflu_slóð": url, "notud_sia": query_list}