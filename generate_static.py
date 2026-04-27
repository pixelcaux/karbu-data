"""
generate_static.py
==================
À lancer UNE FOIS PAR SEMAINE sur ton PC/serveur.
Produit : stations_static.json  (ou .csv)

Ce fichier contient :
  - Toutes les infos OSM (enseigne, services, adresse)
  - Le gov_id pré-calculé par fusion GPS
  - PAS de prix (ceux-ci sont récupérés en temps réel par l'appli)

Dépendances :
    pip install requests pandas tqdm scipy

Utilisation :
    python generate_static.py
    python generate_static.py --output csv      (format CSV)
    python generate_static.py --rayon 150       (rayon de fusion en mètres)
"""

import requests
import pandas as pd
import json
import argparse
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
GOV_PRIX_URL   = "https://donnees.roulez-eco.fr/opendata/instantane"
OUTPUT_JSON    = "stations_static.json"
OUTPUT_CSV     = "stations_static.csv"
RAYON_DEFAUT   = 100  # mètres

# ---------------------------------------------------------------------------
# REQUÊTE OSM
# ---------------------------------------------------------------------------
OVERPASS_QUERY = """
[out:json][timeout:120];
area["ISO3166-1"="FR"]["admin_level"="2"]->.france;
(
  node["amenity"="fuel"](area.france);
  way["amenity"="fuel"](area.france);
  relation["amenity"="fuel"](area.france);
);
out center tags;
"""

FIELDS_MAP = {
    "name"                      : "nom",
    "brand"                     : "enseigne",
    "brand:wikidata"            : "enseigne_wikidata",
    "operator"                  : "operateur",
    "network"                   : "reseau",
    "fuel:diesel"               : "gazole",
    "fuel:octane_95"            : "sp95",
    "fuel:octane_98"            : "sp98",
    "fuel:e10"                  : "e10",
    "fuel:lpg"                  : "gpl",
    "fuel:e85"                  : "e85",
    "fuel:adblue"               : "adblue",
    "fuel:HGV_diesel"           : "diesel_poids_lourds",
    "fuel:cng"                  : "gnv",
    "fuel:hydrogen"             : "hydrogene",
    "socket:type2"              : "prise_type2",
    "socket:chademo"            : "prise_chademo",
    "socket:type2_combo"        : "prise_ccs",
    "car_wash"                  : "lavage_auto",
    "compressed_air"            : "gonflage_pneus",
    "vacuum_cleaner"            : "aspirateur",
    "service:vehicle:oil_change": "vidange",
    "service:vehicle:tyres"     : "pneumatiques",
    "service:vehicle:charging"  : "recharge_electrique",
    "self_service"              : "libre_service",
    "shop"                      : "boutique",
    "food"                      : "restauration",
    "atm"                       : "distributeur_billet",
    "toilets"                   : "toilettes",
    "wifi"                      : "wifi",
    "wheelchair"                : "acces_pmr",
    "hgv"                       : "poids_lourds",
    "motorhome"                 : "camping_car",
    "payment:cash"              : "paiement_especes",
    "payment:credit_cards"      : "paiement_cb",
    "payment:contactless"       : "paiement_sans_contact",
    "opening_hours"             : "horaires",
    "phone"                     : "telephone",
    "website"                   : "site_web",
    "addr:housenumber"          : "numero",
    "addr:street"               : "rue",
    "addr:city"                 : "ville",
    "addr:postcode"             : "code_postal",
}

# ---------------------------------------------------------------------------
# ÉTAPE 1 — OSM
# ---------------------------------------------------------------------------

def fetch_osm() -> pd.DataFrame:
    print("\n[1/3] 🗺️  Téléchargement OSM...")
    try:
        r = requests.post(
            OVERPASS_URL,
            data={"data": OVERPASS_QUERY},
            timeout=180,
            headers={"User-Agent": "stations-static-generator/1.0"}
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
        print(f"     ✅ {len(elements)} stations OSM")
    except Exception as e:
        print(f"     ❌ Erreur OSM : {e}")
        return pd.DataFrame()

    rows = []
    for el in elements:
        tags = el.get("tags", {})
        lat, lon = (el.get("lat"), el.get("lon")) if el["type"] == "node" \
                   else (el.get("center", {}).get("lat"), el.get("center", {}).get("lon"))
        row = {"osm_id": el.get("id"), "lat": lat, "lon": lon}
        for osm_key, col in FIELDS_MAP.items():
            row[col] = tags.get(osm_key, "")
        if not row["enseigne"] and row["nom"]:
            row["enseigne"] = row["nom"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["enseigne"] = df["enseigne"].str.strip()
    return df

# ---------------------------------------------------------------------------
# ÉTAPE 2 — Gov (juste pour récupérer les IDs et noms officiels)
# ---------------------------------------------------------------------------

def fetch_gov_ids() -> pd.DataFrame:
    """
    On télécharge le flux gov UNIQUEMENT pour extraire :
      - gov_id (identifiant officiel de la station)
      - lat / lon
      - enseigne officielle (souvent manquante dans OSM)
      - services officiels déclarés par la station
    PAS les prix — ça c'est le boulot de l'appli en temps réel.
    """
    print("\n[2/3] 🏛️  Téléchargement référentiel gouvernement (IDs + enseignes)...")
    try:
        r = requests.get(GOV_PRIX_URL, timeout=60,
                         headers={"User-Agent": "stations-static-generator/1.0"})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            xml_file = [f for f in z.namelist() if f.endswith(".xml")][0]
            xml_content = z.read(xml_file)

        root = ET.fromstring(xml_content)
        rows = []
        for pdv in root.findall("pdv"):
            lat = float(pdv.get("latitude",  0)) / 100000
            lon = float(pdv.get("longitude", 0)) / 100000

            # Services déclarés officiellement
            services_node = pdv.find("services")
            services_list = []
            if services_node is not None:
                for svc in services_node.findall("service"):
                    if svc.text:
                        services_list.append(svc.text.strip())

            # Horaires 24h/24
            horaires_node = pdv.find("horaires")
            h24 = ""
            if horaires_node is not None:
                h24 = horaires_node.get("automate-24-24", "")

            # Carburants disponibles selon le gov
            carburants_gov = []
            for p in pdv.findall("prix"):
                nom = p.get("nom", "")
                if nom:
                    carburants_gov.append(nom)

            rows.append({
                "gov_id"          : pdv.get("id", ""),
                "lat_gov"         : lat,
                "lon_gov"         : lon,
                "cp_gov"          : pdv.get("cp", ""),
                "ville_gov"       : pdv.findtext("ville", "").strip(),
                "adresse_gov"     : pdv.findtext("adresse", "").strip(),
                "h24"             : h24,
                "services_officiels": " | ".join(services_list),
                "carburants_gov"  : " | ".join(carburants_gov),
            })

        df = pd.DataFrame(rows)
        df = df[df["lat_gov"] != 0]
        print(f"     ✅ {len(df)} stations gouvernement")
        return df

    except Exception as e:
        print(f"     ❌ Erreur gov : {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# ÉTAPE 3 — Fusion GPS → gov_id stocké dans le fichier statique
# ---------------------------------------------------------------------------

def fusionner(df_osm: pd.DataFrame, df_gov: pd.DataFrame, rayon: int) -> pd.DataFrame:
    print(f"\n[3/3] 🔗 Fusion GPS (rayon {rayon}m)...")

    DEG_TO_M = 111_000
    osm_xy = df_osm[["lat", "lon"]].fillna(0).values * DEG_TO_M
    gov_xy = df_gov[["lat_gov", "lon_gov"]].values * DEG_TO_M

    tree = cKDTree(gov_xy)
    distances, indices = tree.query(osm_xy, k=1, workers=-1)
    matched = distances <= rayon

    print(f"     ✅ {matched.sum()} / {len(df_osm)} stations croisées ({matched.mean()*100:.1f}%)")
    print(f"     ⚠️  {(~matched).sum()} stations OSM sans gov_id (stations non déclarées)")

    # Colonnes à récupérer du gov
    gov_cols = ["gov_id", "cp_gov", "ville_gov", "adresse_gov",
                "h24", "services_officiels", "carburants_gov"]

    df_result = df_osm.copy()
    for col in gov_cols:
        df_result[col] = None

    df_result["distance_fusion_m"] = np.round(distances, 1)
    df_result["gov_apparie"]       = matched

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= rayon:
            for col in gov_cols:
                df_result.at[i, col] = df_gov.iloc[idx][col]

            # Si pas d'enseigne OSM, on prend la ville gov comme fallback
            if not df_result.at[i, "enseigne"]:
                df_result.at[i, "enseigne"] = "Station " + str(df_gov.iloc[idx]["ville_gov"])

    df_result["genere_le"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df_result

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def exporter_json(df: pd.DataFrame):
    """Export JSON optimisé pour chargement rapide dans une appli."""
    # Nettoyage : remplacer NaN par None pour JSON propre
    df_clean = df.where(pd.notna(df), None)
    records = df_clean.to_dict(orient="records")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "total"     : len(records),
                "genere_le" : datetime.now().isoformat(),
                "version"   : "1.0"
            },
            "stations": records
        }, f, ensure_ascii=False, separators=(",", ":"))  # compact, sans espaces

    size_kb = __import__("os").path.getsize(OUTPUT_JSON) / 1024
    print(f"\n💾 JSON exporté : {OUTPUT_JSON}  ({size_kb:.0f} Ko, {len(records)} stations)")

def exporter_csv(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 CSV exporté  : {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Génère le fichier statique des stations")
    parser.add_argument("--output", choices=["json", "csv", "both"], default="both",
                        help="Format de sortie (défaut: both)")
    parser.add_argument("--rayon",  type=int, default=RAYON_DEFAUT,
                        help=f"Rayon de fusion GPS en mètres (défaut: {RAYON_DEFAUT})")
    args = parser.parse_args()

    print("=" * 60)
    print("  GÉNÉRATION DU FICHIER STATIQUE DES STATIONS")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    df_osm = fetch_osm()
    if df_osm.empty:
        print("❌ Abandon : données OSM indisponibles.")
        return

    df_gov = fetch_gov_ids()
    if df_gov.empty:
        print("⚠️  Pas de données gov — export OSM seul sans gov_id.")
        df_final = df_osm
    else:
        df_final = fusionner(df_osm, df_gov, args.rayon)

    if args.output in ("json", "both"):
        exporter_json(df_final)
    if args.output in ("csv", "both"):
        exporter_csv(df_final)

    print("\n✅ Terminé ! Ce fichier est valable ~7 jours.")
    print("   Relance generate_static.py chaque semaine pour rester à jour.\n")


if __name__ == "__main__":
    main()