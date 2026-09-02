"""
Generate stations_electrique_static.json.gz from the national IRVE dataset.

The Flutter app expects a gzip JSON list using compact keys:
  a  latitude
  o  longitude
  n  station name
  op operator
  r  brand/network
  ad address
  v  city
  c  postal code
  p  max nominal power in kW
  t  connector types
  u  open 24/7
  h  raw opening hours
  x  tariff text
  d  last update timestamp in milliseconds
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

IRVE_STATIC_URL = "https://transport.data.gouv.fr/resources/84013/download"
OUTPUT_GZ = Path("stations_electrique_static.json.gz")
USER_AGENT = "KarbuFrance-IRVE-Static/1.0"


def main() -> None:
    print("=" * 60)
    print("  GENERATION DU FICHIER STATIQUE DES BORNES IRVE")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/3] Telechargement base nationale IRVE...")
    csv_text = download_csv()
    print(f"     CSV: {len(csv_text.encode('utf-8')) / 1048576:.1f} MB")

    print("\n[2/3] Parsing et dedoublonnage...")
    stations_by_key: dict[str, dict] = {}
    skipped = 0
    for record in read_records(csv_text):
        station = parse_station(record)
        if station is None:
            skipped += 1
            continue
        key = station.pop("k")
        if key not in stations_by_key:
            stations_by_key[key] = station
        else:
            merge_station(stations_by_key[key], station)

    stations = sorted(
        stations_by_key.values(),
        key=lambda item: item.get("n", ""),
    )
    print(f"     {len(stations)} stations, {skipped} lignes ignorees")

    print("\n[3/3] Compression...")
    raw = json.dumps(
        stations,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with gzip.open(OUTPUT_GZ, "wb", compresslevel=9) as gz:
        gz.write(raw)

    print(
        f"     JSON: {len(raw) / 1048576:.1f} MB "
        f"-> gzip: {OUTPUT_GZ.stat().st_size / 1048576:.1f} MB"
    )
    print(f"     {OUTPUT_GZ}")
    print("\nTermine.")


def download_csv() -> str:
    response = requests.get(
        IRVE_STATIC_URL,
        timeout=180,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def read_records(csv_text: str):
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(io.StringIO(csv_text), dialect=dialect)


def parse_station(record: dict[str, str]) -> dict | None:
    lat = parse_float(record.get("consolidated_latitude"))
    lon = parse_float(record.get("consolidated_longitude"))
    if lat is None or lon is None:
        return None
    if lat < 41 or lat > 52 or lon < -6 or lon > 10:
        return None

    name = first(
        record.get("nom_station"),
        record.get("nom_enseigne"),
        record.get("nom_operateur"),
    )
    address = record.get("adresse_station", "")
    updated = parse_date_ms(record.get("date_maj"))
    key = first(
        record.get("id_station_itinerance"),
        record.get("id_station_local"),
        f"{lat:.5f}|{lon:.5f}|{name}",
    )

    return {
        "k": key,
        "a": lat,
        "o": lon,
        "n": name,
        "op": record.get("nom_operateur", ""),
        "r": record.get("nom_enseigne", ""),
        "ad": address,
        "v": city_from_address(address),
        "c": "",
        "p": parse_float(record.get("puissance_nominale")),
        "t": connectors(record),
        "u": record.get("horaires", "").strip() == "24/7",
        "h": empty_to_none(record.get("horaires")),
        "x": tariff(record),
        "d": updated,
    }


def merge_station(target: dict, source: dict) -> None:
    target_power = target.get("p")
    source_power = source.get("p")
    if source_power is not None and (
        target_power is None or source_power > target_power
    ):
        target["p"] = source_power

    target["t"] = sorted(set(target.get("t", [])) | set(source.get("t", [])))
    target["u"] = bool(target.get("u")) or bool(source.get("u"))

    if target.get("x") is None and source.get("x") is not None:
        target["x"] = source["x"]

    target_date = target.get("d")
    source_date = source.get("d")
    if source_date is not None and (
        target_date is None or source_date > target_date
    ):
        target["d"] = source_date


def connectors(record: dict[str, str]) -> list[str]:
    values = []
    if is_true(record.get("prise_type_combo_ccs")):
        values.append("CCS")
    if is_true(record.get("prise_type_2")):
        values.append("Type 2")
    if is_true(record.get("prise_type_chademo")):
        values.append("CHAdeMO")
    if is_true(record.get("prise_type_ef")):
        values.append("Prise E/F")
    if is_true(record.get("prise_type_autre")):
        values.append("Autre")
    return sorted(set(values))


def tariff(record: dict[str, str]) -> str | None:
    raw_tariff = empty_to_none(record.get("tarification"))
    if raw_tariff is not None:
        return raw_tariff
    if is_true(record.get("gratuit")):
        return "Gratuit"
    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_date_ms(value: str | None) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return None


def first(*values: str | None) -> str:
    for value in values:
        clean = (value or "").strip()
        if clean:
            return clean
    return ""


def empty_to_none(value: str | None) -> str | None:
    clean = (value or "").strip()
    return clean or None


def is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes", "oui"}


def city_from_address(address: str) -> str:
    match = re.search(r"\b\d{5}\s+(.+)$", address.strip())
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    main()
