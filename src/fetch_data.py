import time
from pathlib import Path
import httpx

BASE = "https://pokeapi.co/api/v2"
OUT = Path("data/docs")
OUT.mkdir(parents=True, exist_ok=True)


def get_description(species):
    for entry in species["flavor_text_entries"]:
        if entry["language"]["name"] == "en":
            return entry["flavor_text"].replace("\n", " ").replace("\f", " ")
    return "No description available."


for poke_id in range(1, 152):
    p = httpx.get(f"{BASE}/pokemon/{poke_id}").json()
    s = httpx.get(f"{BASE}/pokemon-species/{poke_id}").json()

    types = ", ".join(t["type"]["name"] for t in p["types"])
    abilities = ", ".join(
        a["ability"]["name"] + (" (hidden)" if a["is_hidden"] else "")
        for a in p["abilities"]
    )
    stats = ", ".join(f'{st["stat"]["name"]} {st["base_stat"]}' for st in p["stats"])

    text = f"""Name: {p["name"].capitalize()}
Pokedex number: {poke_id}
Types: {types}
Height: {p["height"] / 10} m
Weight: {p["weight"] / 10} kg
Abilities: {abilities}
Base stats: {stats}
Description: {get_description(s)}
"""
    (OUT / f"{poke_id:03d}_{p['name']}.txt").write_text(text)
    print(f"OK {poke_id} {p['name']}")
    time.sleep(0.2)  # to not saturate the API