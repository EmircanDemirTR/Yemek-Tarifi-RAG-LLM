import json
import re

INPUT = r"C:\Users\emirc\Desktop\VSCode Python\yemek_scraper\eski.jsonl"
OUTPUT = "temiz.jsonl"

def clean_ingredient(ing):
    # "## ..." markdown başlıklarını temizle
    ing = re.sub(r"^#+\s*", "", ing)
    return ing.strip()


def instruction_length(item):
    return sum(len(step) for step in item.get("instructions", []))


# ---------------------------------------------------------------------
# 1) JSONL dosyasını oku
# ---------------------------------------------------------------------
data = []
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            pass  # bozuk satır varsa es geç


print(f"[OK] Toplam {len(data)} tarif yüklendi.")


# ---------------------------------------------------------------------
# 2) URL’ye göre tekrar eden tarifleri temizle
# ---------------------------------------------------------------------
unique_by_url = {}
for item in data:
    url = item.get("url")
    if url:
        unique_by_url[url] = item

data = list(unique_by_url.values())
print(f"[OK] URL bazlı tekrar temizliği sonrası: {len(data)} tarif kaldı.")


# ---------------------------------------------------------------------
# 3) (title + ingredients) kombinasyonuna göre tekrar temizliği
# ---------------------------------------------------------------------
seen = set()
cleaned = []

for item in data:
    title = item.get("title", "").strip().lower()
    ingredients = tuple(item.get("ingredients", []))
    key = (title, ingredients)

    if key not in seen:
        seen.add(key)
        cleaned.append(item)

data = cleaned
print(f"[OK] İçerik bazlı tekrar temizliği sonrası: {len(data)} tarif kaldı.")


# ---------------------------------------------------------------------
# 4) Ingredients içindeki markdown başlıklarını ve boşlukları temizle
# ---------------------------------------------------------------------
for item in data:
    ing = item.get("ingredients", [])
    new_ing = [clean_ingredient(i) for i in ing]
    new_ing = [i for i in new_ing if i]  # boş olanları sil
    item["ingredients"] = new_ing


# ---------------------------------------------------------------------
# 5) Çok kısa veya hatalı tarifleri filtrele
# ---------------------------------------------------------------------

filtered = []
for item in data:
    if len(item.get("ingredients", [])) < 3:
        continue  # çok az malzemeli, büyük ihtimal hatalı

    if instruction_length(item) < 200:
        continue  # talimatlar çok kısa

    if len(item.get("title", "")) < 5:
        continue  # başlık çok kısaysa hatalı olabilir

    filtered.append(item)

data = filtered
print(f"[OK] Hatalı tariflerden sonra: {len(data)} tarif kaldı.")


# ---------------------------------------------------------------------
# 6) Temiz dosyayı yaz
# ---------------------------------------------------------------------
with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[DONE] Temizlik tamamlandı. Çıktı: {OUTPUT}")
