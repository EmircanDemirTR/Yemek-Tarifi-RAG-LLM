"""
False Positive Analizi
"""
import json
import os
import sys
from pathlib import Path

# Windows terminal için UTF-8
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# En son sonuç dosyasını bul
results_dir = Path(__file__).parent / "results"
result_files = sorted(results_dir.glob("evaluation_results_*.json"), reverse=True)

if not result_files:
    print("Sonuç dosyası bulunamadı!")
    exit()

with open(result_files[0], 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 70)
print("🔍 IMPOSSIBLE SORULARDA SKOR ANALİZİ")
print("=" * 70)

for system_key, result in data.items():
    if "error" in result:
        continue
    
    print(f"\n📊 {result['system_name']}")
    print("-" * 60)
    
    impossible_scores = []
    normal_scores = []
    
    for q in result['question_results']:
        score = q.get('top_score', 0)
        if q.get('is_impossible'):
            impossible_scores.append((q['question'][:45], score))
        else:
            normal_scores.append(score)
    
    # Impossible skorları göster
    print("\n  🚫 Impossible Sorular (düşük skor olmalı):")
    for question, score in impossible_scores:
        status = "❌ YÜKSEK" if score >= 0.5 else "✅ Düşük"
        print(f"    [{score:.3f}] {status} - {question}...")
    
    # Normal sorular ile karşılaştır
    avg_impossible = sum(s for _, s in impossible_scores) / len(impossible_scores) if impossible_scores else 0
    avg_normal = sum(normal_scores) / len(normal_scores) if normal_scores else 0
    
    print(f"\n  📈 Skor Karşılaştırması:")
    print(f"    Normal sorular ortalama skor:     {avg_normal:.4f}")
    print(f"    Impossible sorular ortalama skor: {avg_impossible:.4f}")
    print(f"    Fark: {avg_normal - avg_impossible:.4f}")

print("\n" + "=" * 70)
print("💡 SONUÇ VE ÖNERİLER")
print("=" * 70)
print("""
1. Impossible sorular normal sorularla benzer skor alıyor
2. Bu, embedding modellerinin semantik yakınlığa dayalı çalışmasından kaynaklanıyor
3. "Dondurmalı karnıyarık" → "Karnıyarık" ile semantik olarak yakın!

ÇÖZÜM YAKLAŞIMLARI:
├── A) Retrieval seviyesinde: Score threshold yükseltme (0.6-0.7)
├── B) LLM seviyesinde: Context'e dayanarak "bilmiyorum" demesi
└── C) Hybrid: Düşük skorlu sonuçlar için LLM'e "bu sonuç uygun mu?" sorma

RAG sisteminde asıl çözüm B ve C seçenekleridir!
LLM, dönen context'in soruyla uyuşup uyuşmadığını değerlendirebilir.
""")

