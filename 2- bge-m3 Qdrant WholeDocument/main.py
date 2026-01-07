"""
RAG Tarif Arama Sistemi - Ana Uygulama
======================================
BGE-M3 + Qdrant ile akıllı tarif arama

Kullanım:
    python main.py index      # Tarifleri indexle
    python main.py search     # İnteraktif arama modu
    python main.py info       # Veritabanı bilgisi
"""

import sys
import os
import warnings

# Windows terminal için UTF-8 encoding ayarla
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# TensorFlow/Keras uyarılarını sustur
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint

console = Console(force_terminal=True)


def show_banner():
    """Uygulama banner'ını göster"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🍳 RAG TARİF ARAMA SİSTEMİ                             ║
    ║                                                          ║
    ║   BGE-M3 Embedding + Qdrant Vector Database              ║
    ║   ~20,000 Türk Mutfağı Tarifi                            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def cmd_index():
    """Tarifleri indexle"""
    from indexer import index_all_recipes, verify_index
    
    console.print("\n[bold yellow]⚠️  Bu işlem mevcut veritabanını silip yeniden oluşturacak![/bold yellow]")
    confirm = Prompt.ask("Devam etmek istiyor musunuz?", choices=["e", "h"], default="h")
    
    if confirm == "e":
        index_all_recipes(recreate=True)
        verify_index()
    else:
        console.print("[yellow]İşlem iptal edildi.[/yellow]")


def cmd_info():
    """Veritabanı bilgilerini göster"""
    from database import get_database
    from config import COLLECTION_NAME, QDRANT_PATH, MODEL_NAME
    
    db = get_database()
    info = db.get_collection_info()
    
    table = Table(title="📊 Veritabanı Bilgileri")
    table.add_column("Özellik", style="cyan")
    table.add_column("Değer", style="green")
    
    table.add_row("Collection Adı", COLLECTION_NAME)
    table.add_row("Veritabanı Yolu", str(QDRANT_PATH))
    table.add_row("Embedding Modeli", MODEL_NAME)
    table.add_row("Collection Durumu", "✅ Mevcut" if info.get("exists") else "❌ Yok")
    
    if info.get("exists"):
        table.add_row("Vektör Sayısı", f"{info.get('points_count', 0):,}")
        table.add_row("Durum", str(info.get("status", "N/A")))
    
    console.print(table)
    db.close()


def cmd_search():
    """İnteraktif arama modu"""
    from searcher import get_searcher, format_search_results
    from database import get_database
    
    # Veritabanı kontrolü
    db = get_database()
    info = db.get_collection_info()
    
    if not info.get("exists") or info.get("points_count", 0) == 0:
        console.print("[bold red]❌ Veritabanı boş! Önce 'python main.py index' komutunu çalıştırın.[/bold red]")
        return
    
    console.print(f"\n[green]✅ Veritabanı hazır: {info.get('points_count', 0):,} tarif[/green]")
    
    # Searcher başlat
    searcher = get_searcher()
    
    console.print(Panel("""
[bold]Arama Komutları:[/bold]
  • Doğrudan yazın: Genel arama (örn: "tavuklu makarna")
  • /malzeme tavuk, patates: Malzeme bazlı arama
  • /tarif karnıyarık: Tarif adı araması
  • /detay 1: Son aramadaki 1. tarifin detayları
  • /cikis: Çıkış
    """, title="💡 Yardım"))
    
    last_results = []
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]🔍 Arama[/bold cyan]").strip()
            
            if not query:
                continue
            
            # Çıkış kontrolü
            if query.lower() in ["/cikis", "/çıkış", "/exit", "/quit", "q"]:
                console.print("[yellow]👋 Görüşmek üzere![/yellow]")
                break
            
            # Komut kontrolü
            if query.startswith("/malzeme "):
                ingredients = [i.strip() for i in query[9:].split(",")]
                console.print(f"[dim]Malzemeler: {ingredients}[/dim]")
                results = searcher.search_by_ingredients(ingredients, top_k=5)
            
            elif query.startswith("/tarif "):
                recipe_name = query[7:].strip()
                console.print(f"[dim]Aranan tarif: {recipe_name}[/dim]")
                results = searcher.search_recipe_by_name(recipe_name, top_k=5)
            
            elif query.startswith("/detay "):
                try:
                    idx = int(query[7:].strip()) - 1
                    if 0 <= idx < len(last_results):
                        recipe = last_results[idx]
                        show_recipe_details(recipe)
                    else:
                        console.print("[red]Geçersiz numara![/red]")
                except ValueError:
                    console.print("[red]Geçerli bir numara girin![/red]")
                continue
            
            else:
                # Genel arama
                results = searcher.search(query, top_k=5)
            
            # Sonuçları göster
            last_results = results
            show_search_results(results)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Hata: {e}[/red]")
    
    # Temiz çıkış
    db.close()


def show_search_results(results):
    """Arama sonuçlarını göster"""
    if not results:
        console.print("[yellow]❌ Sonuç bulunamadı.[/yellow]")
        return
    
    console.print(f"\n[green]✅ {len(results)} tarif bulundu:[/green]\n")
    
    for i, recipe in enumerate(results, 1):
        score = recipe.get('score', 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.5 else "red"
        
        console.print(Panel(
            f"""[bold]{recipe['title']}[/bold]
[{score_color}]Benzerlik: {score:.2%}[/{score_color}]
[dim]{recipe.get('url', '')}[/dim]

[cyan]Malzemeler:[/cyan] {', '.join(recipe.get('ingredients', [])[:5])}{'...' if len(recipe.get('ingredients', [])) > 5 else ''}
""",
            title=f"[{i}]",
            border_style="blue"
        ))


def show_recipe_details(recipe):
    """Tarif detaylarını göster"""
    console.print(Panel(
        f"""[bold cyan]{recipe['title']}[/bold cyan]
[dim]{recipe.get('url', '')}[/dim]

[bold yellow]📦 Malzemeler:[/bold yellow]
{chr(10).join('  • ' + ing for ing in recipe.get('ingredients', []))}

[bold yellow]📝 Yapılışı:[/bold yellow]
{chr(10).join(f'  {i}. {step}' for i, step in enumerate(recipe.get('instructions', []), 1))}
""",
        title="📗 Tarif Detayı",
        border_style="green"
    ))


def show_help():
    """Yardım mesajını göster"""
    help_text = """
[bold]Kullanım:[/bold]
    python main.py [komut]

[bold]Komutlar:[/bold]
    index     Tarifleri veritabanına indexle (ilk kurulumda)
    search    İnteraktif arama modunu başlat
    info      Veritabanı bilgilerini göster
    help      Bu yardım mesajını göster

[bold]Örnekler:[/bold]
    python main.py index      # Tüm tarifleri indexle
    python main.py search     # Arama modunu başlat
    """
    console.print(Panel(help_text, title="💡 Yardım", border_style="cyan"))


def main():
    """Ana fonksiyon"""
    show_banner()
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "index":
        cmd_index()
    elif command == "search":
        cmd_search()
    elif command == "info":
        cmd_info()
    elif command in ["help", "-h", "--help"]:
        show_help()
    else:
        console.print(f"[red]❌ Bilinmeyen komut: {command}[/red]")
        show_help()


if __name__ == "__main__":
    main()

