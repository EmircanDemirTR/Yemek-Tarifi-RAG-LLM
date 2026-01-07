"""
RAG Pipeline - İnteraktif Arayüz
"""
import os
import sys

# Windows terminal encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

from rag_pipeline import RAGPipeline
from llm_local import get_available_models

console = Console()


def print_welcome():
    """Hoş geldin mesajı"""
    console.print(Panel.fit(
        "[bold cyan]🍳 RAG Tarif Asistanı[/bold cyan]\n"
        "[dim]Retrieval-Augmented Generation ile Türk Mutfağı[/dim]",
        border_style="cyan"
    ))


def print_help():
    """Yardım mesajı"""
    help_text = """
[bold]Komutlar:[/bold]
  [cyan]/rag[/cyan] <soru>      RAG modu (veritabanından context ile)
  [cyan]/llm[/cyan] <soru>      LLM-Only modu (sadece model bilgisi)
  [cyan]/karsilastir[/cyan]     Aynı soruyu her iki modda test et
  [cyan]/model[/cyan] <isim>    Ollama modelini değiştir
  [cyan]/groq[/cyan]            Groq API'ye geç
  [cyan]/modeller[/cyan]        Mevcut modelleri göster
  [cyan]/yardim[/cyan]          Bu menüyü göster
  [cyan]/cikis[/cyan]           Çıkış

[bold]Örnekler:[/bold]
  /rag Mercimek çorbası nasıl yapılır?
  /llm Baklava tarifi ver
  /karsilastir Karnıyarık nasıl yapılır?
"""
    console.print(Panel(help_text, title="Yardım", border_style="green"))


def show_models():
    """Mevcut modelleri göster"""
    table = Table(title="Mevcut LLM Modelleri")
    table.add_column("Model", style="cyan")
    table.add_column("Boyut", style="green")
    table.add_column("Hız", style="yellow")
    table.add_column("Durum", style="magenta")
    
    # Groq
    table.add_row("groq (Llama 3.3 70B)", "70B", "🚀 Çok Hızlı", "✓ API")
    
    # Ollama
    available = get_available_models()
    for model_id, info in available.items():
        status = "✓ Yüklü" if info.get("installed") else "✗ Yüklenmemiş"
        table.add_row(model_id, info["size"], info["speed"], status)
    
    console.print(table)


def display_result(result: dict):
    """Sonucu göster"""
    mode = result.get("mode", "unknown")
    mode_text = "🔍 RAG" if mode == "rag" else "🤖 LLM-Only"
    
    # Cevap paneli
    console.print(Panel(
        result["answer"],
        title=f"{mode_text} Cevap",
        border_style="green"
    ))
    
    # Metrikler
    llm_result = result.get("llm_result", {})
    latency = llm_result.get("latency_ms", 0)
    tokens = llm_result.get("tokens", 0)
    model = llm_result.get("model", "unknown")
    provider = llm_result.get("provider", "unknown")
    
    console.print(f"[dim]Model: {provider}/{model} | Latency: {latency:.0f}ms | Tokens: {tokens}[/dim]")
    
    # RAG modunda bulunan tarifler
    if mode == "rag" and "retrieved_recipes" in result:
        recipes = result["retrieved_recipes"]
        if recipes:
            console.print(f"\n[dim]📚 Bulunan {len(recipes)} tarif:[/dim]")
            for i, r in enumerate(recipes[:3], 1):
                title = r.get("title", "Bilinmiyor")
                score = r.get("score", 0)
                console.print(f"[dim]   {i}. {title} (skor: {score:.2f})[/dim]")


def compare_modes(rag: RAGPipeline, question: str):
    """RAG ve LLM-Only modlarını karşılaştır"""
    console.print(f"\n[bold]Soru:[/bold] {question}\n")
    
    # LLM-Only
    console.print("[bold yellow]1. LLM-Only Modu[/bold yellow]")
    llm_result = rag.query_llm_only(question)
    display_result(llm_result)
    
    # RAG
    console.print("\n[bold cyan]2. RAG Modu[/bold cyan]")
    rag_result = rag.query_rag(question)
    display_result(rag_result)
    
    # Karşılaştırma özeti
    console.print("\n[bold]📊 Karşılaştırma:[/bold]")
    table = Table()
    table.add_column("Metrik", style="cyan")
    table.add_column("LLM-Only", style="yellow")
    table.add_column("RAG", style="green")
    
    table.add_row(
        "Latency",
        f"{llm_result['llm_result']['latency_ms']:.0f}ms",
        f"{rag_result['llm_result']['latency_ms']:.0f}ms"
    )
    table.add_row(
        "Tokens",
        str(llm_result['llm_result'].get('tokens', 0)),
        str(rag_result['llm_result'].get('tokens', 0))
    )
    table.add_row(
        "Context",
        "Yok",
        f"{len(rag_result.get('retrieved_recipes', []))} tarif"
    )
    
    console.print(table)


def main():
    """Ana fonksiyon"""
    print_welcome()
    
    # Varsayılan olarak Groq ile başla
    try:
        rag = RAGPipeline(llm_provider="groq")
        console.print("[green]✓ Groq API hazır[/green]")
    except ValueError:
        console.print("[yellow]⚠ Groq API key bulunamadı, Ollama ile başlanıyor[/yellow]")
        rag = RAGPipeline(llm_provider="ollama")
    
    print_help()
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip()
            
            if not user_input:
                continue
            
            # Komutları işle
            if user_input.lower() in ["/cikis", "/exit", "/q"]:
                console.print("[yellow]Görüşmek üzere! 👋[/yellow]")
                break
            
            elif user_input.lower() in ["/yardim", "/help", "/h"]:
                print_help()
            
            elif user_input.lower() in ["/modeller", "/models"]:
                show_models()
            
            elif user_input.lower() == "/groq":
                try:
                    rag.switch_llm("groq")
                    console.print("[green]✓ Groq API'ye geçildi[/green]")
                except ValueError as e:
                    console.print(f"[red]✗ {e}[/red]")
            
            elif user_input.lower().startswith("/model "):
                model_name = user_input[7:].strip()
                rag.switch_llm("ollama", model_name)
                console.print(f"[green]✓ Model değiştirildi: {model_name}[/green]")
            
            elif user_input.lower().startswith("/karsilastir"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    question = parts[1]
                else:
                    question = Prompt.ask("Soru")
                compare_modes(rag, question)
            
            elif user_input.lower().startswith("/rag "):
                question = user_input[5:].strip()
                if question:
                    with console.status("[bold green]RAG sorgulanıyor..."):
                        result = rag.query_rag(question)
                    display_result(result)
            
            elif user_input.lower().startswith("/llm "):
                question = user_input[5:].strip()
                if question:
                    with console.status("[bold yellow]LLM sorgulanıyor..."):
                        result = rag.query_llm_only(question)
                    display_result(result)
            
            else:
                # Varsayılan: RAG modu
                with console.status("[bold green]RAG sorgulanıyor..."):
                    result = rag.query_rag(user_input)
                display_result(result)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]İptal edildi[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Hata: {e}[/red]")


if __name__ == "__main__":
    main()

