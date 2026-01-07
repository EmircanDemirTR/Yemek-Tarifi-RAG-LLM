#!/usr/bin/env python3
"""Yemek.com tarif scraper

Çektiği verileri recipes.jsonl içine yazar.
"""
import time
import json
import re
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# YAPILANDIRMA - Buradan kolayca değiştirebilirsiniz
# ============================================================
START_PAGE = 1251               # Hangi sayfadan başlayacak (örn: 500)
PAGES_TO_SCRAPE = 750          # Kaç sayfa taranacak (örn: 100)
MAX_WORKERS = 13             # Paralel worker sayısı (5-10 önerilir)
OUTPUT_FILE = 'recipes.jsonl'  # Çıktı dosyası

BASE = 'https://yemek.com'
LISTING_TEMPLATE = BASE + '/tarif/sayfa/{page}/'

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) YemekScraper/1.0'
})
retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504])
session.mount('https://', HTTPAdapter(max_retries=retries))


def get_html(url):
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def extract_recipe_links(listing_html, max_recipes_per_page=14):
    """Extract recipe links from a listing page.
    
    Args:
        listing_html: HTML content of the listing page
        max_recipes_per_page: Maximum number of recipes to extract per page (default 14)
        
    Returns:
        List of recipe URLs (limited to max_recipes_per_page to avoid sidebar/footer links)
    """
    soup = BeautifulSoup(listing_html, 'html.parser')
    anchors = soup.find_all('a', href=True)
    
    # Category page patterns to exclude (these are listing pages, not actual recipes)
    category_keywords = [
        'tarifleri', 'tarifler', 'yemekleri', 'recipes', 
        'kategori', 'category', 'liste', 'list', 'gelenler'
    ]
    
    # Single-word and short category pages (corba, tatli, kek, etc.)
    single_word_categories = [
        'corba', 'kahvaltiliklar', 'tatli', 'pogaca', 'kek', 'kurabiye',
        'borek', 'pasta', 'balik', 'et', 'tavuk', 'sebze', 'makarna',
        'mezeler', 'video', 'dolma', 'sarma', 'pilav', 'kofte', 'kebap',
        'salata', 'atistirmalik', 'icecek', 'sos', 'bakliyat',
        'dolma-sarma', 'pasta-tatli'  # compound categories
    ]
    
    # Collect all unique recipe URLs in the order they appear
    seen = set()
    urls = []
    
    for a in anchors:
        href = a['href']
        # normalize
        full = href if href.startswith('http') else urljoin(BASE, href)
        
        # match recipe urls like https://yemek.com/tarif/<slug>/
        if re.search(r'/tarif/[^/]+/?$', urlparse(full).path):
            # Skip pagination pages
            if '/sayfa/' in full:
                continue
            
            # Extract slug
            slug = urlparse(full).path.rstrip('/').split('/')[-1].lower()
            
            # Skip category pages (ending with keywords)
            is_category = any(slug.endswith(kw) for kw in category_keywords)
            if is_category:
                continue
            
            # Skip single-word category pages
            if slug in single_word_categories:
                continue
            
            # Skip collection pages with specific patterns
            if '-ve-' in slug and ('tarif' in slug or slug.count('-') <= 3):
                # "dolma-ve-sarma-tarifleri", "pasta-ve-tatli" etc.
                continue
            
            # Skip "sizden-gelenler", "video", etc special category pages
            special_categories = ['sizden-gelenler', 'pasta-tatli', 'video', 'blog']
            if slug in special_categories:
                continue
            
            # Skip very short slugs (likely categories like "ye", "sos", etc)
            if len(slug) < 4 or '-' not in slug:
                continue
            
            if full not in seen:
                seen.add(full)
                urls.append(full)
                
                # Stop after collecting max recipes (main content only)
                if len(urls) >= max_recipes_per_page:
                    break
    
    return urls


def gather_section_by_heading(soup, keywords):
    """Find section by heading keyword and return list of text items under it."""
    # Filter out navigation/menu noise
    noise_keywords = ['yemek tarifleri', 'çorba', 'kahvaltılık', 'tatlı', 'poğaça', 
                      'tüm tarif', 'kategoriler', 'börek', 'pasta', 'makarna']
    
    for h in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        text = h.get_text(' ', strip=True).lower()
        if any(k in text for k in keywords):
            items = []
            # Check parent container for better structure
            parent = h.find_parent(['div', 'section', 'article'])
            if parent:
                # Try to find list items first
                for li in parent.find_all('li'):
                    t = li.get_text(' ', strip=True)
                    if t and len(t) > 2:  # filter out empty/too short items
                        items.append(t)
                
                # If no list items, try ordered list for instructions
                if not items:
                    for ol in parent.find_all('ol'):
                        for li in ol.find_all('li'):
                            t = li.get_text(' ', strip=True)
                            if t and len(t) > 2:
                                items.append(t)
            
            # Fallback to sibling scanning
            if not items:
                for sib in h.next_siblings:
                    if getattr(sib, 'name', None) and sib.name in ('h1', 'h2', 'h3', 'h4', 'h5'):
                        break
                    if getattr(sib, 'find_all', None):
                        for li in sib.find_all('li'):
                            t = li.get_text(' ', strip=True)
                            if t and len(t) > 2:
                                items.append(t)
            
            # dedupe while preserving order and filter noise
            seen = set()
            out = []
            for it in items:
                clean = it.lstrip('• ').strip()
                # Skip noise (navigation links, etc)
                clean_lower = clean.lower()
                is_noise = any(nk in clean_lower for nk in noise_keywords)
                # Also skip very long items (likely concatenated menu text)
                if clean and clean not in seen and len(clean) > 2 and len(clean) < 150 and not is_noise:
                    seen.add(clean)
                    out.append(clean)
            
            if out:
                return out
    return []


def parse_recipe_page(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Title - try multiple methods
    title = None
    h1 = soup.find('h1')
    if h1:
        title = h1.get_text(' ', strip=True)
    if not title:
        m = soup.find('meta', property='og:title')
        if m and m.get('content'):
            title = m['content']
    if not title:
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().split('|')[0].strip()

    # Ingredients - look for lists after "malzeme" heading or with measurement units
    ingredients = []
    malzeme_found = False
    
    # Strategy 1: Find main "Malzemeler" heading and collect ALL subsequent ingredient lists
    main_malzeme_h2 = soup.find('h2', string=lambda x: x and 'malzem' in x.lower() if x else False)
    
    if main_malzeme_h2:
        malzeme_found = True
        # Find all elements between "Malzemeler" H2 and "Nasıl Yapılır" H2
        # This handles both flat structures and nested divs with H3/UL inside
        last_subheading = None
        
        # Use find_all_next() to iterate through ALL subsequent elements in document order
        for element in main_malzeme_h2.find_all_next():
            # Stop at next major heading (instructions)
            if element.name == 'h2' and element != main_malzeme_h2:
                heading_text = element.get_text().strip().lower()
                if 'nasıl' in heading_text or 'yapılış' in heading_text or 'püf noktası' in heading_text:
                    break
            
            # Track subsection headings (H3/H4/H5)
            if element.name in ('h3', 'h4', 'h5'):
                subheading_text = element.get_text().strip().lower()
                # Check if it's an ingredient subsection
                if any(keyword in subheading_text for keyword in ['için:', 'için', 'malzem', 'üzeri', 'iç', 'harç', 'sos', 'dolgu', 'krema', 'hamur', 'çikolata']):
                    last_subheading = element.get_text().strip()
            
            # Collect ingredients from lists
            if element.name in ('ul', 'ol'):
                lis = element.find_all('li', recursive=False)
                if lis:
                    # Add subsection marker (e.g., "## Profiterol Hamuru İçin:")
                    if last_subheading and last_subheading not in ingredients:
                        ingredients.append(f"## {last_subheading}")
                    
                    for li in lis:
                        text = li.get_text(' ', strip=True)
                        if text and 3 < len(text) < 150:
                            # Filter navigation noise
                            text_lower = text.lower()
                            if not any(noise in text_lower for noise in ['yemek tarifleri', 'tüm tarif', 'kategoriler', 'video', 'çorba tarifleri']):
                                ingredients.append(text)
                    
                    last_subheading = None  # Reset after this list
            
        # Deduplicate ingredients while preserving order
        if ingredients:
            seen = set()
            deduped = []
            for ing in ingredients:
                ing_lower = ing.lower().strip()
                if ing_lower not in seen:
                    seen.add(ing_lower)
                    deduped.append(ing)
            ingredients = deduped
            
            # Validate collected ingredients
            sample = ' '.join(ingredients[:5]).lower()
            if not any(unit in sample for unit in ['gram', 'adet', 'su bardağı', 'kaşık', 'diş', 'bardağı' ]):
                ingredients = []  # Doesn't look like real ingredients
    
    # Strategy 2: If still no ingredients, try heuristic-based search
    if not ingredients:
        all_lists = soup.find_all(['ul', 'ol'])
        for lst in all_lists:
            lis = lst.find_all('li', recursive=False)
            if not lis or len(lis) < 2:
                continue
            
            items = [li.get_text(' ', strip=True) for li in lis]
            # Check if list items look like ingredients (have units)
            sample = ' '.join(items[:3]).lower()
            if any(unit in sample for unit in ['gram', 'adet', 'su bardağı', 'kaşık', 'diş', 'bardağı']):
                # Additional check: items shouldn't be too long (not paragraphs)
                valid_items = [it for it in items if it and 5 < len(it) < 100]
                if len(valid_items) >= 2:
                    # Final noise filter
                    clean_items = []
                    for it in valid_items:
                        it_lower = it.lower()
                        if not any(noise in it_lower for noise in ['yemek tarifleri', 'tüm tarif', 'kategoriler', 'hakkımızda', 'iletişim']):
                            clean_items.append(it)
                    if clean_items:
                        ingredients = clean_items
                        break

    # Instructions - look for ordered list or numbered sections
    instructions = []
    # Try finding instruction container by class or heading
    inst_containers = soup.find_all('div', class_=lambda c: c and ('recipe' in str(c).lower() or 'instruction' in str(c).lower()))
    for container in inst_containers:
        ols = container.find_all('ol')
        for ol in ols:
            items = [li.get_text(' ', strip=True) for li in ol.find_all('li')]
            if items and len(items) >= 2:
                instructions = [it for it in items if it and len(it) > 10]
                break
        if instructions:
            break
    
    # Fallback to heading-based
    if not instructions:
        instructions = gather_section_by_heading(soup, ['nasıl', 'yap', 'yapılış', 'yapilis', 'hazırlan'])
    
    # Final fallback: any ordered list
    if not instructions:
        for ol in soup.find_all('ol'):
            items = [li.get_text(' ', strip=True) for li in ol.find_all('li')]
            if items and len(items) >= 2:
                # Check if looks like instructions (longer text, action words)
                sample = ' '.join(items[:2]).lower()
                if any(verb in sample for verb in ['ekle', 'karıştır', 'pişir', 'koy', 'çıkar', 'alın', 'yapın']):
                    instructions = [it for it in items if it and len(it) > 10]
                    break
        
        # Last resort: numbered paragraphs
        if not instructions:
            all_text = soup.get_text('\n', strip=True)
            numbered = []
            for line in all_text.splitlines():
                line = line.strip()
                if re.match(r'^\s*\d+[\.\)]\s+', line):
                    clean = re.sub(r'^\s*\d+[\.\)]\s+', '', line)
                    if len(clean) > 15:  # filter noise
                        numbered.append(clean)
            if len(numbered) >= 2:
                instructions = numbered

    return {
        'url': url,
        'title': title or 'Başlık Bulunamadı',
        'ingredients': ingredients,
        'instructions': instructions,
    }


def main(start_page, pages, out_file):
    """Sıralı (tek thread) scraper - yavaş ama güvenli
    
    Parametreler dosya başındaki YAPILANDIRMA bölümünden alınır.
    """
    seen_recipes = set()
    count = 0
    errors = []
    
    end_page = start_page + pages - 1
    print(f'🍳 Yemek.com Scraper başlatılıyor...')
    print(f'📄 Sayfa {start_page} - {end_page} taranacak ({pages} sayfa)\n')
    
    with open(out_file, 'w', encoding='utf-8') as fout:
        for page in range(start_page, start_page + pages):
            listing_url = LISTING_TEMPLATE.format(page=page)
            print(f'[Sayfa {page}/{end_page}] {listing_url}')
            try:
                html = get_html(listing_url)
            except Exception as e:
                err_msg = f'  ❌ Liste sayfası alınamadı: {e}'
                print(err_msg)
                errors.append(err_msg)
                continue

            links = extract_recipe_links(html, max_recipes_per_page=14)
            print(f'  ✓ {len(links)} tarif linki bulundu (sayfa başına limit: 14)')
            
            page_recipes = 0
            for recipe_url in links:
                if recipe_url in seen_recipes:
                    continue
                seen_recipes.add(recipe_url)
                
                try:
                    r_html = get_html(recipe_url)
                    data = parse_recipe_page(r_html, recipe_url)
                    
                    # Basic validation
                    if not data['title'] or data['title'] == 'Başlık Bulunamadı':
                        print(f'   ⚠️  Başlık bulunamadı: {recipe_url}')
                    
                    if not data['ingredients']:
                        print(f'   ⚠️  Malzeme bulunamadı: {data["title"][:40]}')
                    
                    if not data['instructions']:
                        print(f'   ⚠️  Yapılış bulunamadı: {data["title"][:40]}')
                    
                    # Save even if incomplete (for debugging)
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                    fout.flush()  # ensure data is written immediately
                    count += 1
                    page_recipes += 1
                    
                    title_preview = data["title"][:50] + '...' if len(data["title"]) > 50 else data["title"]
                    print(f'   ✓ [{count}] {title_preview}')
                    
                    time.sleep(0.3)  # reduced delay for faster scraping
                    
                except Exception as e:
                    err_msg = f'   ❌ Tarif işlenemedi {recipe_url}: {e}'
                    print(err_msg)
                    errors.append(err_msg)
                    continue
            
            print(f'  → Sayfa {page} tamamlandı: {page_recipes} tarif kaydedildi\n')
            time.sleep(0.5)  # pause between pages

    print(f'\n{"="*60}')
    print(f'✅ Tamamlandı!')
    print(f'📊 Toplam {count} tarif kaydedildi → {out_file}')
    
    if errors:
        print(f'\n⚠️  {len(errors)} hata oluştu:')
        for err in errors[:5]:  # show first 5 errors
            print(f'  {err}')
        if len(errors) > 5:
            print(f'  ... ve {len(errors)-5} hata daha')
    
    print(f'{"="*60}\n')


def main_parallel(start_page, pages, out_file, max_workers):
    """Paralel tarif scraper - çok daha hızlı!
    
    Parametreler dosya başındaki YAPILANDIRMA bölümünden alınır.
    
    Args:
        start_page: Hangi sayfadan başlanacak
        pages: Kaç sayfa taranacak
        out_file: Çıktı dosyası
        max_workers: Aynı anda kaç tarif çekilecek (5-10 arası önerilir)
    """
    end_page = start_page + pages - 1
    print('🚀 Yemek.com Hızlı Scraper başlatılıyor...')
    print(f'📄 Sayfa {start_page} - {end_page} taranacak ({pages} sayfa, {max_workers} paralel worker)\n')
    
    errors = []
    seen_recipes = set()
    file_lock = Lock()  # Thread-safe file writing
    count = 0
    
    def scrape_recipe(recipe_url, recipe_num):
        """Tek bir tarifi çek"""
        nonlocal count
        try:
            time.sleep(0.2)  # Polite delay
            r_html = get_html(recipe_url)
            data = parse_recipe_page(r_html, recipe_url)
            
            # Basic validation
            warnings = []
            if not data['title'] or data['title'] == 'Başlık Bulunamadı':
                warnings.append('başlık yok')
            if not data['ingredients']:
                warnings.append('malzeme yok')
            if not data['instructions']:
                warnings.append('yapılış yok')
            
            # Thread-safe file write
            with file_lock:
                with open(out_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
                title_preview = data["title"][:50] + '...' if len(data["title"]) > 50 else data["title"]
                warning_str = f' [{", ".join(warnings)}]' if warnings else ''
                print(f'   ✓ [{count:3d}] {title_preview}{warning_str}')
            
            return True
            
        except Exception as e:
            err_msg = f'❌ {recipe_url}: {e}'
            errors.append(err_msg)
            return False
    
    # Clear output file
    with open(out_file, 'w', encoding='utf-8') as f:
        pass
    
    # Scrape pages
    for page in range(start_page, start_page + pages):
        print(f'\n[Sayfa {page}/{end_page}] {LISTING_TEMPLATE.format(page=page)}')
        
        try:
            html = get_html(LISTING_TEMPLATE.format(page=page))
        except Exception as e:
            err_msg = f'❌ Sayfa {page} yüklenemedi: {e}'
            print(f'  {err_msg}')
            errors.append(err_msg)
            continue
        
        links = extract_recipe_links(html, max_recipes_per_page=14)
        new_links = [url for url in links if url not in seen_recipes]
        seen_recipes.update(new_links)
        
        print(f'  ✓ {len(new_links)} yeni tarif bulundu')
        
        # Parallel scraping with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(scrape_recipe, url, i): url 
                for i, url in enumerate(new_links, 1)
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                pass  # Results already printed in scrape_recipe
        
        print(f'  → Sayfa {page} tamamlandı')
    
    print(f'\n{"="*60}')
    print(f'✅ Tamamlandı!')
    print(f'📊 Toplam {count} tarif kaydedildi → {out_file}')
    
    if errors:
        print(f'\n⚠️  {len(errors)} hata oluştu:')
        for err in errors[:5]:
            print(f'  {err}')
        if len(errors) > 5:
            print(f'  ... ve {len(errors)-5} hata daha')
    
    print(f'{"="*60}\n')


if __name__ == '__main__':
    # Paralel versiyonu kullan - Ayarları yukarıdaki YAPILANDIRMA bölümünden değiştirin
    main_parallel(start_page=START_PAGE, pages=PAGES_TO_SCRAPE, out_file=OUTPUT_FILE, max_workers=MAX_WORKERS)
