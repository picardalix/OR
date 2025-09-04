# -*- coding: utf-8 -*-
"""
Scraper Mango — uniquement "S'assortit parfaitement" (total look)
CSV: anchor_url,anchor_img,pos_url,pos_img,brand,name,color,price,composition,description
"""

import requests, json, re, os, time, random, hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import csv

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

def fetch_page(url):
    time.sleep(random.uniform(1, 2))
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text

def extract_mango_metadata(html):
    soup = BeautifulSoup(html, 'html.parser')

    name = ""
    h1 = soup.find('h1', class_=re.compile('ProductDetail_title|product-title', re.I))
    if h1: name = h1.get_text(strip=True)

    price = ""
    price_tag = soup.find('meta', {'itemprop': 'price'})
    if price_tag:
        val = price_tag.get('content', '')
        cur_tag = soup.find('meta', {'itemprop': 'priceCurrency'})
        cur = cur_tag.get('content') if cur_tag else 'EUR'
        price = f"{val} {cur}"

    color = ""
    color_tag = soup.find(['p','span'], class_=re.compile('ColorsSelector_label|color-name', re.I))
    if color_tag: color = color_tag.get_text(strip=True)

    description = ""
    desc = soup.find('div', class_=re.compile('Description_description|product-description', re.I))
    if desc:
        p = desc.find('p')
        if p: description = p.get_text(strip=True)

    composition = ""
    comp_text = soup.find(string=re.compile(r'Composition', re.I))
    if comp_text and comp_text.parent:
        full = comp_text.parent.get_text(" ", strip=True)
        m = re.search(r'composition[:\s]*([^.]+)', full, re.I)
        if m:
            composition = m.group(1).strip()
            composition = re.sub(r'\bet entretien\b', '', composition, flags=re.I).strip(" .:-")

    return {
        'brand': 'Mango',
        'name': name,
        'color': color,
        'price': price,
        'composition': composition,
        'description': description
    }

def extract_main_image(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    link = soup.find('link', {'itemprop': 'image'})
    if link and link.get('href'):
        return urljoin(base_url, link['href'])
    og = soup.find('meta', {'property': 'og:image'})
    if og and og.get('content'):
        return urljoin(base_url, og['content'])
    return None

def download_image(img_url, output_dir):
    if not img_url: return ""
    try:
        time.sleep(random.uniform(0.4, 0.9))
        r = requests.get(img_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        filename = hashlib.md5(img_url.encode()).hexdigest() + '.jpg'
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        Image.open(BytesIO(r.content)).convert('RGB').save(path, 'JPEG', quality=90)
        return path
    except Exception as e:
        print(f"[warn] image ko {img_url}: {e}")
        return ""

# ---- Décodage React Flight + filtre "total-look" uniquement ----
def _decode_flight_payloads(html):
    """
    Extrait les payloads React Flight depuis les appels:
    self.__next_f.push([1,"73:[\"$\",\"$L76\",null,{...}]"])
    -> on déséchappe la chaîne, on split sur le premier ':' puis json.loads
    """
    out = []
    # capture la STRING entière (entre " ou ') en gérant les backslashes
    pattern = re.compile(
        r'self\.__next_f\.push\(\[\d+,(?P<q>"|\')(?P<body>(?:\\.|(?!\1).)*)(?P=q)\]\)',
        re.S
    )
    for m in pattern.finditer(html):
        raw = m.group('body')  # ex: 73:[\"$\",\"$L76\",null,{\"location\":\"pdp-cross-selling:total-look:main\",...}]
        try:
            # déséchappe la string JS en s'appuyant sur le parseur JSON
            unescaped = json.loads(f'"{raw}"')
        except Exception:
            continue

        # coupe "73:" et récupère la vraie liste JSON ["$","$L76",null,{...}]
        parts = unescaped.split(':', 1)
        if len(parts) != 2:
            continue
        json_part = parts[1]
        try:
            arr = json.loads(json_part)
        except Exception:
            continue

        if isinstance(arr, list) and len(arr) >= 4 and isinstance(arr[3], dict):
            out.append(arr)

    print(f"[debug] Total payloads React Flight décodés: {len(out)}")
    return out

from urllib.parse import urljoin, urlparse

def _base_domain(url):
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"

def extract_total_look_items(html, base_url):
    items_all = []
    payloads = _decode_flight_payloads(html)
    print(f"[debug] Nombre de payloads trouvés: {len(payloads)}")

    domain = _base_domain(base_url)

    for i, payload in enumerate(payloads):
        obj = payload[3] if isinstance(payload, list) and len(payload) >= 4 else None
        if not isinstance(obj, dict):
            continue

        location = (obj.get('location') or '').lower()
        # on garde UNIQUEMENT le total-look
        if 'total-look' not in location:
            continue

        title = obj.get('title', {})
        title_content = title.get('content', '') if isinstance(title, dict) else str(title)
        print(f"[debug] Payload {i} OK — location: {location} | title: {title_content}")

        for j, item in enumerate(obj.get('crossSellingItems', [])):
            pinfo = item.get('productInfo') or {}
            rel_url = pinfo.get('url') or ''
            # IMPORTANT: l’URL produit doit être jointe sur shop.mango.com
            pos_url = urljoin(domain, rel_url) if rel_url else ''

            pos_img_url = None
            # essaye d’abord via looks[lookId].images[...].img
            looks = item.get('looks') or {}
            look_id = item.get('lookId') or next(iter(looks), None)
            if look_id and look_id in looks:
                images = looks[look_id].get('images') or {}
                # prends la première image avec un champ 'img'
                for key, val in images.items():
                    if isinstance(val, dict) and val.get('img'):
                        pos_img_url = urljoin(domain, val['img'])  # parfois chemin relatif
                        break

            # fallback: certaines structures mettent l’image directement dans productInfo
            if not pos_img_url:
                # champs possibles (selon pages) : 'image', 'img', 'thumbnail', ...
                for k in ('image', 'img', 'thumbnail'):
                    if pinfo.get(k):
                        pos_img_url = urljoin(domain, pinfo[k])
                        break

            items_all.append({
                'pos_url': pos_url,
                'pos_img_url': pos_img_url,
                'product_name': pinfo.get('name', ''),
                'product_id': item.get('productId', '')
            })

    print(f"[debug] Total items extraits: {len(items_all)}")
    return items_all


# ------------------------ Scrape d'une page produit ------------------------
def scrape_mango_product(url, output_dir):
    print(f"Scraping: {url}")
    try:
        html = fetch_page(url)
    except Exception as e:
        print(f"[err] page: {e}")
        return []

    meta = extract_mango_metadata(html)
    anchor_img_url = extract_main_image(html, url)
    anchor_img_path = download_image(anchor_img_url, os.path.join(output_dir, 'images'))

    xsell_items = extract_total_look_items(html, url)
    print(f"[info] Trouvé {len(xsell_items)} items 'S'assortit parfaitement'")

    rows = []
    if not xsell_items:
        # aucune suggestion "S'assortit parfaitement" : une ligne "ancre seule"
        print("[info] Aucun item total-look trouvé, création d'une ligne ancre seule")
        rows.append({
            'anchor_url': url,
            'anchor_img': anchor_img_path or "",
            'pos_url': "",
            'pos_img': "",
            **meta
        })
        return rows

    # Une ligne par article *total-look*
    for i, item in enumerate(xsell_items):
        pos_url = item.get('pos_url', '')
        pos_img_path = ""
        
        if item.get('pos_img_url'):
            pos_img_path = download_image(item['pos_img_url'], os.path.join(output_dir, 'images'))
        elif pos_url:
            # fallback: récupérer l'og:image du produit associé
            try:
                print(f"[info] Récupération image fallback pour {pos_url}")
                phtml = fetch_page(pos_url)
                og = extract_main_image(phtml, pos_url)
                pos_img_path = download_image(og, os.path.join(output_dir, 'images'))
            except Exception as e:
                print(f"[warn] pas d'image pour {pos_url}: {e}")

        row = {
            'anchor_url': url,
            'anchor_img': anchor_img_path or "",
            'pos_url': pos_url,
            'pos_img': pos_img_path or "",
            **meta
        }
        rows.append(row)
        print(f"[info] Ligne {i+1} créée: {item.get('product_name', 'UNKNOWN')}")

    return rows

# ------------------------------- Main --------------------------------------
def main():
    urls = [
        "https://shop.mango.com/fr/fr/p/femme/pantalon/pantalons-elegants/pantalon-de-costume-droit-laine-melangee_17086003",
"https://shop.mango.com/fr/fr/p/femme/vestes/double-face/veste-en-melange-de-laine-avec-col-en-sherpa_17076735",
"https://shop.mango.com/fr/fr/p/femme/pulls-et-cardigans/cardigan/cardigan-col-bateau-avec-boutons_17085811",
"https://shop.mango.com/fr/fr/p/femme/chemises-et-blouses/blouse/blouse-100-soie-satinee-col-montant_17066348",
"https://shop.mango.com/fr/fr/p/femme/chemises-et-blouses/blouse/blouse-col-foulard_17036740",
"https://shop.mango.com/fr/fr/p/femme/top/tricot/top-maille-leopard_17066346",
"https://shop.mango.com/fr/fr/p/teen/jupes/longue/jupe-longue-carreaux_17024779",
"https://shop.mango.com/fr/fr/p/teen/sweat/sweat-shirt-imprime-capuche_17074072",
"https://shop.mango.com/fr/fr/p/teen/chemises-et-blouses/blouses/blouse-fluide-froncee_17045791?c=76",
"https://shop.mango.com/fr/fr/p/femme/jupe/midi/jupe-crayon-simili-cuir_17037779",
"https://shop.mango.com/fr/fr/p/femme/pulls-et-cardigans/cardigan/cardigan-maille-fine-foulard_17045825",
"https://shop.mango.com/fr/fr/p/femme/pantalon/wideleg/pantalon-flare-simili-cuir_17045975?c=32"

    ]
    output_dir = "dataset_mango"
    all_rows = []

    for url in urls:
        all_rows.extend(scrape_mango_product(url, output_dir))

    if all_rows:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'products.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'anchor_url','anchor_img','pos_url','pos_img',
                'brand','name','color','price','composition','description'
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nRésultats sauvegardés dans {csv_path}")
        print(f"Total: {len(all_rows)} lignes")
    else:
        print("Aucun résultat à sauvegarder")

if __name__ == "__main__":
    main()