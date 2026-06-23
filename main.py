import os
import re
import json
import time
import random
import datetime
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# CSV de saída
OUTPUT_CSV = "data_imoveis.csv"

# Volume pequeno para teste inicial
MAX_PAGES = int(os.getenv("MAX_PAGES", 2))
HEADLESS = os.getenv("HEADLESS", "1") == "1"

# Intervalo entre acessos para reduzir chance de bloqueio
MIN_SLEEP = float(os.getenv("MIN_SLEEP", 4))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", 8))

# Insira aqui as URLs de listagem que você encontrar manualmente.
# Exemplo atual: Vitória, imóveis à venda.
LISTING_URLS = [
    {
        "cidade": "Vitória",
        "url": "https://www.olx.com.br/imoveis/venda/estado-es/norte-do-espirito-santo/vitoria",
    },
]

COLUMNS = [
    "id_anuncio",
    "titulo",
    "preco",
    "tipo_imovel",
    "tipo_detalhado",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "condominio",
    "iptu",
    "anunciante_profissional",
    "quantidade_imagens",
    "categoria_olx",
    "url",
    "data_anuncio_olx",
    "data_coleta",
]


@dataclass
class ListingPage:
    cidade: str
    url: str
    page: int


def random_sleep():
    """Espera um tempo aleatório entre acessos."""
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def make_driver():
    """Cria o ChromeDriver automaticamente, sem chromedriver fixo no projeto."""
    options = Options()

    if HEADLESS:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--disable-blink-features=AutomationControlled")

    # User-agent comum de Chrome em Linux
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def build_page_url(base_url, page):
    """Monta a URL da página de listagem."""
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}o={page}"


def parse_money(value):
    """Converte valores como 'R$ 868.323' para 868323."""
    if value is None:
        return None

    value = str(value)
    match = re.search(r"R\$\s*([\d\.\,]+)", value)

    if not match:
        return None

    number = match.group(1).replace(".", "").replace(",", ".")

    try:
        return int(float(number))
    except ValueError:
        return None


def parse_integer(value):
    """Extrai número inteiro de textos como '63m²', '2' ou '5+'."""
    if value is None:
        return None

    value = str(value).strip()

    if value == "5+":
        return 5

    match = re.search(r"\d+", value)

    if not match:
        return None

    return int(match.group(0))


def unix_to_iso(timestamp):
    """Converte timestamp Unix da OLX para ISO, quando possível."""
    if timestamp is None:
        return None

    try:
        return datetime.datetime.fromtimestamp(int(timestamp)).isoformat(timespec="seconds")
    except (ValueError, TypeError, OSError):
        return None


def get_next_data_from_html(html):
    """Extrai o JSON do script __NEXT_DATA__ da página."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})

    if not script or not script.string:
        return None

    return json.loads(script.string)


def get_ads_from_next_data(next_data):
    """Retorna a lista de anúncios dentro do __NEXT_DATA__."""
    if not next_data:
        return []

    return (
        next_data
        .get("props", {})
        .get("pageProps", {})
        .get("ads", [])
    )


def properties_to_dict(properties):
    """Transforma a lista de propriedades da OLX em dicionário por name."""
    result = {}

    if not isinstance(properties, list):
        return result

    for item in properties:
        name = item.get("name")
        value = item.get("value")

        if name:
            result[name] = value

    return result


def infer_tipo_imovel(ad, props):
    """Define casa/apartamento a partir da categoria e do tipo detalhado."""
    category = str(ad.get("categoryName") or ad.get("category") or "").lower()
    detailed_type = str(props.get("real_estate_type") or "").lower()
    text = f"{category} {detailed_type}"

    if "apartamento" in text:
        return "apartamento"

    if "casa" in text:
        return "casa"

    return category or None


def normalize_ad(ad, fallback_city):
    """Converte um anúncio bruto da OLX em uma linha do nosso dataset."""
    if "advertisingId" in ad:
        return None

    if not ad.get("listId") or not ad.get("url"):
        return None

    props = properties_to_dict(ad.get("properties"))
    location_details = ad.get("locationDetails") or {}

    city = location_details.get("municipality") or fallback_city
    neighborhood = location_details.get("neighbourhood")

    return {
        "id_anuncio": ad.get("listId"),
        "titulo": ad.get("subject"),
        "preco": parse_money(ad.get("priceValue") or ad.get("price")),
        "tipo_imovel": infer_tipo_imovel(ad, props),
        "tipo_detalhado": props.get("real_estate_type"),
        "cidade": city,
        "bairro": neighborhood,
        "area_m2": parse_integer(props.get("size")),
        "quartos": parse_integer(props.get("rooms")),
        "banheiros": parse_integer(props.get("bathrooms")),
        "vagas": parse_integer(props.get("garage_spaces")),
        "condominio": parse_money(props.get("condominio")),
        "iptu": parse_money(props.get("iptu")),
        "anunciante_profissional": ad.get("professionalAd"),
        "quantidade_imagens": ad.get("imageCount"),
        "categoria_olx": ad.get("categoryName") or ad.get("category"),
        "url": ad.get("url"),
        "data_anuncio_olx": unix_to_iso(ad.get("date")),
        "data_coleta": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def fetch_listing_page(driver, listing_page):
    """Acessa uma página de listagem e retorna os anúncios extraídos do JSON."""
    page_url = build_page_url(listing_page.url, listing_page.page)

    print(f"Coletando: {listing_page.cidade} | página {listing_page.page}")
    print(page_url)

    driver.get(page_url)
    random_sleep()

    page_text = driver.page_source.lower()

    if "sorry, you have been blocked" in page_text or "access denied" in page_text:
        print("Página bloqueada pela OLX. Pulando esta página.")
        return []

    next_data = get_next_data_from_html(driver.page_source)
    ads = get_ads_from_next_data(next_data)

    rows = []

    for ad in ads:
        row = normalize_ad(ad, listing_page.cidade)

        if row:
            rows.append(row)

    return rows


def save_dataset(rows):
    """Salva o CSV preservando execuções anteriores e removendo duplicatas."""
    new_data = pd.DataFrame(rows, columns=COLUMNS)

    if os.path.exists(OUTPUT_CSV):
        old_data = pd.read_csv(OUTPUT_CSV)

        for column in COLUMNS:
            if column not in old_data.columns:
                old_data[column] = None

        old_data = old_data[COLUMNS]
        data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        data = new_data

    data = data.drop_duplicates(subset=["id_anuncio"], keep="last")
    data = data.sort_values(by=["cidade", "bairro", "preco"], na_position="last")
    data.to_csv(OUTPUT_CSV, index=False)

    return data


def build_listing_pages():
    """Cria a lista de páginas que serão visitadas."""
    pages = []

    for item in LISTING_URLS:
        for page in range(1, MAX_PAGES + 1):
            pages.append(
                ListingPage(
                    cidade=item["cidade"],
                    url=item["url"],
                    page=page,
                )
            )

    return pages


def main():
    driver = make_driver()
    all_rows = []

    try:
        pages = build_listing_pages()

        for listing_page in tqdm(pages, desc="Páginas de listagem"):
            try:
                rows = fetch_listing_page(driver, listing_page)
                print(f"Anúncios extraídos nesta página: {len(rows)}\n")
                all_rows.extend(rows)

            except Exception as error:
                print(f"Erro na página {listing_page.url}")
                print(f"Motivo: {error}\n")

        data = save_dataset(all_rows)

        print(f"Novos registros nesta execução: {len(all_rows)}")
        print(f"Arquivo salvo: {OUTPUT_CSV}")
        print(f"Registros totais no CSV: {len(data)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
