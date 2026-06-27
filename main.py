import os
import re
import json
import time
import random
import datetime
import unicodedata
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


OUTPUT_CSV = "data_imoveis.csv"

# Volume pequeno para teste inicial
MAX_PAGES = int(os.getenv("MAX_PAGES", 2))
HEADLESS = os.getenv("HEADLESS", "1") == "1"

# Intervalo entre acessos para reduzir chance de bloqueio
MIN_SLEEP = float(os.getenv("MIN_SLEEP", 4))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", 8))

#URLs de listagem obtidas manualmente.
LISTING_URLS = [
    {
        "cidade": "Vitória",
        "url": "https://www.olx.com.br/imoveis/venda/estado-es/norte-do-espirito-santo/vitoria",
    },
]

COLUMNS = [
    "id_anuncio",
    "titulo",
    "preco_venda",
    "tipo_imovel",
    "subtipo_imovel",
    "cidade",
    "cidade_norm",
    "bairro",
    "bairro_norm",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "condominio_valor",
    "iptu_valor",
    "condominio_informado",
    "iptu_informado",
    "condominio_suspeito",
    "iptu_suspeito",
    "anunciante_profissional",
    "qtd_imagens",
    "categoria_olx",
    "tipo_detalhado",
    "url",
    "data_anuncio_olx",
    "data_coleta",
    "preco_m2",
]


INTEGER_COLUMNS = [
    "id_anuncio",
    "preco_venda",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "condominio_valor",
    "iptu_valor",
    "qtd_imagens",
]

BOOLEAN_COLUMNS = [
    "condominio_informado",
    "iptu_informado",
    "condominio_suspeito",
    "iptu_suspeito",
    "anunciante_profissional",
]


@dataclass
class ListingPage:
    cidade: str
    url: str
    page: int


def random_sleep():
    """Espera um tempo aleatório entre acessos."""
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def normalize_text(text):
    """Normaliza texto para uso em agrupamentos e comparação."""
    if text is None or pd.isna(text):
        return None

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or None


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
    if value is None or pd.isna(value):
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
    if value is None or pd.isna(value):
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
    if timestamp is None or pd.isna(timestamp):
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

    if "apartamento" in text or "loft" in text or "studio" in text or "cobertura" in text:
        return "apartamento"

    if "casa" in text:
        return "casa"

    return normalize_text(category)


def infer_subtipo_imovel(tipo_detalhado):
    """Cria uma categoria mais simples a partir do tipo detalhado da OLX."""
    text = normalize_text(tipo_detalhado)

    if not text:
        return None

    if "cobertura" in text:
        return "apartamento_cobertura"

    if "loft" in text or "studio" in text:
        return "loft_studio"

    if "apartamento" in text and "padrao" in text:
        return "apartamento_padrao"

    if "casa" in text and "condominio" in text:
        return "casa_condominio"

    if "casa" in text and "rua_publica" in text:
        return "casa_rua_publica"

    if "casa" in text:
        return "casa"

    if "apartamento" in text:
        return "apartamento"

    return text


def is_suspicious_condominio(value):
    """Marca condomínio possivelmente simbólico/placeholder."""
    return value is not None and 0 < value < 50


def is_suspicious_iptu(value):
    """Marca IPTU possivelmente simbólico/placeholder."""
    return value is not None and 0 < value < 20


def calculate_price_per_m2(preco_venda, area_m2):
    """Calcula preço por metro quadrado para EDA e detecção de outliers."""
    if not preco_venda or not area_m2:
        return None

    if area_m2 <= 0:
        return None

    return round(preco_venda / area_m2, 2)


def normalize_ad(ad, fallback_city):
    """Converte um anúncio bruto da OLX em uma linha limpa do dataset."""
    if "advertisingId" in ad:
        return None

    if not ad.get("listId") or not ad.get("url"):
        return None

    props = properties_to_dict(ad.get("properties"))
    location_details = ad.get("locationDetails") or {}

    tipo_detalhado = props.get("real_estate_type")
    cidade = location_details.get("municipality") or fallback_city
    bairro = location_details.get("neighbourhood")

    preco_venda = parse_money(ad.get("priceValue") or ad.get("price"))
    area_m2 = parse_integer(props.get("size"))
    quartos = parse_integer(props.get("rooms"))
    banheiros = parse_integer(props.get("bathrooms"))
    vagas = parse_integer(props.get("garage_spaces"))
    condominio_valor = parse_money(props.get("condominio"))
    iptu_valor = parse_money(props.get("iptu"))

    return {
        "id_anuncio": parse_integer(ad.get("listId")),
        "titulo": ad.get("subject"),
        "preco_venda": preco_venda,
        "tipo_imovel": infer_tipo_imovel(ad, props),
        "subtipo_imovel": infer_subtipo_imovel(tipo_detalhado),
        "cidade": cidade,
        "cidade_norm": normalize_text(cidade),
        "bairro": bairro,
        "bairro_norm": normalize_text(bairro),
        "area_m2": area_m2,
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas,
        "condominio_valor": condominio_valor,
        "iptu_valor": iptu_valor,
        "condominio_informado": condominio_valor is not None,
        "iptu_informado": iptu_valor is not None,
        "condominio_suspeito": is_suspicious_condominio(condominio_valor),
        "iptu_suspeito": is_suspicious_iptu(iptu_valor),
        "anunciante_profissional": bool(ad.get("professionalAd", False)),
        "qtd_imagens": parse_integer(ad.get("imageCount")) or 0,
        "categoria_olx": ad.get("categoryName") or ad.get("category"),
        "tipo_detalhado": tipo_detalhado,
        "url": ad.get("url"),
        "data_anuncio_olx": unix_to_iso(ad.get("date")),
        "data_coleta": datetime.datetime.now().isoformat(timespec="seconds"),
        "preco_m2": calculate_price_per_m2(preco_venda, area_m2),
    }


def fetch_listing_page(driver, listing_page):
    """Acessa uma listagem e retorna anúncios extraídos do __NEXT_DATA__."""
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


def apply_dataset_types(data):
    """Aplica tipos úteis para evitar floats como 1.0 em colunas inteiras."""
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data = data[COLUMNS]

    for column in INTEGER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")

    data["preco_m2"] = pd.to_numeric(data["preco_m2"], errors="coerce").round(2)

    for column in BOOLEAN_COLUMNS:
        data[column] = data[column].fillna(False).astype(bool)

    return data


def remove_unusable_rows(data):
    """Remove registros sem informação mínima para treino do modelo."""
    data = data.dropna(subset=["id_anuncio", "preco_venda", "area_m2"])

    # Evita imóveis sem área válida ou sem preço válido.
    data = data[data["area_m2"] > 0]
    data = data[data["preco_venda"] > 0]

    return data


def save_dataset(rows):
    """Salva o CSV preservando execuções anteriores e removendo duplicatas."""
    new_data = pd.DataFrame(rows, columns=COLUMNS)

    if os.path.exists(OUTPUT_CSV):
        old_data = pd.read_csv(OUTPUT_CSV)

        for column in COLUMNS:
            if column not in old_data.columns:
                old_data[column] = pd.NA

        old_data = old_data[COLUMNS]
        data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        data = new_data

    data = apply_dataset_types(data)
    data = remove_unusable_rows(data)

    # Deduplicação de coleta: mantém o anúncio mais recente por ID.
    data = data.drop_duplicates(subset=["id_anuncio"], keep="last")

    data = data.sort_values(
        by=["cidade_norm", "bairro_norm", "preco_venda"],
        na_position="last",
    )

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

        print(f"Novos registros brutos nesta execução: {len(all_rows)}")
        print(f"Arquivo salvo: {OUTPUT_CSV}")
        print(f"Registros úteis totais no CSV: {len(data)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
