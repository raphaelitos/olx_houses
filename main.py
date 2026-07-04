import os
import re
import json
import time
import random
import datetime
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


OUTPUT_CSV = os.getenv("OUTPUT_CSV", "data_imoveis_vitoria_vila_velha.csv")

MAX_PAGES = int(os.getenv("MAX_PAGES", 20))

HEADLESS = False

MIN_SLEEP = float(os.getenv("MIN_SLEEP", 8))
MAX_SLEEP = float(os.getenv("MAX_SLEEP", 16))

PAGE_TIMEOUT = int(os.getenv("PAGE_TIMEOUT", 45))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))

SAVE_DEBUG_HTML = os.getenv("SAVE_DEBUG_HTML", "1") == "1"
DEBUG_DIR = Path(os.getenv("DEBUG_DIR", "debug_pages"))


LISTING_URLS = [
    {
        "cidade": "Vitória",
        "tipo_url": "apartamento",
        "url": "https://www.olx.com.br/imoveis/venda/apartamentos/estado-es/norte-do-espirito-santo/vitoria",
    },
    {
        "cidade": "Vitória",
        "tipo_url": "casa",
        "url": "https://www.olx.com.br/imoveis/venda/casas/estado-es/norte-do-espirito-santo/vitoria",
    },
    {
        "cidade": "Vila Velha",
        "tipo_url": "apartamento",
        "url": "https://www.olx.com.br/imoveis/venda/apartamentos/estado-es/norte-do-espirito-santo/vila-velha",
    },
    {
        "cidade": "Vila Velha",
        "tipo_url": "casa",
        "url": "https://www.olx.com.br/imoveis/venda/casas/estado-es/norte-do-espirito-santo/vila-velha",
    },
]


COLUMNS = [
    "id_anuncio",
    "titulo",
    "preco_venda",
    "tipo_imovel",
    "subtipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "anunciante_profissional",
    "qtd_imagens",
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_gourmet",
    "tem_lazer",
    "tem_mobiliado",
    "tem_porteira_fechada",
    "tem_elevador",
    "tem_cobertura",
    "tem_garden",
    "tem_sol_manha",
    "tem_frente_mar",
    "preco_m2",
    "preco_m2_suspeito",
    "area_suspeita",
    "preco_suspeito",
    "idade_anuncio_dias",
    "categoria_olx",
    "tipo_detalhado",
    "tipo_url",
    "url",
    "data_anuncio_olx",
    "data_coleta",
]


INTEGER_COLUMNS = [
    "id_anuncio",
    "preco_venda",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "qtd_imagens",
    "idade_anuncio_dias",
]


FLOAT_COLUMNS = [
    "preco_m2",
]


BOOLEAN_COLUMNS = [
    "anunciante_profissional",
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_gourmet",
    "tem_lazer",
    "tem_mobiliado",
    "tem_porteira_fechada",
    "tem_elevador",
    "tem_cobertura",
    "tem_garden",
    "tem_sol_manha",
    "tem_frente_mar",
    "preco_m2_suspeito",
    "area_suspeita",
    "preco_suspeito",
]


@dataclass
class ListingPage:
    cidade: str
    tipo_url: str
    url: str
    page: int


def random_sleep():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def normalize_text(text):
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except TypeError:
        pass

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text)

    return text


def slug_text(text):
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or None


def safe_filename(text):
    text = slug_text(text)
    return text or "sem_nome"


def contains_any(text, terms):
    normalized = normalize_text(text)
    normalized_terms = [normalize_text(term) for term in terms]

    return any(term in normalized for term in normalized_terms)


def make_driver():
    options = Options()

    if HEADLESS:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--disable-blink-features=AutomationControlled")

    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_TIMEOUT)

    return driver


def build_page_url(base_url, page):
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}o={page}"


def parse_money(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

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
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    value = str(value).strip()

    if value == "5+":
        return 5

    match = re.search(r"\d+", value)

    if not match:
        return None

    return int(match.group(0))


def unix_to_iso(timestamp):
    if timestamp is None:
        return None

    try:
        if pd.isna(timestamp):
            return None
    except TypeError:
        pass

    try:
        return datetime.datetime.fromtimestamp(int(timestamp)).isoformat(timespec="seconds")
    except (ValueError, TypeError, OSError):
        return None


def iso_to_datetime(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def calculate_age_days(data_anuncio_olx, data_coleta):
    announcement_date = iso_to_datetime(data_anuncio_olx)
    collection_date = iso_to_datetime(data_coleta)

    if not announcement_date or not collection_date:
        return None

    delta = collection_date - announcement_date

    if delta.days < 0:
        return None

    return delta.days


def get_next_data_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})

    if not script or not script.string:
        return None

    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


def get_ads_from_next_data(next_data):
    if not next_data:
        return []

    return (
        next_data
        .get("props", {})
        .get("pageProps", {})
        .get("ads", [])
    )


def get_ads_from_react_flight(html):
    """
    Extrai anúncios do formato React Flight usado nas URLs filtradas
    por casas/apartamentos.
    """
    marker = r'{\"ads\":['
    start = html.find(marker)

    if start == -1:
        return []

    end_marker = r'],\"searchBoxProps\"'
    end = html.find(end_marker, start)

    if end == -1:
        return []

    fragment = html[start:end + 1] + "}"

    decoded = (
        fragment
        .replace(r'\"', '"')
        .replace(r'\\', '\\')
    )

    try:
        data = json.loads(decoded)
    except json.JSONDecodeError as error:
        print("Falha ao decodificar anúncios do React Flight.")
        print(f"Motivo: {error}")
        return []

    ads = data.get("ads", [])

    if not isinstance(ads, list):
        return []

    return ads


def get_ads_from_html(html):
    """
    Tenta os dois formatos conhecidos da OLX:
    1. __NEXT_DATA__, usado em algumas listagens.
    2. React Flight, usado nas URLs especializadas.
    """
    next_data = get_next_data_from_html(html)
    ads = get_ads_from_next_data(next_data)

    if ads:
        return ads, "__NEXT_DATA__"

    ads = get_ads_from_react_flight(html)

    if ads:
        return ads, "react_flight"

    return [], "not_found"


def properties_to_dict(properties):
    result = {}

    if not isinstance(properties, list):
        return result

    for item in properties:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        value = item.get("value")

        if name:
            result[name] = value

    return result


def get_property_value(props, name):
    return props.get(name)


def get_tipo_imovel_from_url(tipo_url):
    tipo = normalize_text(tipo_url)

    if tipo not in {"casa", "apartamento"}:
        raise ValueError(
            f"tipo_url inválido: {tipo_url}. Use apenas 'casa' ou 'apartamento'."
        )

    return tipo


def get_subtipo_imovel(tipo_detalhado, tipo_imovel):
    """
    O tipo geral vem da URL. Esta função só detalha o subtipo
    quando a OLX entrega real_estate_type.
    """
    text = normalize_text(tipo_detalhado)

    if not text:
        return None

    if tipo_imovel == "apartamento":
        if "cobertura" in text:
            return "apartamento_cobertura"

        if "loft" in text or "studio" in text:
            return "loft_studio"

        if "padrao" in text or "apartamento" in text:
            return "apartamento_padrao"

        return slug_text(text)

    if tipo_imovel == "casa":
        if "condominio" in text:
            return "casa_condominio"

        if "rua publica" in text:
            return "casa_rua_publica"

        if "casa" in text:
            return "casa"

        return slug_text(text)

    return slug_text(text)


def extract_text_flags(titulo, tipo_detalhado, re_features=None, re_complex_features=None, re_types=None):
    """
    Usa textos já presentes na listagem.
    Não abre o anúncio individual.
    """
    text = " ".join(
        str(value)
        for value in [
            titulo,
            tipo_detalhado,
            re_features,
            re_complex_features,
            re_types,
        ]
        if value
    )

    return {
        "tem_suite": contains_any(text, ["suite", "suíte", "suites", "suítes"]),
        "tem_piscina": contains_any(text, ["piscina"]),
        "tem_varanda": contains_any(text, ["varanda", "sacada"]),
        "tem_gourmet": contains_any(text, ["gourmet"]),
        "tem_lazer": contains_any(text, ["lazer", "clube", "academia", "playground", "salão de festas", "salao de festas"]),
        "tem_mobiliado": contains_any(
            text,
            ["mobiliado", "mobiliada", "moveis planejados", "móveis planejados"],
        ),
        "tem_porteira_fechada": contains_any(text, ["porteira fechada"]),
        "tem_elevador": contains_any(text, ["elevador"]),
        "tem_cobertura": contains_any(text, ["cobertura"]),
        "tem_garden": contains_any(text, ["garden"]),
        "tem_sol_manha": contains_any(text, ["sol da manha", "sol da manhã"]),
        "tem_frente_mar": contains_any(
            text,
            ["frente mar", "frente para o mar", "vista mar", "vista para o mar"],
        ),
    }


def calculate_price_per_m2(preco_venda, area_m2):
    if not preco_venda or not area_m2:
        return None

    if area_m2 <= 0:
        return None

    return round(preco_venda / area_m2, 2)


def is_suspicious_area(area_m2):
    if area_m2 is None:
        return False

    return area_m2 < 10 or area_m2 > 1000


def is_suspicious_price(preco_venda):
    if preco_venda is None:
        return False

    return preco_venda < 50000 or preco_venda > 20000000


def is_suspicious_price_per_m2(preco_m2):
    if preco_m2 is None:
        return False

    return preco_m2 < 1000 or preco_m2 > 50000


def normalize_ad(ad, listing_page):
    if not isinstance(ad, dict):
        return None

    if "advertisingId" in ad:
        return None

    if not ad.get("listId") or not ad.get("url"):
        return None

    props = properties_to_dict(ad.get("properties"))
    location_details = ad.get("locationDetails") or {}

    titulo = ad.get("subject")
    tipo_imovel = get_tipo_imovel_from_url(listing_page.tipo_url)
    tipo_detalhado = get_property_value(props, "real_estate_type")

    cidade = location_details.get("municipality") or listing_page.cidade
    bairro = location_details.get("neighbourhood")

    preco_venda = parse_money(ad.get("priceValue") or ad.get("price"))
    area_m2 = parse_integer(get_property_value(props, "size"))
    quartos = parse_integer(get_property_value(props, "rooms"))
    banheiros = parse_integer(get_property_value(props, "bathrooms"))
    vagas = parse_integer(get_property_value(props, "garage_spaces"))

    re_features = get_property_value(props, "re_features")
    re_complex_features = get_property_value(props, "re_complex_features")
    re_types = get_property_value(props, "re_types")

    data_coleta = datetime.datetime.now().isoformat(timespec="seconds")
    data_anuncio_olx = unix_to_iso(ad.get("date"))
    preco_m2 = calculate_price_per_m2(preco_venda, area_m2)

    text_flags = extract_text_flags(
        titulo=titulo,
        tipo_detalhado=tipo_detalhado,
        re_features=re_features,
        re_complex_features=re_complex_features,
        re_types=re_types,
    )

    return {
        "id_anuncio": parse_integer(ad.get("listId")),
        "titulo": titulo,
        "preco_venda": preco_venda,
        "tipo_imovel": tipo_imovel,
        "subtipo_imovel": get_subtipo_imovel(tipo_detalhado, tipo_imovel),
        "cidade": cidade,
        "bairro": bairro,
        "area_m2": area_m2,
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas,
        "anunciante_profissional": bool(ad.get("professionalAd", False)),
        "qtd_imagens": parse_integer(ad.get("imageCount")) or 0,
        **text_flags,
        "preco_m2": preco_m2,
        "preco_m2_suspeito": is_suspicious_price_per_m2(preco_m2),
        "area_suspeita": is_suspicious_area(area_m2),
        "preco_suspeito": is_suspicious_price(preco_venda),
        "idade_anuncio_dias": calculate_age_days(data_anuncio_olx, data_coleta),
        "categoria_olx": ad.get("categoryName") or ad.get("category"),
        "tipo_detalhado": tipo_detalhado,
        "tipo_url": listing_page.tipo_url,
        "url": ad.get("url"),
        "data_anuncio_olx": data_anuncio_olx,
        "data_coleta": data_coleta,
    }


def safe_get(driver, url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            driver.get(url)
            return True
        except Exception as error:
            print(f"Tentativa {attempt}/{MAX_RETRIES} falhou.")
            print(f"Motivo: {error}")
            random_sleep()

    return False


def save_debug_html(html, listing_page, reason):
    if not SAVE_DEBUG_HTML:
        return None

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"debug_{safe_filename(listing_page.cidade)}_"
        f"{safe_filename(listing_page.tipo_url)}_"
        f"pagina_{listing_page.page}_"
        f"{safe_filename(reason)}.html"
    )

    path = DEBUG_DIR / filename
    path.write_text(html, encoding="utf-8")

    return path


def fetch_listing_page(driver, listing_page):
    page_url = build_page_url(listing_page.url, listing_page.page)

    print(
        f"Coletando: {listing_page.cidade} | "
        f"{listing_page.tipo_url} | página {listing_page.page}"
    )
    print(page_url)

    loaded = safe_get(driver, page_url)

    if not loaded:
        print("Falha ao carregar página após tentativas. Pulando.\n")
        return []

    try:
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            lambda current_driver: (
                "__NEXT_DATA__" in current_driver.page_source
                or r'{\"ads\":[' in current_driver.page_source
                or "sorry, you have been blocked" in current_driver.page_source.lower()
                or "access denied" in current_driver.page_source.lower()
            )
        )
    except TimeoutException:
        print("A página carregou, mas os dados dos anúncios não apareceram dentro do timeout.")

    random_sleep()

    html = driver.page_source
    page_text = html.lower()

    if "sorry, you have been blocked" in page_text or "access denied" in page_text:
        print("Página bloqueada pela OLX. Pulando esta página.\n")
        debug_path = save_debug_html(html, listing_page, "bloqueio")
        if debug_path:
            print(f"HTML de debug salvo em: {debug_path}")
        return []

    ads, source = get_ads_from_html(html)
    print(f"Fonte dos dados: {source}. Itens brutos encontrados: {len(ads)}")

    if not ads:
        debug_path = save_debug_html(html, listing_page, "sem_ads")
        if debug_path:
            print(f"HTML de debug salvo em: {debug_path}")
        return []

    rows = []

    for ad in ads:
        row = normalize_ad(ad, listing_page)

        if row:
            rows.append(row)

    print(f"Anúncios válidos após normalização: {len(rows)}")

    if not rows:
        debug_path = save_debug_html(html, listing_page, "sem_linhas_validas")
        if debug_path:
            print(f"HTML de debug salvo em: {debug_path}")

    return rows


def apply_dataset_types(data):
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data = data[COLUMNS]

    for column in INTEGER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")

    for column in FLOAT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").round(2)

    for column in BOOLEAN_COLUMNS:
        data[column] = data[column].fillna(False).astype(bool)

    return data


def load_existing_dataset():
    if os.path.exists(OUTPUT_CSV):
        data = pd.read_csv(OUTPUT_CSV)

        for column in COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA

        data = data[COLUMNS]
        return apply_dataset_types(data)

    return pd.DataFrame(columns=COLUMNS)


def save_dataset(rows):
    if not rows:
        return load_existing_dataset()

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

    data = data.drop_duplicates(subset=["id_anuncio"], keep="last")

    data = data.sort_values(
        by=["cidade", "tipo_imovel", "bairro", "preco_venda"],
        na_position="last",
    )

    data.to_csv(OUTPUT_CSV, index=False)

    return data


def build_listing_pages():
    pages = []

    for item in LISTING_URLS:
        for page in range(1, MAX_PAGES + 1):
            pages.append(
                ListingPage(
                    cidade=item["cidade"],
                    tipo_url=item["tipo_url"],
                    url=item["url"],
                    page=page,
                )
            )

    return pages


def main():
    driver = make_driver()
    total_rows = 0

    try:
        pages = build_listing_pages()

        for listing_page in tqdm(pages, desc="Páginas de listagem"):
            try:
                rows = fetch_listing_page(driver, listing_page)
                total_rows += len(rows)

                print(f"Anúncios extraídos nesta página: {len(rows)}")

                if rows:
                    data = save_dataset(rows)
                    print(f"CSV atualizado. Registros totais: {len(data)}\n")
                else:
                    print("Nenhum registro novo salvo nesta página.\n")

            except Exception as error:
                print(f"Erro na página {listing_page.url}")
                print(f"Motivo: {error}\n")

        final_data = load_existing_dataset()

        print("Coleta finalizada.")
        print(f"Novos registros brutos nesta execução: {total_rows}")
        print(f"Arquivo salvo: {OUTPUT_CSV}")
        print(f"Registros totais no CSV: {len(final_data)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()