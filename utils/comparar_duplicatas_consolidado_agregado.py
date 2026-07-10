import pandas as pd
from pathlib import Path
import unicodedata


DATASET_BASE = "dataset_modelagem_consolidado.csv"
DATASET_NOVO = "dataset_modelagem_agregado.csv"

REPORT_TXT = "relatorio_comparacao_duplicatas_agregado.txt"

DUP_EXATAS_CSV = "duplicatas_candidatas_chave_exata.csv"
DUP_PRECO_ARREDONDADO_CSV = "duplicatas_candidatas_preco_arredondado.csv"
DUP_APROXIMADAS_CSV = "duplicatas_candidatas_aproximadas.csv"


def strip_accents(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def normalize_key_text(value):
    if pd.isna(value):
        return ""

    value = strip_accents(value)
    value = value.lower().strip()
    value = " ".join(value.split())
    return value


def read_dataset(path, origem):
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    df["_origem"] = origem
    df["_row_id"] = range(len(df))

    return df


def prepare_for_comparison(df):
    df = df.copy()

    required_columns = [
        "preco_venda",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
        "cidade",
        "bairro",
        "tipo_imovel",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    for col in ["preco_venda", "area_m2", "quartos", "banheiros", "vagas"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cidade_key"] = df["cidade"].apply(normalize_key_text)
    df["bairro_key"] = df["bairro"].apply(normalize_key_text)
    df["tipo_key"] = df["tipo_imovel"].apply(normalize_key_text)

    df["preco_arredondado"] = (
        pd.to_numeric(df["preco_venda"], errors="coerce")
        .round(-3)
    )

    return df


def add_comparison_columns(base, novo):
    """
    Cria cópias auxiliares para que preço e área estejam disponíveis
    depois do merge, mesmo quando forem usados como chave de junção.
    """
    base = base.copy()
    novo = novo.copy()

    base["preco_venda_base_cmp"] = base["preco_venda"]
    base["area_m2_base_cmp"] = base["area_m2"]
    base["cidade_base_cmp"] = base["cidade"]
    base["bairro_base_cmp"] = base["bairro"]
    base["tipo_imovel_base_cmp"] = base["tipo_imovel"]

    novo["preco_venda_novo_cmp"] = novo["preco_venda"]
    novo["area_m2_novo_cmp"] = novo["area_m2"]
    novo["cidade_novo_cmp"] = novo["cidade"]
    novo["bairro_novo_cmp"] = novo["bairro"]
    novo["tipo_imovel_novo_cmp"] = novo["tipo_imovel"]

    return base, novo


def add_pair_columns(pairs):
    pairs = pairs.copy()

    pairs["dif_preco_abs"] = (
        pairs["preco_venda_base_cmp"] - pairs["preco_venda_novo_cmp"]
    ).abs()

    pairs["dif_area_abs"] = (
        pairs["area_m2_base_cmp"] - pairs["area_m2_novo_cmp"]
    ).abs()

    max_preco = pairs[["preco_venda_base_cmp", "preco_venda_novo_cmp"]].max(axis=1)
    max_area = pairs[["area_m2_base_cmp", "area_m2_novo_cmp"]].max(axis=1)

    pairs["dif_preco_pct"] = (
        pairs["dif_preco_abs"] / max_preco * 100
    ).round(2)

    pairs["dif_area_pct"] = (
        pairs["dif_area_abs"] / max_area * 100
    ).round(2)

    return pairs


def find_exact_duplicates(base, novo):
    key = [
        "cidade_key",
        "bairro_key",
        "tipo_key",
        "preco_venda",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
    ]

    base_cmp, novo_cmp = add_comparison_columns(base, novo)

    pairs = base_cmp.merge(
        novo_cmp,
        on=key,
        suffixes=("_base", "_novo"),
        how="inner",
    )

    return add_pair_columns(pairs)


def find_rounded_price_duplicates(base, novo):
    key = [
        "cidade_key",
        "bairro_key",
        "tipo_key",
        "preco_arredondado",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
    ]

    base_cmp, novo_cmp = add_comparison_columns(base, novo)

    pairs = base_cmp.merge(
        novo_cmp,
        on=key,
        suffixes=("_base", "_novo"),
        how="inner",
    )

    return add_pair_columns(pairs)


def find_approximate_duplicates(base, novo):
    """
    Busca candidatos por bloco estrutural, sem exigir preço idêntico.

    Critério:
    - mesma cidade;
    - mesmo bairro normalizado;
    - mesmo tipo;
    - mesmos quartos, banheiros e vagas;
    - área com diferença <= 5 m² ou <= 5%;
    - preço com diferença <= R$ 50.000 ou <= 5%.
    """
    block_key = [
        "cidade_key",
        "bairro_key",
        "tipo_key",
        "quartos",
        "banheiros",
        "vagas",
    ]

    base_cmp, novo_cmp = add_comparison_columns(base, novo)

    pairs = base_cmp.merge(
        novo_cmp,
        on=block_key,
        suffixes=("_base", "_novo"),
        how="inner",
    )

    pairs = add_pair_columns(pairs)

    area_ok = (
        (pairs["dif_area_abs"] <= 5)
        | (pairs["dif_area_pct"] <= 5)
    )

    preco_ok = (
        (pairs["dif_preco_abs"] <= 50_000)
        | (pairs["dif_preco_pct"] <= 5)
    )

    return pairs[area_ok & preco_ok].copy()


def select_output_columns(pairs):
    wanted = [
        "_row_id_base",
        "_row_id_novo",

        "cidade_base_cmp",
        "cidade_novo_cmp",
        "bairro_base_cmp",
        "bairro_novo_cmp",
        "tipo_imovel_base_cmp",
        "tipo_imovel_novo_cmp",

        "preco_venda_base_cmp",
        "preco_venda_novo_cmp",
        "area_m2_base_cmp",
        "area_m2_novo_cmp",

        "quartos",
        "banheiros",
        "vagas",

        "dif_preco_abs",
        "dif_preco_pct",
        "dif_area_abs",
        "dif_area_pct",
    ]

    existing = [col for col in wanted if col in pairs.columns]

    out = pairs[existing].copy()

    rename_map = {
        "cidade_base_cmp": "cidade_base",
        "cidade_novo_cmp": "cidade_novo",
        "bairro_base_cmp": "bairro_base",
        "bairro_novo_cmp": "bairro_novo",
        "tipo_imovel_base_cmp": "tipo_imovel_base",
        "tipo_imovel_novo_cmp": "tipo_imovel_novo",
        "preco_venda_base_cmp": "preco_venda_base",
        "preco_venda_novo_cmp": "preco_venda_novo",
        "area_m2_base_cmp": "area_m2_base",
        "area_m2_novo_cmp": "area_m2_novo",
    }

    return out.rename(columns=rename_map)


def write_report(base, novo, exatas, arredondadas, aproximadas):
    lines = []

    lines.append("RELATÓRIO DE COMPARAÇÃO DE DUPLICATAS")
    lines.append("")
    lines.append(f"Dataset base: {DATASET_BASE}")
    lines.append(f"Dataset novo: {DATASET_NOVO}")
    lines.append("")
    lines.append(f"Registros dataset base: {len(base)}")
    lines.append(f"Registros dataset novo: {len(novo)}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("RESUMO")
    lines.append("=" * 90)
    lines.append(f"Pares por chave exata: {len(exatas)}")
    lines.append(f"Pares por chave com preço arredondado: {len(arredondadas)}")
    lines.append(f"Pares por aproximação estrutural: {len(aproximadas)}")
    lines.append("")

    if len(novo) > 0:
        linhas_novo_exatas = (
            exatas["_row_id_novo"].nunique()
            if "_row_id_novo" in exatas.columns
            else 0
        )

        linhas_novo_arred = (
            arredondadas["_row_id_novo"].nunique()
            if "_row_id_novo" in arredondadas.columns
            else 0
        )

        linhas_novo_aprox = (
            aproximadas["_row_id_novo"].nunique()
            if "_row_id_novo" in aproximadas.columns
            else 0
        )

        lines.append("Registros únicos do dataset novo envolvidos:")
        lines.append(
            f"  chave exata: {linhas_novo_exatas} "
            f"({round(linhas_novo_exatas / len(novo) * 100, 2)}%)"
        )
        lines.append(
            f"  preço arredondado: {linhas_novo_arred} "
            f"({round(linhas_novo_arred / len(novo) * 100, 2)}%)"
        )
        lines.append(
            f"  aproximação estrutural: {linhas_novo_aprox} "
            f"({round(linhas_novo_aprox / len(novo) * 100, 2)}%)"
        )
        lines.append("")

    lines.append("=" * 90)
    lines.append("INTERPRETAÇÃO")
    lines.append("=" * 90)
    lines.append(
        "A chave exata é o indício mais forte de duplicata, mas ainda não prova "
        "duplicação absoluta, porque os IDs não correspondem entre fontes."
    )
    lines.append(
        "A chave com preço arredondado amplia a busca e deve ser lida como risco "
        "moderado."
    )
    lines.append(
        "A aproximação estrutural é uma triagem ampla: identifica imóveis muito "
        "parecidos, mas pode capturar unidades diferentes do mesmo prédio, "
        "condomínio ou empreendimento."
    )
    lines.append("")

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")


def main():
    base = read_dataset(DATASET_BASE, "base")
    novo = read_dataset(DATASET_NOVO, "novo")

    base = prepare_for_comparison(base)
    novo = prepare_for_comparison(novo)

    exatas = find_exact_duplicates(base, novo)
    arredondadas = find_rounded_price_duplicates(base, novo)
    aproximadas = find_approximate_duplicates(base, novo)

    select_output_columns(exatas).to_csv(DUP_EXATAS_CSV, index=False)
    select_output_columns(arredondadas).to_csv(DUP_PRECO_ARREDONDADO_CSV, index=False)
    select_output_columns(aproximadas).to_csv(DUP_APROXIMADAS_CSV, index=False)

    write_report(base, novo, exatas, arredondadas, aproximadas)

    print("Comparação concluída.")
    print(f"Pares por chave exata: {len(exatas)}")
    print(f"Pares por preço arredondado: {len(arredondadas)}")
    print(f"Pares aproximados: {len(aproximadas)}")
    print(f"Relatório: {REPORT_TXT}")
    print(f"Candidatos exatos: {DUP_EXATAS_CSV}")
    print(f"Candidatos preço arredondado: {DUP_PRECO_ARREDONDADO_CSV}")
    print(f"Candidatos aproximados: {DUP_APROXIMADAS_CSV}")


if __name__ == "__main__":
    main()