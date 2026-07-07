import pandas as pd
from pathlib import Path


RAW_CSV = "data_imoveis_outras_cidades.csv"
MODEL_CSV = "dataset_modelagem_outras_cidades.csv"
AUDIT_CSV = "dataset_auditoria_outras_cidades.csv"
REMOVED_CSV = "dataset_removidos_outras_cidades.csv"
REPORT_TXT = "relatorio_dataset_outras_cidades.txt"


CIDADES_ESPERADAS = [
    "Fundão",
    "Guarapari",
    "Serra",
    "Viana",
    "Cariacica"
]


TARGET_COLUMN = "preco_venda"


# Colunas candidatas para modelagem.
# Observação: preco_m2 NÃO entra aqui porque usa o próprio alvo preco_venda.
MODEL_COLUMNS = [
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
    "area_suspeita",
    "preco_suspeito",
    "idade_anuncio_dias",
]


# Colunas úteis para EDA e auditoria, mas que não devem ser usadas como entrada
# em um modelo que prevê preco_venda.
EDA_ONLY_COLUMNS = [
    "preco_m2",
    "preco_m2_suspeito",
]


AUDIT_COLUMNS = [
    "id_anuncio",
    "titulo",
    "url",
    "categoria_olx",
    "tipo_detalhado",
    "tipo_url",
    "data_anuncio_olx",
    "data_coleta",
    "preco_venda",
    "area_m2",
    "preco_m2",
    "cidade",
    "bairro",
    "tipo_imovel",
    "subtipo_imovel",
    "quartos",
    "banheiros",
    "vagas",
    "qtd_imagens",
    "idade_anuncio_dias",
    "preco_m2_suspeito",
    "preco_m2_suspeito_original",
    "area_suspeita",
    "preco_suspeito",
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
    "area_suspeita",
    "preco_suspeito",
    "preco_m2_suspeito",
]


# Critérios mais rígidos para modelagem inicial.
MIN_PRECO = 50_000
MAX_PRECO = 20_000_000

MIN_AREA = 10
MAX_AREA = 1000

MIN_PRECO_M2_CASA = 700
MAX_PRECO_M2_CASA = 30_000

MIN_PRECO_M2_APARTAMENTO = 1500
MAX_PRECO_M2_APARTAMENTO = 40_000

# Se for None deixa anuncios velhos passarem
MAX_IDADE_ANUNCIO_DIAS = None


def read_raw_dataset(path):
    """Lê o CSV bruto e garante que as colunas esperadas existam."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    expected_columns = set(MODEL_COLUMNS + EDA_ONLY_COLUMNS + AUDIT_COLUMNS)

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    return df


def normalize_types(df):
    """Converte colunas para tipos úteis sem transformar vazios em zero."""
    df = df.copy()

    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    for column in FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(2)

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)

    for column in [
        "tipo_imovel",
        "subtipo_imovel",
        "cidade",
        "bairro",
        "titulo",
        "url",
        "categoria_olx",
        "tipo_detalhado",
        "tipo_url",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


def recompute_suspicious_flags(df):
    """
    Recalcula flags de suspeita com critérios mais rígidos.

    Isso é importante porque o main.py pode ter usado limites antigos.
    """
    df = df.copy()

    if "preco_m2_suspeito" in df.columns:
        df["preco_m2_suspeito_original"] = df["preco_m2_suspeito"]
    else:
        df["preco_m2_suspeito_original"] = pd.NA

    df["preco_m2_suspeito"] = False

    mask_casa = df["tipo_imovel"] == "casa"
    mask_apartamento = df["tipo_imovel"] == "apartamento"

    df.loc[mask_casa, "preco_m2_suspeito"] = (
        df.loc[mask_casa, "preco_m2"].notna()
        & (
            (df.loc[mask_casa, "preco_m2"] < MIN_PRECO_M2_CASA)
            | (df.loc[mask_casa, "preco_m2"] > MAX_PRECO_M2_CASA)
        )
    )

    df.loc[mask_apartamento, "preco_m2_suspeito"] = (
        df.loc[mask_apartamento, "preco_m2"].notna()
        & (
            (df.loc[mask_apartamento, "preco_m2"] <= MIN_PRECO_M2_APARTAMENTO)
            | (df.loc[mask_apartamento, "preco_m2"] > MAX_PRECO_M2_APARTAMENTO)
        )
    )

    df["preco_suspeito"] = (
        df["preco_venda"].notna()
        & (
            (df["preco_venda"] < MIN_PRECO)
            | (df["preco_venda"] > MAX_PRECO)
        )
    )

    df["area_suspeita"] = (
        df["area_m2"].notna()
        & (
            (df["area_m2"] < MIN_AREA)
            | (df["area_m2"] > MAX_AREA)
        )
    )

    return df


def add_removal_reason(df, mask, reason):
    """
    Marca motivo de remoção para auditoria.

    mask=True significa que a linha deve ser removida naquela etapa.
    """
    df.loc[mask & df["motivo_remocao"].isna(), "motivo_remocao"] = reason
    return df


def build_removed_dataset(raw_df):
    """Cria dataset com motivos de remoção, sem alterar o bruto original."""
    df = raw_df.copy()
    df["motivo_remocao"] = pd.NA

    df = add_removal_reason(
        df,
        df[TARGET_COLUMN].isna(),
        "sem_preco_venda",
    )

    df = add_removal_reason(
        df,
        df["area_m2"].isna(),
        "sem_area_m2",
    )

    df = add_removal_reason(
        df,
        df["tipo_imovel"].isna(),
        "sem_tipo_imovel",
    )

    df = add_removal_reason(
        df,
        df["cidade"].isna(),
        "sem_cidade",
    )

    df = add_removal_reason(
        df,
        df["bairro"].isna(),
        "sem_bairro",
    )

    df = add_removal_reason(
        df,
        df["preco_venda"].notna() & (df["preco_venda"] <= 0),
        "preco_venda_invalido",
    )

    df = add_removal_reason(
        df,
        df["area_m2"].notna() & (df["area_m2"] <= 0),
        "area_m2_invalida",
    )

    df = add_removal_reason(
        df,
        df["quartos"].isna() | (df["quartos"] < 1),
        "quartos_ausente_ou_zero",
    )

    df = add_removal_reason(
        df,
        df["banheiros"].isna() | (df["banheiros"] < 1),
        "banheiros_ausente_ou_zero",
    )

    df = add_removal_reason(
        df,
        ~df["cidade"].isin(CIDADES_ESPERADAS),
        "cidade_fora_do_grupo",
    )

    df = add_removal_reason(
        df,
        df["preco_suspeito"],
        "preco_suspeito",
    )

    df = add_removal_reason(
        df,
        df["area_suspeita"],
        "area_suspeita",
    )

    df = add_removal_reason(
        df,
        df["preco_m2_suspeito"],
        "preco_m2_suspeito",
    )

    if MAX_IDADE_ANUNCIO_DIAS is not None:
        df = add_removal_reason(
            df,
            df["idade_anuncio_dias"].notna()
            & (df["idade_anuncio_dias"] > MAX_IDADE_ANUNCIO_DIAS),
            "anuncio_antigo",
        )

    return df[df["motivo_remocao"].notna()].copy()


def apply_model_filters(df):
    """Aplica filtros rígidos para construir o dataset de modelagem."""
    df = df.copy()

    df = df.dropna(
        subset=[
            TARGET_COLUMN,
            "area_m2",
            "tipo_imovel",
            "cidade",
            "bairro",
        ]
    )

    df = df[df["preco_venda"] > 0]
    df = df[df["area_m2"] > 0]

    # Para o primeiro modelo, zeros em quartos/banheiros são tratados como ruído.
    df = df[df["quartos"].notna()]
    df = df[df["banheiros"].notna()]
    df = df[df["quartos"] >= 1]
    df = df[df["banheiros"] >= 1]

    # Mantém só as cidades esperadas para o grupo analisado.
    df = df[df["cidade"].isin(CIDADES_ESPERADAS)]

    # Remove outliers/suspeitos do dataset inicial de modelagem.
    df = df[df["preco_suspeito"] == False]
    df = df[df["area_suspeita"] == False]
    df = df[df["preco_m2_suspeito"] == False]

    if MAX_IDADE_ANUNCIO_DIAS is not None:
        df = df[
            df["idade_anuncio_dias"].isna()
            | (df["idade_anuncio_dias"] <= MAX_IDADE_ANUNCIO_DIAS)
        ]

    return df


def deduplicate_for_modeling(df):
    """
    Deduplicação conservadora.

    Remove apenas duplicatas fortes:
    - mesmo id_anuncio;
    - mesma url.

    Duplicatas aproximadas não são removidas nesta etapa, pois podem representar
    anúncios reais de unidades parecidas, especialmente em empreendimentos.
    Elas devem ser apenas auditadas no relatório.
    """
    df = df.copy()

    stats = {
        "entrada_deduplicacao": len(df),
        "removidos_por_id_anuncio": 0,
        "removidos_por_url": 0,
        "possiveis_duplicatas_chave_exata": 0,
        "possiveis_duplicatas_chave_preco_arredondado": 0,
        "total_removido_por_deduplicacao": 0,
        "saida_deduplicacao": 0,
    }

    before = len(df)

    if "id_anuncio" in df.columns:
        df = df.drop_duplicates(subset=["id_anuncio"], keep="last")

    stats["removidos_por_id_anuncio"] = before - len(df)

    before = len(df)

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")

    stats["removidos_por_url"] = before - len(df)

    approximate_key = [
        "cidade",
        "bairro",
        "tipo_imovel",
        "subtipo_imovel",
        "preco_venda",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
    ]

    existing_key = [column for column in approximate_key if column in df.columns]

    if existing_key:
        stats["possiveis_duplicatas_chave_exata"] = int(
            df.duplicated(subset=existing_key, keep=False).sum()
        )

    if "preco_venda" in df.columns:
        df["preco_venda_arredondado"] = (
            pd.to_numeric(df["preco_venda"], errors="coerce")
            .round(-3)
            .astype("Int64")
        )

        approximate_rounded_key = [
            "cidade",
            "bairro",
            "tipo_imovel",
            "subtipo_imovel",
            "preco_venda_arredondado",
            "area_m2",
            "quartos",
            "banheiros",
            "vagas",
        ]

        existing_rounded_key = [
            column for column in approximate_rounded_key if column in df.columns
        ]

        if existing_rounded_key:
            stats["possiveis_duplicatas_chave_preco_arredondado"] = int(
                df.duplicated(subset=existing_rounded_key, keep=False).sum()
            )

        df = df.drop(columns=["preco_venda_arredondado"])

    stats["saida_deduplicacao"] = len(df)
    stats["total_removido_por_deduplicacao"] = (
        stats["removidos_por_id_anuncio"] + stats["removidos_por_url"]
    )

    return df, stats

def build_model_dataset(raw_df):
    """
    Cria o dataset que será usado em EDA e treino.

    Também retorna estatísticas de deduplicação para o relatório.
    """
    df = raw_df.copy()
    df = apply_model_filters(df)

    before_dedup = len(df)

    df, dedup_stats = deduplicate_for_modeling(df)

    dedup_stats["entrada_modelagem_antes_deduplicacao"] = before_dedup
    dedup_stats["total_removido_por_deduplicacao"] = before_dedup - len(df)

    columns_to_export = MODEL_COLUMNS + EDA_ONLY_COLUMNS
    columns_to_export = [column for column in columns_to_export if column in df.columns]

    return df[columns_to_export].copy(), dedup_stats

def build_audit_dataset(df):
    """Cria um CSV auxiliar para revisar outliers e rastrear anúncios."""
    columns_to_export = [column for column in AUDIT_COLUMNS if column in df.columns]
    return df[columns_to_export].copy()


def append_series(lines, title, series):
    lines.append(title)
    lines.append(str(series))
    lines.append("")


def append_dataframe(lines, title, df):
    lines.append(title)

    if df.empty:
        lines.append("(vazio)")
    else:
        lines.append(df.to_string(index=False))

    lines.append("")


def percentage(part, total):
    if total == 0:
        return 0

    return round((part / total) * 100, 2)


def write_quality_report(raw_df, model_df, removed_df, dedup_stats=None):
    """Gera relatório de cobertura, distribuição, remoções e possíveis problemas."""
    lines = []

    lines.append("RELATÓRIO DO DATASET\n")
    lines.append(f"Arquivo bruto: {RAW_CSV}")
    lines.append(f"Cidades esperadas: {', '.join(CIDADES_ESPERADAS)}")
    lines.append(f"Registros brutos: {len(raw_df)}")
    lines.append(f"Registros no dataset de modelagem: {len(model_df)}")
    lines.append(f"Registros removidos: {len(raw_df) - len(model_df)}")
    lines.append(
        f"Taxa de retenção: {percentage(len(model_df), len(raw_df))}%\n"
    )

    if not removed_df.empty and "motivo_remocao" in removed_df.columns:
        append_series(
            lines,
            "Remoções por primeiro motivo identificado:",
            removed_df["motivo_remocao"].value_counts(dropna=False),
        )

    if dedup_stats:
        lines.append("Deduplicação:")
        lines.append(
            f"Entrada na deduplicação: "
            f"{dedup_stats.get('entrada_deduplicacao', 0)}"
        )
        lines.append(
            f"Removidos por id_anuncio duplicado: "
            f"{dedup_stats.get('removidos_por_id_anuncio', 0)}"
        )
        lines.append(
            f"Removidos por url duplicada: "
            f"{dedup_stats.get('removidos_por_url', 0)}"
        )
        lines.append(
            f"Total removido por deduplicação forte: "
            f"{dedup_stats.get('total_removido_por_deduplicacao', 0)}"
        )
        lines.append(
            f"Possíveis duplicatas por chave exata, não removidas: "
            f"{dedup_stats.get('possiveis_duplicatas_chave_exata', 0)}"
        )
        lines.append(
            f"Possíveis duplicatas por chave com preço arredondado, não removidas: "
            f"{dedup_stats.get('possiveis_duplicatas_chave_preco_arredondado', 0)}"
        )
        lines.append(
            f"Saída da deduplicação: "
            f"{dedup_stats.get('saida_deduplicacao', 0)}"
        )
        lines.append("")
    
    sem_area = removed_df[removed_df["motivo_remocao"] == "sem_area_m2"]

    if not sem_area.empty and {"cidade", "tipo_imovel"}.issubset(sem_area.columns):
        append_series(
            lines,
            "Registros removidos por ausência de área por cidade e tipo:",
            sem_area.groupby(["cidade", "tipo_imovel"])
            .size()
            .sort_values(ascending=False),
        )

    if "cidade" in raw_df.columns:
        cidades_fora = raw_df[~raw_df["cidade"].isin(CIDADES_ESPERADAS)]
        lines.append(
            f"Registros brutos fora das cidades esperadas: {len(cidades_fora)}"
        )
        if not cidades_fora.empty:
            lines.append(str(cidades_fora["cidade"].value_counts(dropna=False)))
        lines.append("")

    if "cidade" in model_df.columns:
        append_series(
            lines,
            "Cobertura por cidade:",
            model_df["cidade"].value_counts(dropna=False),
        )

    if "tipo_imovel" in model_df.columns:
        append_series(
            lines,
            "Cobertura por tipo de imóvel:",
            model_df["tipo_imovel"].value_counts(dropna=False),
        )

    if {"cidade", "tipo_imovel"}.issubset(model_df.columns):
        append_series(
            lines,
            "Cobertura por cidade e tipo de imóvel:",
            model_df.groupby(["cidade", "tipo_imovel"])
            .size()
            .sort_values(ascending=False),
        )

    if "bairro" in model_df.columns:
        bairro_counts = model_df["bairro"].value_counts(dropna=False)

        append_series(
            lines,
            "Top 30 bairros por quantidade de registros:",
            bairro_counts.head(30),
        )

        top_10_bairros = bairro_counts.head(10).sum()
        lines.append(
            f"Registros concentrados nos 10 bairros mais frequentes: "
            f"{top_10_bairros} ({percentage(top_10_bairros, len(model_df))}%)"
        )
        lines.append("")

    if {"cidade", "bairro"}.issubset(model_df.columns):
        append_series(
            lines,
            "Top 30 pares cidade/bairro por quantidade de registros:",
            model_df.groupby(["cidade", "bairro"])
            .size()
            .sort_values(ascending=False)
            .head(30),
        )

    if {"cidade", "tipo_imovel", "bairro"}.issubset(model_df.columns):
        append_series(
            lines,
            "Top 30 pares cidade/tipo/bairro por quantidade de registros:",
            model_df.groupby(["cidade", "tipo_imovel", "bairro"])
            .size()
            .sort_values(ascending=False)
            .head(30),
        )

    if {
    "preco_m2_suspeito",
    "preco_m2_suspeito_original",
    }.issubset(raw_df.columns):
        comparison = (
            raw_df.groupby(["preco_m2_suspeito_original", "preco_m2_suspeito"])
            .size()
            .sort_values(ascending=False)
        )

        append_series(
            lines,
            "Comparação entre preço/m² suspeito original e recalculado:",
            comparison,
        )

    lines.append("Valores ausentes por coluna no dataset de modelagem:")
    lines.append(str(model_df.isna().sum().sort_values(ascending=False)))
    lines.append("")

    numeric_summary_columns = [
        "preco_venda",
        "area_m2",
        "preco_m2",
        "quartos",
        "banheiros",
        "vagas",
        "qtd_imagens",
        "idade_anuncio_dias",
    ]

    existing_numeric_columns = [
        column for column in numeric_summary_columns if column in model_df.columns
    ]

    lines.append("Resumo numérico:")
    if existing_numeric_columns:
        lines.append(str(model_df[existing_numeric_columns].describe()))
    lines.append("")

    if "idade_anuncio_dias" in raw_df.columns:
        idade = pd.to_numeric(raw_df["idade_anuncio_dias"], errors="coerce")

        lines.append("Idade dos anúncios no bruto:")
        lines.append(f"> 90 dias: {int((idade > 90).sum())}")
        lines.append(f"> 180 dias: {int((idade > 180).sum())}")
        lines.append(f"> 365 dias: {int((idade > 365).sum())}")
        lines.append("")

    if "preco_m2" in model_df.columns:
        audit_columns = [
            "cidade",
            "bairro",
            "tipo_imovel",
            "subtipo_imovel",
            "preco_venda",
            "area_m2",
            "preco_m2",
            "quartos",
            "banheiros",
            "vagas",
        ]
        audit_columns = [column for column in audit_columns if column in model_df.columns]

        lowest = model_df.sort_values("preco_m2", ascending=True).head(20)[audit_columns]
        highest = model_df.sort_values("preco_m2", ascending=False).head(20)[audit_columns]

        append_dataframe(lines, "20 menores preços por m² no dataset filtrado:", lowest)
        append_dataframe(lines, "20 maiores preços por m² no dataset filtrado:", highest)

    if "preco_m2" in raw_df.columns:
        raw_audit_columns = [
            "cidade",
            "bairro",
            "tipo_imovel",
            "subtipo_imovel",
            "preco_venda",
            "area_m2",
            "preco_m2",
            "quartos",
            "banheiros",
            "vagas",
            "url",
        ]

        raw_audit_columns = [
            column for column in raw_audit_columns if column in raw_df.columns
        ]

        raw_with_preco_m2 = raw_df[raw_df["preco_m2"].notna()].copy()

        append_dataframe(
            lines,
            "20 menores preços por m² no bruto:",
            raw_with_preco_m2.sort_values("preco_m2", ascending=True)
            .head(20)[raw_audit_columns],
        )

        append_dataframe(
            lines,
            "20 maiores preços por m² no bruto:",
            raw_with_preco_m2.sort_values("preco_m2", ascending=False)
            .head(20)[raw_audit_columns],
        )

    if not removed_df.empty and "preco_m2" in removed_df.columns:
        removed_audit_columns = [
            "motivo_remocao",
            "cidade",
            "bairro",
            "tipo_imovel",
            "subtipo_imovel",
            "preco_venda",
            "area_m2",
            "preco_m2",
            "quartos",
            "banheiros",
            "vagas",
            "url",
        ]

        removed_audit_columns = [
            column for column in removed_audit_columns if column in removed_df.columns
        ]

        removed_price_m2 = removed_df[
            removed_df["motivo_remocao"] == "preco_m2_suspeito"
        ]

        append_dataframe(
            lines,
            "Amostra de registros removidos por preço/m² suspeito:",
            removed_price_m2.head(30)[removed_audit_columns],
        )

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_df = read_raw_dataset(RAW_CSV)
    raw_df = normalize_types(raw_df)
    raw_df = recompute_suspicious_flags(raw_df)

    removed_df = build_removed_dataset(raw_df)
    model_df, dedup_stats = build_model_dataset(raw_df)
    audit_df = build_audit_dataset(raw_df)

    model_df.to_csv(MODEL_CSV, index=False)
    audit_df.to_csv(AUDIT_CSV, index=False)
    removed_df.to_csv(REMOVED_CSV, index=False)

    write_quality_report(raw_df, model_df, removed_df, dedup_stats)

    print("Preparação concluída.")
    print(f"Entrada bruta: {RAW_CSV} ({len(raw_df)} registros)")
    print(f"Dataset de modelagem: {MODEL_CSV} ({len(model_df)} registros)")
    print(f"Dataset de auditoria: {AUDIT_CSV} ({len(audit_df)} registros)")
    print(f"Registros removidos: {REMOVED_CSV} ({len(removed_df)} registros)")
    print(f"Relatório: {REPORT_TXT}")
    print(
        "Registros removidos por deduplicação forte: "
        f"{dedup_stats['total_removido_por_deduplicacao']}"
    )


if __name__ == "__main__":
    main()