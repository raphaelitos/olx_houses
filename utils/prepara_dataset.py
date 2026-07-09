import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO DA EXECUÇÃO
# ============================================================

# Para Vitória/Vila Velha, use:
RAW_CSV = "data_imoveis_vitoria_vila_velha.csv"
PREFIX = ""
CIDADES_ESPERADAS = ["Vitória", "Vila Velha"]

# Para outras cidades, use:
'''
RAW_CSV = "data_imoveis_outras_cidades.csv"
PREFIX = "_outras_cidades"
CIDADES_ESPERADAS = [
    "Fundão",
    "Guarapari",
    "Serra",
    "Viana",
    "Cariacica",
]
'''

MODEL_CSV = f"dataset_modelagem{PREFIX}.csv"
AUDIT_CSV = f"dataset_auditoria{PREFIX}.csv"

REMOVED_GENERAL_CSV = f"dataset_removidos_geral{PREFIX}.csv"
REMOVED_SEM_AREA_CSV = f"dataset_removidos_sem_area{PREFIX}.csv"
REMOVED_PRECO_M2_CSV = f"dataset_removidos_preco_m2_suspeito{PREFIX}.csv"

REPORT_TXT = f"relatorio_dataset{PREFIX}.txt"

TARGET_COLUMN = "preco_venda"


# ============================================================
# LIMITES DE FILTRAGEM
# ============================================================

MIN_PRECO = 40_000
MAX_PRECO = 20_000_000

MIN_AREA = 10
MAX_AREA = 1000

MIN_PRECO_M2_CASA = 700
MAX_PRECO_M2_CASA = 30_000

MIN_PRECO_M2_APARTAMENTO = 1500
MAX_PRECO_M2_APARTAMENTO = 40_000

MAX_IDADE_ANUNCIO_DIAS = None


# ============================================================
# COLUNAS
# ============================================================

RELEVANT_BOOLEAN_COLUMNS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

IGNORED_BOOLEAN_COLUMNS = [
    "anunciante_profissional",
    "tem_porteira_fechada",
    "tem_sol_manha",
    "tem_cobertura",
    "tem_frente_mar",
    "tem_gourmet",
    "tem_garden",
]

SUSPICION_COLUMNS = [
    "area_suspeita",
    "preco_suspeito",
    "preco_m2_suspeito",
    "preco_m2_suspeito_original",
]

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
    "qtd_imagens",
    "idade_anuncio_dias",
    *RELEVANT_BOOLEAN_COLUMNS,
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
    *RELEVANT_BOOLEAN_COLUMNS,
    *SUSPICION_COLUMNS,
]

REMOVED_COLUMNS = [
    "motivo_remocao",
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
    *RELEVANT_BOOLEAN_COLUMNS,
    *SUSPICION_COLUMNS,
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

TEXT_COLUMNS = [
    "titulo",
    "url",
    "categoria_olx",
    "tipo_detalhado",
    "tipo_url",
    "tipo_imovel",
    "subtipo_imovel",
    "cidade",
    "bairro",
]

BOOLEAN_COLUMNS = [
    *RELEVANT_BOOLEAN_COLUMNS,
    *IGNORED_BOOLEAN_COLUMNS,
    "area_suspeita",
    "preco_suspeito",
    "preco_m2_suspeito",
]


# ============================================================
# LEITURA E NORMALIZAÇÃO
# ============================================================

def read_raw_dataset(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    expected_columns = set(
        MODEL_COLUMNS
        + AUDIT_COLUMNS
        + REMOVED_COLUMNS
        + INTEGER_COLUMNS
        + FLOAT_COLUMNS
        + TEXT_COLUMNS
        + BOOLEAN_COLUMNS
    )

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    return df


def normalize_types(df):
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

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


def normalize_city_names(df):
    df = df.copy()

    if "cidade" not in df.columns:
        return df

    df["cidade"] = df["cidade"].replace({
        "Praia Grande": "Fundão",
        "Praia Grande, Fundão": "Fundão",
    })

    return df


def normalize_subtypes(df):
    df = df.copy()

    if "subtipo_imovel" not in df.columns:
        return df

    df["subtipo_imovel"] = df["subtipo_imovel"].replace({
        "venda_apartamento_padrao": "apartamento_padrao",
    })

    return df


# ============================================================
# RECÁLCULO DE FLAGS DE AUDITORIA
# ============================================================

def recompute_suspicious_flags(df):
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


# ============================================================
# REMOÇÕES
# ============================================================

def add_removal_reason(df, mask, reason):
    df.loc[mask & df["motivo_remocao"].isna(), "motivo_remocao"] = reason
    return df


def build_removed_dataset(raw_df):
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
        df["cidade"].notna() & ~df["cidade"].isin(CIDADES_ESPERADAS),
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

    df = df[df["cidade"].isin(CIDADES_ESPERADAS)]

    df = df[df["preco_suspeito"] == False]
    df = df[df["area_suspeita"] == False]
    df = df[df["preco_m2_suspeito"] == False]

    if MAX_IDADE_ANUNCIO_DIAS is not None:
        df = df[
            df["idade_anuncio_dias"].isna()
            | (df["idade_anuncio_dias"] <= MAX_IDADE_ANUNCIO_DIAS)
        ]

    return df


def export_existing_columns(df, columns):
    columns_to_export = [column for column in columns if column in df.columns]
    return df[columns_to_export].copy()


def build_model_dataset(raw_df):
    df = apply_model_filters(raw_df)
    return export_existing_columns(df, MODEL_COLUMNS)


def build_audit_dataset(raw_df):
    return export_existing_columns(raw_df, AUDIT_COLUMNS)


def split_removed_datasets(removed_df):
    removed_sem_area = removed_df[
        removed_df["motivo_remocao"] == "sem_area_m2"
    ].copy()

    removed_preco_m2 = removed_df[
        removed_df["motivo_remocao"] == "preco_m2_suspeito"
    ].copy()

    removed_general = removed_df[
        ~removed_df["motivo_remocao"].isin([
            "sem_area_m2",
            "preco_m2_suspeito",
        ])
    ].copy()

    removed_sem_area = export_existing_columns(removed_sem_area, REMOVED_COLUMNS)
    removed_preco_m2 = export_existing_columns(removed_preco_m2, REMOVED_COLUMNS)
    removed_general = export_existing_columns(removed_general, REMOVED_COLUMNS)

    return removed_general, removed_sem_area, removed_preco_m2


# ============================================================
# RELATÓRIO
# ============================================================

def percentage(part, total):
    if total == 0:
        return 0.0

    return round((part / total) * 100, 2)


def append_section(lines, title):
    lines.append("")
    lines.append("=" * 90)
    lines.append(title)
    lines.append("=" * 90)


def append_subsection(lines, title):
    lines.append("")
    lines.append("-" * 90)
    lines.append(title)
    lines.append("-" * 90)


def append_series(lines, title, series):
    append_subsection(lines, title)

    if series is None or len(series) == 0:
        lines.append("(vazio)")
    else:
        lines.append(str(series))

    lines.append("")


def append_dataframe(lines, title, df, max_rows=None):
    append_subsection(lines, title)

    if df is None or df.empty:
        lines.append("(vazio)")
    else:
        if max_rows is not None:
            df = df.head(max_rows)
        lines.append(df.to_string(index=False))

    lines.append("")


def missing_report(df):
    missing = df.isna().sum().sort_values(ascending=False)
    pct_missing = (missing / len(df) * 100).round(2)

    return pd.DataFrame({
        "coluna": missing.index,
        "ausentes": missing.values,
        "pct_ausente": pct_missing.values,
    })


def describe_numeric(df, columns):
    existing = [column for column in columns if column in df.columns]

    if not existing:
        return pd.DataFrame()

    return df[existing].describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )


def count_approximate_duplicates(df):
    key = [
        "cidade",
        "bairro",
        "tipo_imovel",
        "subtipo_imovel",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
    ]

    existing = [column for column in key if column in df.columns]

    if not existing:
        return 0, 0

    rows = int(df.duplicated(subset=existing, keep=False).sum())

    groups = (
        df.groupby(existing, dropna=False)
        .size()
        .reset_index(name="qtd")
    )

    groups_count = int((groups["qtd"] > 1).sum())

    return rows, groups_count


def write_quality_report(
    raw_df,
    model_df,
    removed_df,
    removed_general,
    removed_sem_area,
    removed_preco_m2,
):
    lines = []

    lines.append("RELATÓRIO DO DATASET")
    lines.append("")
    lines.append(f"Arquivo bruto: {RAW_CSV}")
    lines.append(f"Cidades esperadas: {', '.join(CIDADES_ESPERADAS)}")
    lines.append(f"Registros brutos: {len(raw_df)}")
    lines.append(f"Registros no dataset de modelagem: {len(model_df)}")
    lines.append(f"Registros removidos totais: {len(removed_df)}")
    lines.append(f"Taxa de retenção: {percentage(len(model_df), len(raw_df))}%")
    lines.append("")
    lines.append(f"Dataset de modelagem: {MODEL_CSV}")
    lines.append(f"Dataset de auditoria: {AUDIT_CSV}")
    lines.append(f"Removidos gerais: {REMOVED_GENERAL_CSV}")
    lines.append(f"Removidos sem área: {REMOVED_SEM_AREA_CSV}")
    lines.append(f"Removidos por preço/m² suspeito: {REMOVED_PRECO_M2_CSV}")

    append_section(lines, "COLUNAS EXPORTADAS")

    lines.append("Colunas do dataset de modelagem:")
    lines.append(", ".join(model_df.columns))
    lines.append("")

    lines.append("Flags mantidas:")
    lines.append(", ".join(RELEVANT_BOOLEAN_COLUMNS))
    lines.append("")

    lines.append("Flags descartadas:")
    lines.append(", ".join(IGNORED_BOOLEAN_COLUMNS))
    lines.append("")

    lines.append("Colunas de auditoria removidas da modelagem:")
    lines.append("preco_m2, preco_m2_suspeito, preco_suspeito, area_suspeita")
    lines.append("")

    append_section(lines, "REMOÇÕES")

    append_series(
        lines,
        "Remoções por primeiro motivo identificado",
        removed_df["motivo_remocao"].value_counts(dropna=False)
        if "motivo_remocao" in removed_df.columns
        else pd.Series(dtype="int64"),
    )

    lines.append(f"Removidos gerais: {len(removed_general)}")
    lines.append(f"Removidos por ausência de área: {len(removed_sem_area)}")
    lines.append(f"Removidos por preço/m² suspeito: {len(removed_preco_m2)}")
    lines.append("")

    if not removed_sem_area.empty and {"cidade", "tipo_imovel"}.issubset(removed_sem_area.columns):
        append_series(
            lines,
            "Removidos por ausência de área por cidade e tipo",
            removed_sem_area.groupby(["cidade", "tipo_imovel"], dropna=False)
            .size()
            .sort_values(ascending=False),
        )

    if not removed_preco_m2.empty and {"cidade", "tipo_imovel"}.issubset(removed_preco_m2.columns):
        append_series(
            lines,
            "Removidos por preço/m² suspeito por cidade e tipo",
            removed_preco_m2.groupby(["cidade", "tipo_imovel"], dropna=False)
            .size()
            .sort_values(ascending=False),
        )

    append_dataframe(
        lines,
        "Amostra de removidos por ausência de área",
        removed_sem_area,
        max_rows=30,
    )

    append_dataframe(
        lines,
        "Amostra de removidos por preço/m² suspeito",
        removed_preco_m2,
        max_rows=30,
    )

    append_section(lines, "COBERTURA DO DATASET DE MODELAGEM")

    if "cidade" in model_df.columns:
        append_series(
            lines,
            "Cobertura por cidade",
            model_df["cidade"].value_counts(dropna=False),
        )

    if "tipo_imovel" in model_df.columns:
        append_series(
            lines,
            "Cobertura por tipo de imóvel",
            model_df["tipo_imovel"].value_counts(dropna=False),
        )

    if {"cidade", "tipo_imovel"}.issubset(model_df.columns):
        append_series(
            lines,
            "Cobertura por cidade e tipo de imóvel",
            model_df.groupby(["cidade", "tipo_imovel"], dropna=False)
            .size()
            .sort_values(ascending=False),
        )

    if "bairro" in model_df.columns:
        bairro_counts = model_df["bairro"].value_counts(dropna=False)

        append_series(
            lines,
            "Top 30 bairros por quantidade de registros",
            bairro_counts.head(30),
        )

        top_10 = bairro_counts.head(10).sum()

        lines.append(
            f"Registros concentrados nos 10 bairros mais frequentes: "
            f"{top_10} ({percentage(top_10, len(model_df))}%)"
        )
        lines.append("")

    append_section(lines, "VALORES AUSENTES")

    append_dataframe(
        lines,
        "Valores ausentes no dataset de modelagem",
        missing_report(model_df),
    )

    for col in ["quartos", "banheiros", "vagas"]:
        if col in model_df.columns:
            lines.append(f"{col} ausentes: {int(model_df[col].isna().sum())}")
            lines.append(f"{col} iguais a zero: {int((model_df[col] == 0).sum())}")
            lines.append("")

    append_section(lines, "PERFIL NUMÉRICO")

    append_dataframe(
        lines,
        "Resumo numérico do dataset de modelagem",
        describe_numeric(
            model_df,
            [
                "preco_venda",
                "area_m2",
                "quartos",
                "banheiros",
                "vagas",
                "qtd_imagens",
                "idade_anuncio_dias",
            ],
        ),
    )

    append_section(lines, "DUPLICATAS APROXIMADAS PARA AUDITORIA")

    approx_rows, approx_groups = count_approximate_duplicates(model_df)

    lines.append(
        "Critério usado: cidade, bairro, tipo_imovel, subtipo_imovel, "
        "area_m2, quartos, banheiros e vagas."
    )
    lines.append("")
    lines.append(f"Linhas em grupos aproximados: {approx_rows}")
    lines.append(f"Grupos aproximados: {approx_groups}")
    lines.append(f"Percentual de linhas em grupos aproximados: {percentage(approx_rows, len(model_df))}%")
    lines.append("")
    lines.append(
        "Observação: esses registros não são removidos. "
        "A informação serve apenas para avaliação posterior."
    )

    append_section(lines, "EXTREMOS PARA AUDITORIA")

    audit_source = raw_df.copy()

    if "preco_m2" in audit_source.columns:
        audit_source = audit_source[audit_source["preco_m2"].notna()].copy()

        audit_cols = [
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
            "titulo",
            "url",
        ]

        audit_cols = [col for col in audit_cols if col in audit_source.columns]

        append_dataframe(
            lines,
            "20 menores preços por m² no bruto",
            audit_source.sort_values("preco_m2", ascending=True).head(20)[audit_cols],
        )

        append_dataframe(
            lines,
            "20 maiores preços por m² no bruto",
            audit_source.sort_values("preco_m2", ascending=False).head(20)[audit_cols],
        )

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    raw_df = read_raw_dataset(RAW_CSV)
    raw_df = normalize_types(raw_df)
    raw_df = normalize_city_names(raw_df)
    raw_df = normalize_subtypes(raw_df)
    raw_df = recompute_suspicious_flags(raw_df)

    removed_df = build_removed_dataset(raw_df)
    model_df = build_model_dataset(raw_df)
    audit_df = build_audit_dataset(raw_df)

    removed_general, removed_sem_area, removed_preco_m2 = split_removed_datasets(removed_df)

    model_df.to_csv(MODEL_CSV, index=False)
    audit_df.to_csv(AUDIT_CSV, index=False)
    removed_general.to_csv(REMOVED_GENERAL_CSV, index=False)
    removed_sem_area.to_csv(REMOVED_SEM_AREA_CSV, index=False)
    removed_preco_m2.to_csv(REMOVED_PRECO_M2_CSV, index=False)

    write_quality_report(
        raw_df=raw_df,
        model_df=model_df,
        removed_df=removed_df,
        removed_general=removed_general,
        removed_sem_area=removed_sem_area,
        removed_preco_m2=removed_preco_m2,
    )

    print("Preparação concluída.")
    print(f"Entrada bruta: {RAW_CSV} ({len(raw_df)} registros)")
    print(f"Dataset de modelagem: {MODEL_CSV} ({len(model_df)} registros)")
    print(f"Dataset de auditoria: {AUDIT_CSV} ({len(audit_df)} registros)")
    print(f"Removidos gerais: {REMOVED_GENERAL_CSV} ({len(removed_general)} registros)")
    print(f"Removidos sem área: {REMOVED_SEM_AREA_CSV} ({len(removed_sem_area)} registros)")
    print(f"Removidos por preço/m² suspeito: {REMOVED_PRECO_M2_CSV} ({len(removed_preco_m2)} registros)")
    print(f"Relatório: {REPORT_TXT}")


if __name__ == "__main__":
    main()