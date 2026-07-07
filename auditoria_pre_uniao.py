import pandas as pd
from pathlib import Path


# =========================
# Arquivos de entrada
# =========================

VV_MODEL_CSV = "dataset_modelagem.csv"
VV_AUDIT_CSV = "dataset_auditoria.csv"
VV_REMOVED_CSV = "dataset_removidos.csv"

OUTRAS_MODEL_CSV = "dataset_modelagem_outras_cidades.csv"
OUTRAS_AUDIT_CSV = "dataset_auditoria_outras_cidades.csv"
OUTRAS_REMOVED_CSV = "dataset_removidos_outras_cidades.csv"

REPORT_TXT = "relatorio_auditoria_pre_uniao.txt"


# =========================
# Configurações de auditoria
# =========================

NUMERIC_COLUMNS = [
    "id_anuncio",
    "preco_venda",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "qtd_imagens",
    "idade_anuncio_dias",
    "preco_m2",
]

TEXT_COLUMNS = [
    "titulo",
    "url",
    "cidade",
    "bairro",
    "tipo_imovel",
    "subtipo_imovel",
    "categoria_olx",
    "tipo_detalhado",
    "tipo_url",
    "motivo_remocao",
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
    "preco_m2_suspeito_original",
]

# Chave conservadora: identifica registros muito parecidos, mas não remove.
APPROX_KEY_EXATA = [
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

# Chave com preço arredondado: útil para detectar imóveis de empreendimento ou republicações.
APPROX_KEY_ARREDONDADA_BASE = [
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

# Chave mais “relaxada”, sem vagas, porque vaga ausente pode impedir a detecção.
APPROX_KEY_SEM_VAGAS = [
    "cidade",
    "bairro",
    "tipo_imovel",
    "subtipo_imovel",
    "preco_venda",
    "area_m2",
    "quartos",
    "banheiros",
]

APPROX_KEY_ARREDONDADA_SEM_VAGAS = [
    "cidade",
    "bairro",
    "tipo_imovel",
    "subtipo_imovel",
    "preco_venda_arredondado",
    "area_m2",
    "quartos",
    "banheiros",
]


# =========================
# Utilitários
# =========================

def read_csv_if_exists(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    return normalize_dataframe(df)


def normalize_dataframe(df):
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)

    if "preco_venda" in df.columns:
        df["preco_venda_arredondado"] = (
            pd.to_numeric(df["preco_venda"], errors="coerce")
            .round(-3)
        )

    return df


def existing_columns(df, columns):
    return [column for column in columns if column in df.columns]


def append_line(lines, text=""):
    lines.append(str(text))


def append_section(lines, title):
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def append_subsection(lines, title):
    lines.append("")
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)


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


def pct(part, total):
    if total == 0:
        return 0.0

    return round((part / total) * 100, 2)


def count_duplicate_rows(df, subset):
    subset = existing_columns(df, subset)

    if not subset:
        return 0

    return int(df.duplicated(subset=subset, keep=False).sum())


def count_duplicate_groups(df, subset):
    subset = existing_columns(df, subset)

    if not subset:
        return 0

    grouped = df.groupby(subset, dropna=False).size()
    return int((grouped > 1).sum())


def duplicate_group_summary(df, subset, top_n=20):
    subset = existing_columns(df, subset)

    if not subset:
        return pd.DataFrame()

    grouped = (
        df.groupby(subset, dropna=False)
        .size()
        .reset_index(name="qtd_registros")
        .sort_values("qtd_registros", ascending=False)
    )

    return grouped[grouped["qtd_registros"] > 1].head(top_n)


def sample_duplicate_records(df, subset, top_n_groups=10, records_per_group=5):
    subset = existing_columns(df, subset)

    if not subset:
        return pd.DataFrame()

    grouped = (
        df.groupby(subset, dropna=False)
        .size()
        .reset_index(name="qtd_registros")
        .sort_values("qtd_registros", ascending=False)
    )

    duplicated_groups = grouped[grouped["qtd_registros"] > 1].head(top_n_groups)

    if duplicated_groups.empty:
        return pd.DataFrame()

    samples = []

    for _, row in duplicated_groups.iterrows():
        mask = pd.Series(True, index=df.index)

        for column in subset:
            value = row[column]

            if pd.isna(value):
                mask &= df[column].isna()
            else:
                mask &= df[column].eq(value)

        group_records = df[mask].copy().head(records_per_group)
        group_records["qtd_no_grupo_duplicado"] = row["qtd_registros"]
        samples.append(group_records)

    if not samples:
        return pd.DataFrame()

    result = pd.concat(samples, ignore_index=True)

    preferred_columns = [
        "qtd_no_grupo_duplicado",
        "id_anuncio",
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
        "idade_anuncio_dias",
        "titulo",
        "url",
        "motivo_remocao",
    ]

    preferred_columns = existing_columns(result, preferred_columns)
    return result[preferred_columns]


def make_cross_key(df, subset, key_name):
    subset = existing_columns(df, subset)
    temp = df.copy()

    if not subset:
        temp[key_name] = pd.NA
        return temp

    temp[key_name] = (
        temp[subset]
        .astype("string")
        .fillna("<NA>")
        .agg("||".join, axis=1)
    )

    return temp


def intersection_by_column(left_df, right_df, column):
    if column not in left_df.columns or column not in right_df.columns:
        return pd.DataFrame()

    left = left_df[left_df[column].notna()].copy()
    right = right_df[right_df[column].notna()].copy()

    if left.empty or right.empty:
        return pd.DataFrame()

    return left.merge(
        right,
        on=column,
        how="inner",
        suffixes=("_vv", "_outras"),
    )


def summarize_age(df):
    if "idade_anuncio_dias" not in df.columns:
        return pd.Series(dtype="float64")

    idade = pd.to_numeric(df["idade_anuncio_dias"], errors="coerce")

    summary = {
        "count": int(idade.notna().sum()),
        "ausentes": int(idade.isna().sum()),
        "min": idade.min(),
        "p25": idade.quantile(0.25),
        "mediana": idade.quantile(0.50),
        "media": idade.mean(),
        "p75": idade.quantile(0.75),
        "p90": idade.quantile(0.90),
        "p95": idade.quantile(0.95),
        "p99": idade.quantile(0.99),
        "max": idade.max(),
        "> 30 dias": int((idade > 30).sum()),
        "> 90 dias": int((idade > 90).sum()),
        "> 180 dias": int((idade > 180).sum()),
        "> 365 dias": int((idade > 365).sum()),
        "% > 90 dias": pct(int((idade > 90).sum()), len(df)),
        "% > 180 dias": pct(int((idade > 180).sum()), len(df)),
        "% > 365 dias": pct(int((idade > 365).sum()), len(df)),
    }

    return pd.Series(summary)


def age_by_group(df, group_cols):
    if "idade_anuncio_dias" not in df.columns:
        return pd.DataFrame()

    group_cols = existing_columns(df, group_cols)

    if not group_cols:
        return pd.DataFrame()

    temp = df.copy()
    temp["idade_anuncio_dias"] = pd.to_numeric(
        temp["idade_anuncio_dias"], errors="coerce"
    )

    grouped = temp.groupby(group_cols, dropna=False)["idade_anuncio_dias"]

    result = grouped.agg(
        qtd="count",
        media="mean",
        mediana="median",
        p90=lambda x: x.quantile(0.90),
        max="max",
    ).reset_index()

    result["qtd_mais_180"] = (
        temp.assign(mais_180=temp["idade_anuncio_dias"] > 180)
        .groupby(group_cols, dropna=False)["mais_180"]
        .sum()
        .values
    )

    result["pct_mais_180"] = result.apply(
        lambda row: pct(row["qtd_mais_180"], row["qtd"]),
        axis=1,
    )

    return result.sort_values(["pct_mais_180", "qtd"], ascending=[False, False])


def price_area_profile(df):
    columns = ["preco_venda", "area_m2", "preco_m2"]

    existing = existing_columns(df, columns)

    if not existing:
        return pd.DataFrame()

    return df[existing].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])


def duplicated_value_counts(df, column):
    if column not in df.columns:
        return pd.DataFrame()

    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "qtd"]
    return counts[counts["qtd"] > 1]


# =========================
# Auditorias individuais
# =========================

def analyze_dataset(lines, label, model_df, audit_df, removed_df):
    append_section(lines, f"ANÁLISE INDIVIDUAL — {label}")

    append_line(lines, f"Registros em modelagem: {len(model_df)}")
    append_line(lines, f"Registros em auditoria: {len(audit_df)}")
    append_line(lines, f"Registros removidos: {len(removed_df)}")

    if "id_anuncio" in audit_df.columns:
        total_ids = audit_df["id_anuncio"].notna().sum()
        unique_ids = audit_df["id_anuncio"].nunique(dropna=True)
        append_line(lines, f"IDs não nulos em auditoria: {total_ids}")
        append_line(lines, f"IDs únicos em auditoria: {unique_ids}")
        append_line(lines, f"IDs repetidos em auditoria: {total_ids - unique_ids}")

    if "url" in audit_df.columns:
        total_urls = audit_df["url"].notna().sum()
        unique_urls = audit_df["url"].nunique(dropna=True)
        append_line(lines, f"URLs não nulas em auditoria: {total_urls}")
        append_line(lines, f"URLs únicas em auditoria: {unique_urls}")
        append_line(lines, f"URLs repetidas em auditoria: {total_urls - unique_urls}")

    append_series(
        lines,
        f"{label} — motivos de remoção",
        removed_df["motivo_remocao"].value_counts(dropna=False)
        if "motivo_remocao" in removed_df.columns
        else pd.Series(dtype="int64"),
    )

    append_subsection(lines, f"{label} — duplicatas fortes no dataset de modelagem")
    append_line(
        lines,
        f"Linhas com id_anuncio duplicado: "
        f"{count_duplicate_rows(model_df, ['id_anuncio'])}",
    )
    append_line(
        lines,
        f"Grupos com id_anuncio duplicado: "
        f"{count_duplicate_groups(model_df, ['id_anuncio'])}",
    )
    append_line(
        lines,
        f"Linhas com URL duplicada: "
        f"{count_duplicate_rows(model_df, ['url'])}",
    )
    append_line(
        lines,
        f"Grupos com URL duplicada: "
        f"{count_duplicate_groups(model_df, ['url'])}",
    )

    append_subsection(lines, f"{label} — possíveis duplicatas aproximadas no dataset de modelagem")
    for name, key in [
        ("chave exata com vagas", APPROX_KEY_EXATA),
        ("chave exata sem vagas", APPROX_KEY_SEM_VAGAS),
        ("chave com preço arredondado e vagas", APPROX_KEY_ARREDONDADA_BASE),
        ("chave com preço arredondado sem vagas", APPROX_KEY_ARREDONDADA_SEM_VAGAS),
    ]:
        rows = count_duplicate_rows(model_df, key)
        groups = count_duplicate_groups(model_df, key)
        append_line(lines, f"{name}: {rows} linhas em {groups} grupos")

    append_dataframe(
        lines,
        f"{label} — maiores grupos de possíveis duplicatas por chave exata",
        duplicate_group_summary(model_df, APPROX_KEY_EXATA, top_n=20),
    )

    append_dataframe(
        lines,
        f"{label} — amostra de registros em grupos de possíveis duplicatas por chave exata",
        sample_duplicate_records(model_df, APPROX_KEY_EXATA, top_n_groups=8, records_per_group=4),
    )

    append_dataframe(
        lines,
        f"{label} — maiores grupos de possíveis duplicatas por chave com preço arredondado",
        duplicate_group_summary(model_df, APPROX_KEY_ARREDONDADA_BASE, top_n=20),
    )

    append_dataframe(
        lines,
        f"{label} — amostra de registros em grupos de possíveis duplicatas por chave com preço arredondado",
        sample_duplicate_records(
            model_df,
            APPROX_KEY_ARREDONDADA_BASE,
            top_n_groups=8,
            records_per_group=4,
        ),
    )

    append_series(
        lines,
        f"{label} — idade dos anúncios no dataset de modelagem",
        summarize_age(model_df),
    )

    append_dataframe(
        lines,
        f"{label} — idade por cidade",
        age_by_group(model_df, ["cidade"]),
    )

    append_dataframe(
        lines,
        f"{label} — idade por cidade e tipo de imóvel",
        age_by_group(model_df, ["cidade", "tipo_imovel"]),
    )

    append_dataframe(
        lines,
        f"{label} — perfil numérico de preço, área e preço/m²",
        price_area_profile(model_df),
    )

    if "idade_anuncio_dias" in model_df.columns:
        old_ads = model_df[
            pd.to_numeric(model_df["idade_anuncio_dias"], errors="coerce") > 365
        ].copy()

        preferred_columns = [
            "id_anuncio",
            "cidade",
            "bairro",
            "tipo_imovel",
            "subtipo_imovel",
            "preco_venda",
            "area_m2",
            "preco_m2",
            "idade_anuncio_dias",
            "titulo",
            "url",
        ]

        append_dataframe(
            lines,
            f"{label} — amostra de anúncios com mais de 365 dias",
            old_ads[existing_columns(old_ads, preferred_columns)]
            .sort_values("idade_anuncio_dias", ascending=False)
            .head(30),
        )


# =========================
# Auditorias cruzadas
# =========================

def analyze_cross_datasets(lines, vv_model, vv_audit, vv_removed, outras_model, outras_audit, outras_removed):
    append_section(lines, "ANÁLISE CRUZADA ENTRE OS DOIS CONTEXTOS")

    append_subsection(lines, "Interseção forte entre datasets de modelagem")

    by_id = intersection_by_column(vv_model, outras_model, "id_anuncio")
    by_url = intersection_by_column(vv_model, outras_model, "url")

    append_line(lines, f"Registros em comum por id_anuncio entre modelagens: {len(by_id)}")
    append_line(lines, f"Registros em comum por url entre modelagens: {len(by_url)}")

    preferred_cross_columns = [
        "id_anuncio",
        "url",
        "cidade_vv",
        "bairro_vv",
        "tipo_imovel_vv",
        "preco_venda_vv",
        "area_m2_vv",
        "cidade_outras",
        "bairro_outras",
        "tipo_imovel_outras",
        "preco_venda_outras",
        "area_m2_outras",
    ]

    append_dataframe(
        lines,
        "Amostra de registros em comum por id_anuncio entre modelagens",
        by_id[existing_columns(by_id, preferred_cross_columns)].head(30)
        if not by_id.empty else pd.DataFrame(),
    )

    append_dataframe(
        lines,
        "Amostra de registros em comum por url entre modelagens",
        by_url[existing_columns(by_url, preferred_cross_columns)].head(30)
        if not by_url.empty else pd.DataFrame(),
    )

    append_subsection(lines, "Interseção aproximada entre datasets de modelagem")

    vv_keyed = make_cross_key(vv_model, APPROX_KEY_EXATA, "chave_aproximada")
    outras_keyed = make_cross_key(outras_model, APPROX_KEY_EXATA, "chave_aproximada")

    approx_cross = vv_keyed.merge(
        outras_keyed,
        on="chave_aproximada",
        how="inner",
        suffixes=("_vv", "_outras"),
    )

    append_line(
        lines,
        f"Registros cruzados por chave aproximada exata entre modelagens: {len(approx_cross)}"
    )

    append_dataframe(
        lines,
        "Amostra de cruzamento aproximado entre modelagens",
        approx_cross[existing_columns(approx_cross, preferred_cross_columns)].head(30)
        if not approx_cross.empty else pd.DataFrame(),
    )

    vv_keyed_round = make_cross_key(
        vv_model,
        APPROX_KEY_ARREDONDADA_BASE,
        "chave_aproximada_arredondada",
    )
    outras_keyed_round = make_cross_key(
        outras_model,
        APPROX_KEY_ARREDONDADA_BASE,
        "chave_aproximada_arredondada",
    )

    approx_round_cross = vv_keyed_round.merge(
        outras_keyed_round,
        on="chave_aproximada_arredondada",
        how="inner",
        suffixes=("_vv", "_outras"),
    )

    append_line(
        lines,
        f"Registros cruzados por chave aproximada com preço arredondado entre modelagens: "
        f"{len(approx_round_cross)}"
    )

    append_dataframe(
        lines,
        "Amostra de cruzamento aproximado com preço arredondado entre modelagens",
        approx_round_cross[existing_columns(approx_round_cross, preferred_cross_columns)].head(30)
        if not approx_round_cross.empty else pd.DataFrame(),
    )

    append_subsection(
        lines,
        "Registros removidos de Vitória/Vila Velha por cidade fora do grupo presentes em outras cidades"
    )

    if "motivo_remocao" in vv_removed.columns:
        vv_removed_outside = vv_removed[
            vv_removed["motivo_remocao"] == "cidade_fora_do_grupo"
        ].copy()
    else:
        vv_removed_outside = pd.DataFrame()

    append_line(
        lines,
        f"Registros removidos de Vitória/Vila Velha por cidade_fora_do_grupo: "
        f"{len(vv_removed_outside)}"
    )

    outside_by_id_model = intersection_by_column(
        vv_removed_outside,
        outras_model,
        "id_anuncio",
    )

    outside_by_url_model = intersection_by_column(
        vv_removed_outside,
        outras_model,
        "url",
    )

    outside_by_id_audit = intersection_by_column(
        vv_removed_outside,
        outras_audit,
        "id_anuncio",
    )

    outside_by_url_audit = intersection_by_column(
        vv_removed_outside,
        outras_audit,
        "url",
    )

    outside_by_id_removed = intersection_by_column(
        vv_removed_outside,
        outras_removed,
        "id_anuncio",
    )

    outside_by_url_removed = intersection_by_column(
        vv_removed_outside,
        outras_removed,
        "url",
    )

    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes na MODELAGEM de outras por id: "
        f"{len(outside_by_id_model)}"
    )
    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes na MODELAGEM de outras por url: "
        f"{len(outside_by_url_model)}"
    )
    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes na AUDITORIA de outras por id: "
        f"{len(outside_by_id_audit)}"
    )
    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes na AUDITORIA de outras por url: "
        f"{len(outside_by_url_audit)}"
    )
    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes nos REMOVIDOS de outras por id: "
        f"{len(outside_by_id_removed)}"
    )
    append_line(
        lines,
        f"Removidos VV por cidade_fora_do_grupo presentes nos REMOVIDOS de outras por url: "
        f"{len(outside_by_url_removed)}"
    )

    append_series(
        lines,
        "Cidades dos removidos de Vitória/Vila Velha por cidade_fora_do_grupo",
        vv_removed_outside["cidade"].value_counts(dropna=False)
        if "cidade" in vv_removed_outside.columns
        else pd.Series(dtype="int64"),
    )

    append_dataframe(
        lines,
        "Amostra: removidos de VV por cidade_fora_do_grupo encontrados na modelagem de outras por id",
        outside_by_id_model[
            existing_columns(outside_by_id_model, preferred_cross_columns + ["motivo_remocao_vv"])
        ].head(30)
        if not outside_by_id_model.empty else pd.DataFrame(),
    )

    append_dataframe(
        lines,
        "Amostra: removidos de VV por cidade_fora_do_grupo encontrados na auditoria de outras por id",
        outside_by_id_audit[
            existing_columns(outside_by_id_audit, preferred_cross_columns + ["motivo_remocao_vv"])
        ].head(30)
        if not outside_by_id_audit.empty else pd.DataFrame(),
    )

    append_subsection(lines, "Registros removidos de outras cidades presentes em Vitória/Vila Velha")

    if "motivo_remocao" in outras_removed.columns:
        outras_removed_all = outras_removed.copy()
    else:
        outras_removed_all = pd.DataFrame()

    outras_removed_by_id_vv_model = intersection_by_column(
        outras_removed_all,
        vv_model,
        "id_anuncio",
    )

    outras_removed_by_url_vv_model = intersection_by_column(
        outras_removed_all,
        vv_model,
        "url",
    )

    append_line(
        lines,
        f"Removidos de outras presentes na MODELAGEM de VV por id: "
        f"{len(outras_removed_by_id_vv_model)}"
    )
    append_line(
        lines,
        f"Removidos de outras presentes na MODELAGEM de VV por url: "
        f"{len(outras_removed_by_url_vv_model)}"
    )

    append_dataframe(
        lines,
        "Amostra: removidos de outras encontrados na modelagem de VV por id",
        outras_removed_by_id_vv_model[
            existing_columns(outras_removed_by_id_vv_model, preferred_cross_columns + ["motivo_remocao_outras"])
        ].head(30)
        if not outras_removed_by_id_vv_model.empty else pd.DataFrame(),
    )


# =========================
# Relatório final
# =========================

def write_report():
    vv_model = read_csv_if_exists(VV_MODEL_CSV)
    vv_audit = read_csv_if_exists(VV_AUDIT_CSV)
    vv_removed = read_csv_if_exists(VV_REMOVED_CSV)

    outras_model = read_csv_if_exists(OUTRAS_MODEL_CSV)
    outras_audit = read_csv_if_exists(OUTRAS_AUDIT_CSV)
    outras_removed = read_csv_if_exists(OUTRAS_REMOVED_CSV)

    lines = []

    append_line(lines, "RELATÓRIO DE AUDITORIA PRÉ-UNIÃO DOS DATASETS")
    append_line(lines, "")
    append_line(lines, "Objetivo:")
    append_line(
        lines,
        "Avaliar duplicatas fortes e aproximadas, cruzamentos entre os dois contextos "
        "de coleta, registros removidos por cidade fora do grupo e idade dos anúncios "
        "antes da união final dos CSVs."
    )

    append_section(lines, "RESUMO DOS ARQUIVOS CARREGADOS")
    append_line(lines, f"Vitória/Vila Velha — modelagem: {VV_MODEL_CSV} ({len(vv_model)} registros)")
    append_line(lines, f"Vitória/Vila Velha — auditoria: {VV_AUDIT_CSV} ({len(vv_audit)} registros)")
    append_line(lines, f"Vitória/Vila Velha — removidos: {VV_REMOVED_CSV} ({len(vv_removed)} registros)")
    append_line(lines, f"Outras cidades — modelagem: {OUTRAS_MODEL_CSV} ({len(outras_model)} registros)")
    append_line(lines, f"Outras cidades — auditoria: {OUTRAS_AUDIT_CSV} ({len(outras_audit)} registros)")
    append_line(lines, f"Outras cidades — removidos: {OUTRAS_REMOVED_CSV} ({len(outras_removed)} registros)")

    analyze_dataset(
        lines,
        "Vitória/Vila Velha",
        vv_model,
        vv_audit,
        vv_removed,
    )

    analyze_dataset(
        lines,
        "Outras cidades",
        outras_model,
        outras_audit,
        outras_removed,
    )

    analyze_cross_datasets(
        lines,
        vv_model,
        vv_audit,
        vv_removed,
        outras_model,
        outras_audit,
        outras_removed,
    )

    append_section(lines, "SUGESTÕES DE INTERPRETAÇÃO")
    append_line(
        lines,
        "1. Duplicatas por id_anuncio ou url são duplicatas fortes. Se aparecerem, podem ser removidas com segurança."
    )
    append_line(
        lines,
        "2. Duplicatas por chave aproximada não devem ser removidas automaticamente. Elas podem representar unidades diferentes de um mesmo empreendimento."
    )
    append_line(
        lines,
        "3. Se houver cruzamento por id/url entre modelagens, é sinal de que a união final precisa deduplicar pelo menos por id_anuncio e url."
    )
    append_line(
        lines,
        "4. Se removidos de Vitória/Vila Velha por cidade_fora_do_grupo aparecerem na modelagem de outras cidades, isso confirma que a filtragem por grupo funcionou e que esses dados podem entrar pela base correta."
    )
    append_line(
        lines,
        "5. Anúncios muito antigos não precisam ser removidos agora, mas devem ser avaliados em experimentos separados, por exemplo com todos os dados, até 365 dias e até 180 dias."
    )
    append_line(
        lines,
        "6. Se a proporção de possíveis duplicatas aproximadas for muito alta em bairros específicos, trate isso como concentração de empreendimentos, não necessariamente como duplicação inválida."
    )

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")

    print("Auditoria pré-união concluída.")
    print(f"Relatório gerado: {REPORT_TXT}")


if __name__ == "__main__":
    write_report()