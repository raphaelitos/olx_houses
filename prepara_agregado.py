import pandas as pd
from pathlib import Path
import unicodedata


RAW_CSV = "imoveis_agregado.csv"

MODEL_CSV = "dataset_modelagem_agregado.csv"
AUDIT_CSV = "dataset_auditoria_agregado.csv"
REMOVED_CSV = "dataset_removidos_agregado.csv"
REPORT_TXT = "relatorio_dataset_agregado.txt"


CIDADES_ESPERADAS = [
    "Vitória",
    "Vila Velha",
    "Serra",
    "Guarapari",
    "Cariacica",
    "Viana",
    "Fundão",
]


MIN_PRECO = 20_000
MAX_PRECO = 20_000_000

MIN_AREA = 10
MAX_AREA = 1000

MIN_PRECO_M2_CASA = 700
MAX_PRECO_M2_CASA = 30_000

MIN_PRECO_M2_APARTAMENTO = 1500
MAX_PRECO_M2_APARTAMENTO = 40_000


# Para manter compatibilidade com seu dataset anterior.
# Valores acima disso não removem o registro; viram NaN.
MAX_QUARTOS = 5
MAX_BANHEIROS = 5
MAX_VAGAS = 5


RELEVANT_BOOLEAN_COLUMNS = [
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]


DISCARDED_COLUMNS = [
    "Unnamed: 0",
    "id_anuncio",
    "cidade_busca",
    "tem_gourmet",
    "tem_cobertura",
    "tem_garden",
    "tem_frente_mar",
]


MODEL_COLUMNS = [
    "preco_venda",
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    *RELEVANT_BOOLEAN_COLUMNS,
]


AUDIT_COLUMNS = [
    "preco_venda",
    "preco_m2",
    "tipo_imovel_original",
    "tipo_imovel",
    "cidade_original",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    *RELEVANT_BOOLEAN_COLUMNS,
    "preco_suspeito",
    "area_suspeita",
    "preco_m2_suspeito",
]


REMOVED_COLUMNS = [
    "motivo_remocao",
    *AUDIT_COLUMNS,
]


def strip_accents(text):
    if pd.isna(text):
        return pd.NA

    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def normalize_text_series(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_city(value):
    if pd.isna(value):
        return pd.NA

    raw = str(value).strip()
    key = strip_accents(raw).lower()

    mapping = {
        "vitoria": "Vitória",
        "vila velha": "Vila Velha",
        "serra": "Serra",
        "guarapari": "Guarapari",
        "cariacica": "Cariacica",
        "viana": "Viana",
        "fundao": "Fundão",
        "fundão": "Fundão",
        "praia grande": "Fundão",
        "praia grande, fundao": "Fundão",
        "praia grande, fundão": "Fundão",
    }

    return mapping.get(key, raw)


def normalize_property_type(value):
    if pd.isna(value):
        return pd.NA

    raw = str(value).strip()
    key = strip_accents(raw).lower()

    apartment_values = {
        "apartamento",
        "cobertura",
        "duplex",
        "studio",
        "flat",
        "apart hotel/ flat",
        "apart hotel/flat",
        "apart-hotel",
    }

    house_values = {
        "casa",
        "terrea",
        "térrea",
        "sobrado",
        "casa de condominio",
        "casa de condomínio",
        "triplex",
    }

    if key in apartment_values:
        return "apartamento"

    if key in house_values:
        return "casa"

    return pd.NA


def read_raw_dataset(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    required = [
        "preco_venda",
        "tipo_imovel",
        "cidade",
        "bairro",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
        *RELEVANT_BOOLEAN_COLUMNS,
    ]

    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def normalize_dataset(df):
    df = df.copy()

    df["tipo_imovel_original"] = df["tipo_imovel"]
    df["cidade_original"] = df["cidade"]

    df["tipo_imovel"] = df["tipo_imovel"].apply(normalize_property_type)
    df["cidade"] = df["cidade"].apply(normalize_city)
    df["bairro"] = normalize_text_series(df["bairro"])

    numeric_cols = [
        "preco_venda",
        "area_m2",
        "quartos",
        "banheiros",
        "vagas",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ausências, zeros e negativos nas contagens viram NaN.
    # Para vagas, zero é válido; negativos viram NaN.
    df.loc[df["quartos"].isna() | (df["quartos"] <= 0), "quartos"] = pd.NA
    df.loc[df["banheiros"].isna() | (df["banheiros"] <= 0), "banheiros"] = pd.NA
    df.loc[df["vagas"].isna() | (df["vagas"] < 0), "vagas"] = pd.NA

    # Valores absurdamente altos nas contagens viram NaN, não removem o registro.
    df.loc[df["quartos"] > MAX_QUARTOS, "quartos"] = pd.NA
    df.loc[df["banheiros"] > MAX_BANHEIROS, "banheiros"] = pd.NA
    df.loc[df["vagas"] > MAX_VAGAS, "vagas"] = pd.NA

    for col in ["quartos", "banheiros", "vagas"]:
        df[col] = df[col].round().astype("Int64")

    for col in RELEVANT_BOOLEAN_COLUMNS:
        df[col] = df[col].fillna(False).astype(bool)

    df["preco_m2"] = df["preco_venda"] / df["area_m2"]
    df["preco_m2"] = df["preco_m2"].round(2)

    return df


def add_suspicion_flags(df):
    df = df.copy()

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

    df["preco_m2_suspeito"] = False

    mask_casa = df["tipo_imovel"] == "casa"
    mask_apto = df["tipo_imovel"] == "apartamento"

    df.loc[mask_casa, "preco_m2_suspeito"] = (
        df.loc[mask_casa, "preco_m2"].notna()
        & (
            (df.loc[mask_casa, "preco_m2"] < MIN_PRECO_M2_CASA)
            | (df.loc[mask_casa, "preco_m2"] > MAX_PRECO_M2_CASA)
        )
    )

    df.loc[mask_apto, "preco_m2_suspeito"] = (
        df.loc[mask_apto, "preco_m2"].notna()
        & (
            (df.loc[mask_apto, "preco_m2"] <= MIN_PRECO_M2_APARTAMENTO)
            | (df.loc[mask_apto, "preco_m2"] > MAX_PRECO_M2_APARTAMENTO)
        )
    )

    return df


def add_removal_reason(df, mask, reason):
    df.loc[mask & df["motivo_remocao"].isna(), "motivo_remocao"] = reason
    return df


def build_removed_dataset(df):
    out = df.copy()
    out["motivo_remocao"] = pd.NA

    out = add_removal_reason(out, out["preco_venda"].isna(), "sem_preco_venda")
    out = add_removal_reason(out, out["area_m2"].isna(), "sem_area_m2")
    out = add_removal_reason(out, out["tipo_imovel"].isna(), "tipo_imovel_invalido")
    out = add_removal_reason(out, out["cidade"].isna(), "sem_cidade")
    out = add_removal_reason(out, out["bairro"].isna(), "sem_bairro")
    out = add_removal_reason(out, ~out["cidade"].isin(CIDADES_ESPERADAS), "cidade_fora_do_grupo")
    out = add_removal_reason(out, out["preco_venda"].notna() & (out["preco_venda"] <= 0), "preco_venda_invalido")
    out = add_removal_reason(out, out["area_m2"].notna() & (out["area_m2"] <= 0), "area_m2_invalida")
    out = add_removal_reason(out, out["preco_suspeito"], "preco_suspeito")
    out = add_removal_reason(out, out["area_suspeita"], "area_suspeita")
    out = add_removal_reason(out, out["preco_m2_suspeito"], "preco_m2_suspeito")

    return out[out["motivo_remocao"].notna()].copy()


def build_model_dataset(df):
    out = df.copy()

    out = out.dropna(
        subset=[
            "preco_venda",
            "area_m2",
            "tipo_imovel",
            "cidade",
            "bairro",
        ]
    )

    out = out[out["cidade"].isin(CIDADES_ESPERADAS)]
    out = out[out["preco_venda"] > 0]
    out = out[out["area_m2"] > 0]
    out = out[out["preco_suspeito"] == False]
    out = out[out["area_suspeita"] == False]
    out = out[out["preco_m2_suspeito"] == False]

    existing = [col for col in MODEL_COLUMNS if col in out.columns]
    return out[existing].copy()


def export_existing(df, columns):
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()


def write_report(raw_df, model_df, removed_df):
    lines = []

    lines.append("RELATÓRIO DO DATASET AGREGADO")
    lines.append("")
    lines.append(f"Arquivo bruto: {RAW_CSV}")
    lines.append(f"Registros brutos: {len(raw_df)}")
    lines.append(f"Registros no dataset de modelagem: {len(model_df)}")
    lines.append(f"Registros removidos: {len(removed_df)}")
    lines.append(f"Taxa de retenção: {round(len(model_df) / len(raw_df) * 100, 2)}%")
    lines.append("")

    lines.append("=" * 90)
    lines.append("COLUNAS DO DATASET DE MODELAGEM")
    lines.append("=" * 90)
    lines.append(", ".join(model_df.columns))
    lines.append("")

    lines.append("=" * 90)
    lines.append("REMOÇÕES")
    lines.append("=" * 90)
    lines.append(str(removed_df["motivo_remocao"].value_counts(dropna=False)))
    lines.append("")

    lines.append("=" * 90)
    lines.append("COBERTURA")
    lines.append("=" * 90)
    lines.append("Por cidade:")
    lines.append(str(model_df["cidade"].value_counts(dropna=False)))
    lines.append("")
    lines.append("Por tipo:")
    lines.append(str(model_df["tipo_imovel"].value_counts(dropna=False)))
    lines.append("")
    lines.append("Por cidade/tipo:")
    lines.append(str(model_df.groupby(["cidade", "tipo_imovel"], dropna=False).size().sort_values(ascending=False)))
    lines.append("")

    lines.append("=" * 90)
    lines.append("AUSÊNCIAS")
    lines.append("=" * 90)
    lines.append(str(model_df.isna().sum().sort_values(ascending=False)))
    lines.append("")

    lines.append("=" * 90)
    lines.append("RESUMO NUMÉRICO")
    lines.append("=" * 90)
    numeric_cols = ["preco_venda", "area_m2", "quartos", "banheiros", "vagas"]
    lines.append(str(model_df[numeric_cols].describe()))
    lines.append("")

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_df = read_raw_dataset(RAW_CSV)
    raw_df = normalize_dataset(raw_df)
    raw_df = add_suspicion_flags(raw_df)

    removed_df = build_removed_dataset(raw_df)
    model_df = build_model_dataset(raw_df)
    audit_df = export_existing(raw_df, AUDIT_COLUMNS)
    removed_export = export_existing(removed_df, REMOVED_COLUMNS)

    model_df.to_csv(MODEL_CSV, index=False)
    audit_df.to_csv(AUDIT_CSV, index=False)
    removed_export.to_csv(REMOVED_CSV, index=False)

    write_report(raw_df, model_df, removed_df)

    print("Tratamento concluído.")
    print(f"Entrada bruta: {RAW_CSV} ({len(raw_df)} registros)")
    print(f"Dataset de modelagem: {MODEL_CSV} ({len(model_df)} registros)")
    print(f"Dataset de auditoria: {AUDIT_CSV}")
    print(f"Removidos: {REMOVED_CSV} ({len(removed_df)} registros)")
    print(f"Relatório: {REPORT_TXT}")


if __name__ == "__main__":
    main()