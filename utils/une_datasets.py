import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATASET_VV = "dataset_modelagem.csv"
DATASET_OUTRAS = "dataset_modelagem_outras_cidades.csv"

OUTPUT_CSV = "dataset_modelagem_consolidado.csv"
REPORT_TXT = "relatorio_uniao_datasets.txt"


# ============================================================
# FUNÇÕES
# ============================================================

def read_dataset(path, origem):
    """Lê um dataset de modelagem e adiciona a origem apenas para auditoria interna."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)
    df["_origem_dataset"] = origem

    return df


def normalize_numeric_nullable_int(df, columns):
    """
    Converte colunas numéricas para inteiro anulável.

    Regras:
    - valores ausentes continuam ausentes;
    - valores negativos viram NA;
    - valores não numéricos viram NA;
    - valores válidos são convertidos para Int64.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        values = values.mask(values < 0, pd.NA)

        # Arredonda apenas por segurança caso o CSV tenha vindo com 1.0, 2.0 etc.
        # Como essas colunas representam contagem, o esperado é que sejam inteiras.
        values = values.round()

        df[col] = values.astype("Int64")

    return df


def validate_same_columns(df1, df2):
    """Verifica se os dois datasets têm as mesmas colunas antes da união."""
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    only_1 = sorted(cols1 - cols2)
    only_2 = sorted(cols2 - cols1)

    return only_1, only_2


def build_report(df_vv, df_outras, df_final, only_vv, only_outras):
    lines = []

    lines.append("RELATÓRIO DE UNIÃO DOS DATASETS")
    lines.append("")
    lines.append(f"Dataset Vitória/Vila Velha: {DATASET_VV}")
    lines.append(f"Dataset outras cidades: {DATASET_OUTRAS}")
    lines.append(f"Dataset consolidado: {OUTPUT_CSV}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("RESUMO")
    lines.append("=" * 90)
    lines.append(f"Registros Vitória/Vila Velha: {len(df_vv)}")
    lines.append(f"Registros outras cidades: {len(df_outras)}")
    lines.append(f"Soma esperada: {len(df_vv) + len(df_outras)}")
    lines.append(f"Registros consolidados: {len(df_final)}")
    lines.append(f"Diferença: {len(df_final) - (len(df_vv) + len(df_outras))}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("COLUNAS")
    lines.append("=" * 90)
    lines.append(f"Quantidade de colunas no consolidado: {len(df_final.columns)}")
    lines.append(", ".join(df_final.columns))
    lines.append("")

    lines.append("-" * 90)
    lines.append("Colunas presentes apenas em Vitória/Vila Velha")
    lines.append("-" * 90)
    lines.append(str(only_vv) if only_vv else "(vazio)")
    lines.append("")

    lines.append("-" * 90)
    lines.append("Colunas presentes apenas em outras cidades")
    lines.append("-" * 90)
    lines.append(str(only_outras) if only_outras else "(vazio)")
    lines.append("")

    lines.append("=" * 90)
    lines.append("AUSÊNCIAS EM QUARTOS, BANHEIROS E VAGAS")
    lines.append("=" * 90)

    for col in ["quartos", "banheiros", "vagas"]:
        if col in df_final.columns:
            lines.append(f"{col}:")
            lines.append(f"  ausentes: {int(df_final[col].isna().sum())}")
            lines.append(f"  zeros: {int((df_final[col] == 0).sum())}")
            lines.append(f"  dtype: {df_final[col].dtype}")
            lines.append("")

    lines.append("=" * 90)
    lines.append("COBERTURA")
    lines.append("=" * 90)

    if "cidade" in df_final.columns:
        lines.append("Cobertura por cidade:")
        lines.append(str(df_final["cidade"].value_counts(dropna=False)))
        lines.append("")

    if "tipo_imovel" in df_final.columns:
        lines.append("Cobertura por tipo de imóvel:")
        lines.append(str(df_final["tipo_imovel"].value_counts(dropna=False)))
        lines.append("")

    if {"cidade", "tipo_imovel"}.issubset(df_final.columns):
        lines.append("Cobertura por cidade e tipo de imóvel:")
        lines.append(
            str(
                df_final.groupby(["cidade", "tipo_imovel"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
        )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    df_vv = read_dataset(DATASET_VV, "vitoria_vila_velha")
    df_outras = read_dataset(DATASET_OUTRAS, "outras_cidades")

    # Confere diferenças de colunas antes da união.
    only_vv, only_outras = validate_same_columns(df_vv, df_outras)

    if only_vv or only_outras:
        print("Atenção: os datasets possuem diferenças de colunas.")
        print("Apenas em Vitória/Vila Velha:", only_vv)
        print("Apenas em outras cidades:", only_outras)
        print()

    # União direta.
    df = pd.concat([df_vv, df_outras], ignore_index=True)

    # Remove coluna pouco útil para modelagem.
    if "subtipo_imovel" in df.columns:
        df = df.drop(columns=["subtipo_imovel"])

    # Trata contagens.
    df = normalize_numeric_nullable_int(
        df,
        columns=["quartos", "banheiros", "vagas"],
    )

    # Remove coluna de origem do dataset consolidado final.
    # Ela foi usada apenas para o relatório.
    df_report = df.copy()

    if "_origem_dataset" in df.columns:
        df = df.drop(columns=["_origem_dataset"])

    # Exporta dataset final.
    df.to_csv(OUTPUT_CSV, index=False)

    # Relatório.
    report = build_report(
        df_vv=df_vv,
        df_outras=df_outras,
        df_final=df_report,
        only_vv=only_vv,
        only_outras=only_outras,
    )

    Path(REPORT_TXT).write_text(report, encoding="utf-8")

    print("União concluída.")
    print(f"Registros Vitória/Vila Velha: {len(df_vv)}")
    print(f"Registros outras cidades: {len(df_outras)}")
    print(f"Registros consolidados: {len(df)}")
    print(f"Dataset salvo em: {OUTPUT_CSV}")
    print(f"Relatório salvo em: {REPORT_TXT}")


if __name__ == "__main__":
    main()