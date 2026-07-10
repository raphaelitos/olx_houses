import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATASET_BASE = "dataset_modelagem_consolidado.csv"
DATASET_AGREGADO = "dataset_modelagem_agregado.csv"

DUPLICATAS_EXATAS = "duplicatas_candidatas_chave_exata.csv"

OUTPUT_AUDITAVEL = "dataset_final_auditavel.csv"
OUTPUT_MODELO = "dataset_final_modelagem.csv"
OUTPUT_AGREGADO_FILTRADO = "dataset_modelagem_agregado_filtrado.csv"
OUTPUT_REMOVIDOS_EXATOS = "dataset_agregado_removidos_duplicata_exata.csv"

REPORT_TXT = "relatorio_fusao_final.txt"


# Coluna auxiliar: manter no auditável, remover do dataset de treino.
FONTE_COL = "fonte_dataset"


# ============================================================
# FUNÇÕES
# ============================================================

def read_csv_required(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    return pd.read_csv(path)


def normalize_count_columns(df):
    """
    Garante que quartos, banheiros e vagas sejam numéricos,
    com valores negativos convertidos para ausência.

    O CSV não preserva Int64 ao reler, então essa normalização é refeita aqui.
    """
    df = df.copy()

    for col in ["quartos", "banheiros", "vagas"]:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if col in ["quartos", "banheiros"]:
            df.loc[df[col] <= 0, col] = pd.NA
        else:
            df.loc[df[col] < 0, col] = pd.NA

        df[col] = df[col].round().astype("Int64")

    return df


def get_agregado_rows_to_remove(path):
    """
    Lê duplicatas_candidatas_chave_exata.csv e retorna os índices
    do dataset agregado que devem ser removidos.

    O arquivo de comparação foi gerado com _row_id_novo, que corresponde
    ao índice original do dataset_modelagem_agregado.csv no momento da comparação.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Arquivo de duplicatas exatas não encontrado: {path}"
        )

    df_dup = pd.read_csv(path)

    if df_dup.empty:
        return set(), df_dup

    if "_row_id_novo" not in df_dup.columns:
        raise ValueError(
            "O arquivo de duplicatas exatas não contém a coluna '_row_id_novo'. "
            "Rode novamente o script de comparação antes da fusão."
        )

    rows_to_remove = (
        pd.to_numeric(df_dup["_row_id_novo"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    return set(rows_to_remove), df_dup


def align_columns(base_df, agregado_df):
    """
    Garante que os dois datasets tenham as mesmas colunas antes da união.

    Se houver diferença, a função cria colunas ausentes com NA e ordena
    o agregado conforme as colunas da base.
    """
    base_df = base_df.copy()
    agregado_df = agregado_df.copy()

    base_cols = list(base_df.columns)
    agregado_cols = list(agregado_df.columns)

    only_base = sorted(set(base_cols) - set(agregado_cols))
    only_agregado = sorted(set(agregado_cols) - set(base_cols))

    for col in only_base:
        agregado_df[col] = pd.NA

    for col in only_agregado:
        base_df[col] = pd.NA

    final_cols = list(base_df.columns)

    agregado_df = agregado_df[final_cols]

    return base_df, agregado_df, only_base, only_agregado


def build_report(
    base_df,
    agregado_df,
    agregado_filtrado,
    removidos_exatos,
    final_auditavel,
    final_modelo,
    dup_df,
    only_base,
    only_agregado,
):
    lines = []

    lines.append("RELATÓRIO DE FUSÃO FINAL")
    lines.append("")
    lines.append(f"Dataset base: {DATASET_BASE}")
    lines.append(f"Dataset agregado: {DATASET_AGREGADO}")
    lines.append(f"Duplicatas exatas: {DUPLICATAS_EXATAS}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("RESUMO")
    lines.append("=" * 90)
    lines.append(f"Registros na base original: {len(base_df)}")
    lines.append(f"Registros no agregado original: {len(agregado_df)}")
    lines.append(f"Pares no arquivo de duplicatas exatas: {len(dup_df)}")
    lines.append(f"Registros únicos removidos do agregado: {len(removidos_exatos)}")
    lines.append(f"Registros restantes do agregado: {len(agregado_filtrado)}")
    lines.append(f"Total final auditável: {len(final_auditavel)}")
    lines.append(f"Total final para modelagem: {len(final_modelo)}")
    lines.append("")

    esperado = len(base_df) + len(agregado_filtrado)
    lines.append(f"Total esperado após fusão: {esperado}")
    lines.append(f"Diferença: {len(final_auditavel) - esperado}")
    lines.append("")

    lines.append("=" * 90)
    lines.append("COLUNAS")
    lines.append("=" * 90)
    lines.append("Colunas presentes apenas na base original antes do alinhamento:")
    lines.append(str(only_base) if only_base else "(vazio)")
    lines.append("")
    lines.append("Colunas presentes apenas no agregado antes do alinhamento:")
    lines.append(str(only_agregado) if only_agregado else "(vazio)")
    lines.append("")
    lines.append("Colunas do dataset final auditável:")
    lines.append(", ".join(final_auditavel.columns))
    lines.append("")
    lines.append("Colunas do dataset final de modelagem:")
    lines.append(", ".join(final_modelo.columns))
    lines.append("")

    lines.append("=" * 90)
    lines.append("DISTRIBUIÇÃO POR FONTE")
    lines.append("=" * 90)
    lines.append(str(final_auditavel[FONTE_COL].value_counts(dropna=False)))
    lines.append("")

    if "cidade" in final_auditavel.columns:
        lines.append("=" * 90)
        lines.append("COBERTURA POR CIDADE")
        lines.append("=" * 90)
        lines.append(str(final_auditavel["cidade"].value_counts(dropna=False)))
        lines.append("")

    if "tipo_imovel" in final_auditavel.columns:
        lines.append("=" * 90)
        lines.append("COBERTURA POR TIPO")
        lines.append("=" * 90)
        lines.append(str(final_auditavel["tipo_imovel"].value_counts(dropna=False)))
        lines.append("")

    if {"cidade", "tipo_imovel"}.issubset(final_auditavel.columns):
        lines.append("=" * 90)
        lines.append("COBERTURA POR CIDADE E TIPO")
        lines.append("=" * 90)
        lines.append(
            str(
                final_auditavel
                .groupby(["cidade", "tipo_imovel"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
        )
        lines.append("")

    lines.append("=" * 90)
    lines.append("AUSÊNCIAS EM QUARTOS, BANHEIROS E VAGAS")
    lines.append("=" * 90)

    for col in ["quartos", "banheiros", "vagas"]:
        if col in final_auditavel.columns:
            lines.append(f"{col}:")
            lines.append(f"  ausentes: {int(final_auditavel[col].isna().sum())}")
            lines.append(f"  zeros: {int((final_auditavel[col] == 0).sum())}")
            lines.append(f"  dtype em memória: {final_auditavel[col].dtype}")
            lines.append("")

    lines.append("=" * 90)
    lines.append("POLÍTICA APLICADA")
    lines.append("=" * 90)
    lines.append(
        "1. Registros do agregado presentes em duplicatas_candidatas_chave_exata.csv "
        "foram removidos."
    )
    lines.append(
        "2. Candidatos por preço arredondado não foram removidos automaticamente."
    )
    lines.append(
        "3. Candidatos por aproximação estrutural não foram removidos automaticamente."
    )
    lines.append(
        "4. A base original foi preservada integralmente."
    )
    lines.append(
        "5. A coluna fonte_dataset foi mantida no dataset auditável."
    )
    lines.append(
        "6. A coluna fonte_dataset foi removida do dataset final de modelagem."
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    base = read_csv_required(DATASET_BASE)
    agregado = read_csv_required(DATASET_AGREGADO)

    base = normalize_count_columns(base)
    agregado = normalize_count_columns(agregado)

    rows_to_remove, dup_df = get_agregado_rows_to_remove(DUPLICATAS_EXATAS)

    agregado = agregado.reset_index(drop=True)

    mask_remove = agregado.index.isin(rows_to_remove)

    agregado_removidos = agregado[mask_remove].copy()
    agregado_filtrado = agregado[~mask_remove].copy()

    agregado_removidos.to_csv(OUTPUT_REMOVIDOS_EXATOS, index=False)
    agregado_filtrado.to_csv(OUTPUT_AGREGADO_FILTRADO, index=False)

    base[FONTE_COL] = "base_original"
    agregado_filtrado[FONTE_COL] = "agregado_filtrado"

    base_alinhada, agregado_alinhado, only_base, only_agregado = align_columns(
        base,
        agregado_filtrado,
    )

    final_auditavel = pd.concat(
        [base_alinhada, agregado_alinhado],
        ignore_index=True,
    )

    final_auditavel = normalize_count_columns(final_auditavel)

    final_modelo = final_auditavel.drop(columns=[FONTE_COL])

    final_auditavel.to_csv(OUTPUT_AUDITAVEL, index=False)
    final_modelo.to_csv(OUTPUT_MODELO, index=False)

    report = build_report(
        base_df=base,
        agregado_df=agregado,
        agregado_filtrado=agregado_filtrado,
        removidos_exatos=agregado_removidos,
        final_auditavel=final_auditavel,
        final_modelo=final_modelo,
        dup_df=dup_df,
        only_base=only_base,
        only_agregado=only_agregado,
    )

    Path(REPORT_TXT).write_text(report, encoding="utf-8")

    print("Fusão concluída.")
    print(f"Base original: {len(base)} registros")
    print(f"Agregado original: {len(agregado)} registros")
    print(f"Registros removidos do agregado por chave exata: {len(agregado_removidos)}")
    print(f"Agregado aproveitado: {len(agregado_filtrado)} registros")
    print(f"Dataset final auditável: {OUTPUT_AUDITAVEL} ({len(final_auditavel)} registros)")
    print(f"Dataset final de modelagem: {OUTPUT_MODELO} ({len(final_modelo)} registros)")
    print(f"Agregado filtrado salvo em: {OUTPUT_AGREGADO_FILTRADO}")
    print(f"Removidos por duplicata exata salvos em: {OUTPUT_REMOVIDOS_EXATOS}")
    print(f"Relatório salvo em: {REPORT_TXT}")


if __name__ == "__main__":
    main()