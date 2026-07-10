import pandas as pd


INPUT_CSV = "imoveis_agregado.csv"

OUTPUT_ABSURDOS_CSV = "imoveis_agregado_valores_absurdos.csv"
OUTPUT_LIMPO_CSV = "imoveis_agregado_sem_absurdos.csv"


# Ajuste os limites aqui, se quiser
MAX_QUARTOS = 15
MAX_BANHEIROS = 10
MAX_VAGAS = 15


def main():
    df = pd.read_csv(INPUT_CSV)

    for col in ["quartos", "banheiros", "vagas"]:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_absurdo = (
        (df["quartos"] > MAX_QUARTOS)
        | (df["banheiros"] > MAX_BANHEIROS)
        | (df["vagas"] > MAX_VAGAS)
        | (df["quartos"] < 0)
        | (df["banheiros"] < 0)
        | (df["vagas"] < 0)
    )

    absurdos = df[mask_absurdo].copy()
    limpo = df[~mask_absurdo].copy()

    absurdos.to_csv(OUTPUT_ABSURDOS_CSV, index=False)
    limpo.to_csv(OUTPUT_LIMPO_CSV, index=False)

    print("Separação concluída.")
    print(f"Arquivo original: {INPUT_CSV}")
    print(f"Registros originais: {len(df)}")
    print(f"Registros com valores absurdos: {len(absurdos)}")
    print(f"Registros restantes: {len(limpo)}")
    print()
    print(f"Absurdos salvos em: {OUTPUT_ABSURDOS_CSV}")
    print(f"Base limpa salva em: {OUTPUT_LIMPO_CSV}")

    if len(absurdos) > 0:
        print()
        print("Resumo dos absurdos:")
        print(absurdos[["quartos", "banheiros", "vagas"]].describe())


if __name__ == "__main__":
    main()