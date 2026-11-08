"""Leitura, padronização e limpeza dos CSVs brutos do VRA.

Saída: data/processed/voos.parquet — um voo por linha, com atraso calculado.
"""

from __future__ import annotations

import pandas as pd

from atrasos.config import LIMIAR_ATRASO_MIN, PROCESSED_DIR, RAW_DIR, VOOS_PARQUET

# Ordem das colunas no layout do VRA a partir de abril/2020. Renomeamos por posição
# porque os cabeçalhos têm acentos e já mudaram de grafia entre publicações.
COLUNAS = [
    "empresa",
    "numero_voo",
    "codigo_di",
    "tipo_linha",
    "origem",
    "destino",
    "partida_prevista",
    "partida_real",
    "chegada_prevista",
    "chegada_real",
    "situacao",
    "codigo_justificativa",
]
COLUNAS_DATA = ["partida_prevista", "partida_real", "chegada_prevista", "chegada_real"]

# Atrasos fora desse intervalo são, na prática, erros de registro (fuso, data trocada).
ATRASO_MIN_VALIDO = -120
ATRASO_MAX_VALIDO = 24 * 60


def ler_csv_vra(caminho) -> pd.DataFrame:
    with open(caminho, encoding="utf-8-sig") as f:
        primeira = f.readline()
    pular = 1 if primeira.lower().startswith("atualizado") else 0
    df = pd.read_csv(
        caminho,
        sep=";",
        skiprows=pular,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    if df.shape[1] != len(COLUNAS):
        raise ValueError(f"{caminho.name}: esperado {len(COLUNAS)} colunas, veio {df.shape[1]}")
    df.columns = COLUNAS
    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["empresa", "origem", "destino", "tipo_linha", "codigo_di", "situacao"]:
        df[col] = df[col].str.strip().str.upper()
    df["numero_voo"] = df["numero_voo"].str.strip().str.lstrip("0").replace("", "0")
    for col in COLUNAS_DATA:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    df["cancelado"] = df["situacao"].eq("CANCELADO")
    df["atraso_partida_min"] = (df["partida_real"] - df["partida_prevista"]).dt.total_seconds() / 60
    df["atraso_chegada_min"] = (df["chegada_real"] - df["chegada_prevista"]).dt.total_seconds() / 60
    df["tempo_bloco_previsto_min"] = (df["chegada_prevista"] - df["partida_prevista"]).dt.total_seconds() / 60
    df["atrasado"] = (df["atraso_chegada_min"] > LIMIAR_ATRASO_MIN).astype("int8")
    df["mes"] = df["partida_prevista"].dt.to_period("M").astype(str)

    df = df.drop(columns=["codigo_justificativa"])
    # Duplicatas exatas aparecem quando um voo é republicado no mês seguinte.
    df = df.drop_duplicates(subset=["empresa", "numero_voo", "partida_prevista", "origem"])
    return df


def filtrar_realizados(df: pd.DataFrame) -> pd.DataFrame:
    """Voos realizados com horários consistentes — a população do modelo de atraso."""
    mask = (
        ~df["cancelado"]
        & df["partida_prevista"].notna()
        & df["chegada_prevista"].notna()
        & df["chegada_real"].notna()
        & df["atraso_chegada_min"].between(ATRASO_MIN_VALIDO, ATRASO_MAX_VALIDO)
        & df["tempo_bloco_previsto_min"].between(10, 20 * 60)
    )
    return df.loc[mask].reset_index(drop=True)


def main() -> None:
    arquivos = sorted(RAW_DIR.glob("VRA_*.csv"))
    if not arquivos:
        raise SystemExit("Nenhum CSV em data/raw — rode `python -m atrasos.download` antes.")
    partes = []
    for arq in arquivos:
        bruto = ler_csv_vra(arq)
        partes.append(limpar(bruto))
        print(f"{arq.name}: {len(bruto):,} linhas")
    todos = pd.concat(partes, ignore_index=True)
    todos = todos.drop_duplicates(subset=["empresa", "numero_voo", "partida_prevista", "origem"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    todos.to_parquet(PROCESSED_DIR / "voos_todos.parquet", index=False)
    realizados = filtrar_realizados(todos)
    realizados.to_parquet(VOOS_PARQUET, index=False)

    print(
        f"\nTotal: {len(todos):,} voos | cancelados: {todos['cancelado'].mean():.1%} | "
        f"realizados válidos: {len(realizados):,} | "
        f"taxa de atraso >{LIMIAR_ATRASO_MIN} min: {realizados['atrasado'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
