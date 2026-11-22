"""Séries mensais longas (2015 em diante) para a projeção 2027–2028.

Para cada mês do VRA, baixa o CSV, resume em números mensais e descarta o arquivo bruto
(os meses de 2024 em diante já baixados em data/raw são reaproveitados). Escopo: voos
domésticos (origem e destino em aeródromos brasileiros), no total e por grupo aéreo.

Saída: data/processed/serie_mensal.parquet e reports/projecao/serie_mensal.csv
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
import requests

from atrasos.config import LIMIAR_ATRASO_MIN, PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from atrasos.download import url_mes
from atrasos.limpeza import COLUNAS

SAIDA = PROCESSED_DIR / "serie_mensal.parquet"
# Prefixos ICAO de aeródromos brasileiros
PREFIXOS_BR = ("SB", "SD", "SI", "SJ", "SN", "SS", "SW")
# Grupos aéreos: códigos ICAO que operaram com a mesma marca ao longo do período
GRUPOS = {"Gol": ["GLO"], "LATAM": ["TAM"], "Azul": ["AZU"]}


def _datas(s: pd.Series) -> pd.Series:
    """Os arquivos antigos usam '01/03/2015 05:30'; os novos, '2015-03-01 05:30:00'."""
    s = s.replace("", np.nan)
    amostra = s.dropna().head(50)
    if len(amostra) and amostra.str.contains("/").mean() > 0.5:
        return pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def ler(conteudo: bytes) -> pd.DataFrame:
    texto = conteudo.decode("utf-8-sig", errors="replace")
    pular = 1 if texto.lower().startswith("atualizado") else 0
    df = pd.read_csv(io.StringIO(texto), sep=";", skiprows=pular, dtype=str, keep_default_na=False)
    if df.shape[1] != len(COLUNAS):
        raise ValueError(f"esperado {len(COLUNAS)} colunas, veio {df.shape[1]}")
    df.columns = COLUNAS
    return df


def resumir(df: pd.DataFrame, ano: int, mes: int) -> list[dict]:
    for c in ["empresa", "origem", "destino", "situacao"]:
        df[c] = df[c].str.strip().str.upper()
    dom = df[df["origem"].str[:2].isin(PREFIXOS_BR) & df["destino"].str[:2].isin(PREFIXOS_BR)].copy()
    for c in ["partida_prevista", "chegada_prevista", "chegada_real"]:
        dom[c] = _datas(dom[c])
    dom["cancelado"] = dom["situacao"].eq("CANCELADO")
    atraso = (dom["chegada_real"] - dom["chegada_prevista"]).dt.total_seconds() / 60
    bloco = (dom["chegada_prevista"] - dom["partida_prevista"]).dt.total_seconds() / 60
    dom["valido"] = ~dom["cancelado"] & atraso.between(-120, 24 * 60) & bloco.between(10, 20 * 60)
    dom["atrasado"] = dom["valido"] & (atraso > LIMIAR_ATRASO_MIN)
    dom["atraso_min"] = atraso.where(dom["valido"])

    linhas = []
    escopos = {"Brasil": dom, **{g: dom[dom["empresa"].isin(cods)] for g, cods in GRUPOS.items()}}
    for nome, d in escopos.items():
        validos = int(d["valido"].sum())
        linhas.append(
            {
                "mes": f"{ano}-{mes:02d}",
                "escopo": nome,
                "voos_programados": len(d),
                "cancelados": int(d["cancelado"].sum()),
                "voos_validos": validos,
                "atrasados": int(d["atrasado"].sum()),
                "sem_horario_previsto": int(d["chegada_prevista"].isna().sum()),
            }
        )
    return linhas


def obter(ano: int, mes: int) -> bytes | None:
    local = RAW_DIR / f"VRA_{ano}{mes:02d}.csv"
    if local.exists():
        return local.read_bytes()
    for _ in range(3):
        try:
            r = requests.get(url_mes(ano, mes), timeout=180)
            if r.status_code == 404 or len(r.content) < 10_000:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            continue
    raise RuntimeError(f"falha ao baixar {ano}-{mes:02d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inicio", default="2015-01")
    ap.add_argument("--fim", default="2026-12")
    args = ap.parse_args()

    feitos = pd.read_parquet(SAIDA) if SAIDA.exists() else pd.DataFrame()
    ja = set(feitos["mes"]) if len(feitos) else set()
    linhas = []
    for p in pd.period_range(args.inicio, args.fim, freq="M"):
        if str(p) in ja:
            continue
        conteudo = obter(p.year, p.month)
        if conteudo is None:
            print(f"[--] {p}: não publicado")
            continue
        novas = resumir(ler(conteudo), p.year, p.month)
        linhas += novas
        b = novas[0]
        print(
            f"[ok] {p}: {b['voos_programados']:,} voos, atraso {b['atrasados'] / max(1, b['voos_validos']):.1%}",
            flush=True,
        )
        # salva a cada mês para poder retomar
        parcial = pd.concat([feitos, pd.DataFrame(linhas)], ignore_index=True)
        parcial.sort_values(["escopo", "mes"]).to_parquet(SAIDA, index=False)

    serie = pd.read_parquet(SAIDA).sort_values(["escopo", "mes"]).reset_index(drop=True)
    serie["taxa_atraso"] = serie["atrasados"] / serie["voos_validos"].where(serie["voos_validos"] > 0)
    serie["taxa_cancelamento"] = serie["cancelados"] / serie["voos_programados"].where(serie["voos_programados"] > 0)
    serie.to_parquet(SAIDA, index=False)
    destino = REPORTS_DIR / "projecao"
    destino.mkdir(parents=True, exist_ok=True)
    serie.to_csv(destino / "serie_mensal.csv", index=False)
    print(f"{serie['mes'].nunique()} meses em {SAIDA}")


if __name__ == "__main__":
    main()
