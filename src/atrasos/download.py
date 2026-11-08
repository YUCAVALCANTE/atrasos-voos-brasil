"""Download dos arquivos mensais do VRA (Voo Regular Ativo) do portal de dados abertos da ANAC.

Fonte: https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/historico-de-voos
"""

from __future__ import annotations

import argparse
import time
from urllib.parse import quote

import pandas as pd
import requests

from atrasos.config import PRIMEIRO_MES, RAW_DIR, ULTIMO_MES

BASE_URL = "https://sistemas.anac.gov.br/dadosabertos/" + quote("Voos e operações aéreas/Voo Regular Ativo (VRA)")

MESES = [
    "Janeiro",
    "Fevereiro",
    "Março",
    "Abril",
    "Maio",
    "Junho",
    "Julho",
    "Agosto",
    "Setembro",
    "Outubro",
    "Novembro",
    "Dezembro",
]


def url_mes(ano: int, mes: int) -> str:
    """Monta a URL do CSV de um mês, ex.: .../2025/01 - Janeiro/VRA_20251.csv"""
    pasta = quote(f"{mes:02d} - {MESES[mes - 1]}")
    return f"{BASE_URL}/{ano}/{pasta}/VRA_{ano}{mes}.csv"


def caminho_local(ano: int, mes: int):
    return RAW_DIR / f"VRA_{ano}{mes:02d}.csv"


def baixar_mes(ano: int, mes: int, forcar: bool = False, tentativas: int = 3) -> bool:
    destino = caminho_local(ano, mes)
    if destino.exists() and not forcar:
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = url_mes(ano, mes)
    for i in range(tentativas):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                print(f"[aviso] {ano}-{mes:02d} não publicado ainda ({url})")
                return False
            resp.raise_for_status()
            # Arquivos de meses futuros existem como pastas vazias ou CSV só com cabeçalho.
            if len(resp.content) < 10_000:
                print(f"[aviso] {ano}-{mes:02d} veio vazio ({len(resp.content)} bytes)")
                return False
            destino.write_bytes(resp.content)
            print(f"[ok] {ano}-{mes:02d}: {len(resp.content) / 1e6:.1f} MB")
            return True
        except requests.RequestException as exc:
            espera = 2**i * 5
            print(f"[erro] {ano}-{mes:02d}: {exc} — nova tentativa em {espera}s")
            time.sleep(espera)
    return False


def baixar_periodo(inicio: str = PRIMEIRO_MES, fim: str = ULTIMO_MES, forcar: bool = False) -> list[str]:
    baixados = []
    for periodo in pd.period_range(inicio, fim, freq="M"):
        if baixar_mes(periodo.year, periodo.month, forcar=forcar):
            baixados.append(str(periodo))
    return baixados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", default=PRIMEIRO_MES)
    parser.add_argument("--fim", default=ULTIMO_MES)
    parser.add_argument("--forcar", action="store_true", help="baixa de novo mesmo se já existir")
    args = parser.parse_args()
    meses = baixar_periodo(args.inicio, args.fim, args.forcar)
    print(f"{len(meses)} meses disponíveis em {RAW_DIR}")


if __name__ == "__main__":
    main()
