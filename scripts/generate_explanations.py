"""
Generate LLM explanations for flagged anomaly records via Ollama.

Adds a `llm_explanation` column to each {CATEGORY}_anomalies.csv.
Only processes rows without an explanation (incremental).

Usage:
  python scripts/generate_explanations.py [--category X] [--limit N]

⚠️  Requires UFSC VPN to reach https://ollama.ceos.ufsc.br
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import httpx
import pandas as pd
import yaml

RESULTS_DIR = "outputs/results"
PARAMS_PATH = "conf/base/parameters.yaml"

_FEATURE_LABELS: dict[str, str] = {
    "valor_adjusted": "valor da despesa muito acima do histórico da categoria",
    "mean_value": "desvio em relação à média histórica deste parlamentar nesta verba",
    "year": "ano atípico da despesa",
    "month": "mês atípico da despesa",
    "quarter": "trimestre atípico da despesa",
    "day_of_week": "dia da semana incomum para esta despesa",
}


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def fmt_brl(value: float) -> str:
    try:
        s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    except (TypeError, ValueError):
        return "—"


def decode_feature(feature_name: str, row: pd.Series) -> str:
    if feature_name.startswith("cta_"):
        return f"identidade do parlamentar ({row.get('conta', '?')})"
    if feature_name.startswith("fav_"):
        fav = row.get("favorecido", row.get("conta", "?"))
        if pd.isna(fav) or str(fav) == "nan":
            fav = row.get("conta", "?")
        return f"beneficiário incomum ({fav})"
    return _FEATURE_LABELS.get(feature_name, feature_name)


def build_prompt(row: pd.Series, verba: str) -> str:
    valor_brl = fmt_brl(row.get("valor_adjusted", 0))
    mean_brl = fmt_brl(row.get("mean_value", 0))
    top_feat = decode_feature(str(row.get("top_feature", "")), row)
    shap_val = abs(float(row.get("top_shap_value", 0)))
    conta = row.get("conta", "?")
    month = int(row.get("month", 0))
    year = int(row.get("year", 0))

    return (
        "Você é um auditor público. Explique em 2 a 3 frases simples e diretas por que esta despesa "
        "parlamentar é estatisticamente anômala, sem usar jargão técnico. Seja objetivo e factual.\n\n"
        "Dados da despesa:\n"
        f"- Parlamentar: {conta}\n"
        f"- Categoria de verba: {verba}\n"
        f"- Valor gasto: {valor_brl}\n"
        f"- Mês/Ano: {month:02d}/{year}\n"
        f"- Média histórica deste parlamentar nesta verba: {mean_brl}\n"
        f"- Principal fator de anomalia: {top_feat} (intensidade: {shap_val:.2f})\n\n"
        "Responda apenas com a explicação em português, sem introdução, saudação ou conclusão. "
        "Não use markdown. Não repita os dados acima — apenas explique a anomalia."
    )


def call_ollama(prompt: str, base_url: str, model: str, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    # Connect timeout is short; read timeout is long (model may need to load on cold start)
    timeouts = httpx.Timeout(connect=15.0, read=float(timeout), write=15.0, pool=5.0)
    try:
        resp = httpx.post(url, json=payload, timeout=timeouts)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.ConnectError:
        raise ConnectionError(
            f"\n❌ Não foi possível conectar ao servidor Ollama em {base_url}.\n"
            "   Ative a VPN da UFSC antes de executar este script."
        )
    except httpx.TimeoutException:
        raise TimeoutError(
            f"\n❌ Timeout após {timeout}s aguardando resposta do servidor Ollama.\n"
            "   O modelo pode estar carregando (cold start). Tente aumentar "
            "`ollama_timeout_seconds` em conf/base/parameters.yaml."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"\n❌ Erro HTTP {e.response.status_code}: {e.response.text[:200]}")


def process_category(
    category: str,
    params: dict,
    limit: int = -1,
) -> tuple[int, int]:
    """Returns (processed, total_flagged)."""
    path = os.path.join(RESULTS_DIR, f"{category}_anomalies.csv")
    if not os.path.exists(path):
        print(f"  [pular] {category}: arquivo não encontrado")
        return 0, 0

    df = pd.read_csv(path, index_col=0)
    flagged_mask = df["ensemble_flag"] == True
    flagged = df[flagged_mask]

    if flagged.empty:
        print(f"  [pular] {category}: sem registros flagged")
        return 0, 0

    if "llm_explanation" not in df.columns:
        df["llm_explanation"] = pd.Series("", index=df.index, dtype=object)
    else:
        df["llm_explanation"] = df["llm_explanation"].astype(object).fillna("")

    needs = flagged[df.loc[flagged.index, "llm_explanation"] == ""]

    if needs.empty:
        already = len(flagged)
        print(f"  [ok] {category}: {already} explicações já geradas")
        return 0, already

    # Process highest audit_score records first: most anomalous AND highest value.
    # Falls back to ensemble_score if audit_score not yet computed.
    import numpy as np
    if "audit_score" not in needs.columns and "ensemble_score" in needs.columns:
        needs = needs.copy()
        needs["audit_score"] = (
            needs["ensemble_score"] * np.log1p(needs["valor_adjusted"].clip(lower=0))
        )
    sort_col = "audit_score" if "audit_score" in needs.columns else "ensemble_score"
    needs = needs.sort_values(sort_col, ascending=False)
    to_process = needs if limit < 0 else needs.head(limit)
    already_done = len(flagged) - len(needs)
    print(f"  {category}: {len(to_process)} a explicar "
          f"({already_done} já prontas, {len(flagged)} total)...")

    base_url = params["ollama_base_url"]
    model = params["ollama_model"]
    timeout = params.get("ollama_timeout_seconds", 60)
    verba = category.replace("_", " ").title()

    processed = 0
    skipped = 0
    save_every = 5  # checkpoint frequency

    for i, (idx, row) in enumerate(to_process.iterrows(), start=1):
        conta_short = str(row.get("conta", "?"))[:25]
        try:
            prompt = build_prompt(row, verba)
            explanation = call_ollama(prompt, base_url, model, timeout)
            df.at[idx, "llm_explanation"] = explanation
            processed += 1
            print(f"    [{i}/{len(to_process)}] {conta_short} ✓")
        except ConnectionError as e:
            # VPN is down — save progress and abort everything
            print(e)
            df.to_csv(path)
            print(f"  Salvo progresso parcial: {path}")
            sys.exit(1)
        except (TimeoutError, RuntimeError) as e:
            # Single record failed — log, skip, keep going
            skipped += 1
            print(f"    [{i}/{len(to_process)}] {conta_short} ✗ (pulado: {type(e).__name__})")
        except Exception as e:
            skipped += 1
            print(f"    [{i}/{len(to_process)}] {conta_short} ✗ (erro: {e})")

        # Checkpoint every N records so partial progress is never lost
        if i % save_every == 0:
            df.to_csv(path)

    df.to_csv(path)
    status = f"{processed} geradas"
    if skipped:
        status += f", {skipped} puladas (timeout/erro)"
    print(f"  Salvo: {path} — {status}")
    return processed, len(flagged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gerar explicações LLM para anomalias ALESC via Ollama"
    )
    parser.add_argument("--category", type=str, default=None,
                        help="Categoria específica (ex: DIARIAS). Padrão: todas.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Máximo de novos registros por categoria (-1 = todos)")
    args = parser.parse_args()

    params = load_params()
    print(f"Servidor Ollama : {params['ollama_base_url']}")
    print(f"Modelo          : {params['ollama_model']}")
    print(f"Timeout         : {params.get('ollama_timeout_seconds', 60)}s")
    print("⚠️  Verifique se a VPN da UFSC está ativa antes de continuar.\n")

    if args.category:
        categories = [args.category.upper()]
    else:
        categories = sorted([
            f.replace("_anomalies.csv", "")
            for f in os.listdir(RESULTS_DIR)
            if f.endswith("_anomalies.csv")
        ])

    log: dict = {
        "started_at": datetime.now().isoformat(),
        "model": params["ollama_model"],
        "categories": {},
    }
    total_processed = 0

    for cat in categories:
        print(f"\n── {cat} ──────────────────────────")
        processed, total = process_category(cat, params, limit=args.limit)
        log["categories"][cat] = {"processed": processed, "total_flagged": total}
        total_processed += processed

    log["finished_at"] = datetime.now().isoformat()
    log["total_processed"] = total_processed

    log_path = os.path.join(RESULTS_DIR, "llm_explanations_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Concluído. {total_processed} novas explicações geradas.")
    print(f"   Log: {log_path}")
    if total_processed > 0:
        print("   Execute generate_rankings.py para atualizar top_anomalies.json com as explicações.")


if __name__ == "__main__":
    main()
