"""Streamlit dashboard: 7 pages for ALESC anomaly detection exploration."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def fmt_brl(value: float) -> str:
    """Format a float as Brazilian Real: R$ 1.234,56"""
    try:
        # e.g. 33178.50 → "33.178,50" → "R$ 33.178,50"
        formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {formatted}"
    except (TypeError, ValueError):
        return "—"


# ── CEOS-branded header (mirrors painel-ceos/paineis-frontend header.tsx) ─────

_CEOS_LOGO_PATH = Path(__file__).parent / "assets" / "ceos_navbar.svg"
_CEOS_BG_GRADIENT = (
    "linear-gradient(to right, #073C62 0%, #0D476A 30%, #1F3D50 60%, #3A2F24 100%)"
)
_CEOS_ACCENT_BAR = "#F59E0B"          # same orange as the Licitações painel
_CEOS_LIGHT_BLUE = "rgb(19,230,251)"  # CEOS button accent (lightBlueButton const)


def render_header() -> None:
    """Render the CEOS-branded banner at the top of every page."""
    logo_b64 = base64.b64encode(_CEOS_LOGO_PATH.read_bytes()).decode() if _CEOS_LOGO_PATH.exists() else ""
    logo_html = (
        f'<img src="data:image/svg+xml;base64,{logo_b64}" style="height:50px;width:auto;"/>'
        if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div style="
            background: {_CEOS_BG_GRADIENT};
            border-bottom: 3px solid {_CEOS_ACCENT_BAR};
            height: 80px;
            padding: 0 1.5rem;
            margin: 0 -1rem 1rem -1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        ">
          {logo_html}
          <div style="line-height: 1.2;">
            <div style="font-size: 1.25rem; font-weight: 700; color: {_CEOS_LIGHT_BLUE};">
              Painel CEOS
            </div>
            <div style="font-size: 0.9rem; color: #f9fafb99;">
              Detecção de Anomalias — Despesas Parlamentares ALESC
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# SOD is excluded from this run (mirrors the paper's 5-model ensemble exactly)
SOD_COLS = ["score_sod", "anomaly_sod"]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ALESC — Detecção de Anomalias",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CEOS banner is rendered conditionally *after* page selection (see below) —
# the Home page stays banner-free so the layout is uncluttered for the
# landing experience.

# ── Temporal split constants (mirrors conf/base/parameters.yaml) ──────────────

TRAIN_YEARS = list(range(2011, 2020)) + [2021, 2022]
TEST_YEARS = [2023, 2024, 2025]
INFER_YEARS = [2026]
EXCLUDED_YEARS = [2020]

YEAR_SPLIT_LABEL = {
    **{y: "Treinamento" for y in TRAIN_YEARS},
    **{y: "Avaliação" for y in TEST_YEARS},
    **{y: "Inferência" for y in INFER_YEARS},
    **{y: "Excluído (COVID)" for y in EXCLUDED_YEARS},
}

SPLIT_COLORS = {
    "Treinamento": "#4C72B0",
    "Avaliação": "#E07B54",
    "Inferência": "#2CA02C",
    "Excluído (COVID)": "#999999",
}


# ── Data loaders (cached) ─────────────────────────────────────────────────────

import unicodedata


def _normalize_verba(s: str) -> str:
    """Normalize verba string to match category key: uppercase, no accents, spaces→_."""
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    return s.upper().replace(" ", "_").replace("/", "_")


@st.cache_data(show_spinner=False)
def load_intermediate() -> pd.DataFrame:
    path = "data/02_intermediate/expenses.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_category_data(cat: str) -> pd.DataFrame:
    """All records for a category from intermediate, with split labels and month."""
    df = load_intermediate()
    if df.empty:
        return pd.DataFrame()
    mask = df["verba"].apply(_normalize_verba) == cat
    out = df[mask].copy()
    out["split"] = out["year"].map(YEAR_SPLIT_LABEL).fillna("Outro")
    # month is not in intermediate — derive from vencimento date column
    if "month" not in out.columns and "vencimento" in out.columns:
        out["month"] = pd.to_datetime(out["vencimento"], errors="coerce").dt.month
    elif "month" not in out.columns:
        out["month"] = 0
    return out


@st.cache_data(show_spinner=False)
def load_all_metrics() -> list[dict]:
    path = "outputs/results/all_metrics.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_all_flagged() -> pd.DataFrame:
    """All flagged records across all categories, sorted by ensemble_score desc."""
    frames = []
    for cat in available_categories():
        df = load_anomalies(cat)
        if df.empty or "ensemble_flag" not in df.columns:
            continue
        flagged = df[df["ensemble_flag"] == 1].copy()
        if not flagged.empty:
            flagged["category"] = cat
            frames.append(flagged)
    if not frames:
        return pd.DataFrame()
    import numpy as np
    all_df = pd.concat(frames, ignore_index=True)

    # ── Materiality-weighted audit score ──────────────────────────────────────
    # audit_score = ensemble_score × log1p(valor_adjusted)
    # Prioritizes cases that are BOTH statistically anomalous AND high-value.
    # Used for auditor ranking only; ensemble_score remains the model metric.
    all_df["audit_score"] = (
        all_df["ensemble_score"] * np.log1p(all_df["valor_adjusted"].clip(lower=0))
    )

    all_df = all_df.sort_values("audit_score", ascending=False)
    all_df["global_rank"] = range(1, len(all_df) + 1)
    all_df["category_rank"] = (
        all_df.groupby("category")["audit_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    if "llm_explanation" not in all_df.columns:
        all_df["llm_explanation"] = ""
    else:
        all_df["llm_explanation"] = all_df["llm_explanation"].astype(object).fillna("")
    return all_df


@st.cache_data(show_spinner=False)
def load_anomalies(category: str) -> pd.DataFrame:
    path = f"outputs/results/{category}_anomalies.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def available_categories() -> list[str]:
    results_dir = "outputs/results"
    if not os.path.isdir(results_dir):
        return []
    return sorted([
        f.replace("_anomalies.csv", "")
        for f in os.listdir(results_dir)
        if f.endswith("_anomalies.csv")
    ])


# ── Feature name decoder ──────────────────────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "valor_adjusted": "Valor corrigido pela inflação (R$ Jan/2026)",
    "mean_value": "Gasto médio histórico do parlamentar nesta verba",
    "year": "Ano da despesa",
    "month": "Mês da despesa (1–12)",
    "quarter": "Trimestre (1–4)",
    "day_of_week": "Dia da semana (0=seg, 6=dom)",
    "is_reversal": "É uma devolução/estorno",
}

_FEATURE_GLOSSARY_MD = """
| Feature | Tipo | Descrição |
|---------|------|-----------|
| `cta_XX` | Hash do parlamentar | Bucket XX do hash trick para o campo **Conta** (nome do parlamentar). Indica que *quem* gastou é o fator determinante. |
| `fav_XX` | Hash do favorecido | Bucket XX do hash trick para o campo **Favorecido** (beneficiário do pagamento). Indica que *para quem* se pagou é o fator determinante. |
| `mean_value` | Média histórica | Gasto médio deste parlamentar nesta verba, calculado apenas no conjunto de treino (2011–2022). Valor acima da média histórica é sinal de anomalia. |
| `valor_adjusted` | Valor IPCA | Valor da despesa corrigido pela inflação (IPCA, base Jan/2026). Gastos muito acima do habitual para a categoria. |
| `year` | Temporal | Ano em que a despesa ocorreu. Anomalias temporais (anos atípicos). |
| `month` | Temporal | Mês da despesa (1–12). Padrões sazonais incomuns. |
| `quarter` | Temporal | Trimestre (1–4). Concentração atípica de gastos em um trimestre. |
| `day_of_week` | Temporal | Dia da semana (0=segunda, 6=domingo). Despesas em dias atípicos. |
| `is_reversal` | Flag | Indica devolução ou estorno. |
"""


def decode_feature(feature_name: str, row: pd.Series) -> str:
    """Return a human-readable explanation of a SHAP top feature for a specific record."""
    if feature_name.startswith("cta_"):
        conta = row.get("conta", "?")
        return f"Parlamentar (hash): **{conta}**"
    if feature_name.startswith("fav_"):
        fav = row.get("favorecido", "?")
        if pd.isna(fav) or fav == "nan":
            fav = row.get("conta", "?")
        return f"Favorecido (hash): **{fav}**"
    return _FEATURE_LABELS.get(feature_name, feature_name)


def add_decoded_feature_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'feature_decoded' column to a flagged records DataFrame."""
    if "top_feature" not in df.columns:
        return df
    df = df.copy()
    df["feature_decoded"] = [
        decode_feature(row["top_feature"], row) for _, row in df.iterrows()
    ]
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────

if st.sidebar.button("🔄 Atualizar dados", help="Limpa o cache e recarrega os resultados do pipeline"):
    st.cache_data.clear()
    st.rerun()

PAGES = [
    "🏠 Início",
    "📊 Explorar Dados",
    "🔬 Análise Exploratória",
    "🚨 Anomalias Detectadas",
    "🏆 Ranking",
    "🔍 Auditoria",
    "📈 Tendência Temporal",
    "🧠 Explicabilidade",
]

page = st.sidebar.radio("Navegação", PAGES)

# CEOS-branded banner on every page (Home included).
render_header()

# ── Page: Home ────────────────────────────────────────────────────────────────

if page == "🏠 Início":
    st.title("Painel de Detecção de Anomalias — ALESC")
    st.markdown(
        "**Detecção de anomalias em despesas parlamentares da ALESC (2011–2026)**  \n"
        "Ensemble não supervisionado: **IForest + KNN + LOF + GMM + OCSVM** (5 modelos — replicação do paper) "
        "com votação por **consenso estrito** (todos os modelos devem concordar)."
    )

    # ── Temporal split overview ───────────────────────────────────────────────
    st.subheader("Divisão Temporal dos Dados")
    split_cols = st.columns(4)
    split_cols[0].metric(
        "Treinamento",
        f"{len(TRAIN_YEARS)} anos",
        f"2011–2019, 2021–2022",
        help="Anos usados para ajustar os modelos de detecção de anomalias.",
    )
    split_cols[1].metric(
        "Avaliação (teste)",
        "3 anos",
        "2023 · 2024 · 2025",
        help="Anos usados para calcular scores e detectar anomalias. Modelos NÃO viram esses dados no treino.",
    )
    split_cols[2].metric(
        "Inferência",
        "1 ano",
        "2026 (parcial)",
        help="Dados mais recentes, apenas inferência.",
    )
    split_cols[3].metric(
        "Excluído",
        "2020",
        "COVID-19",
        help="2020 excluído do treino por ser outlier estrutural (pandemia).",
    )

    # Records per year bar chart
    df_int = load_intermediate()
    if not df_int.empty:
        year_counts = (
            df_int.groupby("year").size().reset_index(name="n_records")
        )
        year_counts["split"] = year_counts["year"].map(YEAR_SPLIT_LABEL).fillna("Outro")
        fig_years = px.bar(
            year_counts, x="year", y="n_records", color="split",
            color_discrete_map=SPLIT_COLORS,
            title="Registros por Ano — Divisão Treinamento / Avaliação / Inferência",
            labels={"n_records": "Registros", "year": "Ano", "split": "Conjunto"},
        )
        fig_years.update_layout(bargap=0.1, legend_title_text="Conjunto")
        st.plotly_chart(fig_years, use_container_width=True)

    # ── Pipeline results ──────────────────────────────────────────────────────
    metrics = load_all_metrics()
    if metrics:
        st.subheader("Resultados do Pipeline")
        cols = st.columns(4)
        total_records = sum(m.get("n_records", 0) for m in metrics)
        total_flagged = sum(m.get("n_flagged_ensemble", 0) for m in metrics)
        cols[0].metric("Categorias modeladas", len(metrics))
        cols[1].metric("Registros avaliados", f"{total_records:,}")
        cols[2].metric("Anomalias detectadas", f"{total_flagged:,}")
        cols[3].metric(
            "Taxa de flagging (consenso)",
            f"{100*total_flagged/max(total_records,1):.3f}%",
            help="Consenso estrito: muito menor que baseline IQR. Alta precisão, menor recall.",
        )

        df_m = pd.DataFrame(metrics).sort_values("n_flagged_ensemble", ascending=False)
        df_m["reducao_vs_iqr"] = (
            (df_m["iqr_baseline_pct"] - df_m["flagged_pct"])
            / df_m["iqr_baseline_pct"].clip(lower=0.001) * 100
        ).round(1).astype(str) + "%"
        st.subheader("Anomalias por Categoria de Verba")
        st.dataframe(
            df_m[["category", "n_records", "n_flagged_ensemble", "flagged_pct",
                  "iqr_baseline_pct", "reducao_vs_iqr"]].rename(columns={
                "category": "Categoria",
                "n_records": "Registros avaliados",
                "n_flagged_ensemble": "Sinalizados (consenso)",
                "flagged_pct": "Taxa (%)",
                "iqr_baseline_pct": "Baseline IQR (%)",
                "reducao_vs_iqr": "Redução vs IQR",
            }),
            use_container_width=True,
        )

        with st.expander("Como interpretar a tabela"):
            st.markdown(
                "**Taxa (%)**: proporção dos registros de avaliação (2023–2025) que o ensemble marcou como anômalos.  \n"
                "**Baseline IQR (%)**: proporção de outliers detectados por um simples método IQR no conjunto de treino — "
                "serve como referência ingênua.  \n"
                "**Redução vs IQR**: o ensemble consensual é muito mais preciso; quanto maior a redução, mais seletivo o modelo."
            )
    else:
        st.info("Pipeline não executado. Use: `python -m alesc_poc.pipeline_runner`")

    st.subheader("Logs recentes")
    log_path = "outputs/pipeline.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()[-30:]
        st.code("".join(lines), language="text")

# ── Page: Data Explorer ───────────────────────────────────────────────────────

elif page == "📊 Explorar Dados":
    st.title("Explorar Dados")
    df = load_intermediate()
    if df.empty:
        st.warning("Rode o pipeline de ingestão primeiro: `python -m alesc_poc.pipeline_runner`")
        st.stop()

    # Quick dataset overview
    with st.expander("Visão geral do dataset", expanded=True):
        ov_cols = st.columns(4)
        ov_cols[0].metric("Total de registros", f"{len(df):,}")
        ov_cols[1].metric("Anos disponíveis", f"{int(df['year'].min())}–{int(df['year'].max())}")
        ov_cols[2].metric("Parlamentares distintos", f"{df['conta'].nunique():,}")
        ov_cols[3].metric("Categorias de verba", f"{df['verba'].nunique():,}")

        # Records per category
        cat_counts = df.groupby("verba").size().reset_index(name="registros").sort_values("registros", ascending=False)
        fig_cat = px.bar(
            cat_counts, x="verba", y="registros",
            title="Registros por Categoria de Verba",
            labels={"verba": "Verba", "registros": "Registros"},
        )
        fig_cat.update_xaxes(tickangle=30)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    years = sorted(df["year"].dropna().unique().tolist())
    sel_years = col1.multiselect("Anos", years, default=years[-3:])
    verbas = sorted(df["verba"].dropna().unique().tolist())
    sel_verba = col2.multiselect("Verba", verbas, default=verbas[:3])
    contas = sorted(df["conta"].dropna().unique().tolist())
    sel_conta = col3.multiselect("Parlamentar (Conta)", contas)

    filtered = df.copy()
    if sel_years:
        filtered = filtered[filtered["year"].isin(sel_years)]
    if sel_verba:
        filtered = filtered[filtered["verba"].isin(sel_verba)]
    if sel_conta:
        filtered = filtered[filtered["conta"].isin(sel_conta)]

    st.write(f"Mostrando **{len(filtered):,}** registros (máx. 500 exibidos)")
    disp = filtered[["year", "verba", "descricao", "conta", "favorecido", "valor", "is_reversal"]].head(500).copy()
    disp["valor"] = disp["valor"].apply(fmt_brl)
    st.dataframe(disp, use_container_width=True)

# ── Page: EDA ─────────────────────────────────────────────────────────────────

elif page == "🔬 Análise Exploratória":
    st.title("Análise Exploratória — Perfis EDA")
    reports_dir = "outputs/reports"

    # Show EDA summary JSON if available
    eda_summary_path = "outputs/results/eda_all_summary.json"
    if os.path.exists(eda_summary_path):
        with open(eda_summary_path) as f:
            eda_summary = json.load(f)

        # Top-level global stats
        g_cols = st.columns(4)
        g_cols[0].metric("Total de registros", f"{eda_summary.get('total_records', 0):,}")
        g_cols[1].metric("Categorias", eda_summary.get("n_categories", 0))
        g_cols[2].metric("Anos", f"{eda_summary.get('year_min')}–{eda_summary.get('year_max')}")
        g_cols[3].metric("Favorecido nulo (global)", f"{eda_summary.get('overall_null_favorecido_pct', 0):.1f}%")

        # Per-category summary table
        cats_data = eda_summary.get("categories", [])
        if cats_data:
            df_eda = pd.DataFrame([
                {
                    "Categoria": c["category"],
                    "Registros": c["n_records"],
                    "Anos": f"{c['year_min']}–{c['year_max']}",
                    "Valor médio": fmt_brl(c["valor_mean"]),
                    "Valor mediana": fmt_brl(c["valor_median"]),
                    "Fav. nulo (%)": f"{c['null_favorecido_pct']:.1f}%",
                    "Devoluções (%)": f"{c['reversal_rate_pct']:.1f}%",
                    "Top parlamentar": c["top_conta"][0]["conta"] if c.get("top_conta") else "—",
                }
                for c in cats_data
            ])
            st.subheader("Resumo por Categoria")
            st.dataframe(df_eda, use_container_width=True)

    if not os.path.isdir(reports_dir):
        st.info("Sem relatórios EDA. Execute `/eda` no Claude Code.")
        st.stop()

    html_files = sorted([f for f in os.listdir(reports_dir) if f.startswith("eda_") and f.endswith(".html")])
    if not html_files:
        st.info("Sem arquivos HTML em outputs/reports/. Execute `/eda` para gerar.")
        st.stop()

    selected = st.selectbox("Relatório", html_files)
    with open(os.path.join(reports_dir, selected)) as f:
        html = f.read()
    st.components.v1.html(html, height=800, scrolling=True)

# ── Page: Anomaly Results ─────────────────────────────────────────────────────

elif page == "🚨 Anomalias Detectadas":
    st.title("Anomalias Detectadas")
    cats = available_categories()
    if not cats:
        st.warning("Sem resultados. Execute o pipeline.")
        st.stop()

    cat = st.selectbox("Categoria", cats)
    df_anom = load_anomalies(cat)

    if df_anom.empty:
        st.info(f"Sem arquivo de anomalias para {cat}")
        st.stop()

    # Score scatter
    if "valor_adjusted" in df_anom.columns and "ensemble_score" in df_anom.columns:
        fig = px.scatter(
            df_anom,
            x=df_anom.index,
            y="valor_adjusted",
            color="ensemble_flag",
            color_discrete_map={0: "#4C72B0", 1: "#E07B54"},
            size=df_anom["ensemble_score"].clip(lower=0.01),
            hover_data=["conta", "year", "ensemble_score"],
            title=f"Scatter de Anomalias — {cat}",
            labels={"valor_adjusted": "Valor Corrigido (R$ Jan/2026)", "x": "Índice do Registro"},
        )
        st.plotly_chart(fig, use_container_width=True)

    flagged = df_anom[df_anom["ensemble_flag"] == 1]
    st.metric("Registros sinalizados (consenso)", len(flagged))

    if not flagged.empty:
        flagged_decoded = add_decoded_feature_col(flagged)
        display_cols = [c for c in ["conta", "favorecido", "valor_adjusted", "year",
                                    "ensemble_score", "feature_decoded", "top_shap_value"]
                        if c in flagged_decoded.columns]
        disp = flagged_decoded[display_cols].sort_values("ensemble_score", ascending=False).head(100).copy()
        disp["valor_adjusted"] = disp["valor_adjusted"].apply(fmt_brl)
        st.dataframe(
            disp.rename(columns={
                "conta": "Parlamentar",
                "favorecido": "Favorecido",
                "valor_adjusted": "Valor (R$ Jan/2026)",
                "year": "Ano",
                "ensemble_score": "Score",
                "feature_decoded": "Principal fator SHAP",
                "top_shap_value": "Valor SHAP",
            }),
            use_container_width=True,
        )

    # Boxplot image
    box_path = f"outputs/plots/boxplot_{cat}.png"
    if os.path.exists(box_path):
        st.image(box_path, caption=f"Distribuição de Valores — {cat}")

# ── Page: Leaderboard ─────────────────────────────────────────────────────────

elif page == "🏆 Ranking":
    st.title("Ranking — Parlamentares com Mais Anomalias")
    cats = available_categories()
    if not cats:
        st.warning("Sem resultados.")
        st.stop()

    frames = []
    for cat in cats:
        df_a = load_anomalies(cat)
        if not df_a.empty and "conta" in df_a.columns:
            flagged = df_a[df_a["ensemble_flag"] == 1][["conta", "ensemble_score", "valor_adjusted"]].copy()
            flagged["category"] = cat
            frames.append(flagged)

    if not frames:
        st.info("Nenhum registro sinalizado encontrado.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)
    by_conta = (
        combined.groupby("conta")
        .agg(
            total_flags=("ensemble_score", "count"),
            avg_score=("ensemble_score", "mean"),
            total_valor=("valor_adjusted", "sum"),
            categorias=("category", lambda x: ", ".join(sorted(x.unique()))),
        )
        .sort_values("total_flags", ascending=False)
        .reset_index()
    )
    by_conta["total_valor_fmt"] = by_conta["total_valor"].apply(fmt_brl)
    by_conta = by_conta.drop(columns=["total_valor"]).rename(columns={
        "conta": "Parlamentar",
        "total_flags": "Anomalias",
        "avg_score": "Score médio",
        "total_valor_fmt": "Total (R$ Jan/2026)",
        "categorias": "Categorias",
    })
    st.dataframe(by_conta.head(30), use_container_width=True)

    fig = px.bar(
        by_conta.head(20), x="Parlamentar", y="Anomalias",
        color="Score médio", color_continuous_scale="Reds",
        title="Top 20 Parlamentares por Número de Anomalias Detectadas",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ── Page: Auditoria ───────────────────────────────────────────────────────────

elif page == "🔍 Auditoria":
    st.title("Painel do Auditor")
    st.markdown(
        "Casos ordenados pelo **audit score** — combinação de desvio estatístico e valor monetário.  \n"
        "Explicações em linguagem simples são geradas via LLM — execute "
        "`python scripts/generate_explanations.py` após ativar a VPN da UFSC."
    )
    with st.expander("ℹ️ Como funciona o Audit Score?"):
        st.markdown(
            "### Audit Score = ensemble\\_score × log1p(valor)\n\n"
            "O `ensemble_score` mede o **desvio estatístico** de uma despesa em relação ao "
            "histórico — mas não diferencia uma anomalia de R\\$ 200 de uma de R\\$ 50.000 "
            "com o mesmo perfil de desvio.\n\n"
            "Para **priorizar o trabalho do auditor**, multiplicamos pelo logaritmo do valor "
            "(`log1p` comprime a escala para que um único outlier de R\\$ 1M não domine a lista):\n\n"
            "| Valor | log1p | ensemble = 0,5 → audit\\_score |\n"
            "|------:|------:|-------------------------------:|\n"
            "| R\\$ 200 | 5,3 | **2,6** |\n"
            "| R\\$ 5.000 | 8,5 | **4,3** |\n"
            "| R\\$ 50.000 | 10,8 | **5,4** |\n\n"
            "> **Importante:** o `ensemble_score` continua sendo a métrica do modelo "
            "(reportada no artigo). O `audit_score` é exclusivamente uma ferramenta de "
            "**priorização** — não altera nem substitui o resultado estatístico."
        )

    all_flagged = load_all_flagged()
    if all_flagged.empty:
        st.warning("Sem registros sinalizados. Execute o pipeline primeiro.")
        st.stop()

    tab_rank, tab_profile, tab_data = st.tabs(["🎯 Casos Rankeados", "📊 Perfil por Categoria", "📋 Dados Completos"])

    # ── Tab: Ranked cases ─────────────────────────────────────────────────────
    with tab_rank:
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        top_n_opts = [10, 25, 50, 100]
        top_n = ctrl1.selectbox("Quantos casos mostrar", top_n_opts, index=1)
        cat_opts = ["Todas"] + available_categories()
        cat_filter = ctrl2.selectbox("Filtrar por categoria", cat_opts)
        has_explanation = ctrl3.checkbox("Apenas com explicação LLM", value=False)

        display_df = all_flagged.copy()
        if cat_filter != "Todas":
            display_df = display_df[display_df["category"] == cat_filter]
        if has_explanation and "llm_explanation" in display_df.columns:
            display_df = display_df[display_df["llm_explanation"].fillna("") != ""]
        display_df = display_df.head(top_n)

        if display_df.empty:
            if has_explanation:
                st.info(
                    "Nenhum registro com explicação LLM encontrado.  \n"
                    "Clique **🔄 Atualizar dados** na barra lateral após gerar as explicações."
                )
            else:
                st.info("Nenhum caso encontrado com os filtros selecionados.")
            st.stop()

        # Normalize audit_score for progress bar
        max_audit = float(all_flagged["audit_score"].max()) or 1.0

        for _, rec in display_df.iterrows():
            global_rank = int(rec.get("global_rank", 0))
            cat_rank = int(rec.get("category_rank", 1))
            cat = rec.get("category", "?")
            conta = rec.get("conta", "?")
            fav = rec.get("favorecido", rec.get("conta", "?"))
            if pd.isna(fav) or str(fav) == "nan":
                fav = conta
            valor = fmt_brl(rec.get("valor_adjusted", 0))
            month = int(rec.get("month", 0))
            year = int(rec.get("year", 0))
            audit_score = float(rec.get("audit_score", 0))
            ensemble_score = float(rec.get("ensemble_score", 0))
            top_feat = str(rec.get("top_feature", ""))
            top_shap = float(rec.get("top_shap_value", 0))
            mean_val = fmt_brl(rec.get("mean_value", 0))
            explanation = str(rec.get("llm_explanation", "")).strip()

            with st.container(border=True):
                header_cols = st.columns([1, 6, 3])
                header_cols[0].markdown(
                    f"<div style='font-size:2rem;font-weight:bold;color:#E07B54;'>"
                    f"#{global_rank}</div>",
                    unsafe_allow_html=True,
                )
                header_cols[1].markdown(
                    f"**{conta}**  \n"
                    f"`{cat}` · {valor} · {month:02d}/{year}"
                )
                header_cols[2].markdown(
                    f"**Audit score:** `{audit_score:.3f}`  \n"
                    f"Anomalia: `{ensemble_score:.4f}` · Rank #{cat_rank}"
                )
                st.progress(
                    min(audit_score / max_audit, 1.0),
                    text=f"Audit score: {audit_score:.3f}  (ensemble: {ensemble_score:.4f} × log1p(valor))",
                )

                if explanation:
                    st.info(f"💬 {explanation}")
                else:
                    st.caption(
                        "Explicação LLM não disponível. "
                        "Execute: `python scripts/generate_explanations.py` (com VPN ativa)."
                    )

                detail_col, wf_col = st.columns(2)
                with detail_col:
                    with st.expander("Ver detalhes SHAP"):
                        feat_decoded = decode_feature(top_feat, rec)
                        st.markdown(f"**Principal fator:** {feat_decoded}")
                        st.markdown(
                            f"**Código da feature:** `{top_feat}`  \n"
                            f"**Intensidade SHAP:** `{abs(top_shap):.4f}`  \n"
                            f"**Média histórica (Parlamentar + Verba):** {mean_val}"
                        )

                with wf_col:
                    wf_path = f"outputs/plots/{cat}/waterfall_rank{cat_rank:03d}.png"
                    with st.expander("📷 Waterfall SHAP"):
                        if os.path.exists(wf_path):
                            st.image(wf_path, caption=f"SHAP Waterfall — {cat} rank #{cat_rank}")
                        else:
                            # Find the highest available rank for this category
                            wf_dir = f"outputs/plots/{cat}"
                            max_wf = 0
                            if os.path.isdir(wf_dir):
                                wf_files = [f for f in os.listdir(wf_dir) if f.startswith("waterfall_rank")]
                                if wf_files:
                                    max_wf = max(int(f.replace("waterfall_rank", "").replace(".png", "")) for f in wf_files)
                            if max_wf > 0:
                                st.caption(
                                    f"Waterfall disponível apenas para os top {max_wf} de cada categoria. "
                                    f"Este registro é rank #{cat_rank} em {cat}."
                                )
                            else:
                                st.caption(f"Waterfall não gerado para `{cat}`. Re-execute o pipeline.")

                with st.expander("📋 Dados da despesa"):
                    _DOW = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta",
                            4: "Sexta", 5: "Sábado", 6: "Domingo"}
                    _SKIP = {"ensemble_flag", "global_rank", "category_rank", "audit_score",
                             "llm_explanation", "ensemble_score", "top_feature", "top_shap_value"}
                    _LABELS = {
                        "conta": "Parlamentar", "favorecido": "Favorecido",
                        "valor_adjusted": "Valor (R$ Jan/2026)", "year": "Ano",
                        "month": "Mês", "quarter": "Trimestre", "day_of_week": "Dia da semana",
                        "mean_value": "Média histórica", "verba": "Verba",
                        "descricao": "Descrição", "is_reversal": "É devolução",
                        "category": "Categoria",
                    }
                    rows = []
                    for k, v in rec.items():
                        if k in _SKIP or str(k).startswith(("cta_", "fav_", "score_", "anomaly_")):
                            continue
                        if k not in _LABELS:
                            continue
                        try:
                            if pd.isna(v):
                                continue
                        except (TypeError, ValueError):
                            pass
                        if k == "day_of_week":
                            v = _DOW.get(int(v), str(v))
                        elif k in ("valor_adjusted", "mean_value"):
                            v = fmt_brl(v)
                        elif isinstance(v, float):
                            v = round(v, 2)
                        rows.append({"Campo": _LABELS[k], "Valor": str(v)})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab: Category profiles ────────────────────────────────────────────────
    with tab_profile:
        st.markdown(
            "Distribuição histórica dos gastos por categoria, com as anomalias destacadas. "
            "Gerado ao executar o pipeline completo."
        )
        cats = available_categories()
        if not cats:
            st.info("Sem categorias disponíveis.")
            st.stop()

        prof_cat = st.selectbox("Categoria", cats, key="profile_cat_select")

        df_cat = load_category_data(prof_cat)
        df_anom_cat = load_anomalies(prof_cat)

        if df_cat.empty:
            st.warning("Dados intermediários não encontrados. Execute o pipeline de ingestão.")
        else:
            train_df = df_cat[df_cat["year"].isin(TRAIN_YEARS)]
            eval_normal = df_cat[
                df_cat["year"].isin(TEST_YEARS) &
                ~df_cat.index.isin(df_anom_cat.index if not df_anom_cat.empty else [])
            ]

            # ── Chart 1: Distribution histogram ───────────────────────────────
            st.subheader("Distribuição de Valores")

            hist_frames = []
            for label, sub in [
                ("Treino 2011–2022", train_df),
                ("Normal 2023–2025", eval_normal),
            ]:
                if not sub.empty:
                    hist_frames.append(
                        sub[["valor_adjusted"]].assign(conjunto=label)
                    )
            if not df_anom_cat.empty and "valor_adjusted" in df_anom_cat.columns:
                hist_frames.append(
                    df_anom_cat[["valor_adjusted"]].assign(conjunto="Anômalos")
                )

            if hist_frames:
                hist_df = pd.concat(hist_frames, ignore_index=True)
                fig_hist = px.histogram(
                    hist_df,
                    x="valor_adjusted",
                    color="conjunto",
                    barmode="overlay",
                    opacity=0.6,
                    histnorm="probability density",
                    nbins=80,
                    color_discrete_map={
                        "Treino 2011–2022": "#4C72B0",
                        "Normal 2023–2025": "#2CA02C",
                        "Anômalos": "#E07B54",
                    },
                    labels={"valor_adjusted": "Valor Corrigido (R$ Jan/2026)", "conjunto": "Conjunto"},
                    title=f"Distribuição de Valores — {prof_cat}",
                )
                fig_hist.update_layout(bargap=0.02)
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Chart 2: Boxplot per year + anomaly overlay ───────────────────
            st.subheader("Distribuição por Ano (avaliação 2023–2025)")

            if not eval_normal.empty or not df_anom_cat.empty:
                import plotly.graph_objects as go

                fig_box = go.Figure()

                # Single box trace — plotly groups by x automatically
                if not eval_normal.empty:
                    fig_box.add_trace(go.Box(
                        x=eval_normal["year"].astype(int).astype(str),
                        y=eval_normal["valor_adjusted"],
                        name="Normal",
                        marker_color="#2CA02C",
                        line_color="#2CA02C",
                        fillcolor="rgba(44,160,44,0.25)",
                        boxpoints=False,
                    ))

                # Anomaly scatter overlaid
                if not df_anom_cat.empty and "valor_adjusted" in df_anom_cat.columns:
                    anom = df_anom_cat.copy()
                    anom["year_str"] = anom["year"].astype(int).astype(str)
                    anom["favorecido"] = anom["favorecido"].fillna(anom["conta"]) if "favorecido" in anom.columns else anom["conta"]
                    anom["mes_ano"] = anom["month"].astype(int).apply(lambda m: f"{m:02d}") + "/" + anom["year_str"]
                    anom["valor_fmt"] = anom["valor_adjusted"].apply(fmt_brl)
                    score_col = anom["ensemble_score"] if "ensemble_score" in anom.columns else pd.Series(0.5, index=anom.index)

                    fig_box.add_trace(go.Scatter(
                        x=anom["year_str"],
                        y=anom["valor_adjusted"],
                        mode="markers",
                        name="Anômalo",
                        marker=dict(
                            color=score_col,
                            colorscale="Reds",
                            size=12,
                            symbol="diamond",
                            line=dict(color="white", width=0.5),
                            colorbar=dict(title="Score", thickness=12, len=0.5, x=1.05),
                            cmin=float(score_col.min()),
                            cmax=float(score_col.max()),
                        ),
                        customdata=list(zip(
                            anom["conta"].fillna("?"),
                            anom["favorecido"].fillna("?"),
                            anom["valor_fmt"],
                            anom["mes_ano"],
                            score_col.round(4),
                        )),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Favorecido: %{customdata[1]}<br>"
                            "Valor: %{customdata[2]}<br>"
                            "Mês/Ano: %{customdata[3]}<br>"
                            "Score: %{customdata[4]}"
                            "<extra></extra>"
                        ),
                    ))

                fig_box.update_layout(
                    title=f"Valor por Ano — {prof_cat}",
                    xaxis=dict(title="Ano", type="category",
                               categoryorder="category ascending"),
                    yaxis_title="Valor (R$ Jan/2026)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    boxmode="overlay",
                    height=480,
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Sem dados suficientes para este gráfico.")

        # Also show per-category top 5 from all_flagged
        st.subheader(f"Top 5 anomalias em {prof_cat}")
        cat_top5 = all_flagged[all_flagged["category"] == prof_cat].head(5)
        if not cat_top5.empty:
            display_top5 = cat_top5[
                [c for c in ["conta", "favorecido", "valor_adjusted", "year",
                              "month", "ensemble_score", "llm_explanation"]
                 if c in cat_top5.columns]
            ].copy()
            if "valor_adjusted" in display_top5.columns:
                display_top5["valor_adjusted"] = display_top5["valor_adjusted"].apply(fmt_brl)
            st.dataframe(
                display_top5.rename(columns={
                    "conta": "Parlamentar", "favorecido": "Favorecido",
                    "valor_adjusted": "Valor", "year": "Ano", "month": "Mês",
                    "ensemble_score": "Score", "llm_explanation": "Explicação LLM",
                }),
                use_container_width=True,
            )

    # ── Tab: Full spending data ───────────────────────────────────────────────
    with tab_data:
        st.markdown(
            "Consulte **todos os registros** de uma categoria para contextualizar as anomalias. "
            "Registros sinalizados são destacados. Filtre por parlamentar para ver o histórico completo."
        )

        data_cat = st.selectbox("Categoria", available_categories(), key="data_cat_select")

        df_all = load_category_data(data_cat)
        df_anom_data = load_anomalies(data_cat)

        if df_all.empty:
            st.warning("Dados intermediários não encontrados. Execute o pipeline de ingestão.")
            st.stop()

        # Merge anomaly flags from results CSV onto full category dataset (join on index)
        if not df_anom_data.empty:
            flag_cols = [c for c in ["ensemble_flag", "ensemble_score"] if c in df_anom_data.columns]
            df_all = df_all.join(df_anom_data[flag_cols], how="left")
        if "ensemble_flag" not in df_all.columns:
            df_all["ensemble_flag"] = 0
        else:
            df_all["ensemble_flag"] = df_all["ensemble_flag"].fillna(0).astype(int)
        if "ensemble_score" not in df_all.columns:
            df_all["ensemble_score"] = 0.0
        else:
            df_all["ensemble_score"] = df_all["ensemble_score"].fillna(0.0)

        # ── Filters ──────────────────────────────────────────────────────────
        fcol1, fcol2, fcol3, fcol4 = st.columns([3, 2, 2, 1])

        contas_all = sorted(df_all["conta"].dropna().unique().tolist())
        sel_contas_data = fcol1.multiselect("Parlamentar", contas_all, key="data_contas")

        years_data = sorted(df_all["year"].dropna().unique().astype(int).tolist())
        sel_years_data = fcol2.multiselect("Ano", years_data, key="data_years")

        split_opts = sorted(df_all["split"].unique().tolist()) if "split" in df_all.columns else []
        sel_split_data = fcol3.multiselect("Conjunto", split_opts, key="data_split")

        only_flagged_data = fcol4.checkbox("Só anomalias", key="data_flagged")

        filtered_data = df_all.copy()
        if sel_contas_data:
            filtered_data = filtered_data[filtered_data["conta"].isin(sel_contas_data)]
        if sel_years_data:
            filtered_data = filtered_data[filtered_data["year"].isin(sel_years_data)]
        if sel_split_data:
            filtered_data = filtered_data[filtered_data["split"].isin(sel_split_data)]
        if only_flagged_data:
            filtered_data = filtered_data[filtered_data["ensemble_flag"] == 1]

        # ── Summary metrics ───────────────────────────────────────────────────
        val_col = "valor_adjusted" if "valor_adjusted" in filtered_data.columns else "valor"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros", f"{len(filtered_data):,}")
        m2.metric("Total gasto", fmt_brl(filtered_data[val_col].sum()) if val_col in filtered_data.columns else "—")
        m3.metric("Anomalias", f"{int(filtered_data['ensemble_flag'].sum()):,}")
        flag_rate = 100 * filtered_data["ensemble_flag"].mean() if len(filtered_data) > 0 else 0
        m4.metric("Taxa anomalias", f"{flag_rate:.1f}%")

        # ── Spend-by-parliamentarian bar (only when no single conta selected) ──
        if not sel_contas_data and len(filtered_data) > 0 and val_col in filtered_data.columns:
            top_contas = (
                filtered_data.groupby("conta")[val_col]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )
            fig_spend = px.bar(
                top_contas, x="conta", y=val_col,
                title=f"Top 20 parlamentares por gasto total — {data_cat}",
                labels={"conta": "Parlamentar", val_col: "Total gasto (R$ Jan/2026)"},
                color=val_col, color_continuous_scale="Blues",
            )
            fig_spend.update_xaxes(tickangle=40)
            fig_spend.update_coloraxes(showscale=False)
            st.plotly_chart(fig_spend, use_container_width=True)

        # ── Data table ────────────────────────────────────────────────────────
        display_cols_data = [c for c in [
            "conta", "favorecido", val_col, "year", "month",
            "split", "ensemble_flag", "ensemble_score", "descricao",
        ] if c in filtered_data.columns]

        disp_data = filtered_data[display_cols_data].copy()

        if val_col in disp_data.columns:
            disp_data[val_col] = disp_data[val_col].apply(fmt_brl)
        if "ensemble_score" in disp_data.columns:
            disp_data["ensemble_score"] = disp_data["ensemble_score"].round(4)
        if "ensemble_flag" in disp_data.columns:
            disp_data["ensemble_flag"] = disp_data["ensemble_flag"].map({1: "⚠️ Anômalo", 0: "—"})

        max_rows = 2000
        st.caption(f"Exibindo até {max_rows:,} de {len(filtered_data):,} registros — use os filtros para refinar.")
        st.dataframe(
            disp_data.head(max_rows).rename(columns={
                "conta": "Parlamentar",
                "favorecido": "Favorecido",
                val_col: "Valor (R$ Jan/2026)",
                "year": "Ano",
                "month": "Mês",
                "split": "Conjunto",
                "ensemble_flag": "Anomalia",
                "ensemble_score": "Score",
                "descricao": "Descrição",
            }),
            use_container_width=True,
            height=500,
        )


# ── Page: Temporal Trend ──────────────────────────────────────────────────────

elif page == "📈 Tendência Temporal":
    st.title("Tendência Temporal — Taxa de Anomalias por Ano")

    st.markdown(
        "Os modelos foram **treinados em 2011–2022** (excl. 2020). "
        "Os scores abaixo se referem apenas ao conjunto de **avaliação (2023–2025)** e **inferência (2026)**."
    )

    trend_path = "outputs/plots/temporal_trend.png"
    if os.path.exists(trend_path):
        st.image(trend_path)
    else:
        st.info("Gráfico de tendência temporal ainda não gerado.")

    cats = available_categories()
    if cats:
        rows = []
        for cat in cats:
            df_a = load_anomalies(cat)
            if df_a.empty or "year" not in df_a.columns:
                continue
            by_year = df_a.groupby("year")["ensemble_flag"].agg(["count", "sum"]).reset_index()
            by_year.columns = ["year", "n_records", "n_flagged"]
            by_year["flagged_pct"] = 100 * by_year["n_flagged"] / by_year["n_records"].clip(lower=1)
            by_year["category"] = cat
            rows.append(by_year)

        if rows:
            trend_df = pd.concat(rows, ignore_index=True)
            fig = px.line(
                trend_df, x="year", y="flagged_pct", color="category",
                markers=True,
                title="Taxa de Anomalias (%) por Ano e Categoria — Período de Avaliação",
                labels={"flagged_pct": "% Sinalizados", "year": "Ano", "category": "Categoria"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Page: Explicability ───────────────────────────────────────────────────────

elif page == "🧠 Explicabilidade":
    st.title("Explicabilidade — Importância de Variáveis (SHAP)")
    cats = available_categories()
    if not cats:
        st.warning("Sem resultados.")
        st.stop()

    # Feature glossary always visible
    with st.expander("Glossário de Features — o que significam cta_XX e fav_XX?", expanded=False):
        st.markdown(_FEATURE_GLOSSARY_MD)
        st.markdown(
            "> **Nota sobre hash encoding:** O modelo usa o [hashing trick](https://en.wikipedia.org/wiki/Feature_hashing) "
            "para codificar os nomes dos parlamentares (`Conta`) e beneficiários (`Favorecido`) em vetores binários de "
            "64 e 32 dimensões respectivamente. `cta_15` significa que o parlamentar deste registro tem o nome que "
            "mapeia para o bucket 15 da função hash MurmurHash3. Não é possível reverter o hash para o nome, mas "
            "como a tabela de anomalias inclui o nome original, a decodificação é direta."
        )

    cat = st.selectbox("Categoria", cats)
    df_anom = load_anomalies(cat)

    # Summary SHAP beeswarm plot
    summary_path = f"outputs/plots/{cat}/shap_summary.png"
    if os.path.exists(summary_path):
        st.subheader("Importância Global de Variáveis (SHAP Beeswarm)")
        st.markdown(
            "Cada ponto é um registro sinalizado. Cor **vermelha** = variável com valor alto; "
            "cor **azul** = valor baixo. Posição no eixo X indica impacto no score de anomalia."
        )
        st.image(summary_path)
    else:
        st.info(f"Sem SHAP summary plot para {cat}.")

    # Waterfall plots per record
    waterfall_dir = f"outputs/plots/{cat}"
    if os.path.isdir(waterfall_dir):
        wf_files = sorted([f for f in os.listdir(waterfall_dir) if f.startswith("waterfall_")])
        if wf_files:
            st.subheader("Explicação Individual — Waterfall SHAP")
            st.markdown(
                "Cada barra mostra quanto aquela feature **empurrou o score para anomalia** (vermelho, positivo) "
                "ou para normal (azul, negativo). O valor base `E[f(X)]` é o score médio esperado."
            )

            # Identify which record corresponds to each rank
            flagged_sorted = pd.DataFrame()
            if not df_anom.empty and "ensemble_flag" in df_anom.columns:
                flagged_sorted = (
                    df_anom[df_anom["ensemble_flag"] == 1]
                    .sort_values("ensemble_score", ascending=False)
                    .reset_index(drop=True)
                )

            rank = st.slider("Ranking da anomalia (1 = mais anômalo)", 1, len(wf_files), 1)
            fname = f"waterfall_rank{rank:03d}.png"
            wf_path = os.path.join(waterfall_dir, fname)

            if not flagged_sorted.empty and rank <= len(flagged_sorted):
                record = flagged_sorted.iloc[rank - 1]
                info_cols = st.columns(4)
                info_cols[0].metric("Parlamentar", record.get("conta", "?"))
                fav = record.get("favorecido", "?")
                if pd.isna(fav) or fav == "nan":
                    fav = record.get("conta", "?")
                info_cols[1].metric("Favorecido", str(fav)[:30])
                info_cols[2].metric("Valor (R$)", fmt_brl(record.get("valor_adjusted", 0)))
                info_cols[3].metric("Ano", str(int(record.get("year", 0))))

            if os.path.exists(wf_path):
                st.image(wf_path, caption=f"Rank #{rank} — anomalia mais extrema em {cat}")
            else:
                st.warning(f"Plot {fname} não encontrado.")
        else:
            st.info("Sem waterfall plots. Execute o pipeline completo.")

    # Flagged records table with decoded features
    if not df_anom.empty and "top_feature" in df_anom.columns:
        st.subheader("Registros Sinalizados com Variável Principal Decodificada")
        flagged = df_anom[df_anom["ensemble_flag"] == 1]
        if flagged.empty:
            st.info("Nenhum registro sinalizado nesta categoria.")
        else:
            flagged_decoded = add_decoded_feature_col(flagged)
            display_cols = [c for c in [
                "conta", "favorecido", "valor_adjusted", "year",
                "ensemble_score", "top_feature", "feature_decoded", "top_shap_value",
            ] if c in flagged_decoded.columns]
            disp = flagged_decoded[display_cols].sort_values("ensemble_score", ascending=False).head(50).copy()
            disp["valor_adjusted"] = disp["valor_adjusted"].apply(fmt_brl)
            st.dataframe(
                disp.rename(columns={
                    "conta": "Parlamentar",
                    "favorecido": "Favorecido",
                    "valor_adjusted": "Valor (R$ Jan/2026)",
                    "year": "Ano",
                    "ensemble_score": "Score",
                    "top_feature": "Feature (código)",
                    "feature_decoded": "Feature (descrição)",
                    "top_shap_value": "Valor SHAP",
                }),
                use_container_width=True,
            )
