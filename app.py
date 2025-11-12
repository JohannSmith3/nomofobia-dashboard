# app.py — Dashboard final y profesional (versión completa con normalidad, 2 exploradores, Dunn, y conclusiones ampliadas)
# Autor: Johann Smith Rivera & Julian Mateo Valderrama
# Materia: Estadística No Paramétrica — Universidad Santo Tomás
# Profesor: Javier Sierra
# Título mostrado: "Análisis de Nomofobia y Dependencia al Smartphone"

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import anderson, shapiro, probplot
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
from pathlib import Path
import matplotlib.pyplot as plt
import math

# -------------------- Metadatos --------------------
AUTHORS = "Johann Smith Rivera & Julian Mateo Valderrama"
COURSE = "Estadística No Paramétrica"
UNIVERSITY = "Universidad Santo Tomás"
PROF = "Javier Sierra"
YEAR = "2025"

# -------------------- Configuración de página --------------------
st.set_page_config(
    page_title="Análisis de Nomofobia y Dependencia al Smartphone",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- PORTADA INSTITUCIONAL (blanca, fade-in) --------------------
st.markdown("""
    <style>
    body { background-color: #ffffff; }
    .centered { text-align: center; padding: 36px 18px; }
    .fade-in { animation: fadeIn 1.6s ease-in; }
    @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
    .welcome-title { color: #0F4C81; font-size: 2.4em; font-weight:700; margin-bottom:0.2em; }
    .welcome-sub { color:#333; font-size:1.05em; margin-bottom:1.3em; }
    .launch-btn { background-color: #0F4C81; color: white; padding:10px 22px; border-radius:8px; font-weight:600; }
    .launch-btn:hover { background-color:#123E6C; transform:scale(1.03); }
    </style>
""", unsafe_allow_html=True)

logo_path = Path("logo.png")
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

if not st.session_state["show_dashboard"]:
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.image(str(logo_path), width=220)
    st.markdown(f'<div class="welcome-title">Análisis de Nomofobia y Dependencia al Smartphone</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-sub"><b>{UNIVERSITY}</b> — {COURSE} • Profesor: {PROF}<br>Autores: {AUTHORS} | {YEAR}</div>', unsafe_allow_html=True)
    if st.button("🚀 Iniciar Análisis", key="start_button"):
        st.session_state["show_dashboard"] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------- Header (aparece después de iniciar) --------------------
st.image(str(logo_path), width=120)
st.markdown("# Análisis de Nomofobia y Dependencia al Smartphone")
st.markdown(f"**{UNIVERSITY} — {COURSE}**  • Profesor: {PROF}")
st.markdown(f"**Autores:** {AUTHORS}  • {YEAR}")
st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
st.markdown("---")

# -------------------- CARGA DE DATOS (asumida presente) --------------------
df = pd.read_excel("DATOS REALES.xlsx")
df.columns = df.columns.str.strip()

# limpieza mínima (asumimos columnas presentes)
df["Sexo"] = df["Sexo"].astype(str).str.strip()
df["Estrato"] = df["Estrato"].astype(str).str.strip()
df["Nomofobia?"] = df["Nomofobia?"].astype(str).str.strip()
for col in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Edad", "Mal_uso"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------- SIDEBAR: filtros y opciones --------------------
st.sidebar.header("Parámetros de la visualización")
sexo_options = df["Sexo"].dropna().unique().tolist()
estrato_options = df["Estrato"].dropna().unique().tolist()
nomob_options = df["Nomofobia?"].dropna().unique().tolist()

sexo_sel = st.sidebar.multiselect("Sexo", options=sexo_options, default=sexo_options)
estrato_sel = st.sidebar.multiselect("Estrato", options=estrato_options, default=estrato_options)
nomob_sel = st.sidebar.multiselect("Nomofobia? (Sí/No)", options=nomob_options, default=nomob_options)

show_normality = st.sidebar.checkbox("Mostrar pruebas de normalidad (Shapiro + Anderson-Darling)", value=False)
show_density = st.sidebar.checkbox("Mostrar densidades (violines)", value=True)
bootstrap_spearman = st.sidebar.checkbox("Bootstrapped CI (Spearman)", value=True)
nboots = 1000

# Filter
df_f = df[df["Sexo"].isin(sexo_sel) & df["Estrato"].isin(estrato_sel) & df["Nomofobia?"].isin(nomob_sel)]

# -------------------- CONTEXTO AMPLIADO --------------------
st.subheader("Contexto y objetivos")
st.write(
    "Este trabajo replica la metodología de Fryman & Romine (2021) — agregando ítems de"
    " escalas validadas para construir puntajes de Nomofobia, Ansiedad Social y Autoestima."
    " Objetivos del dashboard:\n\n"
    "1. Evaluar si las horas de uso están asociadas con nomofobia.\n"
    "2. Explorar comorbilidades (ansiedad social, autoestima).\n"
    "3. Aplicar pruebas no paramétricas apropiadas y visualizar resultados de manera accionable."
)
st.markdown("---")

# -------------------- 1) Descriptivas y Visualizaciones detalladas --------------------
st.subheader("1) Estadísticas descriptivas y visualizaciones")

numeric_cols = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"] if c in df_f.columns]
if numeric_cols:
    desc = df_f[numeric_cols].describe().T.rename(columns={"50%": "mediana"})
    st.write("Resumen descriptivo (muestras filtradas):")
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

# Plots por variable numérica: hist + qq + violin/box
for col in numeric_cols:
    st.markdown(f"**Variable:** {col}")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.histogram(df_f, x=col, nbins=30, marginal="box", title=f"Histograma y boxplot — {col}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Matplotlib QQ-plot to show normality visually
        figm, ax = plt.subplots(figsize=(5, 4))
        clean = df_f[col].dropna()
        if len(clean) >= 3:
            probplot(clean, dist="norm", plot=ax)
            ax.set_title(f"QQ-plot — {col}")
        else:
            ax.text(0.1, 0.5, "Insuficientes datos para QQ-plot", fontsize=12)
        st.pyplot(figm)
    # violin (grouped by Estrato if exists)
    if "Estrato" in df_f.columns:
        figv = px.violin(df_f, x="Estrato", y=col, box=True, points="all", title=f"Violin {col} por Estrato")
        st.plotly_chart(figv, use_container_width=True)
    st.markdown("---")

# -------------------- Normality tests (optional) --------------------
def run_normality(series):
    s = series.dropna()
    if len(s) < 3:
        return {"shapiro_p": np.nan, "anderson_stat": np.nan, "anderson_sig": None}
    W, p_sh = shapiro(s)
    ad_res = anderson(s, dist="norm")
    return {"shapiro_p": p_sh, "anderson_stat": ad_res.statistic, "anderson_critical": ad_res.critical_values, "anderson_significance": ad_res.significance_level}

if show_normality:
    st.subheader("Pruebas de normalidad (Shapiro-Wilk y Anderson-Darling)")
    for col in numeric_cols:
        res = run_normality(df_f[col])
        if np.isnan(res["shapiro_p"]):
            st.write(f"{col}: Insuficientes datos para pruebas de normalidad.")
            continue
        st.write(f"**{col}** — Shapiro p = {res['shapiro_p']:.4f}. Anderson-Darling stat = {res['anderson_stat']:.3f}.")
        st.write("_Interpretación:_ p<0.05 en Shapiro sugiere desviación de normalidad. Anderson-Darling compara con valores críticos (ver tabla).")
    st.markdown("---")

# -------------------- 2) Correlaciones Spearman (mapa + tabla con CI bootstrapped) --------------------
st.subheader("2) Correlaciones no paramétricas — Spearman (mapa de calor + tabla con CI)")
spearman_vars = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso", "Edad"] if c in df_f.columns]
if len(spearman_vars) >= 2:
    corr = df_f[spearman_vars].corr(method="spearman")
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Mapa de calor — Correlaciones Spearman")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Detailed table with bootstrap CI for pairwise with target = Horas_Uso (if present)
    target = "Horas_Uso" if "Horas_Uso" in spearman_vars else spearman_vars[0]
    rows = []
    for v in spearman_vars:
        if v == target: continue
        tmp = df_f[[target, v]].dropna()
        if tmp.shape[0] < 5:
            rows.append({"variable": v, "rho": np.nan, "p": np.nan, "rho_CI95": "n<5"})
            continue
        rho, p = stats.spearmanr(tmp[target], tmp[v], nan_policy="omit")
        ci_text = "NA"
        if bootstrap_spearman and tmp.shape[0] >= 10:
            boots = []
            rng = np.random.default_rng(12345)
            x = tmp[target].to_numpy(); y = tmp[v].to_numpy()
            for _ in range(nboots):
                idx = rng.integers(0, len(x), len(x))
                boots.append(stats.spearmanr(x[idx], y[idx]).correlation)
            lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
            ci_text = f"[{lo:.3f}, {hi:.3f}]"
        rows.append({"variable": v, "rho": round(rho, 3), "p": round(p, 4), "rho_CI95": ci_text})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.markdown("---")

# -------------------- 3) Mann–Whitney (Horas_Uso by Nomofobia?) --------------------
st.subheader("3) Test Mann–Whitney — Horas de Uso por Nomofobia (Sí/No)")
if {"Nomofobia?", "Horas_Uso"}.issubset(df_f.columns):
    a = df_f[df_f["Nomofobia?"] == "Sí"]["Horas_Uso"].dropna()
    b = df_f[df_f["Nomofobia?"] == "No"]["Horas_Uso"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        U, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        mu_U = n1 * n2 / 2
        sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mu_U) / sigma_U if sigma_U > 0 else 0
        r = z / math.sqrt(n1 + n2)
        st.write(f"U = {U:.3f}  •  p = {p_u:.4f}  •  z = {z:.3f}  •  r = {r:.3f}")
        if p_u < 0.05:
            st.success("Diferencia estadísticamente significativa entre los grupos (p < 0.05).")
        else:
            st.info("No se detectaron diferencias significativas (p ≥ 0.05).")
        fig_mw = px.box(df_f, x="Nomofobia?", y="Horas_Uso", points="all", color="Nomofobia?", title="Horas de Uso según Nomofobia")
        st.plotly_chart(fig_mw, use_container_width=True)
        # Interpretation
        st.markdown("**Interpretación detallada (Mann–Whitney):**")
        st.write(
            "La prueba compara las distribuciones relativas (órdenes). Un p<0.05 sugiere que la"
            " distribución de horas difiere entre personas con nomofobia y sin nomofobia. El tamaño de efecto r"
            f" = {r:.3f} ayuda a evaluar la magnitud práctica (reglas generales: 0.1 pequeño, 0.3 moderado, 0.5 grande)."
        )
    else:
        st.warning("Insuficientes observaciones en uno de los grupos para Mann–Whitney (mínimo 3 por grupo).")
st.markdown("---")

# -------------------- 4) Kruskal–Wallis (Nomofobia por Estrato) --------------------
st.subheader("4) Kruskal–Wallis (Nomofobia por Estrato)")
if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
    groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
    if len(groups) > 1:
        H, p_kw = stats.kruskal(*groups)
        st.write(f"H = {H:.3f}  •  p = {p_kw:.4f}")
        st.plotly_chart(px.box(df_f, x="Estrato", y="Nomofobia", color="Estrato", points="all",
                               title="Nomofobia por Estrato — Kruskal–Wallis"), use_container_width=True)
        if p_kw < 0.05:
            st.success("Al menos dos estratos difieren en Nomofobia (p < 0.05).")
        else:
            st.info("No se evidencian diferencias significativas entre estratos.")
        st.markdown("**Interpretación (Kruskal–Wallis):**")
        st.write(
            "Kruskal–Wallis evalúa si provienen de la misma distribución. Si p<0.05, se procede a post-hoc"
            " para identificar pares diferentes."
        )
    else:
        st.warning("No hay suficientes grupos para Kruskal–Wallis.")
st.markdown("---")

# -------------------- 5) Post-hoc: Dunn (tabla + heatmap) --------------------
st.subheader("5) Post-hoc: Dunn (comparaciones por pares, Bonferroni)")
if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
    dunn = sp.posthoc_dunn(df_f, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
    st.write("Matriz de p-valores ajustados (Bonferroni):")
    st.dataframe(dunn.style.format("{:.4f}"), use_container_width=True)
    st.plotly_chart(px.imshow(dunn, text_auto=True, color_continuous_scale="Reds", title="Dunn — p-vals ajustados"), use_container_width=True)
    st.caption("Celdas con p < 0.05 indican pares de estratos con diferencias en Nomofobia.")
st.markdown("---")

# -------------------- Explorador A (bivariado avanzado) --------------------
st.subheader("6) Explorador A — análisis bivariado avanzado")
with st.expander("Abrir Explorador A (scatter, trendline, color)"):
    numeric = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
    cat = [c for c in df_f.columns if not np.issubdtype(df_f[c].dtype, np.number)]
    x = st.selectbox("Eje X (num)", numeric, index=0)
    y = st.selectbox("Eje Y (num)", numeric, index=1)
    color = st.selectbox("Color por (categórico)", [None] + cat, index=1 if cat else 0)
    size = st.selectbox("Tamaño por (num, opcional)", [None] + numeric, index=0)
    trend = st.selectbox("Trendline", ["none", "ols", "lowess"], index=1)
    fig = px.scatter(df_f, x=x, y=y, color=color, size=size, trendline=None if trend == "none" else trend,
                     hover_data=["Sexo", "Estrato", "Nomofobia?" ] if "Nomofobia?" in df_f.columns else None,
                     title=f"{y} vs {x}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Usa zoom y selección para investigar puntos atípicos; filtra por la barra lateral para subgrupos.")

st.markdown("---")

# -------------------- Explorador B (comparador correlaciones) --------------------
st.subheader("7) Explorador B — comparador de correlaciones (Spearman personalizado)")
with st.expander("Abrir Explorador B (Spearman)"):
    cand = [c for c in spearman_vars if c in df_f.columns]
    v1 = st.selectbox("Variable A", cand, index=0, key="c1")
    v2 = st.selectbox("Variable B", cand, index=1, key="c2")
    if v1 == v2:
        st.warning("Selecciona dos variables distintas.")
    else:
        rho, p = stats.spearmanr(df_f[v1], df_f[v2], nan_policy="omit")
        st.write(f"Spearman ρ = {rho:.3f}  •  p = {p:.4f}")
        fig_sc = px.scatter(df_f, x=v1, y=v2, color="Nomofobia?" if "Nomofobia?" in df_f.columns else None, trendline="ols",
                            title=f"{v2} vs {v1}")
        st.plotly_chart(fig_sc, use_container_width=True)
        if p < 0.05:
            st.success("Correlación estadísticamente significativa.")
        else:
            st.info("No significativa (p ≥ 0.05).")
        if len(df_f.dropna(subset=[v1, v2])) > 50:
            st.plotly_chart(px.density_contour(df_f, x=v1, y=v2), use_container_width=True)

st.markdown("---")

# -------------------- Explorador C (nuevo): Comparador por grupos categóricos --------------------
st.subheader("8) Explorador C — Comparador por grupos categóricos (tablas + gráficos)")

with st.expander("Abrir Explorador C (comparar medias/medianas por grupo)"):
    # Selección de variables
    cat_var = st.selectbox(
        "Variable categórica para agrupar",
        [c for c in ["Estrato", "Sexo", "Nomofobia?"] if c in df_f.columns]
    )

    numeric_cols = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
    num_var = st.selectbox("Variable numérica a comparar", numeric_cols, index=0)

    # Tabla resumen
    st.write(f"Resumen por **{cat_var}** — variable **{num_var}**:")
    grp = df_f.groupby(cat_var)[num_var].agg(["count", "mean", "median", "std"]).reset_index()
    st.dataframe(grp, use_container_width=True)  # <- corregido: sin formato flotante

    # Boxplot
    fig_box = px.box(
        df_f,
        x=cat_var,
        y=num_var,
        points="all",
        color=cat_var,
        title=f"{num_var} por {cat_var}"
    )
    st.plotly_chart(fig_box, use_container_width=True, key=f"boxplot_{cat_var}_{num_var}")

    # Kruskal–Wallis si hay más de 2 grupos
    uniques = df_f[cat_var].dropna().unique()
    if len(uniques) > 2:
        groups_list = [g[num_var].dropna() for _, g in df_f.groupby(cat_var)]
        try:
            Hc, p_hc = stats.kruskal(*groups_list)
            st.write(f"Kruskal–Wallis: H = {Hc:.3f} • p = {p_hc:.4f}")
            if p_hc < 0.05:
                st.success("Diferencias estadísticamente significativas entre grupos (p < 0.05).")
            else:
                st.info("No se encontraron diferencias significativas entre grupos.")
        except Exception as e:
            st.error(f"No se pudo ejecutar Kruskal–Wallis: {e}")
    else:
        st.info("Kruskal–Wallis no aplica (menos de 3 grupos).")

st.markdown("---")

# -------------------- CONCLUSIONES AMPLIADAS y RECOMENDACIONES --------------------
st.header("Conclusiones")

# Formar conclusiones por test
conclusions = []

# Spearman summary (Horas_Uso pairs)
if "Horas_Uso" in df_f.columns:
    for v in ["Nomofobia", "Ansiedad_social", "Autoestima"]:
        if v in df_f.columns:
            rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[v], nan_policy="omit")
            if np.isnan(rho):
                continue
            text = f"Horas_Uso vs {v}: ρ={rho:.3f}, p={p:.4f}."
            if p < 0.05:
                text += " Asociación estadísticamente significativa."
            else:
                text += " No asociación significativa."
            conclusions.append(text)

# Mann-Whitney conclusion
if 'p_u' in locals():
    conclusions.append(f"Mann–Whitney (Horas_Uso | Nomofobia): p={p_u:.4f}. {'Diferencia significativa entre grupos' if p_u<0.05 else 'Sin diferencia estadísticamente significativa'}.")

# Kruskal conclusion
if 'p_kw' in locals():
    conclusions.append(f"Kruskal–Wallis (Nomofobia ~ Estrato): p={p_kw:.4f}. {'Se detectaron diferencias entre estratos' if p_kw<0.05 else 'No se evidenciaron diferencias entre estratos'}.")

# Dunn highlights (significant pairs)
if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
    sig_pairs = []
    d = dunn.copy()
    for i in d.index:
        for j in d.columns:
            if i == j: continue
            try:
                pv = d.loc[i, j]
                if pv < 0.05:
                    sig_pairs.append(f"{i} vs {j} (p={pv:.3f})")
            except Exception:
                pass
    if sig_pairs:
        conclusions.append("Dunn post-hoc: pares significativos -> " + "; ".join(sig_pairs))
    else:
        conclusions.append("Dunn post-hoc: no se detectaron pares con p<0.05.")

# Print conclusions
st.markdown("**Resumen de hallazgos (detallado):**")
for c in conclusions:
    st.write("• " + c)

# Actionable recommendations (prioritized)
st.markdown("**Recomendaciones accionables (priorizadas):**")
recs = []
# Example priority rules
if any("Ansiedad_social" in s and "significativa" in s for s in conclusions):
    recs.append("Priorizar intervenciones dirigidas a estudiantes con alta ansiedad social para reducir la exposición al smartphone.")
if 'p_u' in locals() and p_u < 0.05:
    recs.append("Diseñar campañas de reducción de tiempo de pantalla y talleres de autocontrol para grupos con nomofobia.")
recs.append("Realizar estudios longitudinales para evaluar causalidad y modelos multivariados que controlen confusores.")
for i, r in enumerate(recs, 1):
    st.write(f"{i}. {r}")

st.info("Las conclusiones y recomendaciones están pensadas para guiar decisiones de intervención y futuras investigaciones.")
st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")



