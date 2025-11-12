# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Dashboard Nomofobia", layout="wide", initial_sidebar_state="expanded")
st.title("Dependencia al Smartphone y Factores Psicológicos")
st.caption("Proyecto final — Estadística No Paramétrica (Fryman & Romine, 2021, replicación local)")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_excel("DATOS REALES.xlsx")
    df.columns = df.columns.str.strip()
    df["Sexo"] = df["Sexo"].astype(str).str.strip()
    df["Estrato"] = df["Estrato"].astype(str)
    df["Nomofobia?"] = df["Nomofobia?"].astype(str).str.strip()
    return df

df = load_data()

# --- FILTROS ---
st.sidebar.header("Filtros de exploración")
sexo_sel = st.sidebar.multiselect("Sexo", df["Sexo"].unique())
estrato_sel = st.sidebar.multiselect("Estrato", sorted(df["Estrato"].unique()))
nomofobia_sel = st.sidebar.multiselect("Nomofobia? (Sí/No)", df["Nomofobia?"].unique())

df_f = df.copy()
if sexo_sel:
    df_f = df_f[df_f["Sexo"].isin(sexo_sel)]
if estrato_sel:
    df_f = df_f[df_f["Estrato"].isin(estrato_sel)]
if nomofobia_sel:
    df_f = df_f[df_f["Nomofobia?"].isin(nomofobia_sel)]

# --- KPI PRINCIPALES ---
st.subheader("📊 Indicadores clave (KPIs)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Participantes", len(df_f))
c2.metric("Promedio Horas de Uso", f"{df_f['Horas_Uso'].mean():.2f}")
c3.metric("Promedio Nomofobia", f"{df_f['Nomofobia'].mean():.2f}")
c4.metric("Proporción con Nomofobia (Sí)", f"{(df_f['Nomofobia?'].value_counts(normalize=True).get('Sí', 0)*100):.1f}%")

# --- CORRELACIONES SPEARMAN ---
st.subheader("📈 Correlaciones (Spearman)")

variables = ["Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"]
results = []
for var in variables:
    rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[var], nan_policy='omit')
    results.append({"Variable": var, "ρ (Spearman)": round(rho, 3), "p-valor": round(p, 4)})

corr_df = pd.DataFrame(results)
st.dataframe(corr_df, use_container_width=True)

# --- DISPERSIÓN HORAS_USO vs NOMOFOBIA ---
st.markdown("### 📉 Relación Horas de Uso vs Nomofobia")
fig = px.scatter(
    df_f, x="Horas_Uso", y="Nomofobia", color="Nomofobia?",
    trendline="ols", hover_data=["Sexo", "Estrato", "Ansiedad_social", "Autoestima"],
    labels={"Horas_Uso": "Horas de Uso (promedio diario)", "Nomofobia": "Puntaje Nomofobia"}
)
st.plotly_chart(fig, use_container_width=True)

# --- MANN–WHITNEY ---
st.subheader("🧮 Prueba Mann–Whitney U: Horas de Uso según Nomofobia (Sí/No)")
if set(df_f["Nomofobia?"].unique()) >= {"Sí", "No"}:
    g1 = df_f[df_f["Nomofobia?"] == "Sí"]["Horas_Uso"]
    g2 = df_f[df_f["Nomofobia?"] == "No"]["Horas_Uso"]
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    st.write(f"**Estadístico U = {stat:.3f}, p = {p:.4f}**")
    if p < 0.05:
        st.success("💡 Diferencia significativa entre grupos.")
    else:
        st.info("No se encontró diferencia significativa entre grupos.")
    fig2 = px.box(df_f, x="Nomofobia?", y="Horas_Uso", color="Nomofobia?",
                  labels={"Nomofobia?": "Nomofobia (Sí/No)", "Horas_Uso": "Horas de Uso"})
    st.plotly_chart(fig2, use_container_width=True)

# --- KRUSKAL–WALLIS ---
st.subheader("🧮 Prueba Kruskal–Wallis: Nomofobia según Estrato Socioeconómico")
groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
if len(groups) > 1:
    H, p_kw = stats.kruskal(*groups)
    st.write(f"**Estadístico H = {H:.3f}, p = {p_kw:.4f}**")
    if p_kw < 0.05:
        st.success("💡 Existen diferencias significativas entre estratos.")
    else:
        st.info("No se encontraron diferencias significativas entre estratos.")
    fig3 = px.box(df_f, x="Estrato", y="Nomofobia", color="Estrato",
                  labels={"Estrato": "Estrato", "Nomofobia": "Puntaje Nomofobia"})
    st.plotly_chart(fig3, use_container_width=True)

# --- EXPLORADOR LIBRE ---
st.subheader("🔎 Explorador interactivo")
with st.expander("Haz tu propio análisis"):
    num_cols = ["Edad", "Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"]
    cat_cols = ["Sexo", "Estrato", "Nomofobia?"]
    x_var = st.selectbox("Eje X", num_cols, index=1)
    y_var = st.selectbox("Eje Y", num_cols, index=2)
    color_var = st.selectbox("Color por", [None] + cat_cols)
    fig_exp = px.scatter(df_f, x=x_var, y=y_var, color=color_var, trendline="lowess",
                         hover_data=cat_cols, labels={x_var: x_var, y_var: y_var})
    st.plotly_chart(fig_exp, use_container_width=True)

# --- PIE DE PÁGINA ---
st.caption("Dashboard interactivo - Nomofobia - Estadística No Paramétrica (2025).")