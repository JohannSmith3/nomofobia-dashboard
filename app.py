# app.py — Dashboard interactivo: Análisis de Nomofobia y Dependencia al Smartphone (Versión Final 2025)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from pathlib import Path

# -------------------- METADATOS --------------------
AUTHORS = "Johann Smith Rivera & Julian Mateo Valderrama"
COURSE = "Estadística No Paramétrica"
UNIVERSITY = "Universidad Santo Tomás"
PROF = "Javier Sierra"
YEAR = "2025"

# -------------------- CONFIGURACIÓN DE PÁGINA --------------------
st.set_page_config(
    page_title="Análisis de Nomofobia y Dependencia al Smartphone",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- ESTILO GLOBAL --------------------
st.markdown("""
    <style>
    /* Fondo suave */
    .stApp {
        background-color: #fff8e7; /* amarillo pastel claro */
        color: #1e1e1e;
    }
    /* Encabezado y logo */
    .logo-container {
        text-align: center;
        animation: fadeIn 2s ease-in;
        margin-bottom: 0px;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .title-block {
        text-align: center;
        padding: 0px 20px 10px 20px;
        border-bottom: 3px solid #0F4C81;
    }
    h1, h2, h3 {
        color: #0F4C81 !important;
    }
    .metric-container {
        background-color: #fffaf0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOGO --------------------
logo_path = Path("logo.png")

if logo_path.exists():
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(str(logo_path), width=160)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    uploaded_logo = st.sidebar.file_uploader("📁 Sube el logo institucional (PNG o SVG)", type=["png", "svg"])
    if uploaded_logo:
        st.image(uploaded_logo, width=160)

# -------------------- ENCABEZADO --------------------
st.markdown(f"""
<div class="title-block">
    <h1>📊 Análisis de Nomofobia y Dependencia al Smartphone</h1>
    <p><strong>{UNIVERSITY}</strong> — {COURSE}<br>
    Profesor: {PROF} | Autores: {AUTHORS} | {YEAR}</p>
</div>
""", unsafe_allow_html=True)

st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
st.markdown("---")

# -------------------- CARGA DE DATOS --------------------
file_path = "DATOS REALES.xlsx"
df = pd.read_excel(file_path)

# Limpieza básica
df.columns = df.columns.str.strip()
df["Sexo"] = df["Sexo"].astype(str).str.strip()
df["Nomofobia?"] = df["Nomofobia?"].astype(str).str.strip()
df["Estrato"] = df["Estrato"].astype(str).str.strip()
df = df.dropna(subset=["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"])

# -------------------- DESCRIPTIVAS --------------------
st.header("📈 Descripción general del estudio")

st.markdown("""
Este estudio busca analizar la **nomofobia** (miedo a estar sin teléfono móvil) y su relación con variables
como las **horas de uso del smartphone**, la **ansiedad social**, la **autoestima**, el **sexo** y el **estrato socioeconómico**.
Se aplican métodos de estadística **no paramétrica** para evaluar dependencias y diferencias entre grupos.
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("📱 Muestra total", len(df))
    st.metric("👩‍🎓 Mujeres", (df["Sexo"] == "Mujer").sum())
with col2:
    st.metric("👨‍🎓 Hombres", (df["Sexo"] == "Hombre").sum())
    st.metric("📊 Estratos únicos", df["Estrato"].nunique())

st.dataframe(df.describe(), use_container_width=True)

# -------------------- ANÁLISIS DE CORRELACIONES --------------------
st.header("🔗 Correlaciones de Spearman")

corr_vars = ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]
corr = df[corr_vars].corr(method="spearman")

fig_corr, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="YlGnBu", center=0, fmt=".2f", ax=ax)
st.pyplot(fig_corr)

st.markdown("""
Las correlaciones muestran cómo las variables se relacionan de forma **monótona**.
Se observa si mayores horas de uso se asocian con mayor nomofobia o niveles de ansiedad social.
""")

# -------------------- EXPLORADOR INTERACTIVO #1 --------------------
st.subheader("🎛️ Explorador de relaciones bivariadas")
x_var = st.selectbox("Variable en el eje X", corr_vars, index=0)
y_var = st.selectbox("Variable en el eje Y", corr_vars, index=1)
color_var = st.selectbox("Color según variable", ["Sexo", "Nomofobia?", "Estrato"], index=0)

fig = px.scatter(
    df, x=x_var, y=y_var, color=color_var, trendline="ols",
    hover_data=["Sexo", "Estrato", "Ansiedad_social", "Autoestima"],
    labels={x_var: x_var, y_var: y_var},
    title=f"Relación entre {x_var} y {y_var} según {color_var}"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------- PRUEBAS NO PARAMÉTRICAS --------------------
st.header("📊 Pruebas No Paramétricas")

st.subheader("Mann-Whitney U — Diferencias por sexo o grupo de nomofobia")

mwu1 = stats.mannwhitneyu(
    df.loc[df["Nomofobia?"] == "Sí", "Horas_Uso"],
    df.loc[df["Nomofobia?"] == "No", "Horas_Uso"]
)
mwu2 = stats.mannwhitneyu(
    df.loc[df["Sexo"] == "Hombre", "Horas_Uso"],
    df.loc[df["Sexo"] == "Mujer", "Horas_Uso"]
)

st.write(f"**Horas de uso según Nomofobia:** p = {mwu1.pvalue:.4f}")
st.write(f"**Horas de uso según Sexo:** p = {mwu2.pvalue:.4f}")

# -------------------- KRUSKAL-WALLIS --------------------
st.subheader("Kruskal-Wallis — Diferencias por Estrato")

kw1 = stats.kruskal(*[g["Nomofobia"].values for _, g in df.groupby("Estrato")])
st.write(f"**Nomofobia según Estrato:** H = {kw1.statistic:.3f}, p = {kw1.pvalue:.4f}")

fig_kw = px.box(df, x="Estrato", y="Nomofobia", color="Estrato",
                title="Distribución de Nomofobia por Estrato Socioeconómico")
st.plotly_chart(fig_kw, use_container_width=True)

# -------------------- POST-HOC (DUNN) --------------------
st.subheader("Prueba Post-Hoc de Dunn (con corrección Bonferroni)")

dunn = sp.posthoc_dunn(df, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
st.dataframe(dunn.style.background_gradient(cmap="YlOrRd", axis=None))

# -------------------- EXPLORADOR INTERACTIVO #2 --------------------
st.subheader("🧭 Explorador interactivo de variables categóricas")

cat_x = st.selectbox("Variable categórica", ["Sexo", "Nomofobia?", "Estrato"])
num_y = st.selectbox("Variable numérica", ["Nomofobia", "Horas_Uso", "Ansiedad_social", "Autoestima"])

fig2 = px.box(df, x=cat_x, y=num_y, color=cat_x,
              title=f"Distribución de {num_y} según {cat_x}")
st.plotly_chart(fig2, use_container_width=True)

# -------------------- CONCLUSIONES --------------------
st.header("📘 Conclusiones")

st.markdown("""
- **La nomofobia se asocia significativamente** con el número de horas de uso diario del smartphone.
- Los **niveles de ansiedad social** tienden a aumentar con la **dependencia al móvil**.
- Existen **diferencias significativas entre estratos**, confirmadas mediante **Kruskal-Wallis y Dunn post hoc**.
- El análisis sugiere que el **estrato medio y alto** presentan niveles ligeramente más altos de nomofobia,
  posiblemente asociados con mayor acceso tecnológico.
- No se hallaron diferencias marcadas por sexo en el uso promedio del teléfono.
- Los resultados confirman la **utilidad de los métodos no paramétricos** en muestras sociales con distribución no normal.
""")
st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
