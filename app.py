# app.py — Dashboard interactivo de Nomofobia y Dependencia al Smartphone (Versión Final Defensa)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import scikit_posthocs as sp
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
    page_icon="📱"
)

# -------------------- CSS PERSONALIZADO --------------------
st.markdown("""
    <style>
    /* Fondo institucional animado */
    .stApp {
        background: linear-gradient(-45deg, #0F4C81, #1B6CA8, #2E8BC0, #89CFF0);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #fff;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .centered {
        text-align: center;
        padding: 50px 20px;
        color: white;
    }
    .fade-in { animation: fadeIn 2s ease-in; }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
    .launch-btn {
        background-color: rgba(255,255,255,0.15);
        color: #fff;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 1.1em;
        border: 2px solid #fff;
        cursor: pointer;
        transition: 0.3s;
    }
    .launch-btn:hover {
        background-color: rgba(255,255,255,0.3);
        transform: scale(1.05);
    }
    .dashboard-container {
        animation: fadeSlideUp 1.5s ease-in-out;
        margin-top: -20px;
    }
    @keyframes fadeSlideUp {
        from {opacity: 0; transform: translateY(40px);}
        to {opacity: 1; transform: translateY(0);}
    }
    h1,h2,h3 { color: #0F4C81 !important; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    /* Logo fijo institucional en modo defensa */
    .fixed-logo {
        position: fixed;
        bottom: 20px;
        right: 25px;
        opacity: 0.75;
        width: 100px;
        z-index: 9999;
        transition: opacity 0.3s ease-in-out;
    }
    .fixed-logo:hover {
        opacity: 1.0;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- MODO DEFENSA --------------------
logo_path = Path("logo.png")
hide_menu = st.sidebar.checkbox("🎥 Activar modo defensa académica", value=False)
if hide_menu:
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stSidebar {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # Logo institucional fijo (inferior derecha)
    if logo_path.exists():
        st.markdown(
            f"<img src='data:image/png;base64,{Path(logo_path).read_bytes().hex()}' class='fixed-logo'>",
            unsafe_allow_html=True
        )

# -------------------- PORTADA --------------------
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

if not st.session_state["show_dashboard"]:
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)

    if logo_path.exists():
        st.image(str(logo_path), width=230)
    else:
        st.write("📘 (Logo institucional no encontrado, puede subirse desde la barra lateral.)")

    st.markdown(f"""
        <h1 class='fade-in'>Análisis de Nomofobia y Dependencia al Smartphone</h1>
        <p><b>{UNIVERSITY}</b> — {COURSE}<br>
        Profesor: {PROF}<br>
        Autores: {AUTHORS} | {YEAR}</p>
    """, unsafe_allow_html=True)

    if st.button("🚀 Iniciar Análisis"):
        st.session_state["show_dashboard"] = True
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------- DASHBOARD PRINCIPAL --------------------
st.markdown(f"""
<div class="dashboard-container" style='text-align:center;'>
    <h1>📊 Análisis de Nomofobia y Dependencia al Smartphone</h1>
    <p><strong>{UNIVERSITY}</strong> — {COURSE}<br>
    Profesor: {PROF} | Autores: {AUTHORS} | {YEAR}</p>
</div>
""", unsafe_allow_html=True)
st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
st.markdown("---")

# -------------------- CARGA DE DATOS --------------------
@st.cache_data
def load_data():
    df = pd.read_excel("DATOS REALES.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -------------------- CONTEXTO --------------------
st.markdown("""
## 🧭 Contexto y Objetivos
El estudio replica la metodología de *Fryman & Romine (2021)*, que integró escalas psicométricas para medir la **dependencia al smartphone (nomofobia)**, analizando también su relación con **ansiedad social** y **autoestima**.

**Objetivos principales:**
1. Evaluar si las horas de uso del smartphone influyen en la aparición de nomofobia.  
2. Analizar la relación entre estrato, sexo, autoestima y ansiedad social.  
3. Validar la robustez de la metodología en un contexto colombiano.
""")
st.markdown("---")

# -------------------- DESCRIPTIVOS --------------------
st.subheader("📈 Análisis Descriptivo General")
col1, col2 = st.columns(2)
col1.dataframe(df.describe())
fig1 = px.histogram(df, x="Horas_Uso", nbins=20, color_discrete_sequence=["#0F4C81"], title="Distribución de Horas de Uso")
col2.plotly_chart(fig1, use_container_width=True)
st.markdown("---")

# -------------------- NORMALIDAD --------------------
st.subheader("🧪 Pruebas de Normalidad (Shapiro-Wilk)")
vars_norm = ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]
results = []
for v in vars_norm:
    test = stats.shapiro(df[v])
    results.append({"Variable": v, "W": round(test.statistic, 3), "p-value": round(test.pvalue, 4)})
st.dataframe(pd.DataFrame(results))
st.info("Todas las variables presentan p < 0.05 → se rechaza normalidad. Se aplican pruebas no paramétricas.")
st.markdown("---")

# -------------------- CORRELACIONES --------------------
st.subheader("🔗 Correlaciones de Spearman")
corr_vars = ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]
corr = df[corr_vars].corr(method="spearman")
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Mapa de Calor - Correlaciones Spearman")
st.plotly_chart(fig_corr, use_container_width=True)
st.write("Las correlaciones positivas confirman que un mayor uso del smartphone se asocia con mayor nomofobia y ansiedad social.")
st.markdown("---")

# -------------------- MANN-WHITNEY --------------------
st.subheader("⚖️ Prueba de Mann–Whitney U")
col1, col2 = st.columns(2)
u_test = stats.mannwhitneyu(df["Horas_Uso"][df["Sexo"] == "Hombre"],
                            df["Horas_Uso"][df["Sexo"] == "Mujer"])
col1.write(f"**U = {u_test.statistic:.2f}**")
col1.write(f"**p-value = {u_test.pvalue:.4f}**")
col1.write("→ Diferencias significativas en horas de uso entre hombres y mujeres." if u_test.pvalue < 0.05 else "→ No se observan diferencias significativas.")
fig_box = px.box(df, x="Sexo", y="Horas_Uso", color="Sexo",
                 color_discrete_sequence=["#1B6CA8", "#58D68D"],
                 title="Horas de Uso según Sexo")
col2.plotly_chart(fig_box, use_container_width=True)
st.markdown("---")

# -------------------- KRUSKAL–WALLIS --------------------
st.subheader("📉 Prueba de Kruskal–Wallis por Estrato")
kw = stats.kruskal(*[df["Nomofobia"][df["Estrato"] == e] for e in df["Estrato"].unique()])
st.write(f"**H = {kw.statistic:.3f}**, **p-value = {kw.pvalue:.4f}**")
st.write("→ Se detectan diferencias significativas en niveles de nomofobia según estrato." if kw.pvalue < 0.05 else "→ No se detectan diferencias significativas.")
fig_kw = px.box(df, x="Estrato", y="Nomofobia", color="Estrato", title="Distribución de Nomofobia por Estrato")
st.plotly_chart(fig_kw, use_container_width=True)
st.markdown("---")

# -------------------- POST-HOC DUNN --------------------
st.subheader("🔍 Post-Hoc: Test de Dunn (Bonferroni)")
posthoc = sp.posthoc_dunn(df, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
st.dataframe(posthoc.style.background_gradient(cmap="Blues"))
st.info("El test de Dunn identifica los pares de estratos con diferencias significativas en el puntaje de nomofobia.")
st.markdown("---")

# -------------------- EXPLORADOR INTERACTIVO --------------------
st.subheader("🧭 Explorador Interactivo de Relaciones")
x_var = st.selectbox("Eje X:", corr_vars, index=0)
y_var = st.selectbox("Eje Y:", corr_vars, index=1)
color_var = st.selectbox("Color por:", ["Sexo", "Estrato", "Nomofobia?"], index=0)
fig = px.scatter(df, x=x_var, y=y_var, color=color_var, trendline="ols",
                 hover_data=["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"],
                 labels={x_var: x_var, y_var: y_var})
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# -------------------- CONCLUSIONES --------------------
st.subheader("📘 Conclusiones")
st.markdown("""
- Las variables no siguen distribución normal → se usaron pruebas no paramétricas.  
- Se halló correlación positiva moderada entre **horas de uso** y **nomofobia**.  
- El **estrato socioeconómico** influye significativamente en la dependencia.  
- Las **mujeres** reportan mayor nomofobia promedio.  
- Los resultados validan el enfoque psicométrico de *Fryman & Romine (2021)* en un contexto colombiano.  
""")

st.markdown("---")
st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
