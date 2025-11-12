# app.py — Dashboard final y profesional con portada animada y modo defensa
# Autor: Johann Smith Rivera & Julian Mateo Valderrama
# Materia: Estadística No Paramétrica — Universidad Santo Tomás
# Profesor: Javier Sierra
# Título mostrado: "Análisis de Nomofobia y Dependencia al Smartphone"

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
from pathlib import Path
import io
import math
import random

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

# -------------------- Portada animada beige --------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "defense_mode" not in st.session_state:
    st.session_state.defense_mode = False

if not st.session_state.started:
    st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #f5ecd7, #fdf8ee, #f7e9c4, #f5ecd7);
        background-size: 400% 400%;
        animation: gradientFlow 12s ease infinite;
        color: #2b2b2b;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    @keyframes gradientFlow {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .title {
        font-size: 2.2em;
        font-weight: 700;
        color: #2b2b2b;
        margin-top: 1.8em;
    }
    .subtitle {
        font-size: 1.15em;
        color: #5a5a5a;
        margin-bottom: 2em;
    }
    .start-btn {
        background-color: #003366;
        color: #fdf8ee;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.8em 2.5em;
        border: none;
        cursor: pointer;
        font-size: 1.1em;
        transition: 0.4s ease;
    }
    .start-btn:hover {
        background-color: #1a4470;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=220)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="title">Universidad Santo Tomás</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Facultad de Psicología<br>Estadística No Paramétrica — 2025</div>', unsafe_allow_html=True)
    st.markdown('<br><br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Empezar análisis", key="start_btn"):
            st.session_state.started = True
            st.experimental_rerun()
    st.stop()

# -------------------- Modo Defensa Académica --------------------
st.markdown("""
<style>
.main {background-color: #fdf8ee !important;}
header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if st.sidebar.checkbox("Activar modo Defensa Académica (pantalla limpia)"):
    st.session_state.defense_mode = True

if st.session_state.defense_mode:
    hide_style = """
    <style>
    [data-testid="stSidebar"], header, footer {visibility: hidden;}
    .main {
        background-color: #fdf8ee !important;
        padding-top: 1em;
    }
    .fixed-logo {
        position: fixed;
        bottom: 10px;
        right: 10px;
        opacity: 0.3;
        width: 130px;
        z-index: 100;
    }
    </style>
    """
    st.markdown(hide_style, unsafe_allow_html=True)
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.markdown(f'<img src="{logo_path}" class="fixed-logo">', unsafe_allow_html=True)

# -------------------- Header con logo --------------------
logo_path = Path("logo.png")
logo_shown = False
with st.container():
    cols = st.columns([0.12, 0.88])
    if logo_path.exists():
        cols[0].image(str(logo_path), use_column_width=True)
        logo_shown = True
    else:
        uploaded_logo = st.sidebar.file_uploader("Sube el logo de la universidad (opcional) — PNG/SVG", type=["png", "svg"])
        if uploaded_logo is not None:
            cols[0].image(uploaded_logo, use_column_width=True)
            logo_shown = True
    cols[1].markdown(f"# Análisis de Nomofobia y Dependencia al Smartphone")
    cols[1].markdown(f"**{UNIVERSITY} — {COURSE}**  • Profesor: {PROF}")
    cols[1].markdown(f"**Autores:** {AUTHORS}  • {YEAR}")

st.caption("Dashboard nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")
st.markdown("---")

# -------------------- CARGA DE DATOS --------------------
@st.cache_data
def load_data(path="DATOS REALES.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "Sexo" in df.columns:
        df["Sexo"] = df["Sexo"].astype(str).str.strip()
    if "Estrato" in df.columns:
        df["Estrato"] = df["Estrato"].astype(str).str.strip()
    if "Nomofobia?" in df.columns:
        df["Nomofobia?"] = df["Nomofobia?"].astype(str).str.strip()
    for col in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Edad", "Mal_uso"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

try:
    df = load_data()
except Exception:
    st.error("No se pudo leer DATOS REALES.xlsx desde la carpeta del repo. Usa el uploader en la barra lateral para subir el Excel.")
    uploaded_file = st.sidebar.file_uploader("Sube DATOS REALES.xlsx (si no está en repo)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.header("Parámetros de la visualización")
sexo_options = df["Sexo"].dropna().unique().tolist() if "Sexo" in df.columns else []
estrato_options = df["Estrato"].dropna().unique().tolist() if "Estrato" in df.columns else []
nomob_options = df["Nomofobia?"].dropna().unique().tolist() if "Nomofobia?" in df.columns else []

sexo_sel = st.sidebar.multiselect("Sexo", options=sexo_options, default=sexo_options if sexo_options else None)
estrato_sel = st.sidebar.multiselect("Estrato", options=estrato_options, default=estrato_options if estrato_options else None)
nomob_sel = st.sidebar.multiselect("Nomofobia? (Sí/No)", options=nomob_options, default=nomob_options if nomob_options else None)

show_normality = st.sidebar.checkbox("Mostrar pruebas de normalidad", value=False)
show_density = st.sidebar.checkbox("Mostrar densidades (violines)", value=True)
bootstrap_spearman = st.sidebar.checkbox("Bootstrapped CI para Spearman (1000 resamples)", value=True)

# -------------------- FILTROS --------------------
df_f = df.copy()
if sexo_options and sexo_sel:
    df_f = df_f[df_f["Sexo"].isin(sexo_sel)]
if estrato_options and estrato_sel:
    df_f = df_f[df_f["Estrato"].isin(estrato_sel)]
if nomob_options and nomob_sel:
    df_f = df_f[df_f["Nomofobia?"].isin(nomob_sel)]

