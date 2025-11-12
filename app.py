# app.py — versión avanzada con mapa de calor, Dunn test y recomendaciones

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
import itertools
import scikit_posthocs as sp

# ------------------ Configuración general ------------------
st.set_page_config(page_title="Dashboard Nomofobia — Avanzado", layout="wide")
st.title("📱 Dashboard Avanzado — Dependencia al Smartphone y Factores Psicológicos")
st.caption("Proyecto Final | Estadística No Paramétrica | Johann Smith (2025)")

# ------------------ Carga de datos ------------------
@st.cache_data
def load_data(path="DATOS REALES.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["Sexo"] = df["Sexo"].astype(str).str.strip()
    df["Estrato"] = df["Estrato"].astype(str).str.strip()
    df["Nomofobia?"] = df["Nomofobia?"].astype(str).str.strip()
    for col in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("No se pudo leer el archivo DATOS REALES.xlsx. Sube el archivo desde la barra lateral.")
    uploaded = st.file_uploader("Sube DATOS REALES.xlsx", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
    else:
        st.stop()

# ------------------ Filtros ------------------
st.sidebar.header("🎚️ Filtros")
sexo_sel = st.sidebar.multiselect("Sexo", df["Sexo"].unique(), default=df["Sexo"].unique())
estrato_sel = st.sidebar.multiselect("Estrato", df["Estrato"].unique(), default=df["Estrato"].unique())
nomofobia_sel = st.sidebar.multiselect("Nomofobia? (Sí/No)", df["Nomofobia?"].unique(), default=df["Nomofobia?"].unique())

df_f = df[df["Sexo"].isin(sexo_sel) & df["Estrato"].isin(estrato_sel) & df["Nomofobia?"].isin(nomofobia_sel)]

# ------------------ KPIs ------------------
st.subheader("📊 Indicadores clave")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Participantes", len(df_f))
k2.metric("Media Horas Uso", f"{df_f['Horas_Uso'].mean():.2f}")
k3.metric("Media Nomofobia", f"{df_f['Nomofobia'].mean():.2f}")
k4.metric("Nomofobia (Sí)", f"{df_f['Nomofobia?'].value_counts(normalize=True).get('Sí',0)*100:.1f}%")

st.markdown("---")

# ------------------ Spearman Correlations ------------------
st.subheader("🔗 Correlaciones (Spearman)")
num_cols = ["Horas_Uso","Nomofobia","Ansiedad_social","Autoestima"]
num_cols = [c for c in num_cols if c in df_f.columns]
corr_matrix = df_f[num_cols].corr(method="spearman")

# Tabla
st.dataframe(corr_matrix.style.format("{:.3f}"), use_container_width=True)

# Heatmap interactivo
heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Mapa de calor — Correlaciones Spearman",
    zmin=-1, zmax=1,
)
st.plotly_chart(heatmap, use_container_width=True)

# Interpretaciones
st.write("**Interpretación:**")
for col in num_cols:
    if col != "Horas_Uso":
        rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[col], nan_policy="omit")
        msg = f"- Correlación {'positiva' if rho>0 else 'negativa'} entre Horas_Uso y {col} (ρ={rho:.3f}, p={p:.4f})"
        msg += " → significativa" if p<0.05 else " → no significativa"
        st.write(msg)

st.markdown("---")

# ------------------ Kruskal-Wallis & Dunn Post-Hoc ------------------
st.subheader("⚖️ Prueba de Kruskal–Wallis y Post-Hoc (Dunn)")
if "Estrato" in df_f.columns and "Nomofobia" in df_f.columns:
    groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
    if len(groups) > 1:
        H, p_kw = stats.kruskal(*groups)
        st.write(f"Estadístico H = {H:.3f} | p = {p_kw:.4f}")
        if p_kw < 0.05:
            st.success("Hay diferencias significativas entre al menos dos estratos.")
            # Dunn test
            dunn = sp.posthoc_dunn(df_f, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
            st.dataframe(dunn.style.format("{:.4f}"), use_container_width=True)
            st.caption("Matriz de p-valores ajustados (Bonferroni) — Prueba de Dunn")
        else:
            st.info("No se detectan diferencias significativas entre estratos.")
    else:
        st.warning("No hay suficientes grupos para aplicar Kruskal-Wallis.")
else:
    st.warning("Columnas requeridas no encontradas (Nomofobia y Estrato).")

# ------------------ Mann-Whitney ------------------
st.subheader("🧪 Prueba U de Mann–Whitney")
if set(df_f["Nomofobia?"].dropna().unique()) >= {"Sí", "No"}:
    g1 = df_f[df_f["Nomofobia?"]=="Sí"]["Horas_Uso"].dropna()
    g2 = df_f[df_f["Nomofobia?"]=="No"]["Horas_Uso"].dropna()
    stat, p = stats.mannwhitneyu(g1, g2)
    st.write(f"U = {stat:.3f} | p = {p:.4f}")
    if p < 0.05:
        st.success("Existe diferencia significativa en Horas de Uso entre grupos con y sin Nomofobia.")
    else:
        st.info("No se encontró diferencia significativa.")
    fig_mw = px.box(df_f, x="Nomofobia?", y="Horas_Uso", points="all", color="Nomofobia?")
    st.plotly_chart(fig_mw, use_container_width=True)

st.markdown("---")

# ------------------ Explorador Interactivo ------------------
st.subheader("🧭 Explorador dinámico")
with st.expander("Abrir panel de exploración"):
    x_var = st.selectbox("Eje X", num_cols, index=0)
    y_var = st.selectbox("Eje Y", num_cols, index=1)
    color_var = st.selectbox("Color por", [None, "Sexo", "Estrato", "Nomofobia?"], index=3)
    trendline = st.selectbox("Línea de tendencia", ["none","ols","lowess"], index=1)
    fig_exp = px.scatter(df_f, x=x_var, y=y_var, color=color_var, trendline=None if trendline=="none" else trendline)
    st.plotly_chart(fig_exp, use_container_width=True)

st.markdown("---")

# ------------------ Panel de Recomendaciones ------------------
st.subheader("💡 Recomendaciones y Conclusiones Estratégicas")
rec = []

# Spearman
for col in num_cols:
    if col != "Horas_Uso":
        rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[col], nan_policy="omit")
        if p < 0.05:
            if rho > 0.5:
                rec.append(f"- Incremento en Horas de uso se asocia **fuertemente** con mayor {col.lower()}. Se sugiere intervención educativa.")
            elif rho > 0:
                rec.append(f"- Existe correlación positiva leve entre Horas de uso y {col.lower()}, podría indicar un patrón inicial de dependencia.")
            else:
                rec.append(f"- Relación negativa entre Horas de uso y {col.lower()}, sugiere posible efecto protector o compensatorio.")
        else:
            rec.append(f"- No hay evidencia significativa de correlación entre Horas de uso y {col.lower()}.")

# Kruskal
if "p_kw" in locals() and p_kw < 0.05:
    rec.append("- Diferencias significativas entre estratos socioeconómicos sugieren influencia del contexto social en la Nomofobia.")
else:
    rec.append("- No se detectan diferencias entre estratos; el fenómeno parece homogéneo por nivel socioeconómico.")

# Mann–Whitney
if "p" in locals() and p < 0.05:
    rec.append("- Usuarios con Nomofobia presentan significativamente más horas de uso diario.")
else:
    rec.append("- El tiempo de uso no difiere significativamente entre quienes reportan y no reportan Nomofobia.")

st.write("\n".join(rec))

st.info("🧠 Interpretación general: los patrones indican que el uso intensivo del smartphone podría asociarse con mayores niveles de nomofobia y ansiedad social, pero no necesariamente con menor autoestima. Se recomienda explorar intervenciones diferenciadas por nivel de exposición y contexto socioeconómico.")

st.markdown("---")

# ------------------ Descargas ------------------
st.subheader("📥 Descargas")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Descargar datos filtrados (CSV)", csv, file_name="datos_filtrados.csv")

summary = "\n".join(rec)
st.download_button("Descargar recomendaciones (TXT)", summary.encode("utf-8"), file_name="recomendaciones.txt")

st.caption("Dashboard nomofobia | Estadística No Paramétrica | 2025")