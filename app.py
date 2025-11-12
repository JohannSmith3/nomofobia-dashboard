# app.py 

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
import itertools
import scikit_posthocs as sp

# ------------------ CONFIGURACIÓN ------------------
st.set_page_config(page_title="Dashboard Nomofobia — Versión Final", layout="wide")
st.title("📱 Dashboard Avanzado — Dependencia al Smartphone y Factores Psicológicos")
st.caption("Proyecto Final | Estadística No Paramétrica | Johann Smith (2025)")

# ------------------ CARGA DE DATOS ------------------
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

# ------------------ FILTROS ------------------
st.sidebar.header("🎚️ Filtros de visualización")
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

# ------------------ CORRELACIONES SPEARMAN ------------------
st.subheader("🔗 Correlaciones (Spearman)")

num_cols = ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]
num_cols = [c for c in num_cols if c in df_f.columns]
corr_matrix = df_f[num_cols].corr(method="spearman")

st.dataframe(corr_matrix.style.format("{:.3f}"), use_container_width=True)

heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Mapa de calor — Correlaciones Spearman",
    zmin=-1, zmax=1,
)
st.plotly_chart(heatmap, use_container_width=True)

# Interpretación automática
for col in num_cols:
    if col != "Horas_Uso":
        rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[col], nan_policy="omit")
        msg = f"- Correlación {'positiva' if rho>0 else 'negativa'} entre Horas_Uso y {col} (ρ={rho:.3f}, p={p:.4f})"
        msg += " → significativa" if p<0.05 else " → no significativa"
        st.write(msg)

st.markdown("---")

# ------------------ MANN–WHITNEY ------------------
st.subheader("🧪 Prueba U de Mann–Whitney (Horas de Uso ~ Nomofobia?)")
if set(df_f["Nomofobia?"].dropna().unique()) >= {"Sí", "No"}:
    g1 = df_f[df_f["Nomofobia?"]=="Sí"]["Horas_Uso"].dropna()
    g2 = df_f[df_f["Nomofobia?"]=="No"]["Horas_Uso"].dropna()
    stat, p = stats.mannwhitneyu(g1, g2)
    st.write(f"U = {stat:.3f} | p = {p:.4f}")
    if p < 0.05:
        st.success("Existe diferencia significativa en Horas de Uso entre quienes presentan Nomofobia y quienes no.")
    else:
        st.info("No se encontró diferencia significativa entre los grupos.")
    fig_mw = px.box(df_f, x="Nomofobia?", y="Horas_Uso", points="all", color="Nomofobia?",
                    title="Distribución de Horas de Uso según Nomofobia (Sí/No)")
    st.plotly_chart(fig_mw, use_container_width=True)

st.markdown("---")

# ------------------ KRUSKAL–WALLIS ------------------
st.subheader("⚖️ Prueba de Kruskal–Wallis (Nomofobia ~ Estrato)")
if "Estrato" in df_f.columns and "Nomofobia" in df_f.columns:
    groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
    if len(groups) > 1:
        H, p_kw = stats.kruskal(*groups)
        st.write(f"Estadístico H = {H:.3f} | p = {p_kw:.4f}")
        fig_kw = px.box(df_f, x="Estrato", y="Nomofobia", color="Estrato", points="all",
                        title="Distribución de Nomofobia por Estrato Socioeconómico")
        st.plotly_chart(fig_kw, use_container_width=True)
        if p_kw < 0.05:
            st.success("Se detectan diferencias significativas entre al menos dos estratos (p < 0.05).")
        else:
            st.info("No se detectan diferencias significativas entre estratos.")
    else:
        st.warning("No hay suficientes grupos para aplicar Kruskal-Wallis.")
else:
    st.warning("Columnas requeridas no encontradas (Nomofobia y Estrato).")

st.markdown("---")

# ------------------ DUNN POST-HOC ------------------
st.subheader("📈 Análisis Post-Hoc — Prueba de Dunn (Bonferroni)")
if "Estrato" in df_f.columns and "Nomofobia" in df_f.columns:
    try:
        dunn = sp.posthoc_dunn(df_f, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
        st.dataframe(dunn.style.format("{:.4f}"), use_container_width=True)
        fig_dunn = px.imshow(dunn, text_auto=True, color_continuous_scale="Blues", 
                             title="Mapa de significancia — Post-Hoc Dunn Test (p-ajustada)")
        st.plotly_chart(fig_dunn, use_container_width=True)
    except Exception as e:
        st.error("No se pudo calcular el test de Dunn. Verifica que existan suficientes observaciones por grupo.")
else:
    st.warning("Datos insuficientes para realizar el test de Dunn.")

st.markdown("---")

# ------------------ EXPLORADOR 1 ------------------
st.subheader("🧭 Explorador Interactivo General")
with st.expander("Abrir explorador de relaciones"):
    num_vars = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
    x_var = st.selectbox("Eje X", num_vars, index=0)
    y_var = st.selectbox("Eje Y", num_vars, index=1)
    color_var = st.selectbox("Color por", [None, "Sexo", "Estrato", "Nomofobia?"], index=3)
    trendline = st.selectbox("Línea de tendencia", ["none","ols","lowess"], index=1)
    fig_exp = px.scatter(df_f, x=x_var, y=y_var, color=color_var, trendline=None if trendline=="none" else trendline,
                         title=f"{y_var} vs {x_var}")
    st.plotly_chart(fig_exp, use_container_width=True)

st.markdown("---")

# ------------------ EXPLORADOR 2: COMPARADOR DE CORRELACIONES ------------------
st.subheader("🧮 Explorador Comparador de Correlaciones")
with st.expander("Analizar correlación entre dos variables numéricas"):
    var1 = st.selectbox("Variable 1", num_cols, index=0, key="var1")
    var2 = st.selectbox("Variable 2", num_cols, index=1, key="var2")
    if var1 != var2:
        rho, p = stats.spearmanr(df_f[var1], df_f[var2], nan_policy="omit")
        st.write(f"ρ = {rho:.3f} | p = {p:.4f}")
        if p < 0.05:
            st.success("Correlación significativa (p < 0.05)")
        else:
            st.info("No se detecta correlación significativa.")
        fig_cmp = px.scatter(df_f, x=var1, y=var2, trendline="ols", color="Nomofobia?",
                             title=f"Relación entre {var1} y {var2}")
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.warning("Selecciona dos variables distintas para comparar.")

st.markdown("---")

# ------------------ PANEL DE RECOMENDACIONES ------------------
st.subheader("💡 Recomendaciones y Conclusiones Estratégicas")
rec = []

# Spearman
for col in num_cols:
    if col != "Horas_Uso":
        rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[col], nan_policy="omit")
        if p < 0.05:
            if rho > 0.5:
                rec.append(f"- **Relación fuerte positiva:** Mayor uso se asocia con mayor {col.lower()}. Se recomienda intervención educativa.")
            elif rho > 0:
                rec.append(f"- **Relación positiva leve:** Podría indicar un patrón inicial de dependencia.")
            else:
                rec.append(f"- **Relación negativa:** Puede reflejar mecanismos compensatorios o protectores.")
        else:
            rec.append(f"- No hay correlación significativa entre horas de uso y {col.lower()}.")

# Kruskal
if "p_kw" in locals() and p_kw < 0.05:
    rec.append("- Se observan diferencias entre estratos, lo cual sugiere influencia del nivel socioeconómico.")
else:
    rec.append("- No se detectan diferencias notables entre estratos en el puntaje de Nomofobia.")

# Mann–Whitney
if "p" in locals() and p < 0.05:
    rec.append("- Usuarios con Nomofobia tienen significativamente más horas de uso diario.")
else:
    rec.append("- No hay diferencias en horas de uso entre grupos con y sin Nomofobia.")

st.write("\n".join(rec))
st.info("🧠 Interpretación general: Los resultados apuntan a que la exposición prolongada al smartphone se relaciona con mayores niveles de nomofobia y ansiedad social, mientras que las diferencias por estrato son menos pronunciadas.")

st.caption("Dashboard nomofobia | Estadística No Paramétrica | 2025")
