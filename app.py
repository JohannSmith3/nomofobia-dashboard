# app.py — Dashboard Nomofobia Final con contexto, descriptivas y análisis ampliado

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import scikit_posthocs as sp

# ------------------ CONFIGURACIÓN ------------------
st.set_page_config(page_title="Dashboard Nomofobia — Proyecto Final", layout="wide")
st.title("📱 Análisis de Nomofobia y Factores Psicológicos en Usuarios de Smartphone")
st.caption("Proyecto Final — Estadística No Paramétrica | Johann Smith (2025)")

# ------------------ CONTEXTO DEL ESTUDIO ------------------
st.markdown("""
### 🧩 Contexto del estudio

La **nomofobia** se define como el miedo irracional a estar sin acceso al teléfono móvil o a perder la conexión con el entorno digital.  
Este estudio analiza **la relación entre el uso del smartphone y variables psicológicas** como:

- **Ansiedad social**
- **Autoestima**
- **Estrato socioeconómico**
- **Presencia o ausencia de nomofobia**

El objetivo principal es **evaluar la existencia de asociaciones y diferencias significativas** usando pruebas **no paramétricas**:  
- **Spearman** (correlaciones)  
- **Mann–Whitney U** (comparación entre dos grupos)  
- **Kruskal–Wallis y Dunn post-hoc** (comparaciones entre múltiples estratos)

""")


st.markdown("---")

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

# ------------------ DESCRIPTIVAS ------------------
st.subheader("📈 Estadísticas descriptivas")

st.markdown("""
En esta sección se muestran las **tendencias generales** de las principales variables del estudio.
Se busca describir el comportamiento general de la muestra antes de aplicar pruebas inferenciales.
""")

desc = df_f[["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]].describe()
st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

# Visualizaciones descriptivas
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df_f, x="Horas_Uso", nbins=20, color="Nomofobia?",
                        title="Distribución de Horas de Uso según Nomofobia",
                        marginal="box", color_discrete_sequence=["#90CAF9", "#E57373"])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.box(df_f, x="Sexo", y="Nomofobia", color="Sexo",
                  title="Distribución de Nomofobia por Sexo")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ------------------ CORRELACIONES SPEARMAN ------------------
st.subheader("🔗 Correlaciones de Spearman")

st.markdown("""
Evalúa la **asociación monotónica** entre el uso del smartphone y las variables psicológicas.  
Esta prueba es adecuada cuando las variables **no cumplen supuestos de normalidad**.
""")

num_cols = ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima"]
corr_matrix = df_f[num_cols].corr(method="spearman")

heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Mapa de calor — Correlaciones Spearman",
    zmin=-1, zmax=1,
)
st.plotly_chart(heatmap, use_container_width=True)

# Interpretaciones automáticas
st.markdown("**Interpretación automática:**")
for col in num_cols:
    if col != "Horas_Uso":
        rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[col], nan_policy="omit")
        msg = f"- Correlación {'positiva' if rho>0 else 'negativa'} entre Horas de Uso y {col} (ρ={rho:.3f}, p={p:.4f})"
        msg += " → significativa" if p<0.05 else " → no significativa"
        st.write(msg)

st.info("💬 Un valor de ρ cercano a ±1 indica una asociación fuerte; valores cercanos a 0 sugieren independencia.")
st.markdown("---")

# ------------------ MANN–WHITNEY ------------------
st.subheader("🧪 Prueba U de Mann–Whitney (Horas de Uso ~ Nomofobia?)")

st.markdown("""
Permite **comparar las horas de uso promedio** entre quienes **presentan nomofobia (Sí)** y quienes **no (No)**.
""")

if set(df_f["Nomofobia?"].dropna().unique()) >= {"Sí", "No"}:
    g1 = df_f[df_f["Nomofobia?"]=="Sí"]["Horas_Uso"].dropna()
    g2 = df_f[df_f["Nomofobia?"]=="No"]["Horas_Uso"].dropna()
    stat, p = stats.mannwhitneyu(g1, g2)
    st.write(f"**U = {stat:.3f} | p = {p:.4f}**")
    if p < 0.05:
        st.success("Existe diferencia significativa en Horas de Uso entre los grupos.")
    else:
        st.info("No se encontró diferencia significativa.")
    fig_mw = px.box(df_f, x="Nomofobia?", y="Horas_Uso", points="all", color="Nomofobia?",
                    title="Comparación de Horas de Uso según Nomofobia (Sí/No)")
    st.plotly_chart(fig_mw, use_container_width=True)
    st.caption("Interpretación: diferencias significativas implican que el tiempo de uso está asociado con la presencia de nomofobia.")

st.markdown("---")

# ------------------ KRUSKAL–WALLIS ------------------
st.subheader("⚖️ Prueba de Kruskal–Wallis (Nomofobia ~ Estrato)")

st.markdown("""
Evalúa si **el puntaje de Nomofobia difiere entre los distintos estratos socioeconómicos**.  
Es una alternativa no paramétrica al ANOVA.
""")

if "Estrato" in df_f.columns and "Nomofobia" in df_f.columns:
    groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
    if len(groups) > 1:
        H, p_kw = stats.kruskal(*groups)
        st.write(f"**Estadístico H = {H:.3f} | p = {p_kw:.4f}**")
        fig_kw = px.box(df_f, x="Estrato", y="Nomofobia", color="Estrato", points="all",
                        title="Puntaje de Nomofobia por Estrato Socioeconómico")
        st.plotly_chart(fig_kw, use_container_width=True)
        if p_kw < 0.05:
            st.success("Se detectan diferencias significativas entre al menos dos estratos (p < 0.05).")
        else:
            st.info("No se detectan diferencias significativas entre estratos.")
    else:
        st.warning("No hay suficientes grupos para aplicar Kruskal-Wallis.")
else:
    st.warning("Columnas requeridas no encontradas (Nomofobia y Estrato).")

st.caption("Interpretación: un p < 0.05 sugiere que el nivel de nomofobia varía según el estrato socioeconómico.")
st.markdown("---")

# ------------------ DUNN POST-HOC ------------------
st.subheader("📈 Análisis Post-Hoc — Prueba de Dunn (Bonferroni)")

st.markdown("""
Si el test de Kruskal–Wallis detecta diferencias, la prueba de Dunn identifica **entre qué grupos específicos** se encuentran esas diferencias.
""")

if "Estrato" in df_f.columns and "Nomofobia" in df_f.columns:
    try:
        dunn = sp.posthoc_dunn(df_f, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")
        st.dataframe(dunn.style.format("{:.4f}"), use_container_width=True)
        fig_dunn = px.imshow(dunn, text_auto=True, color_continuous_scale="Blues", 
                             title="Mapa de significancia — Post-Hoc Dunn Test (p-ajustada)")
        st.plotly_chart(fig_dunn, use_container_width=True)
        st.caption("Interpretación: celdas con valores p < 0.05 indican pares de estratos con diferencias significativas en nomofobia.")
    except Exception as e:
        st.error("No se pudo calcular el test de Dunn. Verifica que existan suficientes observaciones por grupo.")
else:
    st.warning("Datos insuficientes para realizar el test de Dunn.")

st.markdown("---")

# ------------------ EXPLORADOR 1 ------------------
st.subheader("🧭 Explorador Interactivo General")
st.markdown("Permite **examinar relaciones bivariadas** entre las variables cuantitativas o categóricas seleccionadas.")

with st.expander("Abrir explorador"):
    num_vars = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
    x_var = st.selectbox("Eje X", num_vars, index=0)
    y_var = st.selectbox("Eje Y", num_vars, index=1)
    color_var = st.selectbox("Color por", [None, "Sexo", "Estrato", "Nomofobia?"], index=3)
    trendline = st.selectbox("Línea de tendencia", ["none","ols","lowess"], index=1)
    fig_exp = px.scatter(df_f, x=x_var, y=y_var, color=color_var, trendline=None if trendline=="none" else trendline,
                         title=f"Relación entre {x_var} y {y_var}")
    st.plotly_chart(fig_exp, use_container_width=True)

st.caption("Interpretación: las líneas de tendencia y los colores ayudan a identificar posibles agrupaciones o asociaciones visuales.")
st.markdown("---")

# ------------------ EXPLORADOR 2 ------------------
st.subheader("🧮 Explorador Comparador de Correlaciones")
st.markdown("Analiza la **fuerza y dirección de la correlación** entre dos variables numéricas específicas.")

with st.expander("Abrir comparador"):
    var1 = st.selectbox("Variable 1", num_cols, index=0, key="var1")
    var2 = st.selectbox("Variable 2", num_cols, index=1, key="var2")
    if var1 != var2:
        rho, p = stats.spearmanr(df_f[var1], df_f[var2], nan_policy="omit")
        st.write(f"**ρ = {rho:.3f} | p = {p:.4f}**")
        if p < 0.05:
            st.success("Correlación significativa (p < 0.05)")
        else:
            st.info("No se detecta correlación significativa.")
        fig_cmp = px.scatter(df_f, x=var1, y=var2, trendline="ols", color="Nomofobia?",
                             title=f"Relación entre {var1} y {var2}")
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("Interpretación: valores de ρ altos indican fuerte relación monotónica, positiva o negativa.")
    else:
        st.warning("Selecciona dos variables distintas para comparar.")

st.markdown("---")

# ------------------ CONCLUSIONES Y RECOMENDACIONES ------------------
st.subheader("💡 Conclusiones Generales y Recomendaciones")

st.markdown("""
A partir de los análisis realizados, se concluye que:

1. **El uso intensivo del smartphone** presenta una asociación positiva con la **nomofobia y la ansiedad social**, lo cual respalda las hipótesis de dependencia psicológica.
2. **Las diferencias entre estratos** no son siempre significativas, aunque los niveles más altos de uso se concentran en los estratos medios.
3. **No se observaron correlaciones fuertes con la autoestima**, lo que sugiere que la nomofobia podría operar independientemente de la autopercepción personal.
4. Se recomienda profundizar con análisis longitudinales y modelos multivariados para evaluar causalidad.
""")

st.info("🧠 En síntesis: los resultados confirman patrones conductuales coherentes con la literatura sobre dependencia digital y nomofobia, apoyando la necesidad de intervenciones preventivas dirigidas a jóvenes usuarios intensivos de smartphones.")

st.caption("Versión Final — Incluye contexto, descriptivas, interpretación ampliada y conclusiones académicas.")
