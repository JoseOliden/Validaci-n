import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, f_oneway
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

st.title("Validación de Método Analítico Completa")

# -----------------------------
# CARGA DE DATOS
# -----------------------------
file = st.file_uploader("Sube un archivo CSV con tus resultados", type=["EXCEL"])

if file:
    df = pd.read_excel(file)
    st.write("Datos cargados:")
    st.write(df)
    st.write("Columnas detectadas:", df.columns.tolist())

    valores = df["valor"].values

    certificado = st.number_input("Valor certificado (CRM):", step=0.0001)

    # ==========================
    # 1. VERACIDAD
    # ==========================
    media = np.mean(valores)
    sesgo = media - certificado
    sesgo_pct = (sesgo / certificado) * 100

    # ==========================
    # 2. PRECISIÓN
    # ==========================
    sd = np.std(valores, ddof=1)
    rsd = (sd / media) * 100

    # ==========================
    # 3. INCERTIDUMBRE
    # ==========================
    u = sd / np.sqrt(len(valores))
    U = u * 2   # k = 2

    # ==========================
    # 4. SENSIBILIDAD
    # ==========================
    sensibilidad = media / certificado

    # ==========================
    # 5. LOD - LOQ
    # ==========================
    if "blanco" in df.columns:
        sd_blanco = np.std(df["blanco"], ddof=1)
        LOD = 3 * sd_blanco
        LOQ = 10 * sd_blanco
    else:
        LOD = LOQ = None

    # ==========================
    # 6. RANGO LINEAL
    # ==========================
    if "concentracion" in df.columns:
        conc = df["concentracion"].values
        slope, intercept, r_value, p_value, std_err = linregress(conc, valores)
        R2 = r_value**2
    else:
        R2 = None

    # ==========================
    # 7. ROBUSTEZ - ANOVA
    # ==========================
    if "grupo" in df.columns:
        grupos = df.groupby("grupo")["valor"].apply(list)
        anova = f_oneway(*grupos)
        p_anova = anova.pvalue
    else:
        p_anova = None

    # ==========================
    # VISUALIZACIÓN
    # ==========================
    st.header("Gráficos")

    # Histograma
    fig, ax = plt.subplots()
    ax.hist(valores, bins=10)
    st.pyplot(fig)

    # Boxplot
    fig, ax = plt.subplots()
    ax.boxplot(valores)
    st.pyplot(fig)

    # Curva de calibración
    if "concentracion" in df.columns:
        fig, ax = plt.subplots()
        ax.scatter(conc, valores, label="Datos")
        ax.plot(conc, slope*conc + intercept, label="Ajuste lineal")
        ax.set_xlabel("Concentración")
        ax.set_ylabel("Respuesta")
        ax.legend()
        st.pyplot(fig)

    # ==========================
    # RESULTADOS
    # ==========================
    st.header("Resultados de Validación")

    st.write(f"**Media:** {media:.6f}")
    st.write(f"**Sesgo:** {sesgo:.6f}")
    st.write(f"**Sesgo %:** {sesgo_pct:.3f}%")
    st.write(f"**Precisión (RSD %):** {rsd:.3f}%")
    st.write(f"**U (k=2):** {U:.6f}")
    st.write(f"**Sensibilidad:** {sensibilidad:.4f}")

    if LOD:
        st.write(f"**LOD:** {LOD:.6f}")
        st.write(f"**LOQ:** {LOQ:.6f}")

    if R2:
        st.write(f"**R² del ajuste:** {R2:.5f}")
        st.write("**Rango lineal aceptado**" if R2 >= 0.995 else "**Rango lineal rechazado**")

    if p_anova:
        st.write(f"**p-ANOVA:** {p_anova:.5f}")
        st.write("**Método robusto (no hay diferencias)**" if p_anova >= 0.05 else "**Método NO robusto**")

    # ==========================
    # GENERACIÓN DE PDF
    # ==========================
    st.header("Generar reporte PDF")

    if st.button("Descargar PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            doc = SimpleDocTemplate(tmp.name)
            styles = getSampleStyleSheet()
            story = []

            texto = f"""
            <b>REPORTE DE VALIDACIÓN DE MÉTODO</b><br/><br/>
            Media: {media:.6f}<br/>
            Sesgo: {sesgo:.6f}<br/>
            Sesgo %: {sesgo_pct:.3f}%<br/>
            RSD %: {rsd:.3f}<br/>
            Incertidumbre (k=2): {U:.6f}<br/>
            Sensibilidad: {sensibilidad:.4f}<br/><br/>
            LOD: {LOD}<br/>
            LOQ: {LOQ}<br/><br/>
            R²: {R2}<br/><br/>
            ANOVA p-value: {p_anova}<br/>
            """

            story.append(Paragraph(texto, styles["Normal"]))
            doc.build(story)

            st.download_button("Descargar reporte", data=open(tmp.name, "rb").read(),
                               file_name="reporte_validacion.pdf", mime="application/pdf")
