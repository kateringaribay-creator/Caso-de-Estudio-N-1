import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# =========================
# CLASE POO
# =========================

class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def get_numeric_columns(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()

    def get_categorical_columns(self):
        return self.df.select_dtypes(include='object').columns.tolist()

    def missing_values(self):
        return self.df.isnull().sum()

    def descriptive_stats(self):
        return self.df.describe()

    def mode_value(self, column):
        return self.df[column].mode()[0]

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📌 Navegación")

menu = st.sidebar.selectbox(
    "Selecciona un módulo",
    [
        "Home",
        "Carga Dataset",
        "EDA",
        "Conclusiones"
    ]
)

# =========================
# HOME
# =========================

if menu == "Home":

    st.title("📊 Bank Marketing EDA App")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Objetivo")

        st.write(
            """
            Analizar el comportamiento de los clientes de una institución financiera
            para identificar factores relacionados con la aceptación de campañas de marketing.
            """
        )

        st.subheader("🧠 Tecnologías Utilizadas")

        st.write(
            """
            - Python
            - Pandas
            - NumPy
            - Matplotlib
            - Seaborn
            - Streamlit
            """
        )

    with col2:

        st.subheader("👨‍🎓 Datos del Autor")

        st.write(
            """
            Nombre: KATERIN DELFINA GARIBAY FERNANDEZ

            Especialización: Python for Analytics

            Año: 2026
            """
        )

        st.subheader("🏦 Dataset")

        st.write(
            """
            El dataset contiene información de campañas de marketing bancario,
            incluyendo características demográficas, financieras y resultados
            de campañas anteriores.
            """
        )
# =========================
# CARGA DATASET
# =========================

elif menu == "Carga Dataset":

    st.title("📂 Carga del Dataset")

    uploaded_file = st.file_uploader(
        "Sube el archivo BankMarketing.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=';')

        st.success("✅ Archivo cargado correctamente")

        st.subheader("👀 Vista previa")
        st.dataframe(df.head())

        st.subheader("📐 Dimensiones")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Filas", df.shape[0])

        with col2:
            st.metric("Columnas", df.shape[1])

    else:
        st.warning("⚠️ Debes cargar un archivo CSV")
# =========================
# EDA
# =========================

elif menu == "EDA":

    st.title("📊 Análisis Exploratorio de Datos")

    uploaded_file = st.file_uploader(
        "Sube el archivo CSV para análisis",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=';')

        analyzer = DataAnalyzer(df)

        tabs = st.tabs([
            "Información General",
            "Clasificación Variables",
            "Estadísticas",
            "Valores Faltantes",
            "Distribuciones",
            "Categóricas",
            "Bivariado Numérico",
            "Bivariado Categórico",
            "Análisis Dinámico",
            "Hallazgos"
        ])
                

        # =========================================
        # TAB 1 - INFORMACIÓN GENERAL
        # =========================================

        with tabs[0]:

            st.subheader("ℹ️ Información General")

            buffer = io.StringIO()

            df.info(buf=buffer)

            s = buffer.getvalue()

            st.text(s)

            st.subheader("📌 Tipos de Datos")

            tipos_df = pd.DataFrame({
                "Columna": df.columns,
                "Tipo de Dato": df.dtypes.astype(str)
            })

            st.dataframe(tipos_df)

            st.subheader("❗ Conteo de Valores Nulos")

            null_df = pd.DataFrame({
                "Columna": df.columns,
                "Valores Nulos": df.isnull().sum().values
            })

            st.dataframe(null_df)


    
        # =========================================
        # TAB 2 - CLASIFICACIÓN DE VARIABLES
        # =========================================

        with tabs[1]:

            st.subheader("🔎 Clasificación de Variables")

            # VARIABLES NUMÉRICAS
            numeric_cols = analyzer.get_numeric_columns()

            # VARIABLES CATEGÓRICAS
            categorical_cols = analyzer.get_categorical_columns()

            col1, col2 = st.columns(2)

            # =========================
            # NUMÉRICAS
            # =========================

            with col1:

                st.markdown("### 🔢 Variables Numéricas")

                numeric_df = pd.DataFrame({
                    "Variables Numéricas": numeric_cols
                })

                st.dataframe(numeric_df)

                st.metric(
                    "Cantidad de Variables Numéricas",
                    len(numeric_cols)
                )

            # =========================
            # CATEGÓRICAS
            # =========================

            with col2:

                st.markdown("### 🏷️ Variables Categóricas")

                categorical_df = pd.DataFrame({
                    "Variables Categóricas": categorical_cols
                })

                st.dataframe(categorical_df)

                st.metric(
                    "Cantidad de Variables Categóricas",
                    len(categorical_cols)
                )

            # =========================
            # RESUMEN
            # =========================

            st.markdown("---")

            st.subheader("📌 Resumen del Dataset")

            total_variables = len(df.columns)

            col3, col4, col5 = st.columns(3)

            with col3:
                st.metric(
                    "Total Variables",
                    total_variables
                )

            with col4:
                st.metric(
                    "Variables Numéricas",
                    len(numeric_cols)
                )

            with col5:
                st.metric(
                    "Variables Categóricas",
                    len(categorical_cols)
                )

            # =========================
            # INSIGHT
            # =========================

            st.info(
                """
                El dataset contiene una combinación equilibrada de variables
                numéricas y categóricas, lo que permite realizar análisis
                descriptivos y comparativos sobre características de clientes
                y resultados de campañas de marketing.
                """
            )
            # =========================================
        # TAB 3 - ESTADÍSTICAS DESCRIPTIVAS
        # =========================================

        with tabs[2]:

            st.subheader("📈 Estadísticas Descriptivas")

            # =========================
            # DESCRIBE GENERAL
            # =========================

            st.markdown("### 📋 Resumen Estadístico")

            st.dataframe(
                analyzer.descriptive_stats()
            )

            st.markdown("---")

            # =========================
            # SELECCIÓN VARIABLE
            # =========================

            numeric_cols = analyzer.get_numeric_columns()

            variable = st.selectbox(
                "Selecciona una variable numérica",
                numeric_cols
            )

            st.markdown(f"### 📊 Análisis Estadístico de: {variable}")

            # =========================
            # MÉTRICAS
            # =========================

            media = round(df[variable].mean(), 2)
            mediana = round(df[variable].median(), 2)
            moda = round(df[variable].mode()[0], 2)
            desviacion = round(df[variable].std(), 2)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Media", media)

            with col2:
                st.metric("Mediana", mediana)

            with col3:
                st.metric("Moda", moda)

            with col4:
                st.metric("Desv. Estándar", desviacion)

            st.markdown("---")

            # =========================
            # INTERPRETACIÓN
            # =========================

            st.subheader("🧠 Interpretación Básica")

            st.info(
                f"""
                La variable '{variable}' presenta una media de {media},
                una mediana de {mediana} y una desviación estándar de {desviacion}.
                
                Esto permite evaluar la tendencia central y dispersión
                de los datos dentro del dataset.
                """
            )

            # =========================
            # BOXPLOT
            # =========================

            st.subheader("📦 Boxplot")

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.boxplot(
                x=df[variable],
                ax=ax
            )

            st.pyplot(fig)

        # =========================================
        # TAB 4 - ANÁLISIS DE VALORES FALTANTES
        # =========================================

        with tabs[3]:

            st.subheader("🚨 Análisis de Valores Faltantes")

            # =========================
            # CONTEO DE NULOS
            # =========================

            missing = analyzer.missing_values()

            missing_df = pd.DataFrame({
                "Variable": missing.index,
                "Valores Faltantes": missing.values
            })

            st.markdown("### 📋 Conteo de Valores Nulos")

            st.dataframe(missing_df)

            st.markdown("---")

            # =========================
            # TOTAL DE NULOS
            # =========================

            total_missing = missing.sum()

            st.metric(
                "Total de Valores Faltantes",
                int(total_missing)
            )

            st.markdown("---")

            # =========================
            # GRÁFICO
            # =========================

            st.markdown("### 📊 Visualización de Valores Faltantes")

            fig, ax = plt.subplots(figsize=(12, 5))

            sns.barplot(
                x=missing.index,
                y=missing.values,
                ax=ax
            )

            plt.xticks(rotation=90)

            ax.set_xlabel("Variables")
            ax.set_ylabel("Cantidad de Valores Faltantes")

            st.pyplot(fig)

            st.markdown("---")

            # =========================
            # INTERPRETACIÓN
            # =========================

            st.subheader("🧠 Interpretación")

            if total_missing == 0:

                st.success(
                    """
                    El dataset no presenta valores faltantes.
                    Esto facilita el análisis exploratorio y evita
                    procesos adicionales de limpieza de datos.
                    """
                )

            else:

                st.warning(
                    """
                    El dataset presenta valores faltantes en algunas variables.
                    Será necesario considerar técnicas de limpieza o imputación
                    antes de realizar análisis más avanzados.
                    """
                )
        # =========================================
        # TAB 5 - DISTRIBUCIÓN VARIABLES NUMÉRICAS
        # =========================================

        with tabs[4]:

            st.subheader("📊 Distribución de Variables Numéricas")

            numeric_cols = analyzer.get_numeric_columns()

            variable = st.selectbox(
                "Selecciona variable numérica",
                numeric_cols,
                key="histogram"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(
                df[variable],
                kde=True,
                ax=ax
            )

            ax.set_title(f"Distribución de {variable}")

            st.pyplot(fig)

            st.info(
                f"""
                El histograma permite visualizar la distribución de la variable
                {variable}, identificando concentración de valores,
                dispersión y posibles outliers.
                """
            )

        # =========================================
        # TAB 6 - VARIABLES CATEGÓRICAS
        # =========================================

        with tabs[5]:

            st.subheader("🏷️ Análisis de Variables Categóricas")

            categorical_cols = analyzer.get_categorical_columns()

            variable_cat = st.selectbox(
                "Selecciona variable categórica",
                categorical_cols,
                key="categorical"
            )

            conteo = df[variable_cat].value_counts()

            st.dataframe(conteo)

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.countplot(
                data=df,
                x=variable_cat,
                ax=ax
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

            proporciones = round(
                (conteo / conteo.sum()) * 100,
                2
            )

            st.subheader("📌 Proporciones (%)")

            st.dataframe(proporciones)

        # =========================================
        # TAB 7 - BIVARIADO NUMÉRICO VS CATEGÓRICO
        # =========================================

        with tabs[6]:

            st.subheader("📈 Análisis Bivariado")

            numeric_cols = analyzer.get_numeric_columns()

            variable_num = st.selectbox(
                "Selecciona variable numérica",
                numeric_cols,
                key="bivariado_num"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.boxplot(
                data=df,
                x="y",
                y=variable_num,
                ax=ax
            )

            st.pyplot(fig)

            st.info(
                f"""
                El boxplot permite comparar la distribución de
                {variable_num} según la respuesta final de la campaña.
                """
            )

        # =========================================
        # TAB 8 - CATEGÓRICO VS CATEGÓRICO
        # =========================================

        with tabs[7]:

            st.subheader("🔄 Variables Categóricas vs Categóricas")

            categorical_cols = analyzer.get_categorical_columns()

            var1 = st.selectbox(
                "Selecciona primera variable",
                categorical_cols,
                key="cat1"
            )

            var2 = st.selectbox(
                "Selecciona segunda variable",
                categorical_cols,
                key="cat2"
            )

            cross = pd.crosstab(
                df[var1],
                df[var2]
            )

            st.dataframe(cross)

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.heatmap(
                cross,
                annot=True,
                fmt="d",
                cmap="Blues"
            )

            st.pyplot(fig)

        # =========================================
        # TAB 9 - ANÁLISIS DINÁMICO
        # =========================================

        with tabs[8]:

            st.subheader("⚙️ Análisis Dinámico")

            numeric_cols = analyzer.get_numeric_columns()

            selected_cols = st.multiselect(
                "Selecciona variables numéricas",
                numeric_cols
            )

            if len(selected_cols) > 0:

                st.dataframe(
                    df[selected_cols].describe()
                )

                variable_slider = st.selectbox(
                    "Variable para filtrar",
                    selected_cols
                )

                min_val = int(df[variable_slider].min())
                max_val = int(df[variable_slider].max())

                rango = st.slider(
                    "Selecciona rango",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )

                filtered_df = df[
                    (df[variable_slider] >= rango[0]) &
                    (df[variable_slider] <= rango[1])
                ]

                st.write(
                    f"Filas filtradas: {filtered_df.shape[0]}"
                )

                st.dataframe(filtered_df.head())

            else:
                st.warning(
                    "Selecciona al menos una variable."
                )

        # =========================================
        # TAB 10 - HALLAZGOS CLAVE
        # =========================================

        with tabs[9]:

            st.subheader("🧠 Hallazgos Clave")

            st.success(
                """
                1. La mayoría de clientes no aceptó la campaña.
                
                2. Existen diferencias importantes en duración
                de llamadas entre clientes que aceptaron y no aceptaron.
                
                3. Algunas variables categóricas muestran patrones
                relevantes relacionados con la aceptación.
                
                4. El dataset presenta buena calidad de datos,
                sin valores faltantes significativos.
                
                5. El análisis exploratorio permite identificar
                segmentos potenciales para futuras campañas.
                """
            )

            st.markdown("---")

            st.subheader("📌 Conclusión General")

            st.write(
                """
                El EDA permitió comprender mejor el comportamiento
                de los clientes y detectar variables relevantes
                para campañas de marketing bancario.
                """
            )
    # =========================
# CONCLUSIONES
# =========================

elif menu == "Conclusiones":

    st.title("🧠 Conclusiones Finales")

    st.markdown("---")

    conclusiones = [

        "La mayoría de clientes no aceptó la campaña de marketing, lo que evidencia una baja tasa de conversión.",

        "La duración de la llamada presenta una relación importante con la aceptación de la campaña.",

        "Las variables categóricas como trabajo, educación y tipo de contacto muestran diferencias relevantes entre clientes.",

        "El dataset presenta buena calidad de información y no contiene valores faltantes significativos.",

        "El análisis exploratorio permite identificar segmentos potenciales de clientes para mejorar futuras campañas."
    ]

    for i, conclusion in enumerate(conclusiones, start=1):

        st.success(f"Conclusión {i}")

        st.write(conclusion)

        st.markdown("---")

    st.subheader("📌 Reflexión Final")

    st.info(
        """
        El análisis exploratorio de datos permitió comprender mejor
        el comportamiento de los clientes y detectar patrones relevantes
        relacionados con las campañas de marketing bancario.

        Este proyecto integró conceptos fundamentales de Python,
        programación orientada a objetos, análisis estadístico
        y visualización de datos utilizando Streamlit.
        """
    )