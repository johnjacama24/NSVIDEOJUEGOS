import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Cargar modelo, diccionario y dataframe de referencia
# ------------------------------
@st.cache_resource
def cargar_modelo():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)

        dicc_desercion = data["label_encoder_mapping"]
        
        return (
            data["model"],
            dicc_desercion,
            data["diccionario_genero"],
            data["diccionario_estado_civil"],
            data["diccionario_estado_aprendiz"],
            data["dataframe_codificado_top5"],
        )

modelo, dicc_desercion, dicc_genero, dicc_estado_civil, dicc_estado_aprendiz, df_ref = cargar_modelo()

# ------------------------------
# Invertir los diccionarios para mostrar en el selectbox y mapear al código
# ------------------------------
inv_genero = {v: k for k, v in dicc_genero.items()}
inv_estado_civil = {v: k for k, v in dicc_estado_civil.items()}
inv_estado_aprendiz = {v: k for k, v in dicc_estado_aprendiz.items()}


# ------------------------------
# Interfaz de usuario
# ------------------------------
st.title("🧠 Predicción del Riesgo de Deserción MEDIOS GRÁFICOS VISUALES")
st.markdown("Seleccione las opciones correspondientes y presione el botón para predecir.")

# Campos de entrada
edad = st.slider("Edad", 15, 60, 25)
reversiones = st.slider("Cantidad de Reversiones", 0, 10, 0)
quejas = st.slider("Cantidad de quejas", 0, 10, 0)
estrato = st.slider("Estrato", 0, 10, 0)

genero_opcion = st.selectbox("Género", list(dicc_genero.keys()))
estado_civil_opcion = st.selectbox("Estado Civil", list(dicc_estado_civil.keys()))
estado_aprendiz_opcion = st.selectbox("Estado Aprendiz", list(dicc_estado_aprendiz.keys()))

# ------------------------------
# Botón para predecir
# ------------------------------
if st.button("🔍 Realizar predicción"):
    try:
        fila = df_ref.drop(columns=["cluster"]).iloc[0].copy()

        fila["Edad"] = edad
        fila["Cantidad de quejas"] = quejas
        fila["Cantidad de Reversiones"] = reversiones
        fila["Género"] = dicc_genero[genero_opcion]
        fila["Estado Civil"] = dicc_estado_civil[estado_civil_opcion]
        fila["Estado Aprendiz"] = dicc_estado_aprendiz[estado_aprendiz_opcion]
        fila["Estrato"] = estrato

        entrada = pd.DataFrame([fila])
        
        if "Nivel de Desercion" in entrada.columns:
            entrada = entrada.drop(columns=["Nivel de Desercion"])
            
        pred_codificada = modelo.predict(entrada)[0]

        
        st.write(f"🔢 Número de Cluster:", pred_codificada)
        st.write("📊 Riesgo de deserción:", dicc_desercion)
        #st.write("🧪 Tipo:", type(pred_codificada))
        #pred_original = dicc_desercion.get(str(pred_codificada), "Desconocido")
        #st.success(f"✅ Estado del aprendiz predicho: **{pred_original}**")
    
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
