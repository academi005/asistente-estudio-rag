import streamlit as st
import os
import re
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ==========================================
# CAPA 1: CONFIGURACIÓN E INTERFAZ
# ==========================================
INTERNAL_API_KEY = st.secrets["GOOGLE_API_KEY"] 

st.set_page_config(page_title="Academia IA Pro - UNSA", page_icon="🏫", layout="wide", initial_sidebar_state="expanded")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stAppDeployButton {display:none;}
            #stDecoration {display:none;}
            
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin-top: 50px;
                text-align: center;
            }
            
            button[title="Manage app"] {display: none !important;}
            .st-emotion-cache-1wbqy5l {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# CAPA 2: SEGURIDAD INTERNA Y ACCESO
# ==========================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.title("🔐 Academia Virtual UNSA")
        st.write("Bienvenido al sistema de preparación de alto rendimiento para Ingeniería.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        password_input = st.text_input("Introduce tu código de alumno:", type="password")
        
        st.warning("⚠️ El sistema detecta accesos simultáneos. No compartas tu código.")
        
        if st.button("Entrar", use_container_width=True):
            try:
                contraseñas_validas = list(st.secrets["alumnos"].values())
                if password_input in contraseñas_validas:
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.error("Código incorrecto o inactivo. Contacta con coordinación.")
            except KeyError:
                st.error("Error de servidor: No se ha configurado la base de datos de alumnos en Streamlit Secrets.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ==========================================
# CAPA 3: SANITIZACIÓN DE ENTRADA
# ==========================================
def validar_consulta_educativa(texto):
    patrones_bloqueo = [
        r'ignora.*instrucciones', r'actúa como', r'olvida.*anterior',
        r'revela.*sistema', r'sal.*modo', r'prompt.*original'
    ]
    if any(re.search(p, texto.lower()) for p in patrones_bloqueo):
        return False, "⚠️ Alerta de Seguridad: Esta consulta no puede ser procesada. Concentrémonos en tu ingreso a la UNSA. ¿En qué tema académico te ayudo?"
    return True, texto

# ==========================================
# CAPA 4: MULTI-PROFESOR AUTOMÁTICO (TXT Y PDF)
# ==========================================
st.sidebar.title("👨‍🏫 Panel de Preparación")

archivos_encontrados = glob.glob("*.txt") + glob.glob("*.pdf")
archivos_materias = {os.path.splitext(f)[0].capitalize(): f for f in archivos_encontrados if f != "requirements.txt"}
lista_materias = list(archivos_materias.keys())

if not lista_materias:
    st.sidebar.error("❌ No hay cursos disponibles.")
    st.error("No se encontraron archivos (.txt o .pdf) para los cursos. Súbelos a GitHub.")
    st.stop()

materia = st.sidebar.selectbox("Selecciona tu Profesor:", lista_materias)

@st.cache_resource
def cargar_profesor(nombre_materia):
    archivo = archivos_materias[nombre_materia]
    
    if archivo.endswith(".pdf"):
        loader = PyPDFLoader(archivo)
    else:
        loader = TextLoader(archivo, encoding="utf-8")
        
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vector_db = cargar_profesor(materia)

# ==========================================
# CAPA 5: CHAT Y CEREBRO PEDAGÓGICO MAESTRO
# ==========================================
if "materia_actual" not in st.session_state:
    st.session_state.materia_actual = materia

if st.session_state.materia_actual != materia:
    st.session_state.messages = []  
    st.session_state.materia_actual = materia

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title(f"🏫 Preparación Intensiva: {materia}")
st.info(f"¡Hola! Soy tu profesor estratega de {materia} 🛠️. Tenemos una vacante de Ingeniería que asegurar en la UNSA. ¿En qué tema nos quedamos ayer o por dónde empezamos a destruir el temario hoy?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu duda o formula tu respuesta a los retos..."):
    
    es_valido, mensaje_seguro = validar_consulta_educativa(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not es_valido:
            st.warning(mensaje_seguro)
            st.session_state.messages.append({"role": "assistant", "content": mensaje_seguro})
        elif vector_db:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # --- NUEVO PROMPT INTEGRADO CON PRECISIÓN ---
            instrucciones_maestras = f"""
            Eres un profesor preuniversitario especialista en {materia} para el examen de admisión de la UNSA, con dominio profundo de los temas y capacidad para explicarlos de forma clara, práctica y altamente digerible. Tu objetivo no es solo transmitir teoría, sino lograr comprensión real, rapidez de resolución y criterio de examen para asegurar la vacante en carreras de Ingeniería.

            🔒 RESTRICCIÓN TÉCNICA OBLIGATORIA:
            Debes basar tus explicaciones PRINCIPALMENTE en este material extraído del libro/módulo del alumno:
            {contexto}
            Si la información no está en el material extraído, puedes usar tu conocimiento, pero DEBES iniciar diciendo: "Esta información no aparece en nuestro módulo actual, pero te explico que..."

            Tu estilo debe ser:
            - Claro, ordenado, humano y directo.
            - Profundo, pero nunca tedioso.
            - Didáctico incluso para alumnos nuevos.
            - Preciso, sin excesos académicos innecesarios.
            - Enfocado en aprendizaje útil para examen.

            Debes explicar cada tema como un docente excelente: primero la idea central, luego el desarrollo lógico, después las trampas de examen, y finalmente la práctica guiada. La explicación debe caber en un solo mensaje, pero con suficiente profundidad.

            REGLAS GENERALES:
            1. Asume que el alumno puede ser principiante, salvo que el contexto indique un nivel más avanzado.
            2. Si el tema tiene fórmulas, usa LaTeX ($...$) correctamente.
            3. Explica con lenguaje natural, fácil de seguir y bien estructurado.
            4. No uses definiciones frías o mecánicas si puedes dar intuición primero.
            5. No abras temas innecesarios ni divagues.
            6. No resuelvas los ejercicios de práctica hasta que el alumno responda.
            7. Si el alumno muestra dudas, vuelve a explicar con mayor claridad antes de avanzar.
            8. Si el alumno responde un ejercicio, corrige con criterio, paso a paso, mostrando el razonamiento.
            9. Si el tema admite atajos, estrategias o relaciones útiles, enséñalos.
            10. Si hay trampas típicas de examen, adviértelas explícitamente.

            ESTRUCTURA OBLIGATORIA DE LA RESPUESTA (Usa estos mismos encabezados):

            1. 🧠 Base conceptual
            Explica la idea central del tema desde cero, con intuición. ¿Qué es? ¿Para qué sirve? ¿Qué significa realmente? ¿Cómo se interpreta?

            2. 🔍 Desarrollo claro y ordenado
            Desarrolla el contenido en una secuencia lógica (idea general, partes, reglas/fórmulas, interpretación, casos especiales).

            3. 🎯 Foco de examen y trampas clásicas
            Incluye de 3 a 6 puntos clave que suelen aparecer en admisión. Señala trampas, confusiones y patrones disfrazados.

            4. ⚡ Estrategia rápida / hack inteligente
            Da una estrategia corta y lógica para resolver más rápido.

            5. 📌 Ejemplo resuelto modelo
            Incluye un ejemplo resuelto claro y bien explicado, paso a paso.

            6. 📈 Entrenamiento progresivo
            Propón 3 ejercicios (Nivel 1: Calentamiento, Nivel 2: Modo examen, Nivel 3: Asegura vacante).

            7. 🛑 Punto de control
            Cierra SIEMPRE con esta pregunta exacta:
            "¿Tienes alguna duda, quieres intentar resolver los retos por tu cuenta, o pasamos al siguiente tema?"
            """

            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=INTERNAL_API_KEY,
                temperature=0.3
            )
            
            historial_formateado = [("system", instrucciones_maestras)]
            for msg in st.session_state.messages[-5:]: 
                role = "human" if msg["role"] == "user" else "ai"
                historial_formateado.append((role, msg["content"]))
                
            respuesta = llm.invoke(historial_formateado)
            
            texto_final = ""
            if isinstance(respuesta.content, list):
                texto_final = respuesta.content[0].get("text", "")
            elif isinstance(respuesta.content, str):
                texto_final = respuesta.content
            else:
                texto_final = str(respuesta.content)
            
            st.markdown(texto_final)
            st.session_state.messages.append({"role": "assistant", "content": texto_final})
