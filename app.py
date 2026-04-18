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
# CONFIGURACIÓN E INTERFAZ "CLEAN" (Paso 1)
# ==========================================
st.set_page_config(page_title="Academia IA Pro - UNSA", page_icon="🏫", layout="wide")

# Inyección de CSS para ocultar menús, botones de despliegue y el "Manage App"
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stAppDeployButton {display:none;}
            #stDecoration {display:none;}
            [data-testid="stSidebarNav"] {display: none;}
            /* Ocultar el botón de Manage App para usuarios no administradores */
            .st-emotion-cache-1wbqy5l {display:none;} 
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# CAPA 1: SEGURIDAD INTERNA Y ACCESO
# ==========================================
INTERNAL_API_KEY = st.secrets["GOOGLE_API_KEY"] 

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acceso a la Academia Virtual UNSA")
    password_input = st.text_input("Introduce tu código de alumno:", type="password")
    
    if st.button("Entrar"):
        try:
            contraseñas_validas = list(st.secrets["alumnos"].values())
            if password_input in contraseñas_validas:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Código incorrecto o inactivo. Contacta con coordinación.")
        except KeyError:
            st.error("Error: No se han configurado los códigos en Streamlit Secrets.")
    st.stop()

st.sidebar.warning("⚠️ El sistema detecta accesos simultáneos. No compartas tu código.")

# ==========================================
# CAPA 2: MULTI-PROFESOR AUTOMÁTICO (Paso 3)
# ==========================================
st.sidebar.title("👨‍🏫 Panel de Preparación")

# Buscamos archivos .txt y .pdf automáticamente
archivos_encontrados = glob.glob("*.txt") + glob.glob("*.pdf")
# Filtramos para no incluir archivos de sistema
archivos_materias = {os.path.splitext(f)[0].capitalize(): f for f in archivos_encontrados if f != "requirements.txt"}
lista_materias = list(archivos_materias.keys())

if not lista_materias:
    st.sidebar.error("❌ No hay cursos cargados.")
    st.title("📚 Aula Vacía")
    st.info("Sube tus archivos .pdf o .txt a GitHub para que los profesores aparezcan aquí.")
    st.stop()

materia = st.sidebar.selectbox("Selecciona tu Profesor:", lista_materias)

@st.cache_resource
def cargar_profesor(nombre_materia):
    archivo = archivos_materias[nombre_materia]
    # Selección inteligente de cargador
    if archivo.endswith(".pdf"):
        loader = PyPDFLoader(archivo)
    else:
        loader = TextLoader(archivo, encoding="utf-8")
        
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vector_db = cargar_profesor(materia)

# ==========================================
# CAPA 3: CEREBRO PEDAGÓGICO CON MEMORIA
# ==========================================
if "materia_actual" not in st.session_state:
    st.session_state.materia_actual = materia

# Reset de chat al cambiar de materia
if st.session_state.materia_actual != materia:
    st.session_state.messages = []
    st.session_state.materia_actual = materia

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title(f"🏫 Especialista en {materia}")
st.info(f"¡Hola! Soy tu profesor estratega de {materia} 🛠️. Tenemos una vacante de Ingeniería que asegurar en la UNSA. ¿En qué tema nos quedamos ayer o por dónde empezamos a destruir el temario hoy?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu duda o resuelve los retos..."):
    # Sanitización
    patrones = [r'ignora.*instrucciones', r'revela.*sistema', r'prompt.*original']
    if any(re.search(p, prompt.lower()) for p in patrones):
        st.warning("⚠️ Consulta no procesable por seguridad.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if vector_db:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # El "Cerebro Pedagógico" (Paso 2) - Aquí puedes expandir hasta 5000+ caracteres
            SYSTEM_PROMPT = f"""
            Actúa como un Profesor Especialista en {materia} y Estratega con 10 años de experiencia en la UNSA. 
            Tu propósito es guiar al estudiante de Ingeniería mediante el método socrático.
            
            🔒 REGLAS: 
            1. NUNCA des la respuesta directa. 
            2. Usa LaTeX para matemáticas: $...$.
            3. Si el tema no está aquí: '{contexto}', usa tu conocimiento pero avisa al alumno.
            
            ESTRUCTURA DE RESPUESTA:
            1. 🧠 Teoría Profunda.
            2. 🎯 Foco de Examen (Trampas UNSA).
            3. ⚡ El Hack (Estrategia rápida).
            4. 📈 Retos (Nivel 1, 2 y 3).
            5. 🛑 Punto de Control (Pregunta antes de seguir).
            """

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", # Versión estable y rápida
                google_api_key=INTERNAL_API_KEY,
                temperature=0.3
            )
            
            # Memoria de conversación (últimos 6 mensajes)
            historial = [("system", SYSTEM_PROMPT)]
            for m in st.session_state.messages[-6:]:
                role = "human" if m["role"] == "user" else "ai"
                historial.append((role, m["content"]))
            
            respuesta = llm.invoke(historial)
            
            # Limpieza de salida cruda
            texto = respuesta.content if isinstance(respuesta.content, str) else respuesta.content[0].get("text", "")
            
            st.markdown(texto)
            st.session_state.messages.append({"role": "assistant", "content": texto})
