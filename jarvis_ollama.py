from __future__ import print_function

import speech_recognition as sr
import pyttsx3
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import filedialog
import threading
import datetime
import re
import requests
import dateparser
import json
from deep_translator import GoogleTranslator
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
import cv2
from PIL import Image
import numpy as np
from datetime import timedelta
import dateparser
import schedule
import time
from dateutil import parser as dateutil_parser
import subprocess
import sys
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import pyautogui
import os.path
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from aprendizaje_personal import aplicar_personalizaciones, registrar_sinonimo
from conocimiento import guardar_conocimiento, obtener_conocimiento, obtener_tareas_pendientes, agregar_tarea_pendiente, completar_tarea,cargar_conocimiento
from diagnostico_seguridad import (obtener_info_red, conexiones_netstat, procesos_sospechosos, activar_modo_experto, resumen_seguridad_basico, geolocalizar_ip)
from auditoria import auditoria_rapida, auditoria_completa
from herramientas_avanzadas import *


# Detectamos si estamos en Render (o en cualquier entorno 'web')
IS_RENDER = os.getenv("RENDER", "false").lower() in ("1", "true", "yes")

# Importaciones seguras: sólo cargar módulos de audio/GUI si NO estamos en Render
if not IS_RENDER:
    try:
        import speech_recognition as sr
        import pyttsx3
        import tkinter as tk
        # cualquier otra librería que requiera audio/GUIs
    except Exception:
        # Si falla la importación localmente, dejamos variables a None
        sr = None
        pyttsx3 = None
        tk = None
else:
    # En Render: no cargamos estas librerías
    sr = None
    pyttsx3 = None
    tk = None




###########################################
# CONFIGURACIÓN Y VARIABLES GLOBALES
###########################################

personalidad = "Eres un asistente virtual amable, servicial y profesional."
conversation_turns = [
    {"role": "system", "content": f"{personalidad} No repitas la misma información ni hagas las mismas preguntas. Mantén la coherencia y responde de forma clara y completa."}
]
agenda = []
MAX_TURNS = 10
ultima_respuesta = ""
DEFAULT_CITY = "Valencia"
WEATHER_API_KEY = "f6c697a7ea73fcc4358e049ad25ea05b"

global_engine = pyttsx3.init()
global_engine.setProperty("rate", 150)
global_engine.setProperty("volume", 1.0)
engine_lock = threading.Lock()

comandos_autocompletar = [
    "clima",
    "añadir recordatorio reunión mañana a las 10",
    "mostrar agenda",
    "eliminar recordatorio 1",
    "cambia tu personalidad a divertido",
    "hora",
    "salir",
]

###########################
# GOOGLE CALENDAR
###########################

# Permiso de lectura/escritura
SCOPES = ['https://www.googleapis.com/auth/calendar']

def conectar_calendario():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('calendar', 'v3', credentials=creds)

def obtener_eventos_hoy():
    servicio = conectar_calendario()
    ahora = datetime.datetime.utcnow().isoformat() + 'Z'
    final_dia = (datetime.datetime.utcnow().replace(hour=23, minute=59, second=59)).isoformat() + 'Z'

    eventos_resultado = servicio.events().list(
        calendarId='primary',
        timeMin=ahora,
        timeMax=final_dia,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    eventos = eventos_resultado.get('items', [])

    if not eventos:
        return "No tienes eventos en tu calendario hoy."

    respuesta = "Estos son tus eventos para hoy:\n"
    for evento in eventos:
        inicio = evento['start'].get('dateTime', evento['start'].get('date'))
        descripcion = evento.get('summary', 'Sin título')
        respuesta += f"- {descripcion} a las {inicio}\n"

    return respuesta

###########################
# CONTROL ENTORNO DIGITAL
###########################

def abrir_aplicacion(nombre):
    nombre = nombre.lower()
    
    #Cambiar las rutas por las necesarias
    rutas = {
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "spotify": r"C:\Users\josep\AppData\Local\Microsoft\WindowsApps\Spotify.exe",
        "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "bloc de notas": "notepad.exe",
        "explorador": "explorer.exe",
    }

    for app, ruta in rutas.items():
        if app in nombre:
            try:
                subprocess.Popen(ruta)
                return f"Abriendo {app.capitalize()}."
            except Exception as e:
                return f"No pude abrir {app}: {e}"

    return "No encontré esa aplicación registrada."

######################
# CONTROL VOLUMEN
######################

def obtener_control_volumen():
    dispositivos = AudioUtilities.GetSpeakers()
    interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
    return volumen

def subir_volumen(pasos=0.1):
    volumen = obtener_control_volumen()
    actual = volumen.GetMasterVolumeLevelScalar()
    volumen.SetMasterVolumeLevelScalar(min(1.0, actual + pasos), None)
    return "Volumen aumentado."

def bajar_volumen(pasos=0.1):
    volumen = obtener_control_volumen()
    actual = volumen.GetMasterVolumeLevelScalar()
    volumen.SetMasterVolumeLevelScalar(max(0.0, actual - pasos), None)
    return "Volumen reducido."

def silenciar():
    volumen = obtener_control_volumen()
    volumen.SetMute(1, None)
    return "Audio silenciado."

def reanudar_sonido():
    volumen = obtener_control_volumen()
    volumen.SetMute(0, None)
    return "Audio reactivado."

def pausar_musica():
    pyautogui.press('playpause')
    return "Música pausada/reanudada."

def siguiente_cancion():
    pyautogui.press('nexttrack')
    return "Siguiente canción."

def anterior_cancion():
    pyautogui.press('prevtrack')
    return "Canción anterior."

######################
# AGENDA PERSONAL
######################

RUTA_AGENDA = "agenda_personal.json"

def cargar_agenda():
    if os.path.exists(RUTA_AGENDA):
        with open(RUTA_AGENDA, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_evento(descripcion, inicio, fin):
    agenda = cargar_agenda()
    evento = {
        "descripcion": descripcion,
        "inicio": inicio.isoformat(),
        "fin": fin.isoformat()
    }
    agenda.append(evento)
    with open(RUTA_AGENDA, "w", encoding="utf-8") as f:
        json.dump(agenda, f, ensure_ascii=False, indent=2)

def eventos_para_fecha(fecha):
    agenda = cargar_agenda()
    eventos = []
    for evento in agenda:
        inicio = datetime.fromisoformat(evento["inicio"])
        if inicio.date() == fecha.date():
            eventos.append(evento)
    return eventos

def detectar_evento_y_guardar(texto):
    frases_clave = ["tengo", "añade", "agrega", "pon", "estoy ocupado"]
    if any(palabra in texto.lower() for palabra in frases_clave):
        fecha_hora = dateparser.parse(texto, languages=["es"])
        if fecha_hora:
            descripcion = texto
            inicio = fecha_hora
            fin = inicio + timedelta(hours=1)  # Duración por defecto
            guardar_evento(descripcion, inicio, fin)
            return f"Evento guardado: '{descripcion}' el {inicio.strftime('%d/%m/%Y a las %H:%M')}"
        else:
            return "No pude entender bien la fecha y hora. ¿Podrías repetirlo con más detalle?"
    return None




def obtener_eventos_para_fecha_texto(texto):
    try:
        fecha = dateutil_parser.parse(texto, fuzzy=True)
    except Exception:
        return "No pude entender qué fecha deseas consultar."

    eventos = eventos_para_fecha(fecha)
    if not eventos:
        return f"No tienes eventos programados para el {fecha.strftime('%A %d de %B')}."

    respuesta = f"Tienes {len(eventos)} evento(s) el {fecha.strftime('%A %d de %B')}:\n"
    for e in eventos:
        inicio = datetime.fromisoformat(e["inicio"])
        respuesta += f"- {e['descripcion']} a las {inicio.strftime('%H:%M')}\n"
    return respuesta


##############################
# CONFLICO TAREAS Y EVENTOS
##############################

def verificar_conflictos_entre_tareas_y_eventos():
    tareas = cargar_tareas()  # si tu función se llama distinto, ajústalo
    eventos = cargar_agenda()

    conflictos = []
    for tarea in tareas:
        if tarea.get("urgente") and "hora" in tarea:
            hora_tarea = datetime.fromisoformat(tarea["hora"])
            for evento in eventos:
                inicio_evento = datetime.fromisoformat(evento["inicio"])
                fin_evento = datetime.fromisoformat(evento["fin"])
                if inicio_evento <= hora_tarea <= fin_evento:
                    conflictos.append({
                        "tarea": tarea["descripcion"],
                        "evento": evento["descripcion"],
                        "hora": hora_tarea.strftime("%H:%M")
                    })
    return conflictos


def detectar_conflictos(fecha, hora_inicio=None, duracion_minutos=60):
    eventos = eventos_para_fecha(fecha)
    tareas = cargar_conocimiento().get("tareas", [])

    inicio_nuevo = datetime.strptime(f"{fecha} {hora_inicio}", "%Y-%m-%d %H:%M") if hora_inicio else None
    fin_nuevo = inicio_nuevo + timedelta(minutes=duracion_minutos) if inicio_nuevo else None

    conflictos = []

    # Revisar conflictos con eventos
    for evento in eventos:
        ini_evt = datetime.fromisoformat(evento["inicio"])
        fin_evt = datetime.fromisoformat(evento["fin"])
        if inicio_nuevo and fin_nuevo:
            if ini_evt < fin_nuevo and inicio_nuevo < fin_evt:
                conflictos.append(f"Conflicto con evento: {evento['descripcion']} de {ini_evt.strftime('%H:%M')} a {fin_evt.strftime('%H:%M')}")

    # Revisar conflictos con tareas con hora
    for tarea in tareas:
        if not tarea.get("fecha_objetivo") or not tarea.get("hora"):
            continue
        if tarea["fecha_objetivo"] != fecha:
            continue
        hora_tarea = datetime.strptime(f"{fecha} {tarea['hora']}", "%Y-%m-%d %H:%M")
        fin_tarea = hora_tarea + timedelta(minutes=duracion_minutos)
        if inicio_nuevo and fin_nuevo:
            if hora_tarea < fin_nuevo and inicio_nuevo < fin_tarea:
                conflictos.append(f"Conflicto con tarea: {tarea['descripcion']} a las {tarea['hora']}")

    return conflictos


######################
# ANALISIS EMOCIONAL
######################

def detectar_emocion(texto):
    emociones = {
        "tristeza": ["triste", "deprimido", "desanimado", "abrumado"],
        "alegría": ["feliz", "contento", "emocionado", "genial"],
        "estrés": ["estresado", "agotado", "presionado", "cansado"],
        "enojo": ["enojado", "molesto", "furioso", "fastidiado"]
    }

    texto = texto.lower()
    for emocion, palabras_clave in emociones.items():
        if any(palabra in texto for palabra in palabras_clave):
            return emocion
    return None




######################
# VISION POR CAMARA
######################

import cv2

def detectar_rostros_webcam():
    try:
        camara = cv2.VideoCapture(0)  # Usa la primera cámara conectada

        if not camara.isOpened():
            return "No se pudo acceder a la cámara."

        ret, frame = camara.read()
        camara.release()

        if not ret:
            return "No pude capturar una imagen desde la cámara."

        # Mostrar imagen durante 3 segundos
        cv2.imshow("Vista en directo", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        # Convertir a escala de grises para detectar rostros
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        rostros = detector_rostros.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

        if len(rostros) == 0:
            return "No se detectó ningún rostro en la imagen."
        elif len(rostros) == 1:
            return "He detectado una persona frente a la cámara."
        else:
            return f"He detectado {len(rostros)} personas frente a la cámara."
    except Exception as e:
        return f"Ocurrió un error al procesar la cámara: {str(e)}"


def mostrar_video_en_vivo():
    try:
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            return "No se pudo acceder a la cámara."

        # Cargar el modelo de detección de rostro
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        cv2.namedWindow("Cámara en vivo")

        print("Presiona 'q' o 'ESC' para cerrar la cámara")

        while True:
            ret, frame = cam.read()
            if not ret:
                break
            # Convertir a escala de grises para detección
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostros
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Dibujar rectángulos alrededor de los rostros detectados
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Mostrar el video
            cv2.imshow("Cámara en vivo", frame)

            # Presionar 'q' o 'ESC' para salir
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 es ESC
                break

        cam.release()
        cv2.destroyAllWindows()
        return "Cámara cerrada. ¿Te gustaría hacer algo más?"

    except Exception as e:
        return f"Ocurrió un error al mostrar la cámara: {str(e)}"
    
CONFIG_PATH = r"D:\Documentos\Programas\JARVIS\Jarvis 2.0\Jarvis\Yolo_Comp\yolov3-tiny.cfg"
WEIGHTS_PATH = r"D:\Documentos\Programas\JARVIS\Jarvis 2.0\Jarvis\Yolo_Comp\yolov3-tiny.weights"
NAMES_PATH = r"D:\Documentos\Programas\JARVIS\Jarvis 2.0\Jarvis\Yolo_Comp\coco.names"

def detectar_objetos_yolo():
    try:
        # Cargar nombres de clases (coco.names)
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Colores para cada clase
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Cargar YOLO
        net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
        layer_names = net.getUnconnectedOutLayersNames()

        cap = cv2.VideoCapture(0)
        print("Presiona 'q' o ESC para cerrar la cámara.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]

            # Preprocesamiento de la imagen para YOLO
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Detección de objetos (YOLO)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return "Cámara cerrada. ¿Deseas hacer otra cosa?"

    except Exception as e:
        return f"Error al procesar detección YOLO: {str(e)}"
    
    

####################################################################
# RECONOCIMIENTO DE IMAGENES (identifica el texto de la imagen)
####################################################################

# ⚠️ Ruta a tesseract.exe (solo para Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Cambia esto si es necesario
# Establece la carpeta de los archivos de idioma
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

def analizar_imagen(ruta_imagen):
    try:
        img = cv2.imread(ruta_imagen)
        if img is None:
            return "No pude abrir la imagen."

        texto_detectado = pytesseract.image_to_string(img, lang='spa')
        return f"Texto detectado en la imagen:\n{texto_detectado.strip() or 'No se detectó texto'}"
    except Exception as e:
        return f"Error al analizar la imagen: {str(e)}"

###########################################
# GUARDAR TAREAS
###########################################


RUTA_TAREAS = "tareas.json"

def cargar_tareas():
    if os.path.exists(RUTA_TAREAS):
        with open(RUTA_TAREAS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_tareas(tareas):
    with open(RUTA_TAREAS, "w", encoding="utf-8") as f:
        json.dump(tareas, f, ensure_ascii=False, indent=2)
        



def agregar_tarea_pendiente(texto):
    tareas = cargar_conocimiento().get("tareas", [])

    prioridad = "alta" if "urgente" in texto.lower() else "normal"

    # --- Detectar fecha ---
    fecha_match = re.search(r"(hoy|mañana|\d{4}-\d{2}-\d{2})", texto)
    if fecha_match:
        palabra_fecha = fecha_match.group()
        if palabra_fecha == "hoy":
            fecha = datetime.today().strftime("%Y-%m-%d")
        elif palabra_fecha == "mañana":
            fecha = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            fecha = palabra_fecha
    else:
        fecha = None

    # --- Detectar hora ---
    hora_match = re.search(r"\b\d{1,2}:\d{2}\b", texto)
    hora = hora_match.group() if hora_match else None

    # --- Descripción limpia ---
    descripcion = texto
    for palabra in ["urgente", "hoy", "mañana"]:
        descripcion = descripcion.replace(palabra, "")
    if hora:
        descripcion = descripcion.replace(hora, "")
    if fecha_match:
        descripcion = descripcion.replace(fecha_match.group(), "")
    descripcion = descripcion.strip()

    # --- Crear tarea ---
    tarea = {
        "descripcion": descripcion,
        "prioridad": prioridad,
        "fecha_objetivo": fecha,
        "hora": hora,
        "completada": False
    }

    tareas.append(tarea)
    guardar_conocimiento("tareas", tareas)

    # --- Detectar conflictos ---
    if fecha and hora:
        conflictos = detectar_conflictos(fecha, hora)
        if conflictos:
            print("⚠️ Se detectaron conflictos:")
            for c in conflictos:
                print(f"- {c}")

    return f"Tarea añadida: {descripcion} (Prioridad: {prioridad}, Fecha: {fecha or 'sin fecha'}, Hora: {hora or 'sin hora'})"

def obtener_tareas_pendientes(filtrar_prioridad=None):
    tareas = cargar_tareas()
    tareas = [t for t in tareas if not t["completada"]]

    if filtrar_prioridad:
        tareas = [t for t in tareas if t["prioridad"] == filtrar_prioridad]

    if not tareas:
        return "No hay tareas pendientes registradas."

    respuesta = "Estas son tus tareas pendientes:\n"
    for idx, t in enumerate(tareas, 1):
        linea = f"* {t['descripcion']}"
        if t.get("prioridad"):
            linea += f" ({t['prioridad']})"
        if t.get("fecha"):
            linea += f" - para {t['fecha']}"
        respuesta += f"{idx}. {linea}\n"
    return respuesta

def completar_tarea(nombre):
    tareas = cargar_tareas()
    for tarea in tareas:
        if nombre.lower() in tarea["descripcion"].lower():
            tarea["completada"] = True
            guardar_tareas(tareas)
            return f"La tarea '{tarea['descripcion']}' ha sido marcada como completada."
    return "No encontré una tarea con ese nombre."


def revisar_tareas_para_manana():
    tareas = cargar_conocimiento().get("tareas", [])
    mañana = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    para_recordar = [t for t in tareas if t["fecha_objetivo"] == mañana and not t["completada"]]
    if para_recordar:
        print("🔔 Recordatorio de tareas para mañana:")
        for t in para_recordar:
            print(f"- {t['descripcion']} (Prioridad: {t['prioridad']})")

    schedule.every().day.at("21:00").do(revisar_tareas_para_manana)

    while True:
        schedule.run_pending()
        time.sleep(60)


###########################################
# GUARDAR Y ACTUALIZAR EL NOMBRE O APODO
###########################################    

def detectar_y_guardar_nombre(texto):
    patrones = [
        r"me llamo ([\wáéíóúñ]+)",
        r"puedes llamarme ([\wáéíóúñ]+)",
        r"llámame ([\wáéíóúñ]+)"
    ]
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            nombre = match.group(1)
            guardar_conocimiento("nombre", nombre)
            return f"Encantado, {nombre}. Lo recordaré."
    return None


###########################################
# CONECTAR A NOTICIAS ACTUALES
###########################################

def obtener_noticias(pais="us", categoria="general", max_noticias=3):
    api_key = "b27ae607cc144cb0b2d3fb94585f86a2"  # Reemplaza con tu API key real
    url = f"https://newsapi.org/v2/top-headlines?country={pais}&category={categoria}&apiKey={api_key}"
    try:
        r = requests.get(url)
        datos = r.json()
        articulos = datos.get("articles", [])
        
        if not articulos:
            return "No encontré noticias recientes en este momento."
            
        noticias = []
        for art in articulos[:max_noticias]:
            titulo = art["title"]
            fuente = art["source"]["name"]
            url_noticia = art["url"]

            titulo_traducido = GoogleTranslator(source='auto', target='es').translate(titulo)
            noticias.append(f"📰 {titulo_traducido} ({fuente})\n🔗 {url_noticia}")
        
        return "\n\n".join(noticias)
    
    except Exception as e:
        print("Error al obtener noticias:", e)
        return "Hubo un error al consultar las noticias."

def procesar_noticias(texto: str) -> str:
    categorias_validas = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
    for cat in categorias_validas:
        if cat in texto.lower():
            return obtener_noticias(categoria=cat)
    return obtener_noticias()  # Por defecto categoría general


###########################################
# ANALISIS DE DATOS
###########################################

def analizar_archivo_csv_excel(ruta_archivo: str) -> str:
    try:
        # Leer CSV o Excel
        if ruta_archivo.endswith(".csv"):
            df = pd.read_csv(ruta_archivo)
        elif ruta_archivo.endswith(".xlsx"):
            df = pd.read_excel(ruta_archivo)
        else:
            return "Formato de archivo no soportado. Usa .csv o .xlsx"

        resumen = df.describe(include='all').to_string()
        columnas = ", ".join(df.columns)
        mensaje = f"Archivo cargado correctamente. Contiene las columnas: {columnas}.\nResumen estadístico:\n{resumen}"
        return mensaje
    except Exception as e:
        return f"Error al procesar el archivo: {e}"
    
def graficar_columna(ruta_archivo: str, columna: str):
    try:
        if ruta_archivo.endswith(".csv"):
            df = pd.read_csv(ruta_archivo)
        elif ruta_archivo.endswith(".xlsx"):
            df = pd.read_excel(ruta_archivo)
        else:
            return "Formato no compatible."

        if columna not in df.columns:
            return f"La columna '{columna}' no existe."

        plt.figure(figsize=(10,5))
        df[columna].plot(kind='line', title=f"Tendencia de {columna}", grid=True)
        plt.xlabel("Índice")
        plt.ylabel(columna)
        plt.tight_layout()
        plt.savefig("grafico.png")
        os.startfile("grafico.png")  # Abre la imagen en visor predeterminado
        return f"Gráfico generado para la columna: {columna}"
    except Exception as e:
        return f"Error al graficar: {e}"

###########################################
# GUARDAR LOGS Y RECOMENDACIONES SIMPLES
###########################################
HISTORIAL_PATH = "historial_usuario.json"


def es_pregunta_clima(texto: str) -> bool:
    texto = texto.lower()
    patrones = ["clima", "tiempo", "temperatura", "llueve", "lloverá", "va a llover", "hace frío", "hace calor"]
    return any(p in texto for p in patrones)

def guardar_en_historial(texto_usuario):
    registro = {
        "texto": texto_usuario,
        "fecha": datetime.datetime.now().isoformat()
    }

    historial = []
    if os.path.exists(HISTORIAL_PATH):
        try:
            with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
                historial = json.load(f)
        except:
            historial = []

    historial.append(registro)
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial[-100:], f, ensure_ascii=False, indent=2)


def recomendar_accion():
    if not os.path.exists(HISTORIAL_PATH):
        return ""
    
    try:
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            historial = json.load(f)
    except Exception:
        return ""
    
    temas = obtener_conocimiento("temas_interes") or []
    if "tecnología" in temas:
        return "¿Quieres que te muestre las últimas noticias de tecnología?"

    # Filtra preguntas sobre clima en cualquier horario
    preguntas_clima = [h for h in historial if es_pregunta_clima(h["texto"])]

    if len(preguntas_clima) >= 3:
        return "Veo que preguntas seguido por el clima. ¿Te gustaría que te lo diga automáticamente cada mañana?"

    return ""

    

###########################################
# FUNCIÓN PARA GENERAR CON OLLAMA
###########################################

def generar_respuesta_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    data = {"model": "llama3:8b-instruct-q4_0", "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print("Error al generar con Ollama:", e)
        return "Lo siento, hubo un error al generar la respuesta."

###########################################
# VOZ – RECONOCIMIENTO Y SÍNTESIS
###########################################

def escuchar():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="es-ES").lower()
    except:
        return ""

def hablar(texto):
    with engine_lock:
        global_engine.say(texto)
        try:
            global_engine.runAndWait()
        except RuntimeError:
            pass

###########################################
# FILTRO DE LENGUAJE OFENSIVO
###########################################

def filtrar_lenguaje(texto: str) -> bool:
    return any(p in texto.lower() for p in ["ofensivo1", "ofensivo2"])

###########################################
# INTENCIONES
###########################################

def detectar_intencion(texto: str) -> str:
    t = texto.lower()
    if "clima" in t: return "consultar_clima"
    if "cambia tu personalidad a" in t: return "cambiar_personalidad"
    if any(p in t for p in ["recordatorio", "agenda", "recordar", "anotar"]): return "agenda"
    if any(p in t for p in ["salir", "adiós", "hasta luego"]): return "despedida"
    if any(p in t for p in ["noticia", "noticias", "actualidad", "últimas noticias"]): return "noticias"

    return "ninguna"

###########################################
# CLIMA
###########################################

def extraer_ciudad(texto: str) -> str:
    match = re.search(r"clima en ([A-Za-záéíóúÁÉÍÓÚ\s]+)", texto, re.IGNORECASE)
    return match.group(1).strip() if match else None

def consultar_clima(ciudad: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={WEATHER_API_KEY}&units=metric&lang=es"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            d = r.json()
            return f"El clima en {ciudad} es {d['weather'][0]['description']} con {d['main']['temp']}°C."
        return "No pude obtener la información del clima."
    except:
        return "Hubo un error al consultar el clima."

def procesar_consulta_clima(texto: str) -> str:
    ciudad = extraer_ciudad(texto) or DEFAULT_CITY
    return consultar_clima(ciudad)

###########################################
# AGENDA Y RECORDATORIOS
###########################################

def agregar_evento(texto: str) -> str:
    evento = re.sub(r"(añadir recordatorio|recordarme|anotar|agendar)", "", texto.lower()).strip()
    if not evento: return "No se detectó ningún detalle para el recordatorio."
    fecha = dateparser.parse(evento)
    fecha_str = fecha.strftime("%Y-%m-%d %H:%M") if fecha else ""
    agenda.append({"detalle": evento, "fecha": fecha_str})
    return f"Recordatorio añadido: {evento}" + (f" (Fecha: {fecha_str})" if fecha_str else "")

def listar_eventos() -> str:
    if not agenda: return "No tienes recordatorios."
    return "\n".join(f"{i+1}. {e['detalle']}" + (f" - {e['fecha']}" if e['fecha'] else "") for i, e in enumerate(agenda))

def eliminar_evento(texto: str) -> str:
    m = re.search(r"\d+", texto)
    if not m: return "Indica el número del recordatorio a eliminar."
    i = int(m.group()) - 1
    if 0 <= i < len(agenda):
        return f"Recordatorio eliminado: {agenda.pop(i)['detalle']}"
    return "No existe ese recordatorio."

def procesar_agenda(texto: str) -> str:
    t = texto.lower()
    if any(p in t for p in ["añadir recordatorio", "recordarme", "anotar"]): return agregar_evento(texto)
    if any(p in t for p in ["mostrar agenda", "listar recordatorios"]) or t.strip() in ["agenda", "recordatorio"]: return listar_eventos()
    if any(p in t for p in ["eliminar recordatorio", "borrar recordatorio"]): return eliminar_evento(texto)
    return "No entendí el comando de agenda."

###########################################
# PERSONALIDAD Y FECHA/HORA
###########################################

def cambiar_personalidad(texto: str) -> str:
    try:
        nueva = texto.lower().split("cambia tu personalidad a",1)[1].strip()
        global personalidad
        personalidad = nueva
        conversation_turns[0]["content"] = f"{personalidad} No repitas la misma información ni hagas las mismas preguntas. Mantén la coherencia y responde de forma clara y completa."
        return f"Personalidad cambiada a: {nueva}"
    except:
        return "Formato no válido. Usa 'cambia tu personalidad a ...'."

def es_pregunta_fecha_hora(texto: str) -> bool:
    return any(k in texto.lower() for k in ["hora","fecha","día","dia","tiempo"])

def responder_fecha_hora() -> str:
    now = datetime.datetime.now()
    return now.strftime("Hoy es %A, %d de %B de %Y y son las %H:%M.")

###########################################
# HISTORIAL Y CONTEXTO
###########################################

def recortar_historial():
    global conversation_turns
    turns = [t for t in conversation_turns if t["role"] in ("user","assistant")]
    if len(turns) > MAX_TURNS:
        resumen = "Resumen: " + " ".join(t["content"] for t in turns[:-MAX_TURNS])
        conversation_turns = [conversation_turns[0], {"role":"assistant","content":resumen}] + turns[-MAX_TURNS:]

def construir_prompt() -> str:
    return "\n".join(
        f"{'Usuario' if t['role']=='user' else 'Asistente' if t['role']=='assistant' else 'Instrucciones del sistema'}: {t['content']}"
        for t in conversation_turns
    ) + "\nAsistente:"

###########################################
# PROCESAR TEXTO (INTEGRACIÓN CON OLLAMA)
###########################################

# Al inicio del archivo (antes de procesar_texto)
esperando = {}
ultima_respuesta = ""
conversation_turns = [conversation_turns[0]]  # tu sistema inicial

def procesar_texto(texto: str) -> str:
    global esperando, ultima_respuesta, conversation_turns

    texto_original = texto
    texto_lower = texto.strip().lower()

    # === 1. Flujos paso a paso de herramientas ===
    if esperando:
        cmd = esperando.get("cmd")
        if cmd == "nmap_host":
            esperando["host"] = texto_lower
            esperando["cmd"] = "nmap_puertos"
            return "¿Qué rango de puertos te gustaría escanear? (ej. 1-1000)"
        if cmd == "nmap_puertos":
            host = esperando.pop("host")
            puertos = texto_lower
            esperando.clear()
            return escanear_nmap(host, puertos)

        if cmd == "hydra_host":
            esperando["host"] = texto_lower
            esperando["cmd"] = "hydra_servicio"
            return "¿Qué servicio quieres probar? (ej. ssh, ftp...)"

        if cmd == "hydra_servicio":
            esperando["servicio"] = texto_lower
            esperando["cmd"] = "hydra_usuario"
            return "¿Qué usuario usarás para el ataque?"

        if cmd == "hydra_usuario":
            esperando["usuario"] = texto_lower
            esperando["cmd"] = "hydra_dicc"
            return "¿Ruta al diccionario/password list?"

        if cmd == "hydra_dicc":
            host = esperando.pop("host")
            servicio = esperando.pop("servicio")
            usuario = esperando.pop("usuario")
            dicc = texto_lower
            esperando.clear()
            return fuerza_bruta_hydra(host, servicio, dicc, usuario)

        if cmd == "msf_rhosts":
            esperando["rhosts"] = texto_lower
            esperando["cmd"] = "msf_exploit"
            return "¿Exploit a usar? (ruta dentro de Metasploit)"

        if cmd == "msf_exploit":
            esperando["exploit_path"] = texto_lower
            esperando["cmd"] = "msf_payload"
            return "¿Payload deseado?"

        if cmd == "msf_payload":
            rhosts = esperando.pop("rhosts")
            exploit_path = esperando.pop("exploit_path")
            payload = texto_lower
            esperando.clear()
            return exploit_metasploit(rhosts, exploit_path, payload)

    # === 2. Detección de inicio de flujo experto ===
    if "auditaría rápida" in texto_lower or "auditoria rápida" in texto_lower:
        return auditoria_rapida()
    if "auditoría completa" in texto_lower or "auditoria completa" in texto_lower:
        return auditoria_completa()
    if "escanear red" in texto_lower:
        esperando = {"cmd": "nmap_host"}
        return "Modo experto: ¿Cuál es el host o IP a escanear?"
    if "bruteforce" in texto_lower or "fuerza bruta" in texto_lower:
        esperando = {"cmd": "hydra_host"}
        return "Modo experto: ¿Cuál es el host o IP a atacar?"
    if "usar exploit" in texto_lower or "exploit metasploit" in texto_lower:
        esperando = {"cmd": "msf_rhosts"}
        return "Modo experto: ¿IP o rango (RHOSTS)?"
    #Masscan
    if "masscan" in texto_lower or "escanear con masscan" in texto_lower:
        esperando = {"cmd": "masscan_rango"}
        return "Modo experto: ¿Cuál es el rango de IP a escanear con masscan? (ej. 192.168.1.0/24)"
    
    #=== 2.1. Activar modo experto ===
    if "activar modo experto" in texto_lower:
        return activar_modo_experto("Te quiero 3000")

    # === 3. Comandos directos ===
    i = detectar_intencion(texto)
    if i == "consultar_clima": return procesar_consulta_clima(texto)
    if i == "cambiar_personalidad": return cambiar_personalidad(texto)
    if i == "agenda": return procesar_agenda(texto)
    if i == "despedida": return "Hasta luego, fue un placer ayudarte."
    if i == "noticias": return procesar_noticias(texto)

    if "diagnóstico" in texto_lower or "seguridad" in texto_lower:
        info, procesos, conexiones = resumen_seguridad_basico()
        respuesta = info + "\n🧠 Procesos sospechosos (uso alto de CPU):\n"
        if not procesos:
            respuesta += "✅ No se detectaron procesos sospechosos.\n"
        else:
            for p in procesos:
                respuesta += f"⚠️ PID {p['pid']} - {p['name']} - CPU {p['cpu_percent']}%\n"
        respuesta += "\n🌐 Conexiones externas activas:\n"
        if not conexiones:
            respuesta += "✅ No se detectaron conexiones externas sospechosas.\n"
        else:
            for c in conexiones:
                ubicacion = geolocalizar_ip(c["ip"])
                respuesta += f"🔸 IP: {c['ip']}:{c['puerto']} (PID: {c['pid']})\n    📍 Ubicación: {ubicacion}\n"
        return respuesta

    # === 4. Perfil y base de conocimiento ===
    # Aprender nombre
    if "me llamo" in texto_lower:
        match = re.search(r"me llamo ([\w\s]+)", texto, re.IGNORECASE)
        if match:
            nombre = match.group(1).strip().capitalize()
            guardar_conocimiento("nombre", nombre)
            return f"Encantado, {nombre}. Lo recordaré."
    # Ciudad
    if "mi ciudad es" in texto_lower:
        match = re.search(r"mi ciudad es ([\w\s]+)", texto, re.IGNORECASE)
        if match:
            ciudad = match.group(1).strip().capitalize()
            guardar_conocimiento("ciudad", ciudad)
            return f"Perfecto, recordaré que vives en {ciudad}."
    # Intereses
    if "me interesa" in texto_lower or "me interesan" in texto_lower:
        match = re.search(r"me interesa[n]* ([\w\s,]+)", texto, re.IGNORECASE)
        if match:
            temas = [t.strip().lower() for t in match.group(1).split(",")]
            guardar_conocimiento("temas_interes", temas)
            return f"Genial, recordaré que te interesan: {', '.join(temas)}."

    nombre = obtener_conocimiento("nombre")
    if nombre and texto_lower in ["hola", "hey", "buenos días", "buenas"]:
        return f"Hola de nuevo, {nombre}. ¿En qué puedo ayudarte hoy?"
    if "sabes cómo me llamo" in texto_lower or "sabes mi nombre" in texto_lower:
        if nombre:
            return f"Claro que sí, te llamas {nombre}."
        else:
            return "Todavía no me has dicho tu nombre. ¿Cómo te llamas?"

    # === 5. Manejo de emociones ===
    emocion = detectar_emocion(texto)
    if emocion:
        guardar_conocimiento("estado_emocional", emocion)
        emos = {
            "tristeza": "Siento que estés pasando por un momento difícil...",
            "alegría": "¡Me alegra mucho escuchar eso! ...",
            "estrés": "El estrés puede ser pesado. ...",
            "enojo": "Entiendo que estés molesto..."
        }
        return emos.get(emocion, "")

    # === 6. Tareas ===
    if "tareas pendientes" in texto_lower or "qué tareas" in texto_lower:
        if "urgentes" in texto_lower:
            return obtener_tareas_pendientes("urgente")
        elif "bajas" in texto_lower:
            return obtener_tareas_pendientes("baja")
        return obtener_tareas_pendientes()
    if "agrega tarea" in texto_lower or "añade tarea" in texto_lower:
        agregar_tarea_pendiente(texto)
        return "He añadido la nueva tarea según tus indicaciones."
    if "completada la tarea" in texto_lower:
        tarea = texto.split("completada la tarea")[-1].strip()
        return completar_tarea(tarea)

    # === 7. Cámara y visión computacional ===
    if any(f in texto_lower for f in ["ver camara", "mostrar cámara", "activar cámara"]):
        return mostrar_video_en_vivo()
    if any(f in texto_lower for f in ["detectar rostro", "hay alguien"]):
        return detectar_rostros_webcam()
    if "detectar objetos" in texto_lower:
        return detectar_objetos_yolo()
    if "analiza imagen" in texto_lower:
        return seleccionar_y_analizar_imagen()

    # === 8. Análisis de archivos ===
    if any(p in texto_lower for p in ["analiza archivo", "lee archivo", "procesa archivo"]):
        ruta = seleccionar_archivo()
        if ruta:
            return analizar_archivo_csv_excel(ruta)
        return "No se seleccionó ningún archivo."
    if "grafica columna" in texto_lower:
        match = re.search(r"grafica columna ([\w\s]+)", texto, re.IGNORECASE)
        if match:
            columna = match.group(1).strip()
            ruta = seleccionar_archivo()
            if ruta:
                return graficar_columna(ruta, columna)
        return "No se seleccionó ningún archivo."

    # === 9. Control de volumen y apps ===
    if "sube el volumen" in texto_lower: return subir_volumen()
    if "baja el volumen" in texto_lower: return bajar_volumen()
    if "silencia" in texto_lower or "mutea" in texto_lower: return silenciar()
    if "reanuda el sonido" in texto_lower or "activa el sonido" in texto_lower: return reanudar_sonido()
    if "pausa la música" in texto_lower: return pausar_musica()
    if "siguiente canción" in texto_lower: return siguiente_cancion()
    if "canción anterior" in texto_lower: return anterior_cancion()
    if "abre " in texto_lower or "abrir " in texto_lower:
        app = texto.split("abre")[-1].strip() if "abre" in texto_lower else texto.split("abrir")[-1].strip()
        return abrir_aplicacion(app)

    # === 10. Control de conflictos y eventos ===
    if "conflictos" in texto_lower:
        conflictos = verificar_conflictos_entre_tareas_y_eventos()
        if not conflictos:
            return "Todo en orden, no hay conflictos."
        return "He encontrado los siguientes conflictos:\n" + "\n".join(
            f"- Tarea urgente '{c['tarea']}' coincide con evento '{c['evento']}' a las {c['hora']}"
            for c in conflictos
        )
    resp_e = detectar_evento_y_guardar(texto)
    if resp_e: return resp_e
    if any(f in texto_lower for f in ["qué tengo", "eventos", "tengo algo"]):
        return obtener_eventos_para_fecha_texto(texto)

    # === 11. Comandos de conversación IA ===
    conversation_turns.append({"role": "user", "content": texto})
    prompt = construir_prompt()
    respuesta = generar_respuesta_ollama(prompt)
    if respuesta.strip() == ultima_respuesta.strip():
        respuesta = generar_respuesta_ollama(prompt + "\nPor favor, responde de forma diferente.")
    ultima_respuesta = respuesta
    conversation_turns.append({"role": "assistant", "content": respuesta})
    recortar_historial()
    return respuesta

###########################################
# GUI, INTERACCIÓN Y HILOS
###########################################
def seleccionar_y_analizar_imagen():
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.jpeg *.bmp")])
    if ruta:
        return analizar_imagen(ruta)
    return "No se seleccionó ninguna imagen."

def mostrar_en_chat(emisor, mensaje):
    area_chat.configure(state="normal")
    area_chat.insert(tk.END, f"{emisor}: {mensaje}\n")
    area_chat.configure(state="disabled")
    area_chat.yview(tk.END)

def procesar_y_responder(texto_usuario):
    respuesta = procesar_texto(texto_usuario)
    mostrar_en_chat("JARVIS", respuesta)
    hablar(respuesta)

    #Guardar preferencias
    guardar_en_historial(texto_usuario)

    recomendacion = recomendar_accion()
    if recomendacion:
        mostrar_en_chat("JARVIS", recomendacion)
        hablar(recomendacion)

    if respuesta in ["No entendí eso.", "No pude entender qué deseas hacer.", "¿Puedes reformularlo?"]:
        mostrar_aprendizaje_popup(texto_usuario, guardar_sinonimo_desde_gui)



def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    archivo = filedialog.askopenfilename(
        title="Selecciona un archivo CSV o Excel",
        filetypes=[("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xlsx *.xls")]
    )
    return archivo


def ejecutar_comando():
    txt = entrada_texto.get().strip()
    entrada_texto.delete(0, tk.END)
    if txt:
        mostrar_en_chat("Usuario", txt)
        threading.Thread(target=procesar_y_responder, args=(txt,), daemon=True).start()

def escuchar_microfono():
    mostrar_en_chat("JARVIS", "🎤 Escuchando...")
    hablar("Te escucho")
    entrada = escuchar()
    if entrada:
        mostrar_en_chat("Usuario", entrada)
        procesar_y_responder(entrada)
    else:
        mostrar_en_chat("JARVIS", "No entendí lo que dijiste.")
        hablar("No entendí lo que dijiste")


#Función para mostrar el cuadro de aprendizaje
def mostrar_aprendizaje_popup(frase_usuario, guardar_callback):
    ventana = tk.Toplevel()
    ventana.title("Aprender nueva expresión")
    ventana.geometry("400x200")
    
    tk.Label(ventana, text=f"No entendí: \"{frase_usuario}\"", font=("Arial", 12)).pack(pady=10)
    tk.Label(ventana, text="¿Qué querías decir exactamente?").pack()
    
    entrada = tk.Entry(ventana, width=40)
    entrada.pack(pady=5)

    def confirmar():
        nuevo_comando = entrada.get().strip()
        if nuevo_comando:
            guardar_callback(frase_usuario, nuevo_comando)
        ventana.destroy()

    def cancelar():
        ventana.destroy()

    tk.Button(ventana, text="Sí, recuérdalo", command=confirmar).pack(pady=5)
    tk.Button(ventana, text="No", command=cancelar).pack()

def guardar_sinonimo_desde_gui(original, correccion):
    registrar_sinonimo(original, correccion)
    respuesta = procesar_texto(correccion)
    mostrar_en_chat("JARVIS", f"✅ Aprendido: '{original}' significa '{correccion}'.")
    mostrar_en_chat("JARVIS", respuesta)
    hablar(respuesta)

# Autocompletado simple para la entrada de texto
def autocompletar(event):
    texto_actual = entrada_texto.get()
    coincidencias = [cmd for cmd in comandos_autocompletar if cmd.startswith(texto_actual)]
    if coincidencias:
        entrada_texto.delete(0, tk.END)
        entrada_texto.insert(0, coincidencias[0])
        entrada_texto.select_range(len(texto_actual), tk.END)

def iniciar_gui():
    global ventana, entrada_texto, area_chat
    ventana = tk.Tk()
    ventana.title("Asistente JARVIS")
    ventana.geometry("600x500")
    ventana.configure(bg="#f0f0f0")

    area_chat = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, state="disabled", bg="white", font=("Arial", 11))
    area_chat.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    frame_input = tk.Frame(ventana, bg="#f0f0f0")
    frame_input.pack(fill=tk.X, padx=10, pady=5)

    entrada_texto = tk.Entry(frame_input, font=("Arial", 11))
    entrada_texto.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    entrada_texto.bind("<Return>", lambda e=None: ejecutar_comando())
    entrada_texto.bind("<KeyRelease>", autocompletar)

    btn_send = tk.Button(frame_input, text="Enviar", command=ejecutar_comando, bg="#4CAF50", fg="white")
    btn_send.pack(side=tk.LEFT)

    btn_voice = tk.Button(ventana, text="🎤 Hablar", command=lambda: threading.Thread(target=escuchar_microfono, daemon=True).start())
    btn_voice.pack(pady=5)

    # Saludo y acción inicial
    saludo = "Hola, soy JARVIS. ¿En qué puedo ayudarte hoy?"
    mostrar_en_chat("JARVIS", saludo)
    threading.Thread(target=hablar, args=(saludo,), daemon=True).start()

    # Acción inicial útil: mostrar la agenda (si existe)
    if agenda:
        resumen_agenda = listar_eventos()
        mostrar_en_chat("JARVIS", resumen_agenda)
        threading.Thread(target=hablar, args=(resumen_agenda,), daemon=True).start()
    else:
        clima_inicial = procesar_consulta_clima("clima en Valencia")
        mostrar_en_chat("JARVIS", clima_inicial)
        threading.Thread(target=hablar, args=(clima_inicial,), daemon=True).start()

    ventana.mainloop()

if __name__ == "__main__":
    iniciar_gui()
