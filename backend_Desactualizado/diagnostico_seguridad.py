# diagnostico_seguridad.py

import os
import socket
import subprocess
import psutil
import platform
import time
import requests
import json
from datetime import datetime
import hashlib

modo_experto_activado = False

def activar_modo_experto(clave):
    global modo_experto_activado
    if clave == "Te quiero 3000":
        modo_experto_activado = True
        return "✅ Modo experto activado. Ten cuidado con las herramientas que ejecutes."
    else:
        return "❌ Clave incorrecta."

def obtener_info_red():
    hostname = socket.gethostname()
    ip_local = socket.gethostbyname(hostname)
    salida = subprocess.check_output("ipconfig /all", shell=True, encoding="utf-8", errors="ignore")

    return f"🖧 Hostname: {hostname}\nIP local: {ip_local}\n\n[INFO RED]\n{salida}"

def conexiones_netstat():
    salida = subprocess.check_output("netstat -ano", shell=True, encoding="utf-8", errors="ignore")
    return f"[Conexiones activas]\n{salida}"

def procesos_sospechosos(umbral_cpu=10.0):
    sospechosos = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            cpu = proc.info["cpu_percent"]
            if cpu is None:
                cpu = proc.cpu_percent(interval=0.1)
            if cpu > umbral_cpu:
                conexiones = proc.connections()
                sospechosos.append({
                    "pid": proc.pid,
                    "name": proc.name(),
                    "cpu_percent": cpu,
                    "conexiones": conexiones
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return sospechosos

def escanear_red_local_con_ping():
    print("⏳ Escaneando red local con ping simple...")
    red_base = ".".join(socket.gethostbyname(socket.gethostname()).split(".")[:-1])
    vivos = []
    for i in range(1, 255):
        direccion = f"{red_base}.{i}"
        resultado = subprocess.run(["ping", "-n", "1", "-w", "200", direccion], stdout=subprocess.DEVNULL)
        if resultado.returncode == 0:
            vivos.append(direccion)
    return vivos

# Geolocalizar ip
HISTORIAL_IPS_PATH = "historial_ips.json"

def registrar_ip_sospechosa(ip, ubicacion):
    if os.path.exists(HISTORIAL_IPS_PATH):
        with open(HISTORIAL_IPS_PATH, "r", encoding="utf-8") as f:
            historial = json.load(f)
    else:
        historial = []

    historial.append({
        "ip": ip,
        "ubicacion": ubicacion,
        "fecha": datetime.now().isoformat()
    })

    with open(HISTORIAL_IPS_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

def geolocalizar_ip(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
        if response.status_code == 200:
            data = response.json()
            ciudad = data.get("city", "")
            region = data.get("region", "")
            pais = data.get("country", "")
            org = data.get("org", "")
            proveedor = org.lower() if org else ""

            ubicacion = f"{ciudad}, {region}, {pais} - {org}"

            if any(vpn in proveedor for vpn in ["vpn", "ovh", "digitalocean", "amazon", "aws", "google", "cloud", "azure", "linode", "contabo", "hetzner", "proxy"]):
                ubicacion += " ⚠️ Posible VPN/Proxy/Cloud"

            registrar_ip_sospechosa(ip, ubicacion)

            return ubicacion
    except:
        pass
    return "Ubicación desconocida"


#Comparación de hashes de archivos importantes
ARCHIVOS_CRITICOS = [
    r"C:\Windows\System32\drivers\etc\hosts",
    r"C:\Windows\explorer.exe",
    r"C:\Windows\System32\cmd.exe"
    # Añade más rutas importantes
]

HASH_FILE = "hashes_archivos.json"

def calcular_hash_archivo(path):
    try:
        with open(path, "rb") as f:
            contenido = f.read()
        return hashlib.sha256(contenido).hexdigest()
    except Exception as e:
        return None

def guardar_hashes_base():
    hashes = {archivo: calcular_hash_archivo(archivo) for archivo in ARCHIVOS_CRITICOS}
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f)

def verificar_hashes():
    if not os.path.exists(HASH_FILE):
        return "❗ No se encontraron hashes guardados. Ejecuta primero el modo inicial."

    with open(HASH_FILE, "r") as f:
        hashes_guardados = json.load(f)

    cambios = []
    for archivo, hash_original in hashes_guardados.items():
        hash_actual = calcular_hash_archivo(archivo)
        if hash_actual != hash_original:
            cambios.append(f"⚠️ {archivo} ha sido modificado.")

    if not cambios:
        return "✅ Todos los archivos críticos están intactos."
    else:
        return "\n".join(cambios)
    


def resumen_seguridad_basico():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    # Tiempo encendido
    uptime_seconds = time.time() - psutil.boot_time()
    uptime = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))

    # Uso del sistema
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    info = (
        f"🔐 Diagnóstico de seguridad del sistema\n"
        f"📍 Host: {hostname}\n"
        f"🌐 IP local: {ip}\n"
        f"🕒 Uptime: {uptime}\n"
        f"\n🧮 Uso del sistema:\n"
        f"⚙️ CPU: {cpu_percent}%\n"
        f"📈 RAM: {ram.percent}% ({round(ram.used / (1024**3), 2)} GB usados)\n"
        f"💾 Disco: {disk.percent}% usados ({round(disk.free / (1024**3), 2)} GB libres)\n"
    )

    # Procesos con alto uso de CPU
    procesos_sospechosos = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        try:
            if proc.info["cpu_percent"] > 20:  # Puedes ajustar el umbral
                procesos_sospechosos.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Conexiones externas sospechosas
    conexiones_sospechosas = []
    for conn in psutil.net_connections(kind="inet"):
        try:
            if conn.status == "ESTABLISHED" and conn.raddr:
                ip_remota = conn.raddr.ip
                puerto_remoto = conn.raddr.port

                # Ignorar conexiones locales
                if not ip_remota.startswith(("192.168.", "10.", "127.")):
                    conexiones_sospechosas.append({
                        "ip": ip_remota,
                        "puerto": puerto_remoto,
                        "pid": conn.pid,
                        "estado": conn.status
                    })
        except Exception:
            continue

    return info, procesos_sospechosos, conexiones_sospechosas

# -------------------------
# INTEGRACIÓN CON MASSCAN
# -------------------------
def escanear_con_masscan(rango_ip="192.168.1.0/24", puertos="1-1000"):
    if not modo_experto_activado:
        return "❌ Debes activar el modo experto para usar masscan."

    comando = f"masscan -p{puertos} {rango_ip} --rate=1000"
    try:
        print("⏳ Ejecutando masscan...")
        resultado = subprocess.check_output(comando, shell=True, encoding="utf-8", errors="ignore")
        return f"[Resultados masscan]\n{resultado}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error ejecutando masscan:\n{e.output}"
