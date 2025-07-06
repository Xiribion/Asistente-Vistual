# auditoria.py
import os
import json
import time
import hashlib
import psutil
import socket
from diagnostico_seguridad import resumen_seguridad_basico,geolocalizar_ip, guardar_hashes_base, verificar_hashes

AUDITORIA_HASH_FILE = True  # bandera si los hashes ya están guardados

def auditoria_rapida():
    info, procesos, conexiones = resumen_seguridad_basico()
    respuesta = info + "\n🧠 Procesos sospechosos:\n"
    if not procesos:
        respuesta += "✅ No se detectaron procesos sospechosos.\n"
    else:
        for p in procesos:
            respuesta += f"⚠️ PID {p['pid']} - {p['name']} - CPU: {p['cpu_percent']}%\n"

    respuesta += "\n🌐 Conexiones externas activas:\n"
    if not conexiones:
        respuesta += "✅ No se detectaron conexiones externas sospechosas.\n"
    else:
        for c in conexiones:
            ubic = geolocalizar_ip(c["ip"])
            respuesta += (
                f"🔸 IP: {c['ip']}:{c['puerto']} (PID: {c['pid']})\n"
                f"    📍 {ubic}\n"
            )

    return respuesta

def auditoria_completa():
    global AUDITORIA_HASH_FILE
    if AUDITORIA_HASH_FILE:
        guardar_hashes_base()
        AUDITORIA_HASH_FILE = False
        return "🔐 Hashes iniciales guardados. Ejecuta de nuevo para ver verificación."

    programa = auditoria_rapida()
    programa += "\n\n🧾 Verificando integridad de archivos críticos:\n"
    programa += verificar_hashes()
    return programa
