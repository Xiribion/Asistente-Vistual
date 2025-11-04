from flask import Flask, render_template, request, jsonify
# Importa aquí tu integración con Jarvis
#from jarvis_ollama import procesar_texto, escanear_nmap, fuerza_bruta_hydra, exploit_metasploit, escanear_con_masscan, cargar_tareas, obtener_tareas_pendientes  # from tu_module import procesar_texto, herramientas_avanzadas...
from importlib import import_module

def crear_app():
    
    app = Flask(__name__)
    
    # Import lazy del módulo que contiene lógica pesada/local
    try:
        jarvis = import_module("jarvis_ollama")
    except Exception as e:
        # Si falla, registramos y usamos stubs seguros
        jarvis = None
        print("Advertencia: no se pudo importar jarvis_ollama:", e)
    
        # Referencias seguras a las funciones (si no existen, usamos fallback)
    procesar_texto = getattr(jarvis, "procesar_texto", None)
    escanear_nmap = getattr(jarvis, "escanear_nmap", lambda: "No disponible")
    fuerza_bruta_hydra = getattr(jarvis, "fuerza_bruta_hydra", lambda: "No disponible")
    exploit_metasploit = getattr(jarvis, "exploit_metasploit", lambda: "No disponible")
    escanear_con_masscan = getattr(jarvis, "escanear_con_masscan", lambda: "No disponible")
    cargar_tareas = getattr(jarvis, "cargar_tareas", lambda: [])

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/mensaje', methods=['POST'])
    def mensaje():
        data = request.get_json()
        texto_usuario = data.get('mensaje', '')
        respuesta = procesar_texto(texto_usuario)  # Llama a tu función existente
        return jsonify({'respuesta': respuesta})


    tools = {
        'nmap': escanear_nmap,
        'hydra': fuerza_bruta_hydra,
        'metasploit': exploit_metasploit,
        'masscan': escanear_con_masscan,
        'tareas': cargar_tareas
    }

    @app.route('/herramienta/<tool>')
    def herramienta(tool):
        if tool in tools:
            resultado = tools[tool]()
        else:
            resultado = "Herramienta no encontrada."
        return jsonify({'resultado': resultado})
    
    return app

if __name__ == '__main__':
    app= crear_app()
    app.run(debug=True)
