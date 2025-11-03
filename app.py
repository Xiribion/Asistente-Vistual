from flask import Flask, render_template, request, jsonify
# Importa aquí tu integración con Jarvis
from jarvis_ollama import procesar_texto, escanear_nmap, fuerza_bruta_hydra, exploit_metasploit, escanear_con_masscan, cargar_tareas, obtener_tareas_pendientes  # from tu_module import procesar_texto, herramientas_avanzadas...

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
