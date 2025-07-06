import json
import os

RUTA_CONOCIMIENTO = "conocimiento_personal.json"

def cargar_conocimiento():
    if os.path.exists(RUTA_CONOCIMIENTO):
        try:
            with open(RUTA_CONOCIMIENTO, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Error: El archivo de conocimiento está corrupto.")
            return {}
    return {}

def guardar_conocimiento(clave, valor):
    datos = cargar_conocimiento()
    datos[clave] = valor
    with open(RUTA_CONOCIMIENTO, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
        

def obtener_conocimiento(clave):
    return cargar_conocimiento().get(clave)

def agregar_conocimiento_lista(clave, nuevo_valor):
    datos = cargar_conocimiento()
    if clave not in datos or not isinstance(datos[clave], list):
        datos[clave] = []
    if nuevo_valor not in datos[clave]:
        datos[clave].append(nuevo_valor)
        with open(RUTA_CONOCIMIENTO, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        return f"'{nuevo_valor}' añadido a la lista '{clave}'."
    else:
        return f"'{nuevo_valor}' ya estaba en la lista '{clave}'."

def actualizar_preferencia(clave_interna, valor):
    datos = cargar_conocimiento()
    if "preferencias" not in datos:
        datos["preferencias"] = {}
    datos["preferencias"][clave_interna] = valor
    with open(RUTA_CONOCIMIENTO, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

def eliminar_conocimiento(clave):
    datos = cargar_conocimiento()
    if clave in datos:
        del datos[clave]
        with open(RUTA_CONOCIMIENTO, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

def obtener_todo_el_conocimiento():
    return cargar_conocimiento()


def agregar_tarea_pendiente(tarea):
    agregar_conocimiento_lista("tareas_pendientes", tarea)


def obtener_tareas_pendientes():
    tareas = obtener_conocimiento("tareas_pendientes")
    if tareas:
        respuesta = "En este momento, tienes registradas las siguientes tareas pendientes:\n\n"
        for tarea in tareas:
            respuesta += f"* {tarea}\n"
        return respuesta
    else:
        return "Actualmente no tienes tareas pendientes registradas."
    
def completar_tarea(tarea_completada):
    datos = cargar_conocimiento()
    tareas = datos.get("tareas_pendientes", [])
    if tarea_completada in tareas:
        tareas.remove(tarea_completada)
        datos["tareas_pendientes"] = tareas
        with open(RUTA_CONOCIMIENTO, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        return f"He marcado como completada la tarea: {tarea_completada}"
    else:
        return f"No encontré esa tarea registrada."
