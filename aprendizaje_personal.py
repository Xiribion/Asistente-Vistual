import os
import json

PERFIL_PATH = "perfil_usuario.json"

def cargar_perfil_usuario():
    if os.path.exists(PERFIL_PATH):
        with open(PERFIL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"preferencias": {}, "sinonimos": {}, "correcciones": {}}

def guardar_perfil_usuario(perfil):
    with open(PERFIL_PATH, "w", encoding="utf-8") as f:
        json.dump(perfil, f, ensure_ascii=False, indent=2)

def aplicar_personalizaciones(texto_usuario):
    perfil = cargar_perfil_usuario()

    # Correcciones exactas
    if texto_usuario in perfil["correcciones"]:
        return perfil["correcciones"][texto_usuario]
    
    # Sinónimos
    for canonico, variantes in perfil["sinonimos"].items():
        if texto_usuario.lower() in [v.lower() for v in variantes]:
            return canonico

    return texto_usuario

def registrar_sinonimo(nueva_frase, frase_objetivo):
    perfil = cargar_perfil_usuario()
    if frase_objetivo not in perfil["sinonimos"]:
        perfil["sinonimos"][frase_objetivo] = []
    if nueva_frase not in perfil["sinonimos"][frase_objetivo]:
        perfil["sinonimos"][frase_objetivo].append(nueva_frase)
        guardar_perfil_usuario(perfil)

def registrar_correccion(frase_incorrecta, frase_correcta):
    perfil = cargar_perfil_usuario()
    perfil["correcciones"][frase_incorrecta] = frase_correcta
    guardar_perfil_usuario(perfil)
