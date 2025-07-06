import subprocess
from pymetasploit3.msfrpc import MsfRpcClient
from diagnostico_seguridad import modo_experto_activado

def escanear_nmap(host, puertos="1-1024"):
    if not modo_experto_activado:
        return "🔒 Activa el modo experto para usar nmap."
    try:
        import nmap
        nm = nmap.PortScanner()
        nm.scan(host, puertos)
        salida = ""
        for h in nm.all_hosts():
            salida += f"Host: {h} ({nm[h].hostname()}) - Estado: {nm[h].state()}\n"
            for proto in nm[h].all_protocols():
                for port in nm[h][proto]:
                    info = nm[h][proto][port]
                    salida += f"  {proto}/{port} : {info['state']}\n"
        return salida
    except Exception as e:
        return f"Error al ejecutar nmap: {e}"

def fuerza_bruta_hydra(host, servicio, diccionario, usuario):
    if not modo_experto_activado:
        return "🔒 Activa el modo experto para usar Hydra."
    try:
        cmd = [
            'hydra',
            '-l', usuario,
            '-P', diccionario,
            host, servicio
        ]
        resultado = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return resultado
    except subprocess.CalledProcessError as e:
        return f"Hydra falló:\n{e.output}"
    except Exception as e:
        return f"Error al ejecutar hydra: {e}"

def exploit_metasploit(rhosts, exploit_path, payload):
    if not modo_experto_activado:
        return "🔒 Activa el modo experto para usar Metasploit."
    try:
        client = MsfRpcClient('yourpassword')  # asegúrate de haber iniciado msfrpcd
        ex = client.modules.use('exploit', exploit_path)
        ex['RHOSTS'] = rhosts
        ex['PAYLOAD'] = payload
        job = ex.execute()
        return f"Exploit ejecutado. Job ID: {job}."
    except Exception as e:
        return f"Error en Metasploit: {e}"
