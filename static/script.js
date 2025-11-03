function enviarComando() {
    const input = document.getElementById('comandoInput').value;
    if (!input) return;
    // Mostrar comando del usuario
    const consoleOutput = document.getElementById('consoleOutput');
    consoleOutput.innerHTML += `<div class="user-msg">> ${input}</div>`;

    fetch('/mensaje', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ mensaje: input })
    })
    .then(response => response.json())
    .then(data => {
        consoleOutput.innerHTML += `<div class="jarvis-msg">${data.respuesta}</div>`;
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
        document.getElementById('comandoInput').value = '';
        speak(data.respuesta);  // Lectura de voz
    });
}

function speak(texto) {
    if ('speechSynthesis' in window) {
        const msg = new SpeechSynthesisUtterance(texto);
        msg.lang = 'es-ES';
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
    }
}

function ejecutarHerramienta(tool) {
    fetch(`/herramienta/${tool}`)
        .then(response => response.json())
        .then(data => {
            const consoleOutput = document.getElementById('consoleOutput');
            consoleOutput.innerHTML += `<div class="jarvis-msg">${data.resultado}</div>`;
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
            speak(data.resultado);
        });
}

document.addEventListener('DOMContentLoaded', function () {
    const input = document.getElementById('comandoInput');

    input.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            enviarComando();
        }
    });
});