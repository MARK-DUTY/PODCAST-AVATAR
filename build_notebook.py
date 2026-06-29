"""
Generador del cuaderno de Google Colab para el proyecto "Podcast Avatar".

Construimos el archivo .ipynb con Python (json) para garantizar que el JSON
quede 100% valido. Cada celda lleva comentarios en espanol pensados para una
persona sin experiencia en programacion.

v1 (primera version, gratis):
  - pyannote  -> separa las voces (quien habla y cuando)
  - Wav2Lip   -> mueve la boca de cada foto con su audio
  - FFmpeg    -> une todos los pedazos en el video final
Corre en Google Colab usando la GPU gratis.
"""
import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------- portada
cells.append(md(
"""# 🎙️ Podcast Avatar — versión 1 (gratis)

Convierte el **audio de tu podcast** + **2 fotos** (un hombre y una mujer) en un
**video** donde cada foto **mueve la boca** cuando le toca hablar.

El programa **separa las voces solito** (detecta quién habla y cuándo).

---

## 📋 Cómo usarlo (sigue el orden, como una receta)

1. Arriba a la derecha: **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU** y guarda.
2. Ejecuta las celdas **una por una, de arriba hacia abajo** (botón ▶️ a la izquierda de cada una).
3. Cuando una celda te pida algo (token o archivos), hazlo y sigue.
4. Al final se descarga tu video: **podcast_final.mp4**.

> ⚠️ La **primera vez** puede tardar (instala cosas). Si alguna celda marca error,
> cópiame el mensaje rojo y lo arreglamos juntos.
"""))

# ---------------------------------------------------------------- 1 GPU
cells.append(md("## 1️⃣ Revisar que la GPU (tarjeta gráfica gratis) esté encendida"))
cells.append(code(
"""
# Comprobamos que Google nos haya prestado una GPU (acelera muchisimo el trabajo).
import subprocess
res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if res.returncode != 0:
    print("❌ NO hay GPU encendida.")
    print("   Ve a:  Entorno de ejecución → Cambiar tipo de entorno → T4 GPU → Guardar")
    print("   Luego vuelve a ejecutar ESTA celda.")
else:
    print("✅ GPU lista:")
    print(res.stdout.split('\\n')[8] if len(res.stdout.split('\\n')) > 8 else res.stdout)
"""))

# ---------------------------------------------------------------- 2 instalar
cells.append(md(
"""## 2️⃣ Instalar el motor (Wav2Lip) y los modelos

Esto descarga el programa que mueve la boca y sus "cerebros" (modelos).
**Tarda unos 3 a 5 minutos.**

> ⚠️ **MUY IMPORTANTE — esta celda se ejecuta DOS veces:**
> 1. Le das ▶️ la **primera vez**. Al terminar, **la sesión se reinicia sola**
>    (es NORMAL, no es un error; verás un aviso de que el entorno se reinició).
> 2. Cuando termine de reiniciar, **vuelve a darle ▶️ a ESTA MISMA celda 2**.
>    En esta segunda vez (más rápida) saldrá el mensaje **✅ Todo listo**.
>
> 💡 Es normal que aparezca **texto en rojo** con palabras como *"warning"* o
> *"incompatible"*. Eso **NO es error**, puedes ignorarlo. Lo único que importa
> es ver al final el mensaje **✅ Todo listo**.
"""))
cells.append(code(
"""
import os

# 1) Descargar el motor de lip-sync (Wav2Lip)
if not os.path.exists("Wav2Lip"):
    !git clone -q https://github.com/Rudrabha/Wav2Lip

# 2) Librerias con versiones COMPATIBLES entre si (clave para que no truene numpy)
!pip install -q "pyannote.audio>=3.3,<3.4"
!pip install -q "librosa==0.10.2.post1" gdown ffmpeg-python tqdm
# numpy y pandas FIJOS al final, para que manden ellos y no haya choque de versiones
!pip install -q "numpy==2.0.2" "pandas==2.2.2"

# 2b) Arreglar Wav2Lip para la versión nueva de librosa (cambió el formato de una función)
!sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft/' Wav2Lip/audio.py

# 3) Descargar los modelos (pesos) de Wav2Lip
os.makedirs("Wav2Lip/checkpoints", exist_ok=True)
os.makedirs("Wav2Lip/face_detection/detection/sfd", exist_ok=True)
if not os.path.exists("Wav2Lip/checkpoints/wav2lip_gan.pth"):
    !wget -q -O Wav2Lip/checkpoints/wav2lip_gan.pth "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
if not os.path.exists("Wav2Lip/face_detection/detection/sfd/s3fd.pth"):
    !wget -q -O Wav2Lip/face_detection/detection/sfd/s3fd.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

# 4) Reiniciar UNA sola vez para que las versiones nuevas carguen limpias.
#    (Es normal: la sesión se reinicia sola. Después vuelve a darle ▶️ a ESTA celda.)
if not os.path.exists(".instalado_ok"):
    open(".instalado_ok", "w").close()
    print("🔄 Ya quedó lo pesado. Voy a REINICIAR la sesión (es normal, NO es un error).")
    print("   Cuando se reinicie, vuelve a darle ▶️ a ESTA MISMA celda 2 una segunda vez.")
    os.kill(os.getpid(), 9)

# 5) Segunda corrida: comprobamos que todo cargue bien
try:
    import numpy, librosa
    from pyannote.audio import Pipeline
    ok = (os.path.getsize("Wav2Lip/checkpoints/wav2lip_gan.pth") > 1_000_000 and
          os.path.getsize("Wav2Lip/face_detection/detection/sfd/s3fd.pth") > 1_000_000)
    print("numpy:", numpy.__version__, "| librosa:", librosa.__version__)
    print("✅ Todo listo: motor, modelos y separador de voces. Sigue a la celda 3." if ok
          else "⚠️ Algún modelo no descargó bien; vuelve a correr ESTA celda.")
except Exception as e:
    print("⚠️ Aún hay un problema al importar:", e)
    print("👉 Entorno de ejecución → Reiniciar sesión, y vuelve a correr ESTA celda.")
"""))

# ---------------------------------------------------------------- 3 token
cells.append(md(
"""## 3️⃣ Token de Hugging Face (para separar las voces)

La separación de voces usa una herramienta gratis llamada **pyannote**. Necesita
una "llave" (token) gratis. **Solo se hace una vez** (luego ya la tienes):

1. Crea una cuenta gratis en 👉 https://huggingface.co/join
2. Acepta las condiciones (gratis) en estas **dos** páginas (clic en *Agree / Accept*):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Crea tu token aquí 👉 https://huggingface.co/settings/tokens
   (botón *New token* → tipo **Read** → *Create*). Copia el texto que empieza con `hf_...`
4. Ejecuta la celda de abajo y **pégalo** cuando te lo pida.
"""))
cells.append(code(
"""
from getpass import getpass
HF_TOKEN = getpass("Pega tu token de Hugging Face (empieza con hf_) y presiona Enter: ").strip()
print("✅ Token guardado para esta sesión." if HF_TOKEN.startswith("hf_") else "⚠️ Eso no parece un token (debe empezar con hf_).")
"""))

# ---------------------------------------------------------------- 4 subir
cells.append(md(
"""## 4️⃣ Subir tu audio y tus 2 fotos

Te va a pedir **3 archivos, uno por uno**:
1. El **audio** del podcast (`.mp3` o `.wav`).
2. La foto del **primer** hablante (la persona que habla **primero** en el audio).
3. La foto del **segundo** hablante.

> 📸 **Para que se vea bien:** fotos de frente, cara bien iluminada y despejada
> (sin lentes oscuros), boca cerrada o relajada, buena resolución.
"""))
cells.append(code(
"""
from google.colab import files

print("🎧 Sube el AUDIO del podcast (mp3 o wav):")
AUDIO_PATH = list(files.upload().keys())[0]

print("\\n🧑 Sube la foto del PRIMER hablante (el que habla primero):")
IMG_SPEAKER_1 = list(files.upload().keys())[0]

print("\\n👩 Sube la foto del SEGUNDO hablante:")
IMG_SPEAKER_2 = list(files.upload().keys())[0]

print("\\n✅ Recibido:")
print("   Audio :", AUDIO_PATH)
print("   Foto 1:", IMG_SPEAKER_1)
print("   Foto 2:", IMG_SPEAKER_2)
"""))

# ---------------------------------------------------------------- 5 diarizar
cells.append(md(
"""## 5️⃣ Separar las voces (¿quién habla y cuándo?)

Aquí el programa escucha el audio y arma la lista de turnos:
*hablante 1 del segundo 0 al 5, hablante 2 del 5 al 9...* etc.
"""))
cells.append(code(
"""
from pyannote.audio import Pipeline
import torch, os

# Convertimos el audio a un formato simple (mono, 16k) para analizarlo
WAV16 = "audio_16k.wav"
os.system(f'ffmpeg -y -i "{AUDIO_PATH}" -ac 1 -ar 16000 "{WAV16}" -loglevel error')

print("Cargando el separador de voces...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

print("Escuchando y separando voces... (puede tardar segun el largo del audio)")
diar = pipeline(WAV16, num_speakers=2)   # le decimos que son 2 personas

# Lista de turnos ordenados por tiempo
segments = []
for turn, _, speaker in diar.itertracks(yield_label=True):
    segments.append([turn.start, turn.end, speaker])
segments.sort(key=lambda s: s[0])

# Unimos turnos pegados del mismo hablante (menos cortes = mas rapido y fluido)
GAP = 0.4
merged = []
for s in segments:
    if merged and s[2] == merged[-1][2] and s[0] - merged[-1][1] <= GAP:
        merged[-1][1] = s[1]
    else:
        merged.append(list(s))

# Asignamos cada voz a una foto: el PRIMERO que habla = Foto 1
order = []
for s in merged:
    if s[2] not in order:
        order.append(s[2])
spk_to_img = {order[0]: IMG_SPEAKER_1}
if len(order) > 1:
    spk_to_img[order[1]] = IMG_SPEAKER_2
for extra in order[2:]:            # por si detecta de mas, lo mandamos a la foto 1
    spk_to_img[extra] = IMG_SPEAKER_1

print(f"\\n✅ Detecté {len(order)} voz(ces) y {len(merged)} turnos de palabra.")
for i, (a, b, spk) in enumerate(merged[:8]):
    quien = "Foto 1" if spk_to_img[spk] == IMG_SPEAKER_1 else "Foto 2"
    print(f"   Turno {i+1}: {a:5.1f}s → {b:5.1f}s  ({quien})")
if len(merged) > 8:
    print(f"   ... y {len(merged) - 8} turnos más.")
"""))

# ---------------------------------------------------------------- 6 generar clips
cells.append(md(
"""## 6️⃣ Animar la boca de cada turno

Por cada turno, recortamos su pedacito de audio y movemos la boca de la foto
que corresponde. **Esta es la parte que más tarda** (un ratito por turno).
"""))
cells.append(code(
"""
import os
os.makedirs("clips", exist_ok=True)
clip_files = []
MIN_DUR = 0.35   # ignoramos microcortes demasiado breves

total = len(merged)
for i, (start, end, spk) in enumerate(merged):
    dur = end - start
    if dur < MIN_DUR:
        continue
    chunk = f"clips/seg_{i:03d}.wav"
    os.system(f'ffmpeg -y -i "{AUDIO_PATH}" -ss {start:.3f} -t {dur:.3f} -ac 1 -ar 16000 "{chunk}" -loglevel error')
    face = spk_to_img[spk]
    out = f"clips/clip_{i:03d}.mp4"
    quien = "Foto 1" if face == IMG_SPEAKER_1 else "Foto 2"
    print(f"[{i+1}/{total}] {quien}  ({dur:.1f}s)...")
    cmd = (f'cd Wav2Lip && python inference.py '
           f'--checkpoint_path checkpoints/wav2lip_gan.pth '
           f'--face "../{face}" --audio "../{chunk}" --outfile "../{out}" '
           f'--static True --pads 0 15 0 0 --nosmooth')
    os.system(cmd)
    if os.path.exists(out):
        clip_files.append(out)

print(f"\\n✅ Listos {len(clip_files)} clips animados.")
"""))

# ---------------------------------------------------------------- 7 unir
cells.append(md(
"""## 7️⃣ Unir todo en el video final

Igualamos el tamaño de todos los pedazos y los pegamos en orden.

> 📐 Por defecto sale **vertical** (para Reels/TikTok/Shorts). Si lo quieres
> **horizontal** (YouTube), cambia abajo `W, H = 720, 1280` por `W, H = 1280, 720`.
"""))
cells.append(code(
"""
import os
W, H = 720, 1280   # vertical. Para horizontal usa: W, H = 1280, 720

os.makedirs("norm", exist_ok=True)
norm_files = []
for i, c in enumerate(clip_files):
    n = f"norm/n_{i:03d}.mp4"
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=25")
    os.system(f'ffmpeg -y -i "{c}" -vf "{vf}" -c:v libx264 -pix_fmt yuv420p '
              f'-c:a aac -ar 44100 "{n}" -loglevel error')
    if os.path.exists(n):
        norm_files.append(n)

with open("lista.txt", "w") as f:
    for n in norm_files:
        f.write(f"file '{n}'\\n")

os.system('ffmpeg -y -f concat -safe 0 -i lista.txt -c:v libx264 '
          '-pix_fmt yuv420p -c:a aac "podcast_final.mp4" -loglevel error')

if os.path.exists("podcast_final.mp4"):
    print("🎉 ¡VIDEO FINAL listo!: podcast_final.mp4")
else:
    print("⚠️ No se pudo crear el video final. Revisa que haya clips en el paso anterior.")
"""))

# ---------------------------------------------------------------- 8 ver/descargar
cells.append(md("## 8️⃣ Ver y descargar tu video"))
cells.append(code(
"""
from IPython.display import HTML
from base64 import b64encode
from google.colab import files

data = b64encode(open("podcast_final.mp4", "rb").read()).decode()
display(HTML(f'<video width=320 controls src="data:video/mp4;base64,{data}"></video>'))

files.download("podcast_final.mp4")   # se descarga a tu computadora
"""))

# ---------------------------------------------------------------- notebook
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).parent / "Podcast_Avatar.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Cuaderno generado:", out)
print("Celdas:", len(cells))
