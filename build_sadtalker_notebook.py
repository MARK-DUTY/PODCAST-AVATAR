"""
Generador del cuaderno de Colab "Podcast Avatar - SadTalker" (versión con movimiento).

Diferencia con la versión Wav2Lip:
  - Usa SadTalker -> la cara MUEVE LA CABEZA, PARPADEA y "respira" (más realista).
  - Para que sea viable con podcasts largos y 2 personas, en lugar de animar
    cada turno por separado (serían cientos y tardaría horas), anima UN video
    completo por persona y luego RECORTA según quién habla en cada momento.

Piezas (todo gratis, corre en Google Colab con GPU):
  - pyannote   -> separa las voces (quién habla y cuándo)
  - SadTalker  -> anima la cara (cabeza + parpadeo + boca) desde una foto + audio
  - FFmpeg     -> recorta por turnos y une el video final
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md(
"""# 🎬 Podcast Avatar — versión SadTalker (con movimiento)

Igual que la versión anterior, pero ahora la cara **mueve la cabeza, parpadea y
"respira"** (más realista para un podcast), además de mover la boca.

El programa **separa las voces solito** (detecta quién habla y cuándo).

---

## 📋 Cómo usarlo (sigue el orden, como receta)

1. Arriba a la derecha: **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.
2. Ejecuta las celdas **una por una, de arriba hacia abajo** (botón ▶️).
3. Cuando una celda pida algo (token o archivos), hazlo y sigue.
4. Al final se descarga tu video: **podcast_sadtalker.mp4**.

> ⚠️ SadTalker es de 2023 y a veces choca con versiones nuevas de Colab. Ya dejé
> aplicados los arreglos conocidos, pero si una celda marca error en rojo,
> cópiame el mensaje y lo afinamos (como hicimos antes).
>
> 🐢 SadTalker es **más lento** que Wav2Lip. Por eso abajo hay un **MODO PRUEBA**
> para animar solo los primeros segundos y ver rápido cómo queda, antes de hacer
> el podcast completo.
"""))

# 1 - GPU
cells.append(md("## 1️⃣ Revisar que la GPU esté encendida"))
cells.append(code(
"""
import subprocess
res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if res.returncode != 0:
    print("❌ NO hay GPU encendida.")
    print("   Ve a: Entorno de ejecución → Cambiar tipo de entorno → T4 GPU → Guardar")
    print("   Luego vuelve a ejecutar ESTA celda.")
else:
    print("✅ GPU lista.")
    print(res.stdout.split('\\n')[8] if len(res.stdout.split('\\n')) > 8 else res.stdout)
"""))

# 2 - Instalar
cells.append(md(
"""## 2️⃣ Instalar SadTalker, el separador de voces y los modelos

**Tarda ~8 a 12 minutos la primera vez** (baja varias cosas grandes).

> ⚠️ **Esta celda se ejecuta DOS veces** (igual que en el otro cuaderno):
> 1. ▶️ la **primera vez** → instala todo → **se reinicia sola** (es NORMAL, dice
>    "session crashed"; no es error).
> 2. Cuando reinicie, **▶️ otra vez a ESTA misma celda** → sale **✅ Todo listo**.
>
> 💡 El texto en rojo de "warning/incompatible" es **normal**, ignóralo. Solo
> importa ver el **✅ Todo listo** al final de la segunda corrida.
"""))
cells.append(code(
"""
import os

# 1) Descargar SadTalker
if not os.path.exists("SadTalker"):
    !git clone -q https://github.com/OpenTalker/SadTalker

# 2) Versiones COMPATIBLES (mismo "snapshot 2024" que ya nos funcionó)
!pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q "pyannote.audio==3.3.2" "librosa==0.10.2.post1" gdown ffmpeg-python tqdm
# Dependencias propias de SadTalker
!pip install -q safetensors yacs pydub face_alignment==1.3.5 imageio imageio-ffmpeg \
               scikit-image kornia==0.6.8 basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 resampy
# numpy/pandas y compañia fijos al final, para que manden ellos
!pip install -q "numpy==2.0.2" "pandas==2.2.2" "huggingface_hub==0.25.2" "transformers==4.44.2"

# 2b) ARREGLO conocido: basicsr/gfpgan importan algo que torchvision nuevo ya no tiene.
#     Lo redirigimos al lugar nuevo (si no, SadTalker truena al cargar gfpgan).
import subprocess
try:
    deg = subprocess.run(["python", "-c", "import basicsr.data.degradations as m; print(m.__file__)"],
                         capture_output=True, text=True).stdout.strip()
    if deg:
        os.system(f"sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/"
                  f"from torchvision.transforms.functional import rgb_to_grayscale/' '{deg}'")
        print("Arreglo basicsr aplicado en:", deg)
except Exception as e:
    print("Aviso: no pude aplicar el arreglo basicsr (lo revisamos si hay error):", e)

# 3) Modelos de SadTalker
os.makedirs("SadTalker/checkpoints", exist_ok=True)
os.makedirs("SadTalker/gfpgan/weights", exist_ok=True)
base = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
ck = "SadTalker/checkpoints"
def _get(url, dest):
    if not (os.path.exists(dest) and os.path.getsize(dest) > 100000):
        os.system(f'wget -q -O "{dest}" "{url}"')
_get(f"{base}/mapping_00109-model.pth.tar", f"{ck}/mapping_00109-model.pth.tar")
_get(f"{base}/mapping_00229-model.pth.tar", f"{ck}/mapping_00229-model.pth.tar")
_get(f"{base}/SadTalker_V0.0.2_256.safetensors", f"{ck}/SadTalker_V0.0.2_256.safetensors")
_get(f"{base}/SadTalker_V0.0.2_512.safetensors", f"{ck}/SadTalker_V0.0.2_512.safetensors")
gf = "SadTalker/gfpgan/weights"
_get("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth", f"{gf}/alignment_WFLW_4HG.pth")
_get("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", f"{gf}/detection_Resnet50_Final.pth")
_get("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", f"{gf}/GFPGANv1.4.pth")
_get("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth", f"{gf}/parsing_parsenet.pth")

# 4) Reiniciar UNA sola vez para que las versiones nuevas carguen limpias
if not os.path.exists(".instalado_sad"):
    open(".instalado_sad", "w").close()
    print("🔄 Ya quedó lo pesado. Voy a REINICIAR la sesión (es normal, NO es error).")
    print("   Cuando se reinicie, vuelve a darle ▶️ a ESTA MISMA celda 2.")
    os.kill(os.getpid(), 9)

# 5) Segunda corrida: comprobamos
try:
    import numpy, librosa, torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        try:
            from torchaudio.backend.common import AudioMetaData as _AMD
            torchaudio.AudioMetaData = _AMD
        except Exception:
            pass
    from pyannote.audio import Pipeline
    big = os.path.getsize(f"{ck}/SadTalker_V0.0.2_512.safetensors") > 1_000_000
    print("numpy:", numpy.__version__, "| librosa:", librosa.__version__, "| torchaudio:", torchaudio.__version__)
    print("✅ Todo listo: SadTalker, modelos y separador de voces. Sigue a la celda 3." if big
          else "⚠️ Algún modelo no descargó bien; vuelve a correr ESTA celda.")
except Exception as e:
    print("⚠️ Aún hay un problema al importar:", e)
    print("👉 Entorno de ejecución → Reiniciar sesión, y vuelve a correr ESTA celda.")
"""))

# 3 - Token
cells.append(md(
"""## 3️⃣ Token de Hugging Face (para separar las voces)

El mismo token que ya usaste antes (empieza con `hf_`). Si no lo tienes a mano,
sácalo en https://huggingface.co/settings/tokens (tipo **Read**). Y recuerda haber
aceptado las condiciones en:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
"""))
cells.append(code(
"""
from getpass import getpass
HF_TOKEN = getpass("Pega tu token de Hugging Face (empieza con hf_) y Enter: ").strip()
print("✅ Token guardado." if HF_TOKEN.startswith("hf_") else "⚠️ Eso no parece un token (debe empezar con hf_).")
"""))

# 4 - Subir
cells.append(md(
"""## 4️⃣ Subir tu audio y tus 2 fotos

En orden: **audio**, luego foto del que **habla primero**, luego la del segundo.

> 📸 **Para que se vea bien:** fotos de **frente**, cara despejada, **boca cerrada
> o relajada SIN sonreír enseñando dientes**, buena resolución.
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
print("\\n✅ Recibido:\\n  Audio :", AUDIO_PATH, "\\n  Foto 1:", IMG_SPEAKER_1, "\\n  Foto 2:", IMG_SPEAKER_2)
"""))

# 5 - Modo prueba
cells.append(md(
"""## 5️⃣ ¿Prueba rápida o podcast completo?

Como SadTalker es lento, te recomiendo **probar primero solo los primeros segundos**
para ver cómo queda. Cambia el número de abajo:
- `MAX_SEGUNDOS = 30` → anima solo los **primeros 30 segundos** (rápido, para probar).
- `MAX_SEGUNDOS = 0` → anima el **podcast completo** (tarda mucho más).
"""))
cells.append(code(
"""
MAX_SEGUNDOS = 30   # 0 = todo el audio | 30 = solo los primeros 30s (prueba rápida)

import os
if MAX_SEGUNDOS and MAX_SEGUNDOS > 0:
    AUDIO_USAR = "audio_recortado.wav"
    os.system(f'ffmpeg -y -i "{AUDIO_PATH}" -t {MAX_SEGUNDOS} -ac 1 -ar 16000 "{AUDIO_USAR}" -loglevel error')
    print(f"🔎 MODO PRUEBA: usando solo los primeros {MAX_SEGUNDOS} segundos.")
else:
    AUDIO_USAR = "audio_full.wav"
    os.system(f'ffmpeg -y -i "{AUDIO_PATH}" -ac 1 -ar 16000 "{AUDIO_USAR}" -loglevel error')
    print("🎬 MODO COMPLETO: usando todo el audio.")
print("Audio a usar:", AUDIO_USAR)
"""))

# 6 - Diarizacion
cells.append(md("## 6️⃣ Separar las voces (¿quién habla y cuándo?)"))
cells.append(code(
"""
import torch, os, torchaudio
if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio.backend.common import AudioMetaData as _AMD
        torchaudio.AudioMetaData = _AMD
    except Exception:
        pass
from pyannote.audio import Pipeline

print("Cargando el separador de voces...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

print("Escuchando y separando voces...")
diar = pipeline(AUDIO_USAR, num_speakers=2)

segments = []
for turn, _, speaker in diar.itertracks(yield_label=True):
    segments.append([turn.start, turn.end, speaker])
segments.sort(key=lambda s: s[0])

GAP = 0.4
merged = []
for s in segments:
    if merged and s[2] == merged[-1][2] and s[0] - merged[-1][1] <= GAP:
        merged[-1][1] = s[1]
    else:
        merged.append(list(s))

order = []
for s in merged:
    if s[2] not in order:
        order.append(s[2])
spk_to_img = {order[0]: IMG_SPEAKER_1}
if len(order) > 1:
    spk_to_img[order[1]] = IMG_SPEAKER_2
for extra in order[2:]:
    spk_to_img[extra] = IMG_SPEAKER_1

print(f"\\n✅ Detecté {len(order)} voz(ces) y {len(merged)} turnos.")
for i, (a, b, spk) in enumerate(merged[:8]):
    quien = "Foto 1" if spk_to_img[spk] == IMG_SPEAKER_1 else "Foto 2"
    print(f"   Turno {i+1}: {a:5.1f}s → {b:5.1f}s  ({quien})")
"""))

# 7 - SadTalker por persona
cells.append(md(
"""## 7️⃣ Animar la cara de cada persona (SadTalker)

Aquí está el truco eficiente: animamos **UN video completo por persona** (cabeza,
parpadeo y boca) en lugar de cientos de pedacitos. **Esta es la parte lenta.**
"""))
cells.append(code(
"""
import os, glob, shutil

def animar(img, etiqueta):
    out_dir = f"sad_out_{etiqueta}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"🎭 Animando {etiqueta} (puede tardar)...")
    cmd = (f'cd SadTalker && python inference.py '
           f'--driven_audio "../{AUDIO_USAR}" --source_image "../{img}" '
           f'--result_dir "../{out_dir}" --preprocess full --still=False')
    os.system(cmd)
    vids = sorted(glob.glob(f"{out_dir}/**/*.mp4", recursive=True), key=os.path.getmtime)
    if not vids:
        print(f"   ⚠️ No se generó video para {etiqueta}.")
        return None
    print(f"   ✅ {etiqueta} listo:", vids[-1])
    return vids[-1]

full_video = {}
full_video[IMG_SPEAKER_1] = animar(IMG_SPEAKER_1, "persona1")
full_video[IMG_SPEAKER_2] = animar(IMG_SPEAKER_2, "persona2")
print("\\nVideos por persona:", full_video)
"""))

# 8 - Recortar y unir
cells.append(md(
"""## 8️⃣ Recortar por turnos y unir el video final

Tomamos de cada persona solo los trozos donde le toca hablar y los unimos.

> 📐 Sale **horizontal** (YouTube). Para vertical (Reels) cambia `W, H = 1280, 720`
> por `W, H = 720, 1280`.
"""))
cells.append(code(
"""
import os
W, H = 1280, 720   # horizontal (YouTube). Vertical: W, H = 720, 1280

os.makedirs("segs", exist_ok=True)
norm_files = []
for i, (start, end, spk) in enumerate(merged):
    src = full_video.get(spk_to_img[spk])
    if not src or not os.path.exists(src):
        continue
    dur = end - start
    if dur < 0.3:
        continue
    n = f"segs/seg_{i:03d}.mp4"
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=25")
    os.system(f'ffmpeg -y -ss {start:.3f} -t {dur:.3f} -i "{src}" -vf "{vf}" '
              f'-c:v libx264 -pix_fmt yuv420p -an "{n}" -loglevel error')
    if os.path.exists(n):
        norm_files.append(n)

with open("lista.txt", "w") as f:
    for n in norm_files:
        f.write(f"file '{n}'\\n")

# Unir video (sin audio) y luego pegarle el audio original recortado
os.system('ffmpeg -y -f concat -safe 0 -i lista.txt -c:v libx264 -pix_fmt yuv420p "video_mudo.mp4" -loglevel error')
os.system(f'ffmpeg -y -i "video_mudo.mp4" -i "{AUDIO_USAR}" -c:v copy -c:a aac -shortest "podcast_sadtalker.mp4" -loglevel error')

print("🎉 ¡VIDEO FINAL listo!: podcast_sadtalker.mp4" if os.path.exists("podcast_sadtalker.mp4")
      else "⚠️ No se pudo crear el video final. Revisa el paso anterior.")
"""))

# 9 - Ver
cells.append(md("## 9️⃣ Ver y descargar tu video"))
cells.append(code(
"""
from IPython.display import HTML
from base64 import b64encode
from google.colab import files
data = b64encode(open("podcast_sadtalker.mp4", "rb").read()).decode()
display(HTML(f'<video width=480 controls src="data:video/mp4;base64,{data}"></video>'))
files.download("podcast_sadtalker.mp4")
"""))

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

out = Path(__file__).parent / "Podcast_Avatar_SadTalker.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Cuaderno generado:", out, "| Celdas:", len(cells))
