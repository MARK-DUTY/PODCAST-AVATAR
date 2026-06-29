# 🎙️ Podcast Avatar

Convierte el **audio de un podcast** + **2 fotos** (un hombre y una mujer) en un
**video** donde cada foto **mueve la boca** cuando le toca hablar. El programa
**separa las voces solito** (detecta quién habla y cuándo).

Versión 1 (gratis): funciona en **Google Colab** usando la GPU gratis.

---

## ▶️ Abrir en Google Colab (un clic)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MARK-DUTY/PODCAST-AVATAR/blob/main/Podcast_Avatar.ipynb)

> El repo es privado: la primera vez Colab te pedirá **autorizar el acceso a GitHub**
> (das clic en autorizar y listo).

---

## 📋 Cómo se usa

1. Abre el cuaderno en Colab (botón de arriba).
2. Arriba a la derecha: **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.
3. Ejecuta las celdas **en orden** (botón ▶️ de cada una).
4. Cuando te lo pida: pega tu **token de Hugging Face** y sube el **audio** + las **2 fotos**.
5. Al final se descarga tu video: **`podcast_final.mp4`**.

## 🧩 Qué usa por dentro (todo gratis)

- **pyannote** → separa las voces (quién habla y cuándo)
- **Wav2Lip** → mueve la boca de cada foto con su audio
- **FFmpeg** → une todos los pedazos en el video final

## 🛠️ Notas

- Esta v1 mueve la **boca** (la cara queda quieta). El movimiento de cabeza/manos se
  agregará en versiones siguientes.
- El video sale **vertical** (Reels/TikTok) por defecto. Para horizontal (YouTube),
  cambia `W, H = 720, 1280` por `W, H = 1280, 720` en el paso 7.
- El archivo `build_notebook.py` solo sirve para **regenerar** el cuaderno; no se usa al correr.
