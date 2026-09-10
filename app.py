"""Aplicación Streamlit para clasificación acústica experimental del ductus venoso."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from pydub import AudioSegment


# Debe estar en la misma carpeta de GitHub que app.py.
MODEL_PATH = Path(__file__).resolve().parent / "modelo_ductus_final.h5"

THRESHOLD = 0.50
SR = 16_000
DURATION = 2.95
N_SAMPLES = int(SR * DURATION)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8_000
IMG_SIZE = 128


st.set_page_config(
    page_title="Detector experimental de ductus venoso",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 850px; padding-top: 2rem;}
    .warning-box {
        padding: 1rem; border-radius: .6rem; background: #fff4e5;
        border-left: 6px solid #ef8b17; margin-bottom: 1rem;
    }
    .result-ductus {
        padding: 1rem; border-radius: .6rem; color: white;
        background: #1464f4; text-align: center; font-size: 1.35rem;
        font-weight: 700;
    }
    .result-no-ductus {
        padding: 1rem; border-radius: .6rem; color: white;
        background: #e4572e; text-align: center; font-size: 1.35rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelo de inteligencia artificial...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró 'modelo_ductus_final.h5'. "
            "Debe subirlo a la misma carpeta de GitHub que app.py."
        )
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def save_uploaded_temporarily(uploaded_file) -> Path:
    """Guarda temporalmente el audio y conserva su extensión."""
    original_name = getattr(uploaded_file, "name", "audio.wav")
    suffix = Path(original_name).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(uploaded_file.getvalue())
        return Path(temporary.name)


def load_audio(path: Path) -> tuple[np.ndarray, float, float]:
    """Lee el audio con Librosa o, para WMA, mediante FFmpeg/Pydub."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        audio = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(SR)
        y = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
        y /= float(1 << (8 * audio.sample_width - 1))

    if y.size == 0:
        raise ValueError("El archivo de audio está vacío.")

    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    return y.astype(np.float32), rms, peak


def preprocess_audio(path: Path):
    """Replica el preprocesamiento usado para entrenar el modelo."""
    y, rms, peak_original = load_audio(path)
    y, _ = librosa.effects.trim(y, top_db=30)

    if y.size == 0:
        raise ValueError("No se detectó una señal acústica útil.")

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    if len(y) >= N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start + N_SAMPLES]
    else:
        missing = N_SAMPLES - len(y)
        y = np.pad(y, (missing // 2, missing - missing // 2))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)

    image = tf.image.resize(db[..., None], (IMG_SIZE, IMG_SIZE)).numpy()
    image = np.clip((image + 80.0) / 80.0, 0.0, 1.0) * 255.0
    image = np.repeat(image, 3, axis=-1).astype(np.float32)

    return image[None, ...], db, rms, peak_original


def make_spectrogram(db):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    display = librosa.display.specshow(
        db,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Espectrograma Mel del registro")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia Mel")
    fig.colorbar(display, ax=ax, format="%+2.0f dB", label="Intensidad")
    fig.tight_layout()
    return fig


def analyze(model, uploaded_file):
    temporary_path = save_uploaded_temporarily(uploaded_file)
    try:
        model_input, db, rms, peak = preprocess_audio(temporary_path)
        start = time.perf_counter()
        probability_ductus = float(model.predict(model_input, verbose=0).ravel()[0])
        inference_seconds = time.perf_counter() - start
        return probability_ductus, db, rms, peak, inference_seconds
    finally:
        temporary_path.unlink(missing_ok=True)


st.title("Clasificador acústico del ductus venoso")
st.caption("Prototipo de inteligencia artificial para investigación")

st.markdown(
    """
    <div class="warning-box">
    <b>Uso exclusivamente experimental.</b> El resultado analiza el patrón
    acústico, pero no confirma la localización anatómica del ductus venoso.
    No sustituye la valoración médica ni debe modificar decisiones clínicas.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.info("Compruebe que modelo_ductus_final.h5 esté en la raíz del repositorio.")
    st.stop()

st.subheader("1. Seleccione la fuente del audio")
source = st.radio(
    "Fuente",
    ["Grabar con el micrófono", "Subir un archivo"],
    horizontal=True,
    label_visibility="collapsed",
)

if source == "Grabar con el micrófono":
    audio_file = st.audio_input(
        "Grabe aproximadamente 3 segundos del sonido Doppler",
        sample_rate=SR,
    )
else:
    audio_file = st.file_uploader(
        "Seleccione un archivo de audio",
        type=["wav", "wma", "mp3", "m4a", "aac", "flac", "ogg"],
    )

if audio_file is not None:
    st.audio(audio_file)

st.subheader("2. Analice el registro")

if st.button("Analizar sonido", type="primary", use_container_width=True):
    if audio_file is None:
        st.warning("Primero debe grabar o subir un archivo de audio.")
    else:
        try:
            with st.spinner("Procesando el sonido..."):
                probability, db, rms, peak, elapsed = analyze(model, audio_file)

            probability_no_ductus = 1.0 - probability
            prediction = "DUCTUS" if probability >= THRESHOLD else "NO DUCTUS"
            css_class = "result-ductus" if prediction == "DUCTUS" else "result-no-ductus"

            st.markdown(
                f'<div class="{css_class}">Resultado experimental: {prediction}</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            col1.metric("Probabilidad de ductus", f"{probability:.1%}")
            col2.metric("Probabilidad de no ductus", f"{probability_no_ductus:.1%}")

            st.progress(probability, text="Probabilidad estimada de ductus")
            st.write(f"**Umbral de clasificación:** {THRESHOLD:.2f}")
            st.write(f"**Tiempo de inferencia:** {elapsed:.4f} segundos")

            if rms < 0.005:
                st.warning(
                    "La intensidad del audio es muy baja. Acerque el teléfono "
                    "a la salida de audio o aumente moderadamente el volumen."
                )
            elif peak >= 0.99:
                st.warning(
                    "La señal podría estar saturada. Disminuya el volumen o "
                    "aleje ligeramente el teléfono."
                )
            else:
                st.success("El audio presenta una intensidad adecuada para su procesamiento.")

            st.pyplot(make_spectrogram(db), use_container_width=True)
            st.caption(
                "Interpretación experimental: la probabilidad representa la salida "
                "del modelo y no equivale a una confirmación diagnóstica."
            )
        except Exception as error:
            st.error(f"No se pudo analizar el audio: {error}")
            st.info(
                "Si el archivo es WMA, verifique que packages.txt contenga ffmpeg."
            )
