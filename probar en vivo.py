# -*- coding: utf-8 -*-
"""
Prueba experimental en vivo del modelo de sonido del ductus venoso.

Graba 2.95 segundos desde el micrófono, aplica exactamente el mismo
preprocesamiento Mel del entrenamiento y muestra la probabilidad del modelo.

IMPORTANTE: herramienta de investigación. No usar para diagnóstico ni para
modificar decisiones clínicas.

Instalación (Anaconda Prompt):
    pip install sounddevice scipy librosa matplotlib pandas tensorflow
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
import tensorflow as tf
from scipy.io.wavfile import write as write_wav


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
MODEL_PATH = Path(
    r"C:\Users\Azul8\OneDrive\Escritorio\Residentes aprobados"
    r"\Giovanni\sonidos\todos\resultados_modelo\modelo_ductus_final.h5"
)
LIVE_OUTPUT_DIR = Path(
    r"C:\Users\Azul8\OneDrive\Escritorio\Residentes aprobados"
    r"\Giovanni\sonidos\todos\resultados_modelo\pruebas_en_vivo"
)

# Use None para el micrófono predeterminado. Para elegir otro, escriba su índice.
INPUT_DEVICE = None
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


def list_input_devices() -> None:
    """Muestra los dispositivos que pueden utilizarse para grabar."""
    print("\nDISPOSITIVOS DE ENTRADA DISPONIBLES")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            default = "  <-- predeterminado" if index == sd.default.device[0] else ""
            print(f"[{index}] {device['name']}{default}")


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{MODEL_PATH}\n"
            "Compruebe que terminó el entrenamiento y que la ruta es correcta."
        )
    print(f"\nCargando modelo: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("Modelo cargado correctamente.")
    return model


def record_audio() -> np.ndarray:
    """Realiza cuenta regresiva y graba un canal a 16 kHz."""
    print("\nAcerque el micrófono a la salida de audio del ultrasonido.")
    for number in (3, 2, 1):
        print(f"Grabación en {number}...", flush=True)
        time.sleep(1)
    print(f"GRABANDO durante {DURATION:.2f} segundos...", flush=True)
    recording = sd.rec(
        frames=N_SAMPLES,
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
        blocking=True,
    )
    print("Grabación terminada.")
    return recording[:, 0]


def signal_quality(y: np.ndarray) -> dict:
    """Indicadores básicos antes de normalizar el audio."""
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    clipped_fraction = float(np.mean(np.abs(y) >= 0.99))
    return {"rms": rms, "peak": peak, "clipped_fraction": clipped_fraction}


def preprocess_audio(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replica el procesamiento empleado durante el entrenamiento."""
    y = np.asarray(y, dtype=np.float32).squeeze()
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        y = np.zeros(N_SAMPLES, dtype=np.float32)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    if len(y) >= N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start + N_SAMPLES]
    else:
        pad = N_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    image = tf.image.resize(db[..., None], (IMG_SIZE, IMG_SIZE)).numpy()
    image = np.clip((image + 80.0) / 80.0, 0.0, 1.0) * 255.0
    image = np.repeat(image, 3, axis=-1).astype(np.float32)
    return image[None, ...], db


def predict(model: tf.keras.Model, model_input: np.ndarray) -> tuple[float, float]:
    start = time.perf_counter()
    probability = float(model.predict(model_input, verbose=0).ravel()[0])
    inference_seconds = time.perf_counter() - start
    return probability, inference_seconds


def interpretation(probability: float) -> tuple[str, str]:
    if probability >= THRESHOLD:
        return "DUCTUS", "#1464F4"
    return "NO DUCTUS", "#E4572E"


def show_and_save_result(db: np.ndarray, probability: float,
                         timestamp: str, label: str, color: str) -> None:
    """Muestra espectrograma y probabilidad en ventanas individuales."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = librosa.display.specshow(
        db, sr=SR, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX, cmap="magma", ax=ax)
    ax.set(title=f"Espectrograma Mel - {timestamp}", xlabel="Tiempo (s)",
           ylabel="Frecuencia Mel")
    fig.colorbar(image, ax=ax, format="%+2.0f dB", label="Intensidad")
    fig.tight_layout()
    fig.savefig(LIVE_OUTPUT_DIR / f"{timestamp}_espectrograma.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(["Probabilidad"], [probability], color=color, height=.48)
    ax.axvline(THRESHOLD, color="#333333", linestyle="--", linewidth=1.8,
               label=f"Umbral = {THRESHOLD:.2f}")
    ax.text(min(probability + .02, .92), 0, f"{probability:.1%}",
            va="center", fontsize=14, fontweight="bold")
    ax.set(xlim=(0, 1), xlabel="Probabilidad estimada de ductus",
           title=f"Resultado experimental: {label}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(LIVE_OUTPUT_DIR / f"{timestamp}_resultado.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(block=False)


def append_log(row: dict) -> None:
    path = LIVE_OUTPUT_DIR / "registro_pruebas_en_vivo.csv"
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_one_test(model: tf.keras.Model) -> None:
    raw_audio = record_audio()
    quality = signal_quality(raw_audio)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = LIVE_OUTPUT_DIR / f"{timestamp}_audio.wav"
    write_wav(audio_path, SR, np.clip(raw_audio, -1, 1))

    model_input, db = preprocess_audio(raw_audio)
    probability, inference_seconds = predict(model, model_input)
    label, color = interpretation(probability)

    print("\n" + "=" * 55)
    print(f"RESULTADO EXPERIMENTAL: {label}")
    print(f"Probabilidad estimada de ductus: {probability:.2%}")
    print(f"Probabilidad estimada de no ductus: {1-probability:.2%}")
    print(f"Umbral de clasificación: {THRESHOLD:.2f}")
    print(f"Tiempo de inferencia: {inference_seconds:.4f} segundos")
    print(f"RMS: {quality['rms']:.5f} | Pico: {quality['peak']:.3f}")
    if quality["rms"] < 0.005:
        print("ADVERTENCIA: señal muy baja; acerque el micrófono o aumente el volumen.")
    if quality["clipped_fraction"] > 0.01:
        print("ADVERTENCIA: posible saturación; reduzca el volumen o la ganancia.")
    print("Uso exclusivo para investigación; no constituye un diagnóstico.")
    print("=" * 55)

    append_log({
        "fecha_hora": timestamp,
        "archivo_audio": str(audio_path),
        "probabilidad_ductus": probability,
        "probabilidad_no_ductus": 1 - probability,
        "prediccion": label,
        "umbral": THRESHOLD,
        "tiempo_inferencia_segundos": inference_seconds,
        **quality,
    })
    show_and_save_result(db, probability, timestamp, label, color)


def main() -> None:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("PRUEBA EXPERIMENTAL EN VIVO: DUCTUS / NO DUCTUS")
    print("No debe utilizarse para decisiones clínicas.")
    list_input_devices()
    model = load_model()

    while True:
        answer = input("\nPresione ENTER para grabar o escriba Q para terminar: ").strip().lower()
        if answer == "q":
            break
        try:
            run_one_test(model)
        except Exception as exc:
            print(f"\nNo se pudo completar la prueba: {exc}")
            print("Revise el micrófono, INPUT_DEVICE y los permisos de Windows.")
    print("Programa finalizado.")


if __name__ == "__main__":
    main()

