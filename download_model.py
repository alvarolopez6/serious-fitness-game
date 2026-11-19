from pathlib import Path
from urllib.request import urlopen


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)
MODEL_PATH = Path("models/pose_landmarker_full.task")
PARTIAL_PATH = Path("models/pose_landmarker_full.task.part")


def download_model() -> None:
    if MODEL_PATH.exists():
        print(f"Modelo ya disponible: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Descargando el modelo Pose Landmarker Full...")

    try:
        with urlopen(MODEL_URL, timeout=120) as response:
            with PARTIAL_PATH.open("wb") as model_file:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)
    except Exception:
        PARTIAL_PATH.unlink(missing_ok=True)
        raise

    PARTIAL_PATH.replace(MODEL_PATH)
    print(f"Modelo descargado en: {MODEL_PATH}")


if __name__ == "__main__":
    download_model()
