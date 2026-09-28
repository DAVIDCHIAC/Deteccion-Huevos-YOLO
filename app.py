#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 para detección de huevos (bueno/malo)
# Requiere: fastapi uvicorn ultralytics opencv-python-headless pillow numpy python-multipart

import os
# Limitar hilos para no saturar el CPU (t3.micro) y reducir memoria
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["TORCH_NUM_THREADS"] = "2"
os.environ["YOLO_VERBOSE"] = "false"

import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-huevos")

MODEL_PATH = os.getenv("MODEL_PATH", "modelo/best.pt")   # Ruta al modelo YOLO
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.35))
IMGSZ = int(os.getenv("IMGSZ", 224))              # Tamano de inferencia
RETURN_IMAGE = True
MAX_DIM = int(os.getenv("MAX_DIM", 1280))         # Redimensionar antes de YOLO

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 - Detector de Huevos (bueno/malo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo
# -------------------------
logger.info("Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
torch.set_num_threads(2)
model = YOLO(MODEL_PATH)
logger.info("Modelo YOLOv8 cargado correctamente.")

# -------------------------
# Helpers
# -------------------------
def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR a base64 (JPG) redimensionada para respuestas livianas."""
    h, w = img_bgr.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / float(w)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')


def process_frame(frame: np.ndarray) -> dict:
    """Ejecuta YOLO sobre un frame y clasifica cada huevo como bueno/malo."""
    if max(frame.shape[:2]) > MAX_DIM:
        scale = MAX_DIM / float(max(frame.shape[:2]))
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        logger.info("Imagen redimensionada para YOLO a max dim %s", MAX_DIM)

    results = model.predict(source=frame, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)
    if not results:
        return {"success": True, "resultados": [], "num_huevos": 0, "num_buenos": 0,
                "num_malos": 0, "image": None, "message": "Sin detecciones"}

    r = results[0]
    resultados: List[dict] = []
    if len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else 0
            label = model.names[cls_id] if cls_id < len(model.names) else "huevo"
            conf = float(confs[i]) if len(confs) > i else 0.0
            es_bueno = label.lower() in ("bueno", "good", "0")
            color = (0, 200, 0) if es_bueno else (0, 0, 255)
            texto = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, texto, (x1, max(24, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "BUENO" if es_bueno else "MALO", (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            resultados.append({"clase": "bueno" if es_bueno else "malo", "conf": round(conf, 3)})

    num_buenos = sum(1 for x in resultados if x["clase"] == "bueno")
    num_malos = len(resultados) - num_buenos
    img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None
    return {
        "success": True,
        "resultados": resultados,
        "num_huevos": len(resultados),
        "num_buenos": num_buenos,
        "num_malos": num_malos,
        "image": img_b64,
        "message": "OK" if resultados else "No se detectaron huevos"
    }


def _decode_image(image_base64: Optional[str], file_bytes: Optional[bytes]):
    """Devuelve el frame BGR a partir de base64 o bytes."""
    if file_bytes is not None:
        nparr = np.frombuffer(file_bytes, np.uint8)
    elif image_base64:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_base64 = image_base64.strip()
        try:
            img_data = base64.b64decode(image_base64 + "===")
        except Exception as e:
            logger.error("Base64 invalido: %s", e)
            return None, "Base64 invalido o corrupto."
        nparr = np.frombuffer(img_data, np.uint8)
    else:
        return None, "No se recibio ninguna imagen"

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "No se pudo decodificar la imagen"
    return frame, None


# -------------------------
# Pagina web
# -------------------------
WEB_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Detector de Huevos</title>
<style>
  :root { --azul:#007AFF; --verde:#34C759; --rojo:#FF3B30; --gris:#f5f5f5; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--gris); color: #222; }
  header { background: var(--azul); color: #fff; padding: 16px; text-align: center;
           box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  header h1 { font-size: 20px; }
  header p { font-size: 13px; opacity: .9; margin-top: 4px; }
  .container { max-width: 480px; margin: 0 auto; padding: 16px; }
  .panel { background: #fff; border-radius: 14px; padding: 16px; margin-bottom: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .panel-title { font-size: 14px; font-weight: 700; color: #555; margin-bottom: 12px;
                 text-transform: uppercase; letter-spacing: .5px; }
  .btn { display: block; width: 100%; padding: 14px; border: none; border-radius: 12px;
         font-size: 16px; font-weight: 600; color: #fff; cursor: pointer; text-align: center; }
  .btn-primary { background: var(--azul); }
  .btn-primary:disabled { background: #8eb8e8; cursor: not-allowed; }
  .btn-new { background: #8e8e93; margin-top: 10px; }
  .btn-hidden input { display: none; }
  #preview { width: 100%; border-radius: 10px; margin: 12px 0; display: none; }
  .spinner { display: none; text-align: center; padding: 20px 0; }
  .spinner.active { display: block; }
  .loader { border: 4px solid #e0e0e0; border-top: 4px solid var(--azul); border-radius: 50%;
            width: 42px; height: 42px; animation: spin 1s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }
  .result-box { border-radius: 10px; padding: 14px; margin-top: 12px; display: none; }
  .result-box.ok { background: #e9f9ef; border: 1px solid #b7ebc8; }
  .result-box.warn { background: #ffecec; border: 1px solid #ffc1c1; }
  .result-box.info { background: #f0f7ff; border: 1px solid #cfe5ff; }
  .egg-result { font-size: 26px; font-weight: 800; text-align: center; margin: 8px 0; }
  .egg-result .ok { color: var(--verde); }
  .egg-result .bad { color: var(--rojo); }
  .conf { font-size: 14px; color: #666; text-align: center; }
  .status { font-size: 14px; color: #666; margin-top: 8px; text-align: center; }
  .img-result { width: 100%; border-radius: 10px; margin-top: 12px; display: none; }
  .hint { font-size: 13px; color: #777; text-align: center; margin-top: 8px; line-height: 1.4; }
  footer { text-align: center; font-size: 12px; color: #999; padding: 20px 0; }
</style>
</head>
<body>
<header>
  <h1>🥚 Detector de Huevos</h1>
  <p>YOLOv8 - Clasificacion bueno/malo en linea</p>
</header>

<div class="container">
  <div class="panel" id="panelCapture">
    <div class="panel-title">📷 Capturar</div>
    <button class="btn btn-primary btn-hidden" onclick="document.getElementById('fileInput').click()">
      Abrir Camara
    </button>
    <input type="file" id="fileInput" accept="image/*" capture="environment"
           style="display:none">
    <img id="preview" alt="Preview">
    <div class="hint">Encuadra el huevo en la camara (sobre la banda) y toma la foto.</div>
  </div>

  <div id="loading" class="spinner">
    <div class="loader"></div>
    <div class="status">Analizando huevos...</div>
  </div>

  <div class="panel" id="panelResult" style="display:none">
    <div class="panel-title">🧪 Resultado</div>
    <div class="result-box" id="resultBox">
      <div class="egg-result" id="eggText">—</div>
      <div class="conf" id="confText"></div>
      <div class="status" id="statusMsg"></div>
    </div>
    <img class="img-result" id="imgResult" alt="Imagen analizada">
    <button class="btn btn-new" onclick="resetApp()">🔁 Nueva captura</button>
  </div>
</div>

<footer>Detector de huevos - YOLOv8 + FastAPI</footer>

<script>
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const loading = document.getElementById('loading');
  const panelCapture = document.getElementById('panelCapture');
  const panelResult = document.getElementById('panelResult');
  const resultBox = document.getElementById('resultBox');
  const eggText = document.getElementById('eggText');
  const confText = document.getElementById('confText');
  const statusMsg = document.getElementById('statusMsg');
  const imgResult = document.getElementById('imgResult');

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const resizedData = await downscaleImage(file);
    const reader = new FileReader();
    reader.onload = async () => {
      preview.src = reader.result;
      preview.style.display = 'block';
      await sendToServer(reader.result);
    };
    reader.readAsDataURL(resizedData);
  });

  function downscaleImage(file, maxDim = 1280) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          const scale = maxDim / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('no blob')),
                      'image/jpeg', 0.75);
      };
      img.onerror = () => reject(new Error('no image'));
      img.src = URL.createObjectURL(file);
    });
  }

  async function sendToServer(imageBase64) {
    loading.classList.add('active');
    panelCapture.style.opacity = '.5';
    try {
      const resp = await fetch('/predict_json/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 })
      });
      const data = await resp.json();
      loading.classList.remove('active');
      panelCapture.style.opacity = '1';
      showResult(data);
    } catch (err) {
      loading.classList.remove('active');
      panelCapture.style.opacity = '1';
      showResult({ success: false, message: 'Error de conexion con el servidor.' });
    }
  }

  function showResult(data) {
    panelResult.style.display = 'block';
    imgResult.style.display = 'none';

    if (data && data.success && data.num_huevos > 0) {
      const malos = data.resultados.filter(r => r.clase === 'malo');
      if (malos.length === 0) {
        resultBox.className = 'result-box ok';
        eggText.className = 'egg-result ok';
        eggText.textContent = '✅ ' + data.num_huevos + ' huevo(s) BUENO(S)';
      } else {
        resultBox.className = 'result-box warn';
        eggText.className = 'egg-result bad';
        eggText.textContent = '⚠️ ' + malos.length + ' huevo(s) MALO(S)';
      }
      const desc = data.resultados.map(r =>
        (r.clase === 'bueno' ? 'Bueno' : 'Malo') + ' (' + Math.round(r.conf * 100) + '%)'
      ).join(' | ');
      confText.textContent = desc;
      statusMsg.textContent = data.message || 'OK';
      speakResult(malos.length === 0 ? 'Huevo bueno' : 'Huevo malo');
    } else {
      resultBox.className = 'result-box info';
      eggText.className = 'egg-result';
      eggText.textContent = 'No se detectaron huevos';
      confText.textContent = '';
      statusMsg.textContent = (data && data.message) || 'Reintenta con mejor angulo e iluminacion.';
    }

    if (data && data.image) {
      imgResult.src = 'data:image/jpeg;base64,' + data.image;
      imgResult.style.display = 'block';
    }
    panelResult.scrollIntoView({ behavior: 'smooth' });
  }

  function speakResult(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES';
    u.rate = 0.95;
    window.speechSynthesis.speak(u);
  }

  function resetApp() {
    fileInput.value = '';
    preview.style.display = 'none';
    panelResult.style.display = 'none';
    panelCapture.style.opacity = '1';
    loading.classList.remove('active');
    imgResult.style.display = 'none';
  }
</script>
</body>
</html>
"""


# -------------------------
# Rutas API
# -------------------------
@app.get("/", response_class=HTMLResponse)
def web():
    return HTMLResponse(WEB_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """Recibe una imagen (multipart o base64) y devuelve los huevos detectados."""
    try:
        logger.info("Peticion recibida en /predict/")
        if file:
            contents = await file.read()
            frame, err = _decode_image(None, contents)
        else:
            frame, err = _decode_image(image_base64, None)
        if err:
            return {"error": err}
        logger.info("Procesando imagen con YOLOv8...")
        return process_frame(frame)
    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}


@app.post("/predict_json/")
async def predict_json(request: Request):
    """Permite enviar imagen como JSON con campo 'image_base64'."""
    try:
        logger.info("Peticion JSON recibida en /predict_json/")
        body = await request.json()
        image_base64 = body.get("image_base64")
        frame, err = _decode_image(image_base64, None)
        if err:
            return {"error": err}
        logger.info("Procesando imagen JSON con YOLOv8...")
        result = process_frame(frame)
        logger.info("Huevos detectados: %s", result["resultados"])
        return result
    except Exception as e:
        logger.exception("Error en /predict_json/: %s", e)
        return {"error": str(e)}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)