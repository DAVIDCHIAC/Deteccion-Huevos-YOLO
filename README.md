# Detección de Huevos Bueno/Malo — YOLOv8 + FastAPI

Sistema de clasificación automática de huevos (**bueno/malo**) mediante visión por computadora, pensado para funcionar **en línea sobre una banda transportadora**: se le acerca el celular a cada huevo y la página web indica si está bueno o malo, con su nivel de confianza.

## Características

- Detección + clasificación con un solo modelo **YOLOv8n** (detección de objetos con clase embebida).
- **2 clases**: `bueno` (calidades AA, A, B) y `malo` (calidades C, D, E).
- **Interfaz web móvil**: el usuario fotografía el huevo desde el navegador del celular y recibe el veredicto con voz (TTS) y resaltado visual (cuadro verde = bueno, rojo = malo).
- **Optimizado para instancias pequeñas**: redimensionado a `1280px`, hilos de CPU limitados e `imgsz=224` para correr bien en una `t3.micro` de AWS.
- Modelo entrenado con **mAP50 99.4%** (precisión ~98%, recall ~98%).

## Tecnologías

| Capa | Tecnología |
|------|-----------|
| Backend | FastAPI + Uvicorn |
| Detección | Ultralytics YOLOv8 (`yolov8n`) |
| Imágenes | OpenCV (`cv2`) |
| Deep learning | PyTorch (entrenado con CUDA, corre en CPU) |
| Nube | AWS EC2 (Ubuntu) + systemd |

## Estructura del proyecto

```
.
├── app.py                    # API FastAPI + interfaz web embebida
├── modelo/
│   └── best.pt               # Modelo YOLOv8 entrenado (5.9 MB)
├── scripts/
│   ├── build_dataset.py      # Convierte el dataset crudo a 2 clases + split
│   └── train.py              # Entrenamiento de YOLOv8n
├── prueba/
│   ├── huevo_bueno.jpg       # Imagen de ejemplo (bueno)
│   └── huevo_malo.jpg        # Imagen de ejemplo (malo)
├── README.md
└── .gitignore
```

## Dataset

El dataset original contiene **5,580 recortes de huevos (224×224)** etiquetados en formato YOLO con 6 calidades: **A, AA, B, C, D, E**.

Para este proyecto se fusionan en 2 clases:

| Clase nueva | Calidades originales | Cantidad |
|-------------|----------------------|----------|
| `bueno` (0) | A, AA, B | 2,790 |
| `malo` (1)  | C, D, E  | 2,790 |

Reorganización con split 80/20 (determinístico):

```bash
python scripts/build_dataset.py \
    --src "ruta/Self-Collected Dataset/Yolo Dataset" \
    --dst "dataset" \
    --val 0.2
```

Esto genera `dataset/images/{train,val}` con sus etiquetas y `dataset/data.yaml`.

## Instalación

Requisitos: **Python 3.10+**, para entrenar se recomienda GPU con CUDA.

```bash
git clone <tu-repo-url>
cd <carpeta-del-repo>

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics fastapi "uvicorn[standard]" opencv-python-headless numpy python-multipart
```

> Para CPU (solo inferencia): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Entrenamiento

```bash
python scripts/train.py --data dataset/data.yaml --epochs 80 --imgsz 224 --batch 32
```

El mejor modelo queda en `runs/huevos/weights/best.pt`. Copiar a `modelo/best.pt` para usarlo en la API.

Métricas obtenidas en validación (80 épocas, `yolov8n`, 224px):

| Clase  | Precisión | Recall | mAP50 |
|--------|-----------|--------|-------|
| bueno  | 96.7%     | 99.4%  | 99.3% |
| malo   | 99.5%     | 97.2%  | 99.4% |
| **total** | **98.1%** | **98.3%** | **99.4%** |

## Ejecutar la API

```bash
python app.py
```

Servidor disponible en `http://localhost:8080` (por defecto carga `modelo/best.pt`).

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Página web para capturar/subir foto desde el celular. |
| `GET` | `/health` | Estado del servicio (`{"status":"ok"}`). |
| `POST` | `/predict/` | Imagen por `multipart/form-data` (campo `file`). |
| `POST` | `/predict_json/` | Imagen en Base64 dentro de un JSON. |

### Respuesta

```json
{
  "success": true,
  "resultados": [ { "clase": "bueno", "conf": 0.97 }, { "clase": "malo", "conf": 0.96 } ],
  "num_huevos": 2,
  "num_buenos": 1,
  "num_malos": 1,
  "image": "<imagen anotada en Base64>",
  "message": "OK"
}
```

### Ejemplos

**cURL (multipart):**

```bash
curl -X POST -F "file=@prueba/huevo_bueno.jpg" http://localhost:8080/predict/
```

**Python:**

```python
import requests, base64

b64 = base64.b64encode(open("prueba/huevo_malo.jpg", "rb").read()).decode()
r = requests.post("http://localhost:8080/predict_json/", json={"image_base64": b64})
print(r.json()["resultados"])  # -> [{"clase": "malo", "conf": 0.97}]
```

Con la imagen de ejemplo `prueba/huevo_bueno.jpg` se espera `bueno` (~97%).

## Configuración (variables de entorno)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `modelo/best.pt` | Ruta del modelo YOLO. |
| `CONF_THRESH` | `0.35` | Umbral de confianza de detección. |
| `IMGSZ` | `224` | Tamaño de entrada a la red. |
| `MAX_DIM` | `1280` | Tamaño máximo de la imagen antes de inferir. |
| `PORT` | `8080` | Puerto del servidor. |

## Despliegue en AWS EC2

> Probado en instancia Ubuntu `t3.micro` (con 4 GiB de swap).

1. Lanzar instancia Ubuntu y abrir el puerto `8080` en el grupo de seguridad.
2. Subir el proyecto:

   ```bash
   scp -i "tu_llave.pem" app.py -r modelo scripts prueba ubuntu@<IP>:/home/ubuntu/proyecto/
   ```

3. Preparar e instalar dependencias en la instancia:

   ```bash
   sudo apt update
   sudo apt install -y libgl1 libglib2.0-0 python3-pip python3-venv
   cd /home/ubuntu/proyecto
   python3 -m venv venv
   source venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install ultralytics fastapi "uvicorn[standard]" opencv-python-headless pillow numpy python-multipart
   ```

4. (Opcional) Servicio systemd `/etc/systemd/system/huevos.service` con `Restart=always` y `ExecStart=/home/ubuntu/proyecto/venv/bin/python3 /home/ubuntu/proyecto/app.py`; habilitar con `systemctl enable --now huevos`.
5. Abrir `http://<IP>:8080` desde el celular y fotografiar los huevos sobre la banda.

## Notas de funcionamiento

- La página web redimensiona la foto a `1280px` antes de subirla (respuestas rápidas en datos móviles).
- Encuadra el huevo de cerca, similar a las imágenes de entrenamiento (el huevo ocupa ~50% del cuadro).
- El modelo detecta varios huevos a la vez: en la banda con +1 huevo, cada uno se anota con su clase.

## Prueba rápida

```bash
curl -X POST -F "file=@prueba/huevo_bueno.jpg" http://localhost:8080/predict/
```

Respuesta esperada: `"resultados": [{"clase": "bueno", "conf": ~0.96}]`.