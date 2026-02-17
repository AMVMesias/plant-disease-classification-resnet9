# 🌿 Plant Disease Classification — ResNet9

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.07%25-00C853?style=for-the-badge)

**Clasificación de enfermedades en plantas usando una red neuronal ResNet9 entrenada desde cero con TensorFlow/Keras.**

</div>

---

## 📋 Descripción

Este proyecto implementa un modelo de **Deep Learning** basado en la arquitectura **ResNet9** para identificar **38 clases** de enfermedades y estados saludables en hojas de plantas. El modelo fue entrenado desde cero (sin transfer learning) y alcanzó una accuracy de **99.07%** en el conjunto de validación.

### 🎯 Objetivo
Detectar automáticamente si una hoja de planta está sana o enferma, y clasificar la enfermedad específica, a partir de una fotografía.

---

## 📊 Resultados del Entrenamiento

| Métrica | Valor |
|---------|-------|
| **Accuracy Final (Validación)** | **99.07%** |
| **Loss Final (Validación)** | 0.1338 |
| **Mejor Accuracy en Entrenamiento** | 96.95% |
| **Épocas** | 15 |
| **Total de Parámetros** | 6,594,214 (~25 MB) |

### 📈 Evolución del Entrenamiento

| Época | Train Acc | Val Acc | Train Loss | Val Loss |
|:-----:|:---------:|:-------:|:----------:|:--------:|
| 1 | 44.28% | 63.94% | 2.5411 | 1.9435 |
| 5 | 84.64% | 78.86% | 1.0675 | 1.3023 |
| 10 | 91.02% | 81.03% | 0.6032 | 1.0534 |
| 13 | 94.23% | 88.73% | 0.3663 | 0.5640 |
| 14 | 95.40% | 97.97% | 0.2933 | 0.1895 |
| **15** | **96.95%** | **99.07%** | **0.2156** | **0.1338** |

---

## 🧠 Arquitectura del Modelo

El modelo utiliza la arquitectura **ResNet9**, una variante compacta de ResNet con conexiones residuales:

```
Input (256×256×3)
    │
    ▼
┌─────────────────┐
│  Conv2D (64)    │  → BatchNorm → ReLU
└────────┬────────┘
         ▼
┌─────────────────┐
│  Conv2D (128)   │  → BatchNorm → ReLU → MaxPool(4)
└────────┬────────┘
         ▼
┌─────────────────┐
│  ResBlock (128) │  ← Conexión residual (skip connection)
│  Conv→BN→ReLU   │
│  Conv→BN→ReLU   │
│  + Input        │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Conv2D (256)   │  → BatchNorm → ReLU → MaxPool(4)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Conv2D (512)   │  → BatchNorm → ReLU → MaxPool(4)
└────────┬────────┘
         ▼
┌─────────────────┐
│  ResBlock (512) │  ← Conexión residual (skip connection)
│  Conv→BN→ReLU   │
│  Conv→BN→ReLU   │
│  + Input        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ GlobalAvgPool2D │
│  Dropout (0.5)  │
│  Dense (38)     │  → Softmax
└─────────────────┘
```

**Total: 6,594,214 parámetros** (6,589,734 entrenables)

---

## ⚡ Técnicas de Optimización

| Técnica | Descripción |
|---------|-------------|
| **Mixed Precision (FP16)** | Usa `float16` para cálculos en GPU, acelerando el entrenamiento significativamente |
| **One Cycle LR** | Scheduler que sube el LR al 30% del entrenamiento y luego baja gradualmente |
| **AdamW Optimizer** | Adam con weight decay desacoplado (`1e-4`) para mejor regularización |
| **Gradient Clipping** | `clipnorm=0.1` para prevenir explosión de gradientes |
| **Data Augmentation** | Flips aleatorios, cambios de brillo, contraste y saturación |
| **L2 Regularization** | Weight decay de `1e-4` en todas las capas convolucionales |
| **Dropout** | 50% antes de la capa de clasificación final |
| **tf.data Pipeline** | Carga de datos optimizada con prefetch y paralelización |

---

## 📦 Dataset

- **Nombre:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Fuente:** Kaggle
- **Imágenes de Entrenamiento:** 70,295
- **Imágenes de Validación:** 17,572
- **Clases:** 38 (diferentes enfermedades y estados sanos de plantas)
- **Tamaño de Imagen:** 256 × 256 px (RGB)

---

## 🛠️ Configuración e Instalación

### Requisitos
- Python 3.12+
- TensorFlow 2.x
- GPU recomendada (NVIDIA T4 o superior)
- Cuenta de Kaggle con API Token

### Configuración de Credenciales

> ⚠️ **Las credenciales de Kaggle se manejan de forma segura y NO se suben al repositorio.**

1. Ve a [kaggle.com/settings](https://kaggle.com/settings) → sección **API** → **Create New API Token**
2. Crea un archivo `credentials.json` en la raíz del proyecto con el siguiente formato:

```json
{
  "kaggle_username": "TU_USUARIO_AQUI",
  "kaggle_key": "TU_API_KEY_AQUI"
}
```

3. Este archivo está protegido por `.gitignore` y **nunca se subirá a GitHub**.
4. El notebook lee automáticamente este archivo y configura las credenciales de Kaggle.

### Ejecución

1. Abre `train_resnet9_optimized.ipynb` en **Google Colab**
2. Sube tu archivo `credentials.json` al entorno de Colab
3. Selecciona **GPU T4** como acelerador de hardware
4. Ejecuta todas las celdas secuencialmente

---

## 📁 Estructura del Proyecto

```
plant-disease-classification-resnet9/
├── 📓 train_resnet9_optimized.ipynb   # Notebook principal (entrenamiento completo)
├── 🔐 credentials.json               # Credenciales de Kaggle (NO se sube a Git)
├── 📄 .gitignore                      # Archivos ignorados por Git
└── 📖 README.md                       # Este archivo
```

### Archivos Generados en Google Drive (`Moviles AI99/`)
```
Moviles AI99/
├── mejor_modelo_resnet9.keras    # Mejor modelo (checkpoint)
├── modelo_resnet9_final.keras    # Modelo final
├── clases.json                   # Diccionario de clases (38 clases)
├── history.json                  # Historial de entrenamiento
└── resultados_resnet9.png        # Gráficas de accuracy y loss
```

---

## 🔒 Seguridad

Los siguientes archivos sensibles están excluidos del repositorio mediante `.gitignore`:

- `credentials.json` — Credenciales del usuario
- `kaggle.json` / `**/kaggle.json` — Token API de Kaggle
- `.kaggle/` — Carpeta de configuración de Kaggle
- `secrets.json` / `*.secrets.json` — Cualquier archivo de secretos
- `.env` / `.env.*` — Variables de entorno

---

## 👨‍💻 Autor

**Mesias Mariscal V.**

---

<div align="center">

*Proyecto académico — Universidad — Desarrollo de Aplicaciones Móviles (3er Parcial)*

</div>
