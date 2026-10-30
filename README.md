# Mapeo de inundaciones con Sentinel-1 y U-Net — Sen1Floods11 y Lima 2017

**Redes Neuronales y Aprendizaje Profundo — UNI FIIS, 2026-II · Examen parcial, Pregunta 2**

Segmentación píxel a píxel del agua de inundación a partir de radar Sentinel-1 (VV, VH) con una U-Net implementada desde cero en PyTorch. El repositorio incluye:

1. Una **U-Net propia** de 2 canales de entrada, con profundidad, conexiones de salto y tipo de *upsampling* configurables.
2. Un **estudio de ablación** con presupuesto de cómputo fijo:
   - etiquetas doradas vs. doradas + débiles;
   - pérdida BCE, Dice, combinada y BCE con peso en bordes;
   - con y sin conexiones de salto, y profundidad 4 vs. 3.
3. Una **evaluación en distribución** (IoU, F1, precisión, recall, omisión, comisión y matriz de confusión) comparada con la línea base clásica de Otsu.
4. Una **región de prueba propia para el Perú**: Sentinel-1 del 17-mar-2017 sobre Lima, comparado con la delineación **Copernicus EMS EMSN038** e inferido con la estrategia *overlap-tile*.
5. Un **análisis cualitativo** de errores característicos del SAR.

**Informe:** [`Informe_P2_Inundaciones_Sentinel1.pdf`](Informe_P2_Inundaciones_Sentinel1.pdf), en formato NeurIPS y 10 páginas.

## Resultados principales

| Evaluación | Modelo final (E5b: dorado, Dice, profundidad 3) | Otsu (VH) |
|---|---|---|
| Prueba en distribución (90 chips) — IoU agua | **0.663** (F1 0.798) | 0.554 |
| Bolivia (evento no visto) — IoU agua | 0.572 | **0.718** |
| Lima, 17-mar-2017 vs EMSN038 — IoU agua | 0.001 | 0.005 |

- **Ablación (a):** añadir etiquetas débiles no cambia el IoU de prueba (0.640 → 0.641), pero en Bolivia lo eleva de 0.408 a **0.716**, porque corrige el *recall*.
- **Ablación (b):** Dice, BCE + Dice y BCE con peso en bordes superan a la BCE pura; las diferencias entre las tres son menores que el ruido de validación.
- **Ablación (c):** la profundidad 3 (1.93 M parámetros) iguala o supera a la profundidad 4 (7.76 M). Sin conexiones de salto, cae la precisión.
- **Lima:** la retrodispersión VH dentro de los polígonos EMSN038 (−18.9 dB) está lejos de la del agua abierta (−25.4 dB) y no aumenta al retirarse la inundación. La brecha refleja sobre todo el desajuste de la referencia óptica (huellas de lodo cartografiadas días después), no solo un fallo del modelo.

Tablas completas en `outputs/resultados/` y figuras en `outputs/figuras/`.

## Ejecución con un único comando

```bash
pip install -r requirements.txt
earthengine authenticate              # una sola vez (cuenta con acceso a Earth Engine)
python run_all.py --ee-project <ID_DE_SU_PROYECTO_EE>
```

`run_all.py` ejecuta el notebook completo sin interfaz y guarda:

| Carpeta | Contenido |
|---|---|
| `outputs/P2_ejecutado.ipynb` | notebook con todas las salidas |
| `outputs/resultados/` | `tabla_ablacion.csv`, `tabla_evaluacion.csv`, `tabla_brecha_lima.csv`, `experimentos.json`, `normalizacion.json` |
| `outputs/figuras/` | curvas, matriz de confusión, galerías de error y figuras de Lima |
| `outputs/checkpoints/` | pesos de cada corrida (`.pt`) |

Opciones: `--quick` (prueba de humo de ~5 min), `--n-weak N` (chips débiles a usar, por defecto 1500), `--out` y `--data` (carpetas).

**Alternativa en Google Colab:** abrir `notebooks/P2_Inundaciones_UNet_Sentinel1.ipynb`, activar la GPU T4, escribir el ID de Earth Engine en la celda de parámetros y ejecutar todo. Los resultados se guardan en `MyDrive/P2_inundaciones/`.

## Requisitos

- **GPU recomendada.** En una T4, las 7 corridas de la ablación toman alrededor de 70 min en total. En CPU funciona, pero es mucho más lento.
- **Datos:** Sen1Floods11 v1.1 se descarga solo desde el bucket público `gs://sen1floods11`, sin credenciales.
- **Earth Engine:** un proyecto de Google Cloud registrado en Earth Engine (uso no comercial/académico). Se usa para obtener las escenas Sentinel-1 de Lima ya calibradas y en dB (`COPERNICUS/S1_GRD`), además del agua permanente de JRC y los límites de país LSIB.
- **EMSN038:** el notebook descarga solo el paquete oficial `EMSN038_Geodatabase.zip` desde <https://mapping.emergency.copernicus.eu/activations/EMSN038/>. Como etiqueta usa la capa `DamAssess_A` (extensión de inundación OBIA sobre WorldView-2) con `source = "river"`, es decir, solo la inundación observada, y como área de interés la capa `EMSN038_AOI`. Si la descarga automática falla, basta colocar el zip en `outputs/EMSN038/` (en Colab: `MyDrive/P2_inundaciones/EMSN038/`).

## Datos y particiones

| Conjunto | Chips | Uso |
|---|---|---|
| Sen1Floods11 dorado — `train` | 252 | entrenamiento |
| Sen1Floods11 dorado — `valid` | 89 | selección de modelo (nunca se elige con prueba) |
| Sen1Floods11 dorado — `test` | 90 | evaluación en distribución |
| Sen1Floods11 dorado — Bolivia | 15 | región no vista interna al dataset (referencia) |
| Sen1Floods11 débil (Otsu sobre VH), sin Bolivia | 1,500 de 4,160 | ablación (a) |
| Lima, 17-mar-2017 (S1B, órbita 171 desc.) + EMSN038 | escena completa + chips 512×512 | generalización al Perú |

Preprocesamiento igual al de los autores del dataset: dB recortado a [−50, 1] y escalado a [0, 1], seguido de una estandarización por banda con estadísticas **solo** del conjunto de entrenamiento dorado. Los píxeles sin datos (−1 o NaN) se excluyen de la pérdida y de las métricas.

## Estructura

```
├── run_all.py                                  # comando único
├── requirements.txt
├── Informe_P2_Inundaciones_Sentinel1.pdf       # informe (formato NeurIPS)
├── notebooks/
│   └── P2_Inundaciones_UNet_Sentinel1.ipynb    # todo el código, documentado por secciones
├── outputs/                                    # resultados de la corrida reportada (sin pesos)
│   ├── resultados/                             # tablas CSV/JSON
│   └── figuras/
└── informe/                                    # fuente LaTeX del informe
    ├── main.tex
    └── make_tables.py                          # genera las tablas del informe desde outputs/resultados
```

Para regenerar el informe tras una nueva corrida: `python informe/make_tables.py --res outputs/resultados`, copiar `outputs/figuras/*` a `informe/figuras/` y compilar `informe/main.tex` con `pdflatex`.

## Referencias principales

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.* CVPR Workshops.
- Copernicus Emergency Management Service (2017). *EMSN038: Post-disaster situation analyses of flood and landslides in Lima, Peru.*
- Material del curso: Clase 9 — U-Net (notas y notebook), Prof. Aldo Camargo.
