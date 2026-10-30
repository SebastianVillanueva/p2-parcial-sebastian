"""Ejecuta de punta a punta la Pregunta 2 (U-Net + Sentinel-1) con un único comando.

    python run_all.py --ee-project MI_PROYECTO_EE

Corre el notebook `notebooks/P2_Inundaciones_UNet_Sentinel1.ipynb` sin interfaz y guarda
una copia ejecutada (con todas las salidas) en `outputs/P2_ejecutado.ipynb`.
Tablas -> outputs/resultados/ · figuras -> outputs/figuras/ · pesos -> outputs/checkpoints/
"""
import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee-project", default=os.environ.get("P2_EE_PROJECT", ""),
                    help="ID del proyecto de Google Earth Engine (necesario para la región de Lima)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"), help="carpeta de resultados")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="carpeta de datos (Sen1Floods11)")
    ap.add_argument("--n-weak", type=int, default=1500, help="nº de chips con etiqueta débil (de 4,385)")
    ap.add_argument("--quick", action="store_true", help="prueba rápida: pocos pasos de entrenamiento")
    args = ap.parse_args()

    if not args.ee_project:
        sys.exit("Falta --ee-project (ID del proyecto de Earth Engine). Ver README, sección 'Requisitos'.")
    os.makedirs(args.out, exist_ok=True)
    os.environ.update(P2_EE_PROJECT=args.ee_project, P2_OUT=args.out, P2_DATA=args.data,
                      P2_N_WEAK=str(args.n_weak), P2_QUICK="1" if args.quick else "0", MPLBACKEND="Agg")

    src = os.path.join(ROOT, "notebooks", "P2_Inundaciones_UNet_Sentinel1.ipynb")
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": ROOT}})
    t0 = time.time()
    try:
        client.execute()
    finally:
        out_nb = os.path.join(args.out, "P2_ejecutado.ipynb")
        nbformat.write(nb, out_nb)
        print(f"Notebook ejecutado guardado en {out_nb} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
