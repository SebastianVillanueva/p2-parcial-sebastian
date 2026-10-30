"""Genera las tablas y los números del informe (LaTeX) a partir de los resultados del notebook.

    python informe/make_tables.py --res outputs/resultados

Escribe informe/generado/{numeros,tab_ablacion,tab_eval,tab_lima}.tex
"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

NOMBRES = {  # nombre de corrida -> etiqueta legible
    "E1_dorado_bce": "E1", "E2_dorado+debil_bce": "E2", "E3_dice": "E3", "E4_bce+dice": "E4",
    "E6_bce_borde": "E6", "E5a_sin_saltos": "E5a", "E5b_depth3": "E5b",
}
DATOS = {"gold": "dorado", "gold+weak": "dorado + débil"}
PERD = {"bce": "BCE", "dice": "Dice", "bce+dice": "BCE + Dice", "bce_borde": "BCE con peso en bordes"}


def tx(s):
    return str(s).replace("\u2212", "$-$").replace("_", "\\_").replace("%", "\\%")


def f3(x):
    return "--" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="carpeta resultados/ del notebook")
    a = ap.parse_args()
    out = os.path.join(HERE, "generado"); os.makedirs(out, exist_ok=True)

    E = json.load(open(os.path.join(a.res, "experimentos.json")))
    best = max(E, key=lambda k: E[k]["val"]["IoU"])
    orden = [k for k in NOMBRES if k in E] + [k for k in E if k not in NOMBRES]

    # ---------- tabla de ablación ----------
    best_val = max(E[k]["val"]["IoU"] for k in E)
    filas = []
    for k in orden:
        r, c = E[k], E[k]["cfg"]
        arq = ("sin saltos" if not c.get("skip", True) else "con saltos") + f", prof. {c.get('depth', 4)}"
        v = f3(r["val"]["IoU"]); v = rf"\textbf{{{v}}}" if r["val"]["IoU"] == best_val else v
        filas.append(f"{NOMBRES.get(k, k)} & {DATOS.get(c['data'], c['data'])} & {PERD.get(c['loss'], c['loss'])} & {arq} & "
                     f"{r['params_M']:.2f} & {v} & {f3(r['test']['IoU'])} & {f3(r['test']['F1'])} & "
                     f"{f3((r.get('bolivia') or {}).get('IoU'))} \\\\")
    tab = r"""\begin{table}[t]
\centering\small
\caption{Estudio de ablación. Todas las corridas usan el mismo presupuesto (%d pasos, lote 16, recortes de 256\,px). La selección se hace con el IoU de agua en \emph{validación} (en negrita el mejor); las columnas de prueba y Bolivia se reportan solo como referencia.}
\label{tab:ablacion}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lllcrcccc}
\toprule
 & Datos & Pérdida & Arquitectura & Parám. (M) & IoU val & IoU test & F1 test & IoU Bolivia \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (E[best]["steps"], "\n".join(filas))
    open(os.path.join(out, "tab_ablacion.tex"), "w").write(tab)

    # ---------- tabla de evaluación en distribución ----------
    ev = pd.read_csv(os.path.join(a.res, "tabla_evaluacion.csv"), index_col=0)
    filas = []
    for idx, r in ev.iterrows():
        filas.append(f"{tx(idx)} & {f3(r['IoU'])} & {f3(r['IoU_chip_medio'])} & {f3(r['F1'])} & {f3(r['precision'])} & "
                     f"{f3(r['recall'])} & {f3(r['omision'])} & {f3(r['comision'])} \\\\")
    tab = r"""\begin{table}[t]
\centering\small
\caption{Clase \emph{agua}: modelo final (%s) frente a la línea base clásica (umbral de Otsu sobre VH). \emph{test}: chips dorados de los eventos vistos en entrenamiento; \emph{bolivia}: evento nunca visto. IoU agregado sobre todos los píxeles válidos e IoU promedio por chip (métrica de los autores del dataset).}
\label{tab:eval}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lccccccc}
\toprule
 & IoU & IoU/chip & F1 & Precisión & Recall & Omisión & Comisión \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (NOMBRES.get(best, best), "\n".join(filas))
    open(os.path.join(out, "tab_eval.tex"), "w").write(tab)

    # ---------- tabla de Lima ----------
    lima_csv = os.path.join(a.res, "tabla_brecha_lima.csv")
    if os.path.exists(lima_csv):
        li = pd.read_csv(lima_csv, index_col=0)
        filas = [f"{tx(idx)} & {f3(r['IoU'])} & {f3(r['IoU_chip_medio'])} & {f3(r['F1'])} & {f3(r['precision'])} & {f3(r['recall'])} & "
                 f"{f3(r['omision'])} & {f3(r['comision'])} \\\\" for idx, r in li.iterrows()]
        tab = r"""\begin{table}[t]
\centering\small
\caption{Generalización geográfica: el modelo entrenado solo con Sen1Floods11 evaluado sobre Lima (Sentinel-1 del 17-mar-2017 contra la delineación EMSN038), frente a la prueba en distribución.}
\label{tab:lima}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lccccccc}
\toprule
 & IoU & IoU/chip & F1 & Precisión & Recall & Omisión & Comisión \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(filas)
    else:
        tab = r"\begin{center}\fbox{\parbox{0.9\linewidth}{\centering Tabla de Lima pendiente (ejecutar §10--12 del notebook).}}\end{center}"
    open(os.path.join(out, "tab_lima.tex"), "w").write(tab)

    # ---------- números sueltos para el texto ----------
    def g(k, split, m):
        return f3(((E.get(k) or {}).get(split) or {}).get(m))
    n = {
        "BestRun": NOMBRES.get(best, best).replace("_", r"\_"),
        "BestLoss": PERD.get(E[best]["cfg"]["loss"], E[best]["cfg"]["loss"]),
        "BestData": DATOS.get(E[best]["cfg"]["data"], E[best]["cfg"]["data"]),
        "Steps": str(E[best]["steps"]),
        "IoUvalEone": g("E1_dorado_bce", "val", "IoU"), "IoUvalEtwo": g("E2_dorado+debil_bce", "val", "IoU"),
        "IoUvalBest": f3(E[best]["val"]["IoU"]), "IoUtestBest": f3(E[best]["test"]["IoU"]),
        "FonetestBest": f3(E[best]["test"]["F1"]), "PrectestBest": f3(E[best]["test"]["precision"]),
        "RectestBest": f3(E[best]["test"]["recall"]),
        "IoUbolBest": f3((E[best].get("bolivia") or {}).get("IoU")),
        "IoUvalNoSkip": g("E5a_sin_saltos", "val", "IoU"), "IoUvalDepthThree": g("E5b_depth3", "val", "IoU"),
        "MinutesTotal": f"{sum(r['minutes'] for r in E.values()):.0f}",
    }
    if "Otsu VH (test)" in ev.index: n["IoUotsuTest"] = f3(ev.loc["Otsu VH (test)", "IoU"])
    if os.path.exists(lima_csv):
        li = pd.read_csv(lima_csv, index_col=0); L = li.iloc[1]
        n.update(IoULima=f3(L["IoU"]), PrecLima=f3(L["precision"]), RecLima=f3(L["recall"]), FoneLima=f3(L["F1"]))
    extra = os.path.join(a.res, "lima_extra.json")
    if os.path.exists(extra):
        x = json.load(open(extra))
        for k, v in x.items():
            key = "".join(ch for ch in k.title() if ch.isalpha())
            n[f"Lx{key}"] = f"{v:.1f}" if isinstance(v, float) else str(v)
    with open(os.path.join(out, "numeros.tex"), "w") as f:
        for k, v in n.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("Mejor corrida:", best, "| archivos en", out)


if __name__ == "__main__":
    main()
