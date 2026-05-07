"""
benchmark.py
============
Compara todas las heurísticas de heuristicas.py en múltiples episodios.

Métricas registradas por episodio:
  - score          : puntaje total acumulado
  - exito          : True si score > 0 (aterrizaje sin crash)
  - combustible    : % restante al final (100 - consumido)
  - duracion_pasos : número de pasos del episodio
  - max_score      : mejor puntaje histórico de ese agente

Uso:
  python benchmark.py [--episodios N] [--semilla S] [--salida resultados.csv]

Ejemplo:
  python benchmark.py --episodios 100 --semilla 42
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sin ventana gráfica (headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from heuristicas import AGENTES, crear_agente


# ─────────────────────────────────────────────────────────────────────────────
# Función: ejecutar un episodio completo con un agente dado
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_episodio(env, agente, semilla=None):
    """
    Corre un episodio completo.
    Retorna dict con métricas del episodio.
    """
    # Resetear estado interno si el agente lo soporta
    if hasattr(agente, "reset"):
        agente.reset()

    obs, _ = env.reset(seed=semilla)
    done   = False
    score  = 0.0
    fuel   = 100.0
    pasos  = 0

    while not done:
        accion, _ = agente.predict(obs)

        # Consumo de combustible (igual que entorno_local.py)
        if accion == 2:
            fuel -= 0.4
        elif accion in [1, 3]:
            fuel -= 0.1
        fuel = max(0.0, fuel)

        obs, reward, terminated, truncated, _ = env.step(accion)
        score += reward
        pasos += 1
        done   = terminated or truncated

    return {
        "score":          score,
        "exito":          score > 0,
        "combustible":    fuel,
        "duracion_pasos": pasos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de benchmark
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_benchmark(n_episodios=100, semilla_base=0, archivo_csv="resultados_benchmark.csv"):
    print(f"\n{'='*65}")
    print(f"  BENCHMARK LUNARLANDER — {n_episodios} episodios por agente")
    print(f"{'='*65}\n")

    env = gym.make("LunarLander-v3")

    todos_resultados = []   # lista de dicts con columna 'agente'
    resumen          = {}   # dict agente → estadísticas agregadas

    for nombre_agente in AGENTES:
        agente = crear_agente(nombre_agente)
        scores     = []
        exitos     = []
        fuels      = []
        pasos_list = []

        print(f"▶ Evaluando: {nombre_agente} ({n_episodios} episodios)...")
        t0 = time.time()

        for ep in range(n_episodios):
            semilla = semilla_base + ep
            res = ejecutar_episodio(env, agente, semilla=semilla)
            scores.append(res["score"])
            exitos.append(res["exito"])
            fuels.append(res["combustible"])
            pasos_list.append(res["duracion_pasos"])

            # Guardar fila individual
            todos_resultados.append({
                "agente":           nombre_agente,
                "episodio":         ep + 1,
                "score":            res["score"],
                "exito":            int(res["exito"]),
                "combustible":      res["combustible"],
                "duracion_pasos":   res["duracion_pasos"],
            })

        elapsed = time.time() - t0
        resumen[nombre_agente] = {
            "score_medio":   np.mean(scores),
            "score_std":     np.std(scores),
            "score_max":     np.max(scores),
            "tasa_exito":    np.mean(exitos) * 100,
            "fuel_medio":    np.mean(fuels),
            "pasos_medios":  np.mean(pasos_list),
            "tiempo_total":  elapsed,
        }

        r = resumen[nombre_agente]
        print(f"   Score:  {r['score_medio']:7.1f} ± {r['score_std']:.1f}  "
              f"| Max: {r['score_max']:.1f}")
        print(f"   Éxito:  {r['tasa_exito']:.0f}%  "
              f"| Combustible: {r['fuel_medio']:.1f}%  "
              f"| Pasos: {r['pasos_medios']:.0f}")
        print()

    env.close()

    # ── Guardar CSV detallado ─────────────────────────────────────────────────
    df = pd.DataFrame(todos_resultados)
    df.to_csv(archivo_csv, index=False)
    print(f"✔ Resultados detallados guardados en: {archivo_csv}\n")

    # ── Guardar CSV resumen ───────────────────────────────────────────────────
    resumen_csv = archivo_csv.replace(".csv", "_resumen.csv")
    df_resumen = pd.DataFrame(resumen).T.reset_index().rename(columns={"index": "agente"})
    df_resumen.to_csv(resumen_csv, index=False)
    print(f"✔ Resumen guardado en: {resumen_csv}\n")

    return df, resumen


# ─────────────────────────────────────────────────────────────────────────────
# Generar gráficos comparativos
# ─────────────────────────────────────────────────────────────────────────────
def generar_graficos(df, resumen, archivo_salida="benchmark_graficos.png"):
    agentes    = list(resumen.keys())
    colores    = plt.cm.Set2(np.linspace(0, 1, len(agentes)))
    n          = len(agentes)
    etiquetas  = [a.replace(" ", "\n") for a in agentes]

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0a0a1a")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

    style_ax = dict(facecolor="#111122")
    style_title = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    style_tick  = {"labelsize": 8}

    # ── 1. Score medio con barras de error ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2], **style_ax)
    medias = [resumen[a]["score_medio"] for a in agentes]
    stds   = [resumen[a]["score_std"]   for a in agentes]
    bars   = ax1.bar(range(n), medias, color=colores, yerr=stds,
                     capsize=5, error_kw=dict(color="#fff", linewidth=1.2),
                     edgecolor="white", linewidth=0.5)
    ax1.axhline(200, color="#ffcc00", linestyle="--", linewidth=1.2,
                label="Umbral 'Solved' (200)")
    ax1.axhline(0,   color="#ff4444", linestyle=":",  linewidth=1.0)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(etiquetas, fontsize=8, color="#aaa")
    ax1.tick_params(axis="y", labelsize=8)
    ax1.set_ylabel("Puntaje", color="#aaa", fontsize=9)
    ax1.set_title("Score Promedio ± Desviación Estándar", **style_title)
    ax1.legend(fontsize=8, facecolor="#222233", labelcolor="white")
    for bar, m in zip(bars, medias):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{m:.0f}", ha="center", va="bottom", color="white", fontsize=8)

    # ── 2. Tasa de éxito ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2], **style_ax)
    tasas = [resumen[a]["tasa_exito"] for a in agentes]
    wedges, texts, autotexts = ax2.pie(
        tasas, labels=None, autopct="%1.0f%%",
        colors=colores, startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="#222233", linewidth=1.5)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(7)
    ax2.set_title("Tasa de Éxito (%)", **style_title)
    legend_patches = [mpatches.Patch(color=colores[i], label=agentes[i])
                      for i in range(n)]
    ax2.legend(handles=legend_patches, fontsize=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.25), facecolor="#222233",
               labelcolor="white", ncol=2)

    # ── 3. Box plot de scores ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2], **style_ax)
    datos_scores = [df[df["agente"] == a]["score"].values for a in agentes]
    bp = ax3.boxplot(datos_scores, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(color="#888"),
                     capprops=dict(color="#888"),
                     flierprops=dict(marker="o", color="#888", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(200, color="#ffcc00", linestyle="--", linewidth=1.2)
    ax3.set_xticks(range(1, n + 1))
    ax3.set_xticklabels(etiquetas, fontsize=8, color="#aaa")
    ax3.tick_params(axis="y", labelsize=8)
    ax3.set_ylabel("Puntaje", color="#aaa", fontsize=9)
    ax3.set_title("Distribución de Scores (Box Plot)", **style_title)

    # ── 4. Combustible promedio restante ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2], **style_ax)
    fuels = [resumen[a]["fuel_medio"] for a in agentes]
    bars4 = ax4.barh(range(n), fuels, color=colores, edgecolor="white",
                     linewidth=0.5)
    ax4.set_yticks(range(n))
    ax4.set_yticklabels(etiquetas, color="#aaa", fontsize=7)
    ax4.tick_params(axis="x", labelsize=8, labelcolor="#aaa")
    ax4.set_xlabel("Combustible (%)", color="#aaa", fontsize=9)
    ax4.set_title("Combustible Restante Promedio", **style_title)
    ax4.axvline(50, color="#ffcc00", linestyle="--", linewidth=1)
    for bar, f in zip(bars4, fuels):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{f:.1f}%", va="center", color="white", fontsize=8)

    # ── 5. Curva de aprendizaje (score acumulado por episodio) ─────────────────
    ax5 = fig.add_subplot(gs[2, :], **style_ax)
    for i, nombre in enumerate(agentes):
        sub  = df[df["agente"] == nombre].sort_values("episodio")
        roll = sub["score"].rolling(10, min_periods=1).mean()
        ax5.plot(sub["episodio"], roll, color=colores[i],
                 linewidth=1.5, label=nombre, alpha=0.85)
    ax5.axhline(200, color="#ffcc00", linestyle="--", linewidth=1.2,
                label="Umbral 'Solved'")
    ax5.axhline(0, color="#ff4444", linestyle=":", linewidth=0.8)
    ax5.tick_params(axis="both", labelsize=8)
    ax5.set_xlabel("Episodio", color="#aaa", fontsize=9)
    ax5.set_ylabel("Score (media móvil 10ep)", color="#aaa", fontsize=9)
    ax5.set_title("Score por Episodio (Media Móvil de 10 episodios)", **style_title)
    ax5.legend(fontsize=8, facecolor="#222233", labelcolor="white",
               loc="lower right", ncol=3)

    # Título global
    fig.suptitle("LUNARLANDER · Comparativa de Heurísticas",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    plt.savefig(archivo_salida, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✔ Gráficos guardados en: {archivo_salida}\n")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Imprimir ranking final
# ─────────────────────────────────────────────────────────────────────────────
def imprimir_ranking(resumen):
    print("\n" + "═" * 80)
    print("  RANKING FINAL (ordenado por score medio)")
    print("═" * 80)
    header = f"{'#':>2}  {'Agente':<28} {'Score μ':>8} {'Std':>7} {'Max':>8} {'Éxito':>7} {'Fuel%':>7} {'Pasos':>7}"
    print(header)
    print("─" * 80)

    ranking = sorted(resumen.items(), key=lambda x: x[1]["score_medio"], reverse=True)
    for i, (nombre, r) in enumerate(ranking, 1):
        medalla = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"{medalla}{i:>2}  {nombre:<28} "
              f"{r['score_medio']:>8.1f} "
              f"{r['score_std']:>7.1f} "
              f"{r['score_max']:>8.1f} "
              f"{r['tasa_exito']:>6.0f}% "
              f"{r['fuel_medio']:>7.1f} "
              f"{r['pasos_medios']:>7.0f}")
    print("═" * 80)

    ganador = ranking[0][0]
    print(f"\n  ✅ Ganador recomendado: {ganador}")
    print(f"     Score promedio:  {ranking[0][1]['score_medio']:.1f}")
    print(f"     Tasa de éxito:   {ranking[0][1]['tasa_exito']:.0f}%")
    print(f"     Combustible:     {ranking[0][1]['fuel_medio']:.1f}%\n")
    return ganador


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de heurísticas LunarLander")
    parser.add_argument("--episodios", type=int, default=100,
                        help="Número de episodios por agente (default: 100)")
    parser.add_argument("--semilla",   type=int, default=0,
                        help="Semilla base para reproducibilidad (default: 0)")
    parser.add_argument("--salida",    type=str, default="resultados_benchmark.csv",
                        help="Nombre del CSV de salida")
    args = parser.parse_args()

    df, resumen = ejecutar_benchmark(
        n_episodios  = args.episodios,
        semilla_base = args.semilla,
        archivo_csv  = args.salida,
    )
    generar_graficos(df, resumen, archivo_salida="benchmark_graficos.png")
    imprimir_ranking(resumen)