"""
global_aggregation.py — Cómputo del Modelo Global Federado
===========================================================
Este script es ejecutado por el COORDINADOR (servidor federado).
Recibe los pesos entrenados de cada cliente y produce el modelo global
usando tres métodos de agregación:

  1. FedAvg      — Promedio ponderado por cantidad de muestras (McMahan et al., 2017)
  2. FedMedian   — Mediana coordenada a coordenada (Yin et al., 2018)
                   Robusto ante clientes malintencionados (ataques Byzantine).
  3. FedAdam     — Agregación con momentum adaptativo estilo Adam (Reddi et al., 2020)
                   Converge más rápido que FedAvg en datos heterogéneos.

Uso
---
    python global_aggregation.py \\
        --weights_dir . \\
        --n_clients 4 \\
        --method all \\
        --output_dir global_models

    # o un método específico:
    python global_aggregation.py --method fedavg
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from TheModel import build_model

# ─────────────────────────────────────────────────────────────
# 0. Argumentos CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Federated model aggregation")
    parser.add_argument("--weights_dir", type=str, default=".",
                        help="Directorio con los archivos client_X_weights.npz")
    parser.add_argument("--n_clients", type=int, default=4,
                        help="Número de clientes")
    parser.add_argument("--method", type=str, default="all",
                        choices=["fedavg", "fedmedian", "fedadam", "all"],
                        help="Método de agregación a usar")
    parser.add_argument("--output_dir", type=str, default="global_models",
                        help="Carpeta donde guardar los modelos globales")
    # Hiperparámetros de FedAdam
    parser.add_argument("--fedadam_lr", type=float, default=0.01,
                        help="Learning rate del servidor para FedAdam (τ)")
    parser.add_argument("--fedadam_beta1", type=float, default=0.9)
    parser.add_argument("--fedadam_beta2", type=float, default=0.99)
    parser.add_argument("--fedadam_epsilon", type=float, default=1e-3)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 1. Carga de pesos de los clientes
# ─────────────────────────────────────────────────────────────
def load_client_weights(weights_dir: str, n_clients: int):
    """
    Carga los archivos client_X_weights.npz y devuelve:
      - all_weights: lista de listas de arrays (un sublist por cliente)
      - n_samples  : array con el número de muestras de cada cliente
    """
    all_weights = []
    n_samples   = []

    for i in range(n_clients):
        path = os.path.join(weights_dir, f"client_{i}_weights.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró: {path}")

        data = np.load(path, allow_pickle=True)

        # Recuperar capas en orden numérico
        layer_keys = sorted(
            [k for k in data.files if k.startswith("layer_")],
            key=lambda k: int(k.split("_")[1])
        )
        weights = [data[k] for k in layer_keys]
        all_weights.append(weights)
        n_samples.append(int(data["n_samples"][0]))

        print(f"  Cliente {i}: {data['n_samples'][0]} muestras, "
              f"{len(weights)} arrays de pesos")

    return all_weights, np.array(n_samples)


# ─────────────────────────────────────────────────────────────
# 2. FedAvg — Promedio Federado
# ─────────────────────────────────────────────────────────────
def fedavg(all_weights: list, n_samples: np.ndarray) -> list:
    """
    FedAvg (McMahan et al., 2017 — "Communication-Efficient Learning of
    Deep Networks from Decentralized Data").

    Algoritmo:
        w_global = Σ (n_i / N) * w_i
    donde n_i = muestras del cliente i y N = total de muestras.

    Es el estándar de facto del aprendizaje federado. Simple y eficiente,
    pero sensible a clientes con datos atípicos o maliciosos.
    """
    total = n_samples.sum()
    fractions = n_samples / total   # pesos relativos de cada cliente

    global_weights = []
    for layer_idx in range(len(all_weights[0])):
        # Suma ponderada de los arrays de la capa layer_idx
        aggregated = sum(
            fractions[client] * all_weights[client][layer_idx]
            for client in range(len(all_weights))
        )
        global_weights.append(aggregated)

    return global_weights


# ─────────────────────────────────────────────────────────────
# 3. FedMedian — Mediana Coordinada
# ─────────────────────────────────────────────────────────────
def fedmedian(all_weights: list, **kwargs) -> list:
    """
    FedMedian / Coordinate-wise Median (Yin et al., 2018 — "Byzantine-Robust
    Distributed Learning: Towards Optimal Statistical Rates").

    Algoritmo:
        w_global[j] = median({w_i[j] para todo cliente i})
    donde [j] denota la j-ésima coordenada (elemento escalar) del vector de pesos.

    Ventajas sobre FedAvg:
    - Robusto ante ataques Byzantine: incluso si hasta ⌊(n-1)/2⌋ clientes
      envían gradientes maliciosos, la mediana no se ve afectada.
    - Tolera datos heterogéneos (non-IID) mejor que el promedio.

    Desventaja: ligeramente más costoso computacionalmente, pero
    perfectamente viable para modelos pequeños/medianos.
    """
    global_weights = []
    for layer_idx in range(len(all_weights[0])):
        # Apilar: shape (n_clients, *shape_capa)
        stacked = np.stack(
            [all_weights[client][layer_idx] for client in range(len(all_weights))],
            axis=0
        )
        # Mediana a lo largo del eje de clientes (axis=0)
        global_weights.append(np.median(stacked, axis=0))

    return global_weights


# ─────────────────────────────────────────────────────────────
# 4. FedAdam — Agregación Adaptativa con Momentum
# ─────────────────────────────────────────────────────────────
def fedadam(
    all_weights: list,
    n_samples: np.ndarray,
    global_weights_prev: list,
    m_prev: list | None = None,
    v_prev: list | None = None,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.99,
    epsilon: float = 1e-3,
    round_num: int = 1,
) -> tuple[list, list, list]:
    """
    FedAdam (Reddi et al., 2020 — "Adaptive Federated Optimization").

    Algoritmo (vista del servidor):
        Δ = FedAvg(Δ_i) = promedio ponderado de las actualizaciones locales
        m_t = β₁ * m_{t-1} + (1 - β₁) * Δ          # momento de primer orden
        v_t = β₂ * v_{t-1} + (1 - β₂) * Δ²          # momento de segundo orden
        m̂_t = m_t / (1 - β₁ᵗ)                        # corrección de sesgo
        v̂_t = v_t / (1 - β₂ᵗ)                        # corrección de sesgo
        w_global = w_prev + τ * m̂_t / (√v̂_t + ε)   # actualización tipo Adam

    donde τ es el learning rate del servidor.

    Ventajas sobre FedAvg:
    - Converge más rápido, especialmente en datos heterogéneos (non-IID).
    - El momentum acumula información de rondas anteriores.
    - El segundo momento adapta el lr por parámetro, evitando oscilaciones.
    - Compatible con cualquier optimizador local (SGD, Adam, etc.).

    Returns
    -------
    (new_global_weights, m_t, v_t)
        Los pesos globales actualizados y los estados del optimizador para
        pasarlos a la siguiente ronda.
    """
    # Calcular la "actualización" de cada cliente: Δ_i = w_i - w_global_prev
    total = n_samples.sum()
    fractions = n_samples / total

    delta_global = []  # actualización agregada ≈ "pseudo-gradiente del servidor"
    for layer_idx in range(len(all_weights[0])):
        delta_i_list = [
            all_weights[client][layer_idx] - global_weights_prev[layer_idx]
            for client in range(len(all_weights))
        ]
        delta_agg = sum(fractions[c] * delta_i_list[c] for c in range(len(all_weights)))
        delta_global.append(delta_agg)

    # Inicializar momentos si es la primera ronda
    if m_prev is None:
        m_prev = [np.zeros_like(d) for d in delta_global]
    if v_prev is None:
        v_prev = [np.zeros_like(d) for d in delta_global]

    new_weights = []
    m_new = []
    v_new = []

    for layer_idx, (delta, m_l, v_l, w_prev) in enumerate(
        zip(delta_global, m_prev, v_prev, global_weights_prev)
    ):
        # Actualizar momentos
        m_t = beta1 * m_l + (1 - beta1) * delta
        v_t = beta2 * v_l + (1 - beta2) * (delta ** 2)

        # Corrección de sesgo
        m_hat = m_t / (1 - beta1 ** round_num)
        v_hat = v_t / (1 - beta2 ** round_num)

        # Actualizar pesos globales
        w_new = w_prev + lr * m_hat / (np.sqrt(v_hat) + epsilon)

        new_weights.append(w_new)
        m_new.append(m_t)
        v_new.append(v_t)

    return new_weights, m_new, v_new


# ─────────────────────────────────────────────────────────────
# Utilidad: asignación robusta de pesos (Keras 3)
# ─────────────────────────────────────────────────────────────
def _apply_weights(model: tf.keras.Model, weights_list: list) -> None:
    """
    Asigna pesos directamente con w.assign() en lugar de set_weights().
    En Keras 3, set_weights() puede no aplicar los pesos silenciosamente
    cuando el modelo fue construido con la API funcional. w.assign() es
    la forma más robusta de actualizar tensores en TF2/Keras 3.
    """
    model_weights = model.weights   # lista de tf.Variable (trainable + non-trainable)

    if len(model_weights) != len(weights_list):
        raise ValueError(
            f"Incompatibilidad de pesos: el modelo tiene {len(model_weights)} "
            f"tensores pero se recibieron {len(weights_list)}."
        )

    for tf_var, np_array in zip(model_weights, weights_list):
        tf_var.assign(tf.cast(np_array, tf_var.dtype))

    # Verificación rápida: el primer tensor debe haber cambiado
    first_after = model.weights[0].numpy().mean()
    first_expected = float(np.array(weights_list[0]).mean())
    if abs(first_after - first_expected) > 1e-5:
        raise RuntimeError(
            f"Los pesos NO se aplicaron correctamente. "
            f"Esperado: {first_expected:.5f}, Obtenido: {first_after:.5f}"
        )


# ─────────────────────────────────────────────────────────────
# 5. Evaluación del modelo global
# ─────────────────────────────────────────────────────────────
def evaluate_global_model(global_weights: list, method_name: str):
    """
    Carga el conjunto de prueba completo de MNIST y evalúa el modelo global.
    """
    print(f"\n  Evaluando modelo global [{method_name}] en MNIST test...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_model()
    _apply_weights(model, global_weights)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  [{method_name}]  Test Loss: {loss:.4f}  |  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return loss, acc


# ─────────────────────────────────────────────────────────────
# 6. Guardado del modelo global
# ─────────────────────────────────────────────────────────────
def save_global_model(global_weights: list, method_name: str, output_dir: str):
    """Guarda el modelo global como .weights.h5 y los pesos como .npz."""
    os.makedirs(output_dir, exist_ok=True)

    model = build_model()
    _apply_weights(model, global_weights)
    model_path = os.path.join(output_dir, f"global_model_{method_name}.weights.h5")
    model.save_weights(model_path)

    # También guardar pesos crudos como .npz (para la siguiente ronda federada)
    npz_path = os.path.join(output_dir, f"global_weights_{method_name}.npz")
    np.savez_compressed(
        npz_path,
        **{f"layer_{i}": w for i, w in enumerate(global_weights)}
    )

    print(f"  Modelo guardado: {model_path}")
    print(f"  Pesos crudos  : {npz_path}")


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print(f"  Agregación Federada — {args.n_clients} clientes  |  método: {args.method}")
    print(f"{'='*65}\n")

    # Cargar pesos de todos los clientes
    print("Cargando pesos locales...\n")
    all_weights, n_samples = load_client_weights(args.weights_dir, args.n_clients)

    results = {}

    # ── FedAvg ──────────────────────────────────────────────
    if args.method in ("fedavg", "all"):
        print(f"\n{'─'*50}")
        print("  [1/3] FedAvg — Promedio Ponderado")
        print(f"{'─'*50}")
        fedavg_weights = fedavg(all_weights, n_samples)
        loss, acc = evaluate_global_model(fedavg_weights, "FedAvg")
        save_global_model(fedavg_weights, "fedavg", args.output_dir)
        results["FedAvg"] = {"loss": loss, "accuracy": acc}

    # ── FedMedian ───────────────────────────────────────────
    if args.method in ("fedmedian", "all"):
        print(f"\n{'─'*50}")
        print("  [2/3] FedMedian — Mediana Coordinada (Byzantine-Robust)")
        print(f"{'─'*50}")
        fedmedian_weights = fedmedian(all_weights)
        loss, acc = evaluate_global_model(fedmedian_weights, "FedMedian")
        save_global_model(fedmedian_weights, "fedmedian", args.output_dir)
        results["FedMedian"] = {"loss": loss, "accuracy": acc}

    # ── FedAdam ─────────────────────────────────────────────
    if args.method in ("fedadam", "all"):
        print(f"\n{'─'*50}")
        print("  [3/3] FedAdam — Agregación Adaptativa con Momentum")
        print(f"{'─'*50}")

        # Para FedAdam necesitamos los pesos previos del servidor global.
        # En la primera ronda, se usa FedAvg como inicialización.
        if args.method == "all":
            global_prev = fedavg_weights   # ya calculados arriba
        else:
            # Si sólo corremos FedAdam, inicializar con FedAvg
            print("  (Inicializando w_prev con FedAvg para FedAdam...)")
            global_prev = fedavg(all_weights, n_samples)

        fedadam_weights, _, _ = fedadam(
            all_weights,
            n_samples,
            global_weights_prev=global_prev,
            lr=args.fedadam_lr,
            beta1=args.fedadam_beta1,
            beta2=args.fedadam_beta2,
            epsilon=args.fedadam_epsilon,
            round_num=1,
        )
        loss, acc = evaluate_global_model(fedadam_weights, "FedAdam")
        save_global_model(fedadam_weights, "fedadam", args.output_dir)
        results["FedAdam"] = {"loss": loss, "accuracy": acc}

    # ── Resumen comparativo ─────────────────────────────────
    print(f"\n{'='*65}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'='*65}")
    print(f"  {'Método':<15} {'Loss':>10} {'Accuracy':>12}")
    print(f"  {'─'*40}")
    for method, metrics in results.items():
        print(f"  {method:<15} {metrics['loss']:>10.4f} {metrics['accuracy']:>11.2%}")
    print(f"{'='*65}\n")

    best = max(results, key=lambda m: results[m]["accuracy"])
    print(f"  ✅ Mejor método: {best}  "
          f"(Accuracy = {results[best]['accuracy']:.2%})\n")

    # ── Gráficas comparativas ───────────────────────────────
    if len(results) > 1:
        plot_comparison(results, args.output_dir)


# ─────────────────────────────────────────────────────────────
# 8. Visualización comparativa de métodos
# ─────────────────────────────────────────────────────────────
def plot_comparison(results: dict, output_dir: str):
    """
    Genera dos gráficas comparativas de los métodos de agregación:
      1. Barras: Accuracy y Loss por método
      2. Matrices de confusión lado a lado
    """
    os.makedirs(output_dir, exist_ok=True)
    methods   = list(results.keys())
    accs      = [results[m]["accuracy"] * 100 for m in methods]
    losses    = [results[m]["loss"]           for m in methods]
    colors    = ["#4C72B0", "#DD8452", "#55A868"][:len(methods)]

    # ── Fig 1: Accuracy y Loss ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparativa de Métodos de Agregación Federada", fontsize=14, fontweight="bold")

    # Accuracy
    bars = axes[0].bar(methods, accs, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Test Accuracy (%)", fontsize=12)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(max(0, min(accs) - 3), min(100, max(accs) + 3))
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Loss
    bars2 = axes[1].bar(methods, losses, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Test Loss", fontsize=12)
    axes[1].set_ylabel("Sparse Categorical Crossentropy")
    axes[1].set_ylim(0, max(losses) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, losses):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(output_dir, "comparison_metrics.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Gráfica guardada: {path1}")

    # ── Fig 2: Matrices de confusión ─────────────────────────
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    class_names = [str(i) for i in range(10)]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Matrices de Confusión — Modelos Globales", fontsize=14, fontweight="bold")

    for ax, method, color in zip(axes, methods, colors):
        weights_path = os.path.join(output_dir, f"global_weights_{method.lower()}.npz")
        if not os.path.exists(weights_path):
            ax.set_title(f"{method}\n(no disponible)")
            continue

        data = np.load(weights_path)
        layer_keys = sorted([k for k in data.files if k.startswith("layer_")],
                            key=lambda k: int(k.split("_")[1]))
        global_w = [data[k] for k in layer_keys]

        model = build_model()
        _apply_weights(model, global_w)

        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=False, annot_kws={"size": 7})
        acc_val = results[method]["accuracy"] * 100
        ax.set_title(f"{method}  —  {acc_val:.2f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

        # Classification report en consola
        print(f"\n{'─'*55}")
        print(f"  Classification Report — {method}")
        print(f"{'─'*55}")
        print(classification_report(y_test, y_pred, target_names=class_names))

    plt.tight_layout()
    path2 = os.path.join(output_dir, "comparison_confusion_matrices.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Gráfica guardada: {path2}")


if __name__ == "__main__":
    main()