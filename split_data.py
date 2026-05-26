"""
split_data.py — División MNIST en N partes estadísticamente equivalentes
=========================================================================
⚠️  ARCHIVO CONFIDENCIAL — NO subir al repositorio del equipo.

Este script descarga MNIST y lo divide en n = 4 subconjuntos (uno por
integrante del equipo) de forma IID (Independent and Identically
Distributed), garantizando que cada cliente tenga:
  - Aproximadamente el mismo número de muestras.
  - Una distribución de clases estadísticamente equivalente.

Salida
------
Genera los archivos (en la carpeta ./client_data/):
    client_0_train.npz, client_0_test.npz
    client_1_train.npz, client_1_test.npz
    client_2_train.npz, client_2_test.npz
    client_3_train.npz, client_3_test.npz

Cada .npz contiene las llaves 'x' e 'y'.

Uso
---
    python split_data.py --n_clients 4 --seed 42
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from collections import Counter


# ─────────────────────────────────────────────
# Argumentos CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Split MNIST into N IID client datasets")
    parser.add_argument("--n_clients", type=int, default=4, help="Número de clientes (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    parser.add_argument("--output_dir", type=str, default="client_data", help="Carpeta de salida")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Carga de MNIST
# ─────────────────────────────────────────────
def load_mnist():
    """Descarga MNIST con TensorFlow y normaliza a [0, 1] con shape (N, 28, 28, 1)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    # Agregar canal: (N, 28, 28) → (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


# ─────────────────────────────────────────────
# División IID estratificada
# ─────────────────────────────────────────────
def iid_split(x: np.ndarray, y: np.ndarray, n_clients: int, seed: int):
    """
    División IID estratificada por clase:
      1. Por cada clase, mezcla aleatoriamente sus índices.
      2. Divide los índices en n_clients partes iguales.
      3. Cada cliente recibe la misma proporción de cada clase.

    Esto garantiza distribuciones estadísticamente equivalentes.

    Returns
    -------
    list of (x_i, y_i) tuples, uno por cliente.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(n_clients)]

    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        # Dividir en n_clients trozos (algunos pueden tener 1 muestra más)
        splits = np.array_split(cls_indices, n_clients)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # Mezclar índices dentro de cada cliente
    clients = []
    for idx_list in client_indices:
        idx = np.array(idx_list)
        rng.shuffle(idx)
        clients.append((x[idx], y[idx]))

    return clients


# ─────────────────────────────────────────────
# Verificación estadística
# ─────────────────────────────────────────────
def verify_distribution(clients, split_name="Train"):
    """Imprime la distribución de clases de cada cliente para verificación."""
    print(f"\n{'─'*60}")
    print(f"  Verificación de distribución [{split_name}]")
    print(f"{'─'*60}")
    for i, (x_c, y_c) in enumerate(clients):
        counts = Counter(y_c.tolist())
        sorted_counts = [counts.get(cls, 0) for cls in range(10)]
        print(f"  Cliente {i}  |  n={len(y_c):5d}  |  "
              f"dist={sorted_counts}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────
# Guardado
# ─────────────────────────────────────────────
def save_clients(clients, output_dir, split_name):
    """Guarda los datos de cada cliente como archivo .npz."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (x_c, y_c) in enumerate(clients):
        path = os.path.join(output_dir, f"client_{i}_{split_name}.npz")
        np.savez_compressed(path, x=x_c, y=y_c)
        print(f"  Guardado: {path}  (n={len(y_c)})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  MNIST Split — {args.n_clients} clientes  |  seed={args.seed}")
    print(f"{'='*60}\n")

    print("Cargando MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")

    print("\nDividiendo train set...")
    train_clients = iid_split(x_train, y_train, args.n_clients, seed=args.seed)
    verify_distribution(train_clients, "Train")

    print("Dividiendo test set...")
    test_clients = iid_split(x_test, y_test, args.n_clients, seed=args.seed + 1)
    verify_distribution(test_clients, "Test")

    print(f"Guardando en '{args.output_dir}/'...")
    save_clients(train_clients, args.output_dir, "train")
    save_clients(test_clients, args.output_dir, "test")

    print("\n✅ División completada exitosamente.")
    print(f"   Comparte client_i_train.npz y client_i_test.npz sólo")
    print(f"   con el integrante i correspondiente.")
    print(f"   NO subas esta carpeta ni este script al repositorio.\n")


if __name__ == "__main__":
    main()
