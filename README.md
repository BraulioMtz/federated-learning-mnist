# Aprendizaje Federado con MNIST — Mini-ResNet

Implementación de un flujo completo de **Federated Learning** sobre el dataset MNIST,
usando TensorFlow y tres métodos de agregación global.

---

## Estructura del repositorio

```
.
├── TheModel.py             # Arquitectura del modelo global (Mini-ResNet)
├── local_training.ipynb    # Entrenamiento local del cliente (cada integrante lo ejecuta)
├── global_aggregation.py   # Agregación del modelo global (3 métodos)
└── README.md
```

> ⚠️ **`split_data.py`** y la carpeta **`client_data/`** son confidenciales y **no se incluyen** en este repositorio.

---

## Modelo: Mini-ResNet

La arquitectura elegida es un **Mini-ResNet** con bloques residuales, distinta a la vista en clase:

```
Input (28×28×1)
  → Conv2D(32) + BN + ReLU         [Stem]
  → ResBlock(32)                    [28×28]
  → ResBlock(64, stride=2)          [14×14]
  → ResBlock(64)                    [14×14]
  → ResBlock(128, stride=2)         [7×7]
  → GlobalAveragePooling
  → Dropout(0.4)
  → Dense(128) + BN + ReLU
  → Dense(10, softmax)
```

Ventajas: los bloques residuales permiten gradientes más estables, lo que es especialmente
valioso en el contexto federado, donde los clientes entrenan por separado con datos parciales.

---

## Flujo de trabajo federado

```
Coordinador                         Clientes (×4)
──────────────────                  ────────────────────────────
                   ─ global weights →
                                    [Ronda local]
                                    local_training.ipynb
                                    (entrenar, evaluar, guardar pesos)
                   ← client_X_weights.npz ─
[Agregar con
 global_aggregation.py]
─ nuevo global model →  (siguiente ronda)
```

---

## Métodos de Agregación Global

### 1. FedAvg — Promedio Ponderado Federado
**Referencia:** McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*

Calcula el promedio de los pesos de cada cliente, ponderado por su cantidad de muestras de entrenamiento:

```
w_global = Σ (n_i / N) * w_i
```

Es el método estándar. Simple y eficiente, pero sensible a clientes con datos atípicos o a ataques adversariales.

---

### 2. FedMedian — Mediana Coordinada (Byzantine-Robust)
**Referencia:** Yin et al. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.*

En lugar del promedio, usa la **mediana coordenada a coordenada** sobre los pesos de todos los clientes:

```
w_global[j] = median({w_i[j]  para todo i})
```

**Mejora sobre FedAvg:** Es robusto ante ataques Byzantine. Si hasta la mitad de los clientes envían pesos maliciosos o corruptos, la mediana no se ve afectada, a diferencia del promedio que se desvía significativamente. También maneja mejor la heterogeneidad de datos (non-IID).

---

### 3. FedAdam — Agregación Adaptativa con Momentum
**Referencia:** Reddi et al. (2020). *Adaptive Federated Optimization.* ICLR 2021.

Aplica el optimizador **Adam en el servidor**, usando las actualizaciones de los clientes como "pseudo-gradientes":

```
Δ       = FedAvg de las actualizaciones locales Δ_i = w_i - w_prev
m_t     = β₁ * m_{t-1} + (1 - β₁) * Δ          ← momento 1er orden
v_t     = β₂ * v_{t-1} + (1 - β₂) * Δ²          ← momento 2do orden
w_global = w_prev + τ * m̂_t / (√v̂_t + ε)
```

**Mejora sobre FedAvg:** El momentum acumula información de rondas anteriores, acelerando la convergencia. El segundo momento adapta el learning rate por parámetro, evitando oscilaciones. Especialmente útil cuando los datos entre clientes son heterogéneos (non-IID), escenario donde FedAvg converge lentamente.

---

## Instalación

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

## Uso

### Paso 0 – División de datos *(sólo el coordinador, fuera del repo)*
```bash
python split_data.py --n_clients 4 --seed 42
# Distribuir client_i_train.npz y client_i_test.npz a cada integrante
```

### Paso 1 – Entrenamiento local *(cada integrante)*
Abrir `local_training.ipynb`, configurar `CLIENT_ID` y ejecutar todas las celdas.
Al finalizar, enviar `client_X_weights.npz` al coordinador.

### Paso 2 – Agregación global *(coordinador)*
```bash
# Todos los métodos
python global_aggregation.py --weights_dir . --n_clients 4 --method all

# Sólo FedAvg
python global_aggregation.py --method fedavg

# FedAdam con hiperparámetros personalizados
python global_aggregation.py --method fedadam --fedadam_lr 0.005
```

Los modelos se guardan en `global_models/`.

---

## Equipo
Proyecto para la materia Cloud Computing — ITESM
