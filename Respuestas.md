## Respuestas:

### 1) Diferencias entre `QDA`y `TensorizedQDA`

1. Paraleliza sobre las 𝑘 clases, no sobre las 𝑛 observaciones.


2. `self.inv_covs` es una lista de \( k \) matrices (inversas de las matrices de covarianza), cada una con forma (13, 13).  
   `self.means` es una lista de \( k \) vectores columna, cada uno con forma (13, 1).

    Luego, al aplicar `np.stack` sobre estas listas se obtienen:

    - `tensor_inv_cov.shape = (k, 13, 13)`: se apilan las matrices inversas de covarianza por clase.  
    - `tensor_means.shape = (k, 13, 1)`: se apilan los vectores de medias por clase.

    Para una observación \( x \), el modelo QDA calcula el logaritmo de la probabilidad a posteriori para cada clase \( k \).  
    En `TensorizedQDA`, al tensorizar estos parámetros, el cálculo de las probabilidades a posteriori se ejecuta en paralelo para todas las clases, dado un mismo \( x \). Esto se hace de la siguiente forma:

    Primero, se centra el vector \( x \) restando las medias:

    ```python
    unbiased_x = x - self.tensor_means
    ```
    Donde:
    x tiene forma (13, 1), un vector columna con 13 características. self.tensor_means tiene forma (k, 13, 1), una pila de vectores de medias, uno por clase. El resultado es un tensor de forma (k, 13, 1), que contiene los vectores centrados por clase.

    Luego se calcula el producto interno mediante un producto tensorial:
    ```python
    inner_prod = unbiased_x.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_x
    ```
    Aquí, unbiased_x.transpose(0, 2, 1) reordena los ejes para que tenga forma (k, 1, 13). Este producto matricial da como resultado un tensor de forma (k, 1, 1).

    Finalmente, se calcula la log-verosimilitud condicional para cada clase:
    ```python
    return 0.5 * np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()
    ```
    La función LA.det(self.tensor_inv_cov) calcula el determinante de cada una de las 𝑘
    matrices de forma paralela, devolviendo un vector de forma (k,). A este vector se le aplica el logaritmo componente a componente, y luego se le resta el producto interno, que también se convierte a forma (k,) mediante flatten().

    Por último, para predecir la clase de 𝑥, se suma a estas log-verosimilitudes el logaritmo de las probabilidades a priori de cada clase, y se selecciona la clase con mayor valor:
    ```python
    np.argmax(self.log_a_priori + self._predict_log_conditionals(x))
    ```

### 2) Optimización
