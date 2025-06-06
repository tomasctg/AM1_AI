# Consigna QDA

**Notación**: en general notamos

* $k$ la cantidad de clases
* $n$ la cantidad de observaciones
* $p$ la cantidad de features/variables/predictores

**Sugerencia:** combinaciones adecuadas de `transpose`, `stack`, `reshape` y, ocasionalmente, `flatten` y `diagonal` suele ser más que suficiente. Se recomienda *fuertemente* explorar la dimensionalidad de cada elemento antes de implementar las clases.

## Tensorización

En esta sección nos vamos a ocupar de hacer que el modelo sea más rápido para generar predicciones, observando que incurre en un doble `for` dado que predice en forma individual un escalar para cada observación, para cada clase. Paralelizar ambos vía tensorización suena como una gran vía de mejora de tiempos.

### 1) Diferencias entre `QDA`y `TensorizedQDA`

1. ¿Sobre qué paraleliza `TensorizedQDA`? ¿Sobre las $k$ clases, las $n$ observaciones a predecir, o ambas?
Sobre las K clases solo paraleliza.
2. Analizar los shapes de `tensor_inv_covs` y `tensor_means` y explicar paso a paso cómo es que `TensorizedQDA` llega a predecir lo mismo que `QDA`.

### 2) Optimización

Debido a la forma cuadrática de QDA, no se puede predecir para $n$ observaciones en una sola pasada (utilizar $X \in \mathbb{R}^{p \times n}$ en vez de $x \in \mathbb{R}^p$) sin pasar por una matriz de $n \times n$ en donde se computan todas las interacciones entre observaciones. Se puede acceder al resultado recuperando sólo la diagonal de dicha matriz, pero resulta ineficiente en tiempo y (especialmente) en memoria. Aún así, es *posible* que el modelo funcione más rápido.

3. Implementar el modelo `FasterQDA` (se recomienda heredarlo de `TensorizedQDA`) de manera de eliminar el ciclo for en el método predict.
4. Mostrar dónde aparece la mencionada matriz de $n \times n$, donde $n$ es la cantidad de observaciones a predecir.
5. Demostrar que
$$
diag(A \cdot B) = \sum_{cols} A \odot B^T = np.sum(A \odot B^T, axis=1)
$$ es decir, que se puede "esquivar" la matriz de $n \times n$ usando matrices de $n \times p$. También se puede usar, de forma equivalente,
$$
np.sum(A^T \odot B, axis=0).T
$$
queda a preferencia del alumno cuál usar.
6. Utilizar la propiedad antes demostrada para reimplementar la predicción del modelo `FasterQDA` de forma eficiente en un nuevo modelo `EfficientQDA`.
7. Comparar la performance de las 4 variantes de QDA implementadas hasta ahora (no Cholesky) ¿Qué se observa? A modo de opinión ¿Se condice con lo esperado?

## Cholesky

Hasta ahora todos los esfuerzos fueron enfocados en realizar una predicción más rápida. Los tiempos de entrenamiento (teóricos al menos) siguen siendo los mismos o hasta (minúsculamente) peores, dado que todas las mejoras siguen llamando al método `_fit_params` original de `QDA`.

La descomposición/factorización de [Cholesky](https://en.wikipedia.org/wiki/Cholesky_decomposition#Statement) permite factorizar una matriz definida positiva $A = LL^T$ donde $L$ es una matriz triangular inferior. En particular, si bien se asume que $p \ll n$, invertir la matriz de covarianzas $\Sigma$ para cada clase impone un cuello de botella que podría alivianarse. Teniendo en cuenta que las matrices de covarianza son simétricas y salvo degeneración, definidas positivas, Cholesky como mínimo debería permitir invertir la matriz más rápido.

*Nota: observar que calcular* $A^{-1}b$ *equivale a resolver el sistema* $Ax=b$.

### 3) Diferencias entre implementaciones de `QDA_Chol`

8. Si una matriz $A$ tiene fact. de Cholesky $A=LL^T$, expresar $A^{-1}$ en términos de $L$. ¿Cómo podría esto ser útil en la forma cuadrática de QDA?
7. Explicar las diferencias entre `QDA_Chol1`y `QDA` y cómo `QDA_Chol1` llega, paso a paso, hasta las predicciones.
8. ¿Cuáles son las diferencias entre `QDA_Chol1`, `QDA_Chol2` y `QDA_Chol3`?
9. Comparar la performance de las 7 variantes de QDA implementadas hasta ahora ¿Qué se observa?¿Hay alguna de las implementaciones de `QDA_Chol` que sea claramente mejor que las demás?¿Alguna que sea peor?

### 4) Optimización

12. Implementar el modelo `TensorizedChol` paralelizando sobre clases/observaciones según corresponda. Se recomienda heredarlo de alguna de las implementaciones de `QDA_Chol`, aunque la elección de cuál de ellas queda a cargo del alumno según lo observado en los benchmarks de puntos anteriores.
13. Implementar el modelo `EfficientChol` combinando los insights de `EfficientQDA` y `TensorizedChol`. Si se desea, se puede implementar `FasterChol` como ayuda, pero no se contempla para el punto.
13. Comparar la performance de las 9 variantes de QDA implementadas ¿Qué se observa? A modo de opinión ¿Se condice con lo esperado?



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

3. Ver en archivo QDA.py nueva clase FasterQDA que hereda de TensorizedQDA. 
4. En este caso tendremos a $X \in \mathbb{R}^{p \times n}$ con 𝑛 observaciones columna, cada una con 𝑝 características.  $\mu_k \in \mathbb{R}^{k \times p \times 1}$ y $\Sigma_k^{-1} \in \mathbb{R}^{k \times p \times p}$, seguiran siendo calculados como en los puntos anteriores. Las medias centradas por clase ahora se tendra que calculara para n observaciones como: $X_{\text{centered}} = X - \mu_k \in \mathbb{R}^{k \times p \times n}$. 
De esta forma el producto: 
$\ (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k)  \in \mathbb{R}^{k \times n \times n}$ contiene la mencionada matriz de $n \times n$.

5. Sea $A \in \mathbb{R}^{n \times m} $, $B \in \mathbb{R}^{m \times n} $. La componente i-th de la diagonal de $AB$ se puede escribir como $[AB]_{ii} = \sum_{k=1}^m A_{ik} B_{ki}$. Entonces el vector de la diagonal de $AB$ se define como: 
    $$\operatorname{diag}(AB) = \left[
    \sum_{k=1}^m A_{1k} B_{k1},\ 
    \sum_{k=1}^m A_{2k} B_{k2},\ 
    \dots,\ 
    \sum_{k=1}^m A_{nk} B_{kn}
    \right]$$
    Por otro lado como $B^T\in\mathbb{R}^{n \times m}$ entonces $(A \odot B^T)_{ij} = A_{ij} B_{ji}$. Luego sumando las columnas por fila se tiene 
    $$
    \sum_{\text{cols}} A \odot B^{T}
    = \left[
    \sum_{k=1}^m A_{1k} B_{k1},\ 
    \sum_{k=1}^m A_{2k} B_{k2},\ 
    \dots,\ 
    \sum_{k=1}^m A_{nk} B_{kn}
    \right]$$

6. Ver clase EfficientQDA en QDA.py especificamente en 
    ```python 
    diag_inner_prod = np.sum(inv_cov_x * unbiased_x, axis=1)  # (k, n)
    ```
7. Tomando como baseline el modelo QDA basico, vemos los siguientes resultados.

    | Model         | Test Median (ms) | Mean Accuracy  | Test Speedup  | Test Mem Reduction  |
    |---------------|------------------|----------------|---------------|---------------------|
    | QDA           | 3.487574         | 0.982407       | 1.000000      | 1.000000            |
    | TensorizedQDA | 1.374187         | 0.982593       | 2.537918      | 0.638765            |
    | LDA           | 1.432470         | 0.716296       | 2.434659      | 1.742188            |
    | FasterQDA     | 0.155948         | 0.983333       | 22.363701     | 0.069789            |
    | EfficientQDA  | 0.110844         | 0.986111       | 31.463808     | 0.099771            |

### 3) Cholesky
8. Siendo $A=LL^T$, la inversa de $A$:
    $$A^{-1} = L^{-T}L^{-1}$$
    Retomando la ecuacion:
    $$
    \log{f_j(x)} = -\frac{1}{2}\log |\Sigma_j| - \frac{1}{2} (x-\mu_j)^T \Sigma_j^{-1} (x- \mu_j) + C
    $$
    Si $$\Sigma_j = L_{j}L_{j}^{T}$$
    Luego $$\Sigma_j^{-1} = L_{j}^{-T}L_{j}^{-1}$$ 
    Definiendo a $$\delta(x) = (x-\mu_j)^T \Sigma_j^{-1} (x- \mu_j)$$ 
    Replazando se tiene
    $$\delta(x) = (x-\mu_j)^T L_{j}^{-T}L_{j}^{-1} (x- \mu_j)$$ 
    Como $$(L_{j}^{-1}(x-\mu_j))^{T} = (x-\mu_j)^{T}L_{j}^{-T}$$ 
    Si definimos $L_{j}^{-1} (x- \mu_j) = z$ entonces 
    $$\delta(x) = z^{T}z = ||z||^{2} = ||L_{j}^{-1}(x- \mu_j)||^{2}$$
    Invertir $L_j$ la cual es una matriz triangular, es rapido que invert $\Sigma_j$ completa.

    Por otro remplanzado la factorizacion en el termino $-\frac{1}{2}\log |\Sigma_j|$ tenemos:
    $$-\frac{1}{2}\log |\Sigma_j| = -\frac{1}{2}\log |L_{j}L_{j}^{T}|$$ 
    Donde $$|L_{j}L_{j}^{T}|=|L_{j}||L_{j}^{T}|$$ Como el determinante de la transpuesta de una matriz traingular es igual al determinante de la matriz original entonces
    $$|L_{j}||L_{j}^{T}| = |L_{j}|^2$$
    Finalmente como $L_{j}$ es traingular su determinante se calcula como el producto de los elementos en su diagonal. 

9. La diferencia mas notable entre `QDA_Chol1`y `QDA` se encuentra en la etapa de `training`. En `QDA` se calcula la matriz de covarianza para cada clase $\Sigma_j$ mientras que en `QDA_Chol1` aplicando la factorizacion de Cholesky y considerando el resultado del punto 8, se logra un algoritmo en donde se computa la inversa de una matriz traingular inferior, siendo este computo mas rapido. Esto se muestra en el siguiente bloque de codigo.
```python
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]
``` 
La funcion `cholesky` obtiene en este caso con el flag `lower=True` la matriz inferior resultante de la factorizacion. Luego para cada clase se calcula la inversa. 
El resultado del punto 8 tambien simplifica la etapa de prediccion usando $\delta(x) = z^{T}z = ||z||^{2} = ||L_{j}^{-1}(x- \mu_j)||^{2}$ entonces el codigo se simplifica como:
```python
    y = L_inv @ unbiased_x
    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()
```

10. La principal diferencia entre `QDA_Chol1`, `QDA_Chol2` y `QDA_Chol3`, yase en la forma que $y$ se calcula y por lo tanto como se trata a $L_{j}$. En `QDA_Chol1` y `QDA_Chol3` en la etapa de `training` se calcula la inversa de la matrix triangular inferior $L_{j}$ en el primer caso de la forma clasica 
```python
LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
```
y en el segundo caso se hace uso del algoritmo `DTRTRI` el cual computa la inversa de una matriz triangular superior o inferior 
```python
dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
```
En el caso de `QDA_Chol2` simplemente se calcula la matrix trinagular inferior $L_{j}$. Lo cual implica diferencia con las otras dos en que al momento de predecir es necesario calcular $y$ y por lo tanto obtener la inversa de $L_{j}$. Pero que que calcular $L_{j}^{-1}(x- \mu_j)$ equivale a resolver el sistema $L_{j}y=(x- \mu_j)$, entonces es posible usar `solve_traingular` un metodo que resulve la equacion ``a x = b`` para `x`, asumiendo que a es una matriz tringular. 

```python
y = solve_triangular(L, unbiased_x, lower=True)
```
11. 
    | model          | test_median_ms | mean_accuracy | test_speedup | test_mem_reduction |
    |----------------|----------------|---------------|--------------|--------------------|
    | QDA            | 5.278476       | 0.982407      | 1.000000     | 1.000000           |
    | TensorizedQDA  | 1.844894       | 0.982593      | 2.861128     | 0.635025           |
    | FasterQDA      | 0.119947       | 0.985741      | 44.006553    | 0.069381           |
    | EfficientQDA   | 0.147977       | 0.983333      | 35.670922    | 0.099187           |
    | QDA_Chol1      | 2.431347       | 0.986111      | 2.171009     | 0.985795           |
    | QDA_Chol2      | 5.343377       | 0.982222      | 0.987854     | 1.000627           |
    | QDA_Chol3      | 2.216614       | 0.984444      | 2.381324     | 1.000627           |

12. Basado en el benchmark anterior se implementara las optimizaciones y tensorizaciones en base al modelo `QDA_Chol3` que presenta la mejor performance de las 3. 