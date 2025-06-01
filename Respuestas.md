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

