# Guía de parámetros — Preprocesamiento (Bloque A)

Documentación para los tesistas. Explica **qué hace cada parámetro** del
preprocesador (`core/preprocessor.py`, pestaña **A · Pre** de la GUI) y
**cómo elegir su valor** según el tipo de nube de puntos.

---

## El concepto clave: la escala característica `d̄`

Antes de cualquier limpieza, el programa calcula **`d̄` (d-barra)**: la
**mediana de la distancia de cada punto a su vecino más cercano**, sobre una
muestra de la nube.

Intuitivamente, `d̄` es *"cuán juntos están los puntos en promedio"*: la
resolución real de la nube.

**¿Por qué importa?** Porque casi todos los umbrales espaciales del
preproceso se expresan como **múltiplos de `d̄`**, no en números absolutos.
Así, el mismo parámetro (por ejemplo "factor = 2.5") significa lo correcto
tanto si la nube está en milímetros, en metros o en unidades arbitrarias.

> **No hay que configurar `d̄`**: se calcula solo. Pero conviene mirar su
> valor en el log (`ℹ️ Escala característica d̄ = ...`) para entender qué
> tan densa es la nube.

El orden del pipeline es:

```
validar → calcular d̄ → dedup → ROR → SOR → voxel → suavizado MLS → normales
```

---

## 0. Validación

Limpieza de puntos inválidos. Es lo primero que ocurre.

| Parámetro | Qué hace | Cuándo activarlo |
|---|---|---|
| **Quitar NaN / Inf** | Elimina puntos con coordenadas no numéricas o infinitas. | **Siempre ON.** Los escáneres láser generan estos valores por mediciones fallidas. No cuesta nada. |
| **Quitar puntos en el origen (0,0,0)** | Elimina puntos exactamente en el origen. | **ON para datos de láser** (los *returns* fallidos suelen registrarse en (0,0,0)). Considerar OFF solo si tu objeto legítimamente tiene puntos en el origen del sistema de coordenadas. |

---

## 1. Deduplicación

Elimina puntos repetidos o casi coincidentes usando una rejilla: conserva
un punto por celda de lado `dedup_factor · d̄`.

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Activar** | — | ON por defecto. Útil cuando hay solapamiento de escaneos (puntos duplicados en zonas registradas desde varias posiciones). |
| **Factor** (× d̄) | `0.1 – 0.3` | Es la fracción de `d̄` bajo la cual dos puntos se consideran "el mismo". <br>• **Bajo (0.1)**: solo fusiona duplicados casi exactos (conservador). <br>• **Alto (0.3+)**: empieza a hacer un downsampling ligero. <br>Por defecto **0.2** es un buen punto medio. |

> ⚠️ No subir demasiado el factor: si se acerca al voxel posterior, estás
> haciendo el mismo trabajo dos veces. La deduplicación es para *duplicados*,
> no para reducir densidad (eso es la voxelización).

---

## 2. Radius Outlier Removal (ROR)

Elimina **puntos aislados** ("fliers"): un punto se borra si tiene **menos
de N vecinos** dentro de un radio dado. Va **antes** que el SOR a propósito,
para que los puntos sueltos no contaminen la estadística del SOR.

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Activar ROR** | — | ON para láser (partículas en el aire, *mixed pixels* en bordes generan fliers). |
| **Factor radio** (× d̄) | `2.0 – 5.0` | Radio de búsqueda = `factor · d̄`. <br>• **Bajo (2.0)**: agresivo, exige que los vecinos estén muy cerca → borra más. <br>• **Alto (5.0)**: permisivo → borra solo puntos muy aislados. |
| **Mín. vecinos** | `3 – 8` | Cuántos vecinos debe tener un punto dentro del radio para sobrevivir. <br>• **Bajo (3)**: conservador, borra poco. <br>• **Alto (8+)**: agresivo, puede comerse bordes legítimos de zonas poco densas. |

> **Regla práctica**: si ves "agujeros" o pérdida de detalle en zonas finas,
> *baja* el mínimo de vecinos o *sube* el factor de radio (menos agresivo).

---

## 3. Statistical Outlier Removal (SOR)

Elimina **ruido de superficie**: para cada punto mide la distancia media a
sus `k` vecinos; si esa distancia se aleja más de `σ` desviaciones estándar
del promedio global, lo borra. Bueno para ruido difuso (no para fliers
aislados — de eso se encarga el ROR).

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Activar SOR** | — | ON por defecto. |
| **k vecinos** | `10 – 30` | Cuántos vecinos se usan para evaluar cada punto. <br>• **Bajo (10)**: sensible al detalle local, más ruidoso en la decisión. <br>• **Alto (30)**: estadística más estable, pero más lento. <br>Para nubes densas de láser, `20` funciona bien. |
| **σ ratio** | `1.0 – 3.0` | Umbral de tolerancia (en desviaciones estándar). **Es el control principal de agresividad.** <br>• **Bajo (1.0)**: muy agresivo, borra bastante (riesgo de comerse detalle real). <br>• **Alto (3.0)**: permisivo, solo borra ruido evidente. <br>Por defecto **2.0**. Si la nube es muy ruidosa, baja a 1.5; si pierdes detalle, sube a 2.5–3.0. |

---

## 4. Voxelización

Reduce y **uniformiza la densidad**: divide el espacio en cubos de lado
`voxel_factor · d̄` y reemplaza todos los puntos de cada cubo por su
centroide. Va **al final de la limpieza** (no antes) para no mezclar
outliers dentro de la superficie.

**Por qué es importante para láser**: los escaneos tienen densidad muy
desigual (cerca del escáner = denso, lejos = ralo). Uniformar la densidad
**mejora muchísimo la reconstrucción posterior**, sobre todo Ball Pivoting,
y estabiliza Poisson.

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Activar** | — | ON recomendado para láser. OFF solo si tu nube ya es uniforme y quieres máxima densidad. |
| **Factor** (× d̄) | `1.0 – 2.5` | Lado del voxel = `factor · d̄`. **Controla el balance detalle ↔ peso.** <br>• **Bajo (1.0)**: conserva casi toda la resolución (más puntos, más lento, más detalle). <br>• **Alto (2.5)**: reduce mucho los puntos (más rápido y limpio, menos detalle). <br>Por defecto **1.5**. |

> **Cómo decidir**: mira cuántos puntos quedan en el log tras la voxelización.
> Si la malla final sale demasiado pesada o lenta, sube el factor. Si pierdes
> detalle fino, bájalo.

---

## 5. Suavizado MLS (no destructivo)

Reduce el ruido **reposicionando** cada punto sobre la superficie local
(no borra puntos). Para cada punto ajusta un plano con sus `k` vecinos
(PCA) y proyecta el punto sobre ese plano. Reemplaza al antiguo "filtro de
rugosidad" que *eliminaba* aristas.

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Activar** | — | ON para reducir ruido manteniendo la geometría. OFF si la nube ya es muy limpia y quieres cero alteración de las posiciones originales. |
| **Iteraciones** | `1 – 3` | Cuántas veces se repite el suavizado. <br>• **1**: suavizado leve (recomendado por defecto). <br>• **2–3**: más suave, pero cada iteración aleja un poco más los puntos de su posición original → riesgo de "redondear" detalle. |
| **Vecinos (k)** | `10 – 24` | Tamaño del vecindario para ajustar el plano local. <br>• **Bajo (10)**: respeta más el detalle local, suaviza menos. <br>• **Alto (24)**: superficie más suave, puede aplanar features pequeños. |
| **Preservar aristas** | — | **ON recomendado.** Detecta zonas de alta curvatura (aristas, esquinas) y **reduce el suavizado ahí**, evitando redondearlas. Dejarlo OFF solo si quieres un suavizado uniforme y agresivo en formas puramente orgánicas. |

> ⚠️ El suavizado es opcional y *modifica posiciones*. Si tu tesis necesita
> preservar las coordenadas medidas tal cual (p. ej. para análisis métrico),
> considera dejarlo OFF o en 1 iteración suave.

---

## 6. Normales

Calcula la **normal** (dirección perpendicular a la superficie) de cada
punto y las orienta de forma consistente. **Este paso es crítico**: la
reconstrucción Poisson depende casi por completo de tener buenas normales.

| Parámetro | Rango típico | Cómo elegirlo |
|---|---|---|
| **Factor radio** (× voxel) | `3.0 – 5.0` | Radio de búsqueda para estimar la normal = `factor · tamaño_voxel`. <br>• **Bajo (3.0)**: normales más sensibles al detalle, pero más ruidosas. <br>• **Alto (5.0)**: normales más suaves y estables. <br>Por defecto **4.0**. |
| **Máx. vecinos** | `20 – 50` | Tope de vecinos usados por punto (límite de cómputo). `30` es un buen valor general. |
| **Orientación k** | `15 – 50` | Vecinos usados para **orientar** las normales de forma coherente (que todas apunten "hacia afuera"). <br>• Valores bajos pueden dejar normales mal orientadas en zonas complejas. <br>• Subirlo da orientación más robusta pero más lento. Por defecto **30**. |

> **Nota técnica (orientación de normales con láser):** si se conoce la
> **posición del escáner**, lo ideal es orientar las normales *hacia el
> sensor* (mucho más fiable que el método geométrico). El preprocesador ya
> soporta esto vía el parámetro `sensor_center`, pero todavía falta cablear
> la lectura de esa posición desde los archivos E57. Es una mejora pendiente.

---

## Resumen: ¿por dónde empezar a ajustar?

1. **Deja los valores por defecto** y corre el preproceso una vez.
2. Mira el **log**: el valor de `d̄` y cuántos puntos quedan tras cada paso.
3. Ajusta **un parámetro a la vez** y observa el efecto:
   - ¿Queda ruido? → baja `σ ratio` (SOR) o activa/ajusta el suavizado MLS.
   - ¿Se pierde detalle / aparecen agujeros? → sube `σ ratio`, sube el factor
     de radio del ROR, o baja el factor de voxel.
   - ¿Demasiado pesado/lento? → sube el factor de voxel.
4. **Anota los parámetros usados** para cada nube: como todo es relativo a
   `d̄`, deberían transferirse bien entre nubes similares (reproducibilidad).

> El preproceso es **determinista** (usa una semilla fija): con los mismos
> parámetros y la misma nube, siempre obtendrás el mismo resultado. Esto es
> importante para poder defender los resultados de la tesis.
