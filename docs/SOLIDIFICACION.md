# Guía de parámetros — Solidificación (Bloque C)

Documentación para los tesistas. Explica **cómo funciona la cascada de
solidificación** (`core/solidifier.py`, pestaña **C · Sol** de la GUI),
**qué hace cada parámetro** y **cómo se valida la calidad** del sólido final.

---

## Objetivo del bloque

Convertir la malla reconstruida (que normalmente tiene agujeros, bordes
abiertos y defectos no-manifold) en un **sólido cerrado** exportable a
STL/OBJ/PLY, **perdiendo la menor fidelidad geométrica posible** respecto
a la superficie reconstruida.

Los dos criterios de aceptación de un resultado son:

1. **Cierre topológico** ("watertight"): toda arista tiene exactamente dos
   triángulos adyacentes y no hay vértices *pinch*. Se verifica con un
   chequeo O(T) propio (`MeshSolidifier.is_topologically_closed`).

   > ⚠️ **Nota técnica**: no se usa `is_watertight()` de Open3D como
   > criterio interno porque su definición incluye un test de
   > auto-intersecciones de coste **cuadrático** en el número de
   > triángulos: en mallas cerradas de >100k triángulos tarda horas.
   > El cierre topológico es el criterio relevante para exportar sólidos.

2. **Fidelidad geométrica**: distancia de **Chamfer simétrica** entre la
   malla de entrada y la candidata (media de las distancias punto a punto
   entre muestreos uniformes de ambas superficies), **normalizada por la
   diagonal del bounding box**. Un error de 0.01 significa "en promedio,
   la superficie se movió un 1% del tamaño del objeto".

---

## La cascada de estrategias

Se prueban en orden de mayor a menor fidelidad esperada. La primera que
cumpla **ambos** criterios (cerrada + error ≤ tolerancia) se acepta. Si
ninguna cumple, se devuelve la **mejor candidata** (cerrada primero, menor
error después) con una advertencia en el log.

```
E1 repair → E2 poisson → E3 voxel → E4 hull → (passthrough)
```

### Estrategia 1 — Reparación topológica (`repair`)

Repara la malla original **sin regenerarla**: es la vía de máxima fidelidad
porque conserva la geometría existente.

1. Limpieza estándar (vértices/triángulos duplicados, degenerados, huérfanos).
2. `merge_close_vertices` con **límite de seguridad**: el eps solicitado
   (`merge_eps_factor × voxel`) se recorta al 50% de la arista mediana.
   Sin este límite, un eps mayor que la arista típica colapsa la malla
   (en pruebas destruía hasta el 90% de los vértices).
3. Eliminación de componentes conexos pequeños (<1% de los triángulos).
4. **Cierre real de agujeros** (*peel & fill*, ver abajo).

### Estrategia 2 — Envolvente Poisson (`poisson`) ★ nueva

Reemplaza al Convex Hull como fallback principal. A diferencia del hull,
**preserva las concavidades**:

1. Muestrea uniformemente la superficie de la malla (con normales
   interpoladas de la propia malla).
2. Reconstruye con **Screened Poisson** (`fallback_poisson_depth`, por
   defecto 8): al ser una superficie implícita extraída por marching
   cubes, el resultado es **cerrado por construcción**.
3. Conserva solo el componente conexo principal (elimina burbujas) y
   recorta si la envolvente se extiende mucho más allá del bbox original.
4. *Peel & fill* de seguridad.

En el conejo de Stanford: hull = 2.9% de error medio; envolvente
Poisson = **0.49%** (6× más fiel), ambas cerradas.

### Estrategia 3 — Voxelización + Ball Pivoting (`voxel`)

Voxeliza la malla y re-triangula los centros de voxel con Ball Pivoting
(dos radios) + *peel & fill*. Es más robusta que antes (se corrigió un
error de API que hacía que esta estrategia **siempre** lanzara excepción
y cayera al hull) pero rara vez logra el cierre completo: queda como
respaldo intermedio.

### Estrategia 4 — Convex Hull (`hull`)

Último recurso. La envolvente convexa destruye toda concavidad, por lo que
normalmente **excede la tolerancia de fidelidad** y solo se usa si ninguna
otra estrategia produjo algo cerrado.

---

## Peel & fill (cierre de agujeros)

El cierre anterior solo hacía `merge_close_vertices`, que no triangula
nada (y además invocaba la API con un argumento inexistente, por lo que
nunca se ejecutó). El método actual itera hasta lograr el cierre o agotar
las rondas:

1. **Peel**: eliminar defectos no-manifold (aristas con >2 triángulos y
   vértices *pinch*) que impiden el cierre.
2. **Fan fill**: cerrar cada loop frontera con un abanico de triángulos a
   su centroide. Es *manifold-seguro por construcción* (cada arista
   frontera recibe exactamente un triángulo nuevo y el centroide es un
   vértice nuevo) y el winding se toma del triángulo dueño de cada arista,
   por lo que funciona aunque la malla no esté orientada de forma
   consistente.
3. **Fill**: los bordes que el abanico no pudo cerrar (cadenas no simples)
   se triangulan con `o3d.t.geometry.TriangleMesh.fill_holes()` (API
   tensor de Open3D ≥0.19). *Nota*: `fill_holes` puede introducir defectos
   no-manifold nuevos; se eliminan en la ronda siguiente.
4. **Merge** fino (30% de la arista mediana), solo en la primera ronda:
   repetido sobre los parches nuevos reintroduce defectos no-manifold.
5. Si una ronda no progresa y quedan pocas aristas (≤24), se **pela
   localmente** la ranura degenerada para que la siguiente ronda la
   rellene. Se guarda siempre el mejor estado alcanzado y nunca se
   devuelve uno peor.

---

## Parámetros (pestaña C · Sol)

| Parámetro | Default | Notas |
|---|---|---|
| Voxel size auto / manual | auto | Escala de referencia para merge y voxelización (1.5% del tamaño del objeto). |
| Estrategia 1–4 | todas ON | Desactivar solo para experimentar/depurar. |
| Profundidad Poisson | 8 | 8 ≈ misma fidelidad que 9 con la mitad de triángulos. Subir a 9–10 solo para objetos con detalle muy fino. |
| Merge eps factor | 0.5 | Se recorta automáticamente al 50% de la arista mediana. |
| Cerrar agujeros | ON | Activa *peel & fill* en la Estrategia 1. |
| Validar fidelidad (Chamfer) | ON | Sin esto, se acepta la primera estrategia cerrada aunque deforme el objeto. |
| Error máx. (% diagonal) | 2.0% | Tolerancia de fidelidad. Bajar a 1% para objetos con concavidades importantes; subir si el escaneo es muy incompleto y hay que "inventar" superficie para cerrar. |

Las `stats` del bloque incluyen ahora `fidelity_error` (error de la
estrategia aceptada) y `strategies_tried` (qué probó la cascada y con qué
resultado), útiles para reportar en la memoria.

---

## Validación

Script: `testing/test_solid_quality.py` — corre el pipeline completo
(pre → rec → sol) sobre un dataset y mide, con código independiente del
solidificador: cierre topológico, orientabilidad, volumen (tetraedros
firmados) y Chamfer sólido↔malla reconstruida.

```bash
python testing/test_solid_quality.py testing/conejo/bunny/reconstruction/bun_zipper_res2.ply
python testing/test_solid_quality.py input.ply --max-points 120000
python testing/test_solid_quality.py <nube> --force-fallback   # sin E1, prueba los fallbacks
```

Resultados de referencia (Stanford bunny res2, parámetros por defecto):

| Métrica | Antes | Después |
|---|---|---|
| Cerrada (watertight) | **No** (la E1 se aceptaba sin verificar) | **Sí** (E1 repair) |
| Orientable | No | Sí |
| Volumen | no calculable | 0.000746 m³ |
| Chamfer medio del sólido | 0.23% pero malla abierta | **0.29%** y cerrada |
| Chamfer medio (fallback sin E1) | 2.6% (hull, cóncavo destruido) | **0.49%** (poisson) |
| Estrategia voxel | excepción silenciosa siempre | funcional |

Por estrategia aislada (bunny): repair 0.40% ✓cerrada · poisson 0.48%
✓cerrada · voxel 1.6% ✗ · hull 2.97% ✓cerrada (rechazada por tolerancia).

En `input.ply` (escaneo parcial de 1.7M pts, submuestreado a 120k para el
test): antes → hull con 3.6% de error (toda concavidad perdida); después →
repair cerrada con **0.76–0.78%** de error. Nota: al ser un escaneo
parcial (una "sábana" de superficie), el sólido resultante es una concha
delgada que envuelve la superficie observada — el volumen pequeño es
esperable y preferible a inventar un bloque convexo.
