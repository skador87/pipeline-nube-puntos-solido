# Priorización de la deuda técnica

> Contraste de la lista de pendientes de
> [`ESTADO_Y_PENDIENTES.md`](ESTADO_Y_PENDIENTES.md) (secciones 5 y 6) con el
> código real, con la evidencia que respalda cada prioridad.
> Fecha: julio 2026, sobre el commit `1ccde2a`.

---

## 1. Método de priorización

Cada ítem se puntúa en tres ejes de 1 a 5:

| Eje | Pregunta | 1 | 5 |
|---|---|---|---|
| **Impacto** | ¿Cuánto frena el trabajo de los tesistas? | Molestia menor | Bloquea una línea de trabajo |
| **Riesgo** | ¿Qué pasa si no se arregla? | Nada relevante | Resultado incorrecto o app inutilizable |
| **Esfuerzo** | ¿Cuánto cuesta arreglarlo? | Cambio local | Rediseño con validación pesada |

**Prioridad = (Impacto + Riesgo) × (6 − Esfuerzo)**

El esfuerzo entra invertido: a igual impacto y riesgo, primero lo barato. El
resultado no se sigue de forma ciega — en la sección 4 se documentan las dos
correcciones al orden que la fórmula por sí sola no captura.

---

## 2. Evidencia medida

Todo lo que sigue se midió en el entorno validado (Python 3.11.15, Open3D
0.19.0) sobre `testing/conejo/bunny/reconstruction/bun_zipper_res2.ply`.

### 2.1 Línea base del pipeline

Ejecución del harness con parámetros por defecto, **antes** de cualquier
cambio. Coincide con la referencia documentada:

| Métrica | Valor |
|---|---|
| Estrategia | `repair` |
| `is_closed` | `True` |
| `fidelity_stats` | `0.003968` (0.3968 %) |
| Volumen | `0.000746` |
| Diagonal bbox | `0.255290` |
| d̄ (escala característica) | `0.002052` |

### 2.2 Los defaults absolutos de Ball Pivoting cuelgan el proceso

Con `bp_radius = 1.0` sobre la nube preprocesada del bunny (5.838 puntos):

| Configuración | Radio en unidades de d̄ | Radio vs. diagonal del objeto | Resultado |
|---|---|---|---|
| `bp_radius = 1.0` (default actual) | 487 · d̄ | **4.03 × la diagonal** | **no termina en 300 s** (proceso terminado a la fuerza) |
| segundo radio (`radius × 2`) | 975 · d̄ | 8.07 × la diagonal | — |
| `bp_radius = 2 · d̄` | 2 · d̄ | 0.017 × la diagonal | 11.243 tri en **0.8 s** |

`alpha = 0.1` no cuelga, pero degrada:

| Configuración | Alpha en unidades de d̄ | Triángulos | Tiempo |
|---|---|---|---|
| `alpha = 0.1` (default actual) | 49 · d̄ (0.40 × diagonal) | 3.336 | 0.3 s |
| `alpha = 5 · d̄` | 5 · d̄ | 10.622 | 0.8 s |

El default actual descarta ~2/3 de los triángulos: la envolvente se vuelve
tan gruesa que pierde el detalle de la oreja y la pata del conejo.

**Corrección al documento:** el pendiente 1.1 dice que Ball Pivoting «produce
basura». Es peor que eso — **no termina**. Y como no existe forma de cancelar
una operación en curso (ver ítem N1), la única salida es matar la aplicación.

### 2.3 Reparto del tiempo de reconstrucción

Instrumentando las cuatro etapas de `MeshReconstructor.run()` por separado
(23.782 triángulos de Poisson):

| Etapa | Tiempo | % del total |
|---|---|---|
| Poisson | 0.59 s | 15.6 % |
| `_filter_phantom_webbing` | 1.51 s | **40.1 %** |
| `_filter_by_mesh_thickness` | 1.65 s | **43.9 %** |
| Suavizado Taubin | 0.02 s | 0.4 % |
| **Total** | **3.77 s** | |

Los dos filtros triángulo-a-triángulo se llevan el **84 %** del tiempo, a
**133 µs/triángulo**. Extrapolado a las mallas de 360k triángulos que produce
una nube de 120k puntos: ~48 s solo en filtros, coherente con los 37–55 s
que reporta el documento.

**El pendiente 2.1 queda confirmado**: el documento pedía «perfilar primero»
antes de vectorizar. Ya está perfilado, y los filtros son efectivamente el
cuello de botella. Poisson no lo es.

### 2.4 Estado de la validación automática

`grep` de `pytest` y `unittest` sobre todo el repositorio: **cero
coincidencias**. `requirements.txt` no declara ninguna dependencia de test.
El único mecanismo de validación es `testing/test_solid_quality.py`, que
imprime métricas pero **no afirma nada**: no hay condición que falle, hay que
leer los números a ojo y compararlos mentalmente con la referencia.

---

## 3. Ítems nuevos detectados en el código

Cuatro cosas que no están en la sección 5 del documento.

### N1 · No se puede cancelar una operación en curso
`main_gui.py:147` — `WorkerThread` no tiene mecanismo de cancelación, y
`_worker_busy()` solo rechaza lanzar una operación nueva. Combinado con 2.2,
una reconstrucción mal parametrizada obliga a matar la aplicación y perder
todo el estado. **(3+4) × 3 = 21**

### N2 · `_update_solid_label` pisa la información de la malla
`main_gui.py:2602` escribe el resultado del sólido en `lbl_mesh_info`, que es
la misma etiqueta donde `_update_mesh_label` puso la malla reconstruida. Tras
ejecutar C se pierde de la vista el conteo de la malla de B, y no vuelve salvo
re-ejecutando. Dos productos compitiendo por una etiqueta. **(2+2) × 5 = 20**

### N3 · Las `stats` de cada bloque se descartan
`_on_result()` (`main_gui.py:2449`) pasa `result["stats"]` a la función que
actualiza la etiqueta y no las guarda en ningún sitio. Es prerrequisito del
pendiente 1.3: no se puede exportar un reporte de algo que no se conserva.
**(3+3) × 5 = 30** — pero en la práctica es parte del coste de 1.3.

### N4 · `except Exception` silencioso por triángulo
`reconstructor.py:295` y `:378` capturan cualquier excepción dentro del bucle
y conservan el triángulo sin registrar nada. Es una decisión defendible
(preferir un falso negativo a romper la ejecución), pero enmascara errores
sistemáticos: si el filtro fallara en el 100 % de los triángulos, el log diría
«0 tri eliminados» y nadie se enteraría. **(2+2) × 5 = 20**

---

## 4. Tabla de prioridades

| # | Ítem | Fase doc. | Imp. | Riesgo | Esf. | **Prioridad** |
|---|---|---|---|---|---|---|
| 1 | **3.1a** Test de regresión mínimo (aserciones sobre el harness) | 3 | 4 | 5 | 1 | **45** |
| 2 | **1.1** `bp_radius` / `alpha` relativos a d̄ | 1 | 4 | 5 | 2 | **36** |
| 3 | **N3** Conservar las `stats` de cada bloque | — | 3 | 3 | 1 | **30** |
| 4 | **1.2** Panel de calidad en la UI | 1 | 3 | 3 | 2 | **24** |
| 5 | **1.3** Reporte de sesión exportable | 1 | 4 | 3 | 3 | **21** |
| 5 | **N1** Cancelar operación en curso | — | 3 | 4 | 3 | **21** |
| 7 | **N2** Etiqueta de malla pisada por el sólido | — | 2 | 2 | 1 | **20** |
| 7 | **N4** `except` silencioso en los filtros | — | 2 | 2 | 1 | **20** |
| 9 | **2.1** Vectorizar los filtros de artefactos | 2 | 4 | 2 | 3 | **18** |
| 9 | **3.2** Diagnóstico de normales | 3 | 3 | 3 | 3 | **18** |
| 11 | **3.1b** Suite completa de smoke tests de GUI | 3 | 4 | 4 | 4 | **16** |
| 11 | **3.3** Métricas geométricas adicionales | 3 | 2 | 2 | 2 | **16** |
| 11 | **4.1** Mensajes de error e interfaz (`ux-copy`) | 4 | 2 | 2 | 2 | **16** |
| 11 | **4.3** Presets guardables por tipo de dato | 4 | 3 | 1 | 2 | **16** |
| 15 | **2.2** Progreso real por etapa | 2 | 2 | 1 | 2 | **12** |
| 15 | **4.2** Crítica de usabilidad (`design-critique`) | 4 | 2 | 1 | 2 | **12** |
| 15 | **4.4** Comparación visual antes/después | 4 | 3 | 1 | 3 | **12** |
| 15 | **4.5** Controles de visualización | 4 | 2 | 1 | 2 | **12** |
| 19 | Limpieza menor (`nano *`, `vscode/`) | — | 1 | 1 | 1 | **10** |
| 20 | **4.6** Formalizar el tema oscuro (`design-system`) | 4 | 1 | 1 | 3 | **6** |
| 21 | **3.4** Procesamiento por bloques con solape | 3 | 3 | 2 | 5 | **5** |

### Dos correcciones al orden del documento

**a) La suite de tests hay que partirla en dos.** El documento la deja entera
en la Fase 3 anotando que «es lo más urgente de esta fase». Puntuada como un
solo bloque de seis escenarios da 16 — el esfuerzo alto la hunde. Pero dentro
hay dos cosas de coste muy distinto:

- **3.1a**, envolver el harness ya existente en aserciones sobre el bunny
  (cerrado, fidelidad ≤ 0.5 %, estrategia `repair`), es **esfuerzo 1** y sube
  a **45, el primer lugar de la tabla**. Es lo único que convierte «los
  números se ven parecidos» en «el pipeline no empeoró», y protege todos los
  demás ítems: sin esto, cada cambio de la Fase 1 y 2 se valida a ojo.
- **3.1b**, los seis escenarios de GUI con Qt offscreen y un doble de
  `Viewport3D`, sigue siendo esfuerzo 4 y se queda en la Fase 3.

**b) 1.1 sube dentro de la Fase 1.** El documento lo lista primero, lo cual
coincide con la puntuación, pero por la razón equivocada: no es un problema de
calidad de malla sino de **disponibilidad**. Un método del pipeline cuelga la
aplicación con sus valores de fábrica, sin forma de cancelar. Para una defensa
de tesis en que se comparen los tres métodos de reconstrucción, es el único
ítem de la lista capaz de arruinar la demostración en vivo.

### Qué NO subir de prioridad

- **3.4 (procesamiento por bloques)** es el ítem más caro de la lista y su
  problema tiene una mitigación que ya funciona: submuestrear. Último.
- **4.6 (tema oscuro)** no habilita nada; es higiene que conviene hacer
  cuando ya no se vaya a tocar más la UI, no antes.
- **2.2 (progreso por etapa)** es percepción, no capacidad. Además su coste
  real baja mucho si se hace *después* de 2.1, porque los filtros vectorizados
  son el sitio natural donde emitir progreso por lotes.

---

## 5. Plan por fases (revisado)

**Fase 1 — Corrección y visibilidad** · ítems 1, 2, 3, 4, 5, 7
1. 3.1a · Test de regresión mínimo sobre el bunny ← *nuevo, va primero*
2. 1.1 · `bp_radius` y `alpha` relativos a d̄
3. N3 + 1.2 + N2 · Conservar `stats` y panel de calidad (con la etiqueta separada)
4. 1.3 · Reporte de sesión exportable

**Fase 2 — Rendimiento y feedback** · ítems 6, 9, 15
5. N1 · Cancelación de operaciones
6. 2.1 · Vectorizar los filtros (84 % del tiempo, ya perfilado)
7. 2.2 · Progreso por etapa, apoyado en los lotes de 2.1

**Fase 3 — Robustez** · ítems 9, 11, 21
8. 3.2 · Diagnóstico de normales
9. 3.1b · Suite completa de smoke tests
10. 3.3 · Métricas adicionales
11. 3.4 · Procesamiento por bloques

**Fase 4 — UX** · sin cambios de orden interno respecto al documento.

**Transversal, en cualquier momento:** N4 y la limpieza menor son de esfuerzo
1 y se pueden hacer junto a cualquier otro cambio que toque esos archivos.
