# Estado del proyecto y trabajo pendiente

> Documento de contexto para retomar el trabajo en una sesión nueva.
> Última actualización: julio 2026, commit `1ccde2a`.

---

## 1. Qué es esto

Pipeline que convierte nubes de puntos en sólidos 3D cerrados exportables.
GUI en PyQt5 + Vispy, núcleo geométrico en Open3D. Trabajo de tesis de
pregrado (dos tesistas; el dueño de este repo es el profesor guía).

Lectura previa recomendada: [`README.md`](../README.md),
[`PREPROCESAMIENTO.md`](PREPROCESAMIENTO.md) y
[`SOLIDIFICACION.md`](SOLIDIFICACION.md).

**Pipeline:** `A · Preprocesamiento → B · Reconstrucción → C · Solidificación`

---

## 2. Entorno de ejecución (importante)

No usar el `python` global. El entorno es conda:

```
C:\Users\Skador\miniforge3\envs\tesis_pointcloud\python.exe
```

Dos requisitos no obvios para ejecutar scripts desde consola:

1. **`PATH` debe incluir `...\envs\tesis_pointcloud\Library\bin`** — si no,
   `import open3d` falla con *«DLL load failed while importing pybind»*.
2. **`PYTHONUTF8=1`** — los logs llevan emojis y en consola cp1252 revientan
   con `UnicodeEncodeError`.

Ejemplo (PowerShell):

```powershell
$env:PATH = "C:\Users\Skador\miniforge3\envs\tesis_pointcloud\Library\bin;" + $env:PATH
$env:PYTHONUTF8 = "1"
& "C:\Users\Skador\miniforge3\envs\tesis_pointcloud\python.exe" main_gui.py
```

Versiones validadas: Python 3.11.15, Open3D 0.19.0, numpy 2.4.6, scipy 1.17.1,
PyQt5 5.15.11, Vispy 0.16.2, laspy 2.7.0.

---

## 3. Estado actual

Repositorio en GitHub, todo sincronizado:
`https://github.com/skador87/pipeline-nube-puntos-solido` (público, rama `main`).

| Commit | Contenido |
|---|---|
| `1ccde2a` | README y requirements.txt |
| `dd395b5` | Carga LAS/LAZ (LiDAR) vía laspy |
| `2b304d1` | Visor robusto, modos Básico/Avanzado, edición interactiva, S3DIS |
| `ccd20fc` | Fix crash al re-ejecutar, orden A/B/C, tooltips, export por producto |
| `75b377b` | Commit inicial |

Lo que ya funciona y está validado: cascada de solidificación con criterio
doble (cierre topológico + fidelidad Chamfer), cierre real de agujeros,
modos Básico/Avanzado, barra de edición interactiva con medición, carga
`.ply/.pcd/.xyz/.txt/.las/.laz/.e57`, exportación por producto.

---

## 4. Trampas conocidas (no repetir estos errores)

- **`is_watertight()` y `get_volume()` de Open3D son O(T²)** en mallas
  cerradas (test de auto-intersecciones por fuerza bruta). En una malla de
  360k triángulos tardan horas. Usar `MeshSolidifier.is_topologically_closed()`
  (O(T)) y volumen por tetraedros firmados.
- **No recrear visuales de Vispy** en cada ejecución: deja buffers GL
  huérfanos que provocan *access violations* nativos. Se actualizan con
  `set_data()` y se ocultan con `visible`.
- **No hacer render de picking en GPU** (`visual_at`): era la causa de los
  crashes al rotar. Se resuelve por bounds en CPU.
- **No soltar la referencia al `QThread`** en `finished`: destruirlo mientras
  su limpieza interna corre aborta el proceso. Se libera en el siguiente
  `_start_worker()` tras `wait()`.
- **PowerShell 5.1 mangla comillas dobles** al pasar here-strings a
  ejecutables nativos. Para mensajes de commit con comillas, usar
  `git commit -F <archivo>`.

---

## 5. Trabajo pendiente acordado

> La priorización completa, con el criterio (impacto / riesgo / esfuerzo) y la
> evidencia medida que la respalda, está en
> [`PRIORIZACION_DEUDA_TECNICA.md`](PRIORIZACION_DEUDA_TECNICA.md).

### Fase 1 — Corrección y visibilidad ✅ COMPLETADA

**1.1 `bp_radius` y `alpha` en unidades absolutas.** ✅
Los defaults eran `bp_radius=1.0` y `alpha=0.1` mientras todo el
preprocesamiento usaba umbrales relativos a `d̄`. Medido: con `bp_radius=1.0`
la esfera era **4.03× la diagonal del objeto** (487·d̄) y Ball Pivoting **no
terminaba en 300 s** sobre el bunny — no «producía basura», se colgaba.
Ahora `MeshReconstructor` calcula `d̄` sobre la nube que recibe y resuelve
ambos como múltiplos (`bp_radius_factor=2.0`, `alpha_factor=5.0`). Se
conservan las claves antiguas para el modo `"absolute"`, que avisa en el log
si el valor es desproporcionado.

**1.2 Panel de calidad en la UI.** ✅
Grupo «Calidad del sólido» en el panel derecho: estrategia ganadora (con
color según su fiabilidad), error de fidelidad contra la tolerancia vigente,
cierre topológico, volumen, agujeros de la malla de entrada y la cascada
completa de estrategias probadas. Se añadió `volume` a las `stats` del
solidificador (tetraedros firmados, O(T)).

**1.3 Reporte de sesión exportable.** ✅
`Archivo → Exportar reporte de sesión…` (Ctrl+R), en Markdown o JSON.
Incluye dataset, entorno, y por cada bloque ejecutado sus parámetros,
métricas y duración. `WorkerThread` ahora cronometra cada bloque y
`MainWindow` conserva la bitácora de la sesión.

### Fase 2 — Rendimiento y feedback

**2.1 Vectorizar los filtros de artefactos.** ← siguiente
`_filter_phantom_webbing` y `_filter_by_mesh_thickness` iteran triángulo por
triángulo con consultas KDTree individuales. **Ya está perfilado** (no hace
falta repetirlo): sobre el bunny, de 3.77 s de reconstrucción se van
**1.51 s (40.1 %) en webbing y 1.65 s (43.9 %) en thickness — el 84 % del
total**, a 133 µs/triángulo, mientras que Poisson son solo 0.59 s (15.6 %).
Extrapolado a 360k triángulos da ~48 s, coherente con los 37–55 s
observados. Vectorizar con consultas por lotes (`cKDTree.query` acepta
arrays); el caso difícil es `query_ball_point` con radio variable por
triángulo en el filtro de webbing.

**2.2 Progreso real por etapa.**
`main_gui.py:180-227` — el progreso salta 5 → 20 → 100, así que una
reconstrucción de 50 s parece congelada. Emitir progreso por etapa desde los
módulos de `core/` (ya reciben `log_callback`; podría añadirse un
`progress_callback` opcional sin romper firmas).

### Fase 3 — Robustez y calidad de datos

**3.1 Suite de tests en el repositorio.** Parcialmente hecha: ya existen
`testing/test_regresion_bunny.py` (aserciones sobre las métricas del
pipeline) y `testing/test_gui_calidad_reporte.py` (smoke test sin pantalla
del panel de calidad y del reporte). Falta el resto de los escenarios de la
sección 6: exportación real a disco, modos Básico/Avanzado, transformaciones,
medición y loaders.
**3.2 Diagnóstico de normales** antes de reconstruir. La calidad de Poisson
depende de que estén bien orientadas; los escaneos de una sola cara (paredes
S3DIS) son el peor caso y hoy el problema se detecta solo al final.
**3.3 Métricas geométricas adicionales** en `stats`: Hausdorff p95, ratio de
volumen, preservación de curvatura. Argumentos cuantitativos para la memoria.
**3.4 Procesamiento por bloques** con solape, para nubes de 1M+ puntos sin
submuestrear (hoy hay que bajar la densidad).

### Fase 4 — UX

**4.1 Mensajes de error e interfaz** (`ux-copy`). ✅
`_show_error()` pasó de volcar la excepción a un diálogo con título,
explicación accionable y detalle técnico plegado. Los nombres internos
(`repair`, `ball_pivoting`, `poisson`) ya no llegan a la interfaz: los combos
muestran nombres legibles y devuelven la clave por `currentData()`. Las
etiquetas crípticas de la barra de edición (`∠°`, `×`) ahora son «Ángulo» y
«Factor» con sufijo/prefijo en el propio campo.

**4.2 Crítica de usabilidad de la GUI** (`design-critique`). ✅
Ver [`PRIORIZACION_DEUDA_TECNICA.md`](PRIORIZACION_DEUDA_TECNICA.md) §6.
El hallazgo principal era que **cada acción existía 3 o 4 veces** (menú +
barra + panel derecho, y «Deshacer» además en la barra de edición, donde
significaba otra cosa). Resuelto: la barra solo lleva acciones globales de
archivo, el panel derecho es información, y el menú mantiene todo por
teclado. Los dos «Deshacer» ahora se llaman distinto.

**4.3 Presets guardables** por tipo de dato (bunny / escaneo propio / S3DIS /
LiDAR aéreo). ← siguiente candidato de esta fase
**4.4 Comparación visual antes/después** — superponer nube y sólido, o vista
dividida.
**4.5 Controles de visualización** — tamaño de punto (hoy fijo en 2 px, se
satura con nubes densas tipo S3DIS), wireframe, escala de referencia.

**4.6 Formalizar el tema oscuro** (`design-system`). ✅
Los colores viven en `ui/theme.py` como tokens (superficies, texto, acento,
semánticos, escala tipográfica, espaciado, radios) y el stylesheet se genera
desde ahí. `testing/test_ui_tema_formato.py` falla si alguien mete un color
fuera de los tokens o una combinación que no cumpla WCAG AA.

### Limpieza menor

- Archivos con nombres accidentales: `nano test_imports.py` (con espacio) y
  `nano_test_main.py` — el `nano` del editor quedó pegado al nombre. Ojo: el
  README referencia el primero como comprobación de instalación, hay que
  actualizarlo si se renombra.
- Carpeta `vscode/` duplicada junto a `.vscode/`.

---

## 6. Validación

**Antes de dar por buena cualquier modificación, correr estas tres:**

```powershell
& $PY testing\test_ui_tema_formato.py
& $PY testing\test_regresion_bunny.py
& $PY testing\test_gui_calidad_reporte.py
```

Ambas devuelven código de salida 0/1, así que sirven tal cual en cualquier
automatización. **No requieren pytest** (no está instalado en el entorno
validado); si algún día se instala, las funciones `test_*` de la primera se
recogen sin tocar nada.

- `test_regresion_bunny.py` — fija la referencia del pipeline: sólido
  cerrado y orientable, estrategia `repair`, fidelidad ≤ 0.4365 % (10 % sobre
  la referencia de 0.3968 %), volumen dentro del 5 % de 0.000746. Además
  comprueba que `bp_radius` y `alpha` siguen siendo proporcionales a `d̄`
  **antes** de reconstruir, para que una regresión de escala falle en
  milisegundos en vez de colgar la ejecución.
- `test_gui_calidad_reporte.py` — smoke test sin pantalla
  (`QT_QPA_PLATFORM=offscreen` + doble de `Viewport3D`): corre A→B→C contra
  la GUI real y verifica que las métricas llegan al panel de calidad y al
  reporte, y que re-ejecutar B invalida el panel del sólido anterior.

**Harness de métricas** (mide con código independiente del pipeline):

```powershell
& $PY testing\test_solid_quality.py testing\conejo\bunny\reconstruction\bun_zipper_res2.ply
```

Resultado esperado con parámetros por defecto: estrategia `repair`,
`is_closed=True`, `fidelity_stats ≈ 0.0040` (0.40 %), volumen `≈ 0.000746`.

Opciones: `--max-points N` para submuestrear, `--force-fallback` para
desactivar la reparación y evaluar los respaldos, `--quiet` para omitir el log.

⚠️ **El error de fidelidad tiene ruido de muestreo**: la distancia de Chamfer
se estima muestreando puntos al azar sobre las mallas, sin semilla fija. En
tres corridas consecutivas sobre el mismo código se observó 0.003962 /
0.003970 / 0.004021. Una diferencia de ese orden **no es una regresión**; para
afirmar que algo empeoró hay que ver el valor fuera de esa banda, o comparar
el conteo de vértices/triángulos, que sí es estable.

**Escenarios de GUI que faltan por cubrir** (los originales se escribieron en
una carpeta temporal de sesión y se perdieron):

1. Ciclo A→B→C completo y re-ejecución de cada bloque (el escenario que
   causaba el crash), verificando que se invalidan los productos aguas abajo.
   *Parcial*: `test_gui_calidad_reporte.py` ya cubre el ciclo completo y la
   re-ejecución de B; faltan la re-ejecución de A y de C.
2. Exportación real a PLY/STL/OBJ/OFF y nube a XYZ, comprobando bytes en disco.
3. Modos Básico/Avanzado: mapeo categoría → parámetro numérico, independencia
   entre bloques A/B/C, re-aplicación de presets al volver de Avanzado.
4. Transformaciones con verificación geométrica: desplazamiento exacto,
   intercambio de ejes al rotar 90°, diagonal ×2 al escalar, deshacer que
   restaura el estado original, alineación entre productos, normales
   renormalizadas tras escalar.
5. Medición: distancia 3-4-5 → 5.0000 y componentes ΔX/ΔY/ΔZ en el log.
6. Loader: `.txt` estilo S3DIS (limpio y con líneas corruptas), `.las` con RGB
   16-bit, `.las` UTM con intensidad → gris y recentrado, `.laz` comprimido.

Truco para probar la GUI sin pantalla: `QT_QPA_PLATFORM=offscreen` y sustituir
`main_gui.Viewport3D` por un doble que implemente `show_point_cloud`,
`show_mesh`, `set_edit_mode`, `set_measure_points`, `clear_measure`, más las
señales `transform_committed` y `point_picked`. **Implementación de
referencia**: la clase `ViewportDoble` de `testing/test_gui_calidad_reporte.py`,
junto con `_ejecutar_bloque()`, que corre un `WorkerThread` de forma síncrona
(llamando a `.run()` directamente, sin hilo) y entrega el resultado a
`MainWindow._on_result()`. Reutilizarlas para los escenarios que faltan.

---

## 7. Restricciones de diseño

- **No romper** las firmas `run(params) -> (resultado, stats)` de los módulos
  de `core/`, ni las claves que `ParamPanel.get_*_params()` ya expone en
  `main_gui.py`. Se pueden **agregar** parámetros con defaults sensatos.
- Estilo del proyecto: métodos privados que loguean vía `_log()` /
  `log_callback`, `stats` como dict, docstrings breves en español.
- El modo Básico debe seguir garantizando la robustez: los parámetros críticos
  quedan fijos en su configuración validada y las categorías solo mueven
  parámetros "de gusto" dentro de rangos seguros.

---

## 8. Plugins disponibles en la sesión

Instalados: **Design** (`accessibility-review`, `design-critique`,
`design-handoff`, `design-system`, `research-synthesis`, `user-research`,
`ux-copy`) y **Engineering** (`architecture`, `code-review`, `debug`,
`deploy-checklist`, `documentation`, `incident-response`, `standup`,
`system-design`, `tech-debt`, `testing-strategy`).

Útiles aquí: `tech-debt` (priorizar con criterio defendible),
`testing-strategy` (diseñar la suite de la sección 6), `ux-copy` (fase 4.1),
`design-critique` (fase 4.2), `design-system` (fase 4.6).

No aplican: `deploy-checklist`, `incident-response`, `standup`,
`design-handoff` — son para servicios web en producción y equipos con
diseñador aparte.

⚠️ El `code-review` del plugin **choca de nombre** con el `/code-review`
propio del entorno; verificar cuál se ejecuta.

**Ojo:** el plugin Engineering es ingeniería de software genérica, no geometría
computacional. Aporta proceso (cómo priorizar, cómo planificar tests), no
conocimiento de Poisson, KDTree ni mallas.
