# Point Cloud Processor — de nube de puntos a sólido 3D

Aplicación de escritorio que convierte una **nube de puntos** (escaneo láser,
fotogrametría, LiDAR) en un **sólido cerrado exportable** a STL/OBJ/PLY, apto
para análisis o impresión 3D.

Desarrollada como trabajo de tesis de pregrado. Interfaz en PyQt5 + Vispy,
núcleo geométrico en Open3D.

```
nube de puntos  →  A · Preprocesamiento  →  B · Reconstrucción  →  C · Solidificación  →  sólido
   .ply .las         limpieza de ruido       malla triangular       cierre topológico      .stl .obj
   .txt .e57         normales                                       control de calidad
```

---

## Instalación

Requiere **Python 3.11** y las dependencias nativas de Open3D. La vía más
confiable en Windows es un entorno conda (Miniforge/Miniconda):

```bash
conda create -n tesis_pointcloud python=3.11 -y
```

```bash
conda activate tesis_pointcloud
```

```bash
pip install -r requirements.txt
```

### Nota para Windows

Open3D necesita que el directorio `Library\bin` del entorno esté en el `PATH`;
si no, `import open3d` falla con *«DLL load failed while importing pybind»*.
Activar el entorno con `conda activate` normalmente lo resuelve. Si ejecutas
desde VS Code, el repositorio ya incluye la configuración en
[`.vscode/settings.json`](.vscode/settings.json).

Para que los mensajes del log se muestren correctamente en consola:

```bash
set PYTHONUTF8=1
```

### Verificar la instalación

```bash
python "nano test_imports.py"
```

Debe imprimir `=== TODOS LOS IMPORTS OK ===`.

---

## Uso

Lanzar la interfaz gráfica:

```bash
python main_gui.py
```

Flujo de trabajo típico:

1. **Cargar nube** (`📂 Cargar nube`) — formatos soportados: `.ply`, `.pcd`,
   `.xyz`, `.txt`, `.las`, `.laz` y `.e57` (los dos últimos requieren las
   dependencias opcionales).
2. Ajustar parámetros y ejecutar **A → B → C** por separado, o
   **▶ Ejecutar todo** para el pipeline completo.
3. Revisar el resultado en el visor 3D y **exportar** el producto que
   corresponda: nube, malla o sólido.

### Modos de uso

| Modo | Para quién | Qué muestra |
|---|---|---|
| **Básico** (por defecto) | Uso general | Categorías simples por proceso (*Suave/Media/Agresiva*, *Bajo/Medio/Alto*…). Los parámetros críticos para la robustez del sólido quedan fijos en su configuración validada. |
| **Avanzado** | Especialistas | Los ~40 parámetros numéricos del pipeline. Muestra los valores que el modo básico está aplicando, por lo que sirve también para aprender el efecto de cada uno. |

Todos los controles tienen **texto flotante** explicando qué hacen, sus valores
típicos y cuándo conviene subirlos o bajarlos.

### Herramientas de edición y medición

La barra sobre el visor 3D permite **mover, rotar y escalar** la pieza, con el
cursor (arrastrando) o de forma paramétrica (botones `−`/`+` con paso
configurable y eje activo X/Y/Z), y **medir distancias** entre dos puntos: el
resultado y sus componentes ΔX/ΔY/ΔZ aparecen en el terminal de la aplicación.

---

## Estructura del proyecto

```
core/
  io_loader.py      Carga multi-formato (.ply .pcd .xyz .txt .las .laz .e57)
  preprocessor.py   A · limpieza: dedup, ROR, SOR, voxel, MLS, normales
  reconstructor.py  B · malla: Poisson / Ball Pivoting / Alpha Shape
  solidifier.py     C · sólido: cascada de 4 estrategias + control de calidad
main_gui.py         Interfaz gráfica (PyQt5 + Vispy)
utils/helpers.py    Utilidades de logging y E/S
testing/
  test_solid_quality.py   Harness de validación de fidelidad geométrica
  conejo/bunny/           Stanford Bunny (datos de prueba incluidos)
docs/
  PREPROCESAMIENTO.md     Guía de parámetros del bloque A
  SOLIDIFICACION.md       Guía de la cascada de solidificación (bloque C)
```

Cada módulo de `core/` expone la misma interfaz:
`Clase(entrada, log_callback).run(params) -> (resultado, stats)`.

---

## Documentación

- **[docs/PREPROCESAMIENTO.md](docs/PREPROCESAMIENTO.md)** — qué hace cada
  parámetro de limpieza y cómo elegirlo según el tipo de nube. Explica la
  escala característica `d̄`, base de todos los umbrales relativos.
- **[docs/SOLIDIFICACION.md](docs/SOLIDIFICACION.md)** — cómo funciona la
  cascada de solidificación, los dos criterios de aceptación (cierre
  topológico + fidelidad geométrica) y los resultados de referencia.

---

## Validación

El harness mide la calidad del sólido con código independiente del pipeline:
cierre topológico, orientabilidad, volumen y distancia de Chamfer respecto a la
malla reconstruida.

```bash
python testing/test_solid_quality.py testing/conejo/bunny/reconstruction/bun_zipper_res2.ply
```

Resultado esperado con los parámetros por defecto:

| Métrica | Valor |
|---|---|
| Estrategia usada | `repair` |
| Cerrado (watertight) | sí |
| Error de fidelidad (Chamfer / diagonal) | ≈ 0.4 % |

Opciones útiles: `--max-points N` para submuestrear nubes grandes y
`--force-fallback` para desactivar la reparación y evaluar las estrategias de
respaldo.

---

## Datos de prueba

El repositorio incluye el **Stanford Bunny** (`testing/conejo/bunny/`), con el
que se reproducen todos los resultados de referencia.

Los datasets grandes quedan **fuera del repositorio** a propósito
(ver [`.gitignore`](.gitignore)) y deben obtenerse por separado:

- `input.ply` — escaneo propio, 1.7 M puntos (104 MB)
- `Stanford3dDataset_v1.2_Aligned_Version` — S3DIS, interiores con RGB;
  se coloca en `testing/`

---

## Nota técnica

El proyecto **no usa `is_watertight()` ni `get_volume()` de Open3D** como
criterio interno de estanqueidad: ambos incluyen un test de auto-intersecciones
de coste cuadrático que, en mallas cerradas grandes, tarda horas y congela la
interfaz. En su lugar se usa `MeshSolidifier.is_topologically_closed()`
(manifold sin frontera, O(T)) y el volumen se calcula por suma de tetraedros
firmados.
