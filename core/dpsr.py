# core/dpsr.py
# DPSR — Reconstrucción de superficie por Poisson diferenciable.
#
# Implementación propia en PyTorch de la formulación de:
#     Peng, Jiang, Liao, Niemeyer, Pollefeys y Geiger,
#     «Shape As Points: A Differentiable Poisson Solver», NeurIPS 2021.
#     https://arxiv.org/abs/2106.03452
#
# No es una copia del código de los autores (MIT): está escrita desde las
# ecuaciones del artículo, para poder documentar cada paso en la memoria y
# para no arrastrar sus dependencias (PyTorch3D, torch-scatter), que en
# Windows son problemáticas. Solo se necesita PyTorch y scikit-image.
#
# ⚠️ Qué es y qué NO es
# ────────────────────
# Esto NO es un modelo de IA preentrenado: no hay pesos, ni dataset, ni
# checkpoint que descargar. Es un solucionador de la ecuación de Poisson
# escrito de forma DERIVABLE, lo que permite propagar gradientes desde la
# malla resultante hasta los puntos y normales de entrada.
#
# En su uso actual (`run(params)` del reconstructor) se emplea solo la pasada
# hacia adelante, que equivale a un Poisson clásico no cribado resuelto por
# FFT sobre una grilla densa. La ventaja frente al Poisson de Open3D no está
# en la precisión sino en que la salida es CERRADA POR CONSTRUCCIÓN: es una
# superficie de nivel de una función continua, así que no puede tener bordes.
#
# El valor de que sea derivable se materializa cuando se añada encima el
# bucle de optimización por forma (optimizar posiciones y normales para
# minimizar la distancia a la nube observada). Eso está pendiente.
#
# Formulación
# ───────────
# Dados puntos orientados {(p_i, n_i)} normalizados a [0,1)³:
#
#   1. Se rasteriza el campo de normales sobre una grilla densa v(x) por
#      splatting trilineal.
#   2. La ecuación ∇²χ = ∇·v se resuelve en el dominio espectral. Con el
#      dominio en [0,1)³ y frecuencias enteras k:
#          ∇·  →  i2π (k · v̂)
#          ∇²  →  −(2π)² |k|²
#      de donde
#          χ̂ = i2π(k·v̂) / (−(2π)²|k|²) = −i (k·v̂) / (2π |k|²)
#   3. Se atenúan las altas frecuencias con una gaussiana espectral, que
#      cumple el mismo papel que el suavizado del Poisson clásico.
#   4. Se desplaza χ para que su nivel cero pase por los puntos de entrada.
#   5. Marching cubes en el nivel 0.

import numpy as np
import open3d as o3d


# ══════════════════════════════════════════════════════════════
#  IMPORTACIÓN TEMPRANA DE TORCH  (no mover de sitio)
# ══════════════════════════════════════════════════════════════
# ⚠️ TRAMPA: torch DEBE importarse antes de que Open3D ejecute trabajo
# pesado. Open3D carga su propio runtime OpenMP (`libiomp5md.dll` de
# `Library\bin`) la primera vez que estima normales o similar; a partir de
# ese momento `torch/lib/fbgemm.dll` ya no encuentra los símbolos que
# necesita y su importación falla con:
#
#     OSError: [WinError 127] No se encontró el proceso especificado.
#     Error loading "...torch\lib\fbgemm.dll" or one of its dependencies.
#
# Importar aquí, al cargar el módulo, garantiza el orden correcto: como
# `main_gui` importa `core.dpsr` al arrancar, torch queda cargado mucho
# antes de que se ejecute el bloque A. Si se dejara la importación dentro
# de las funciones, ejecutar A y luego B con DPSR reventaría.
#
# Sigue siendo una dependencia OPCIONAL: el try/except permite que el
# módulo se importe sin torch y que el método simplemente no aparezca.
try:
    import torch                                    # noqa: F401
    _TORCH_ERROR = None
except Exception as _e:                             # ImportError u OSError
    torch = None
    _TORCH_ERROR = _e


# ══════════════════════════════════════════════════════════════
#  DISPONIBILIDAD  (mismo patrón que check_e57_support / check_las_support)
# ══════════════════════════════════════════════════════════════

def check_dpsr_support() -> tuple[bool, str]:
    """
    ¿Están las dependencias opcionales del método DPSR?

    Returns
    -------
    (disponible, mensaje)
    """
    faltan = []
    if torch is None:
        # Distinguir «no instalado» de «instalado pero no carga»: el
        # segundo caso suele ser el conflicto de OpenMP descrito arriba y
        # el mensaje de «instálalo» sería engañoso.
        if isinstance(_TORCH_ERROR, ImportError):
            faltan.append("torch")
        else:
            return False, (
                f"torch está instalado pero no se pudo cargar: "
                f"{_TORCH_ERROR}. En Windows suele deberse a un conflicto "
                f"de runtimes OpenMP entre Open3D y torch."
            )
    try:
        from skimage.measure import marching_cubes      # noqa: F401
    except ImportError:
        faltan.append("scikit-image")

    if faltan:
        return False, (
            f"Método DPSR no disponible: falta {' y '.join(faltan)}. "
            f"Instalar con: pip install torch --index-url "
            f"https://download.pytorch.org/whl/cu121 && pip install "
            f"scikit-image"
        )

    if torch.cuda.is_available():
        return True, (f"Método DPSR disponible en GPU "
                      f"({torch.cuda.get_device_name(0)})")
    return True, "Método DPSR disponible (solo CPU: será más lento)"


def dispositivo_por_defecto() -> str:
    """'cuda' si hay GPU utilizable, 'cpu' en otro caso."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
#  NÚCLEO
# ══════════════════════════════════════════════════════════════

def _fftfreqs(res: int, device):
    """Frecuencias enteras (ciclos por dominio) para `rfftn` 3D.
    Devuelve (res, res, res//2+1, 3)."""
    import torch
    fx = torch.fft.fftfreq(res, d=1.0 / res, device=device)
    fy = torch.fft.fftfreq(res, d=1.0 / res, device=device)
    fz = torch.fft.rfftfreq(res, d=1.0 / res, device=device)
    kx, ky, kz = torch.meshgrid(fx, fy, fz, indexing="ij")
    return torch.stack([kx, ky, kz], dim=-1)


def _pesos_trilineales(pts, res):
    """Índices y pesos de los 8 vecinos de cada punto. Compartido por el
    splatting y la interpolación, para que ambos usen exactamente el mismo
    núcleo (si divergieran, el nivel cero dejaría de pasar por los puntos)."""
    p  = pts * res
    p0 = p.floor()
    w1 = p - p0
    w0 = 1.0 - w1
    p0 = p0.long()

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                peso = ((w0[:, 0] if dx == 0 else w1[:, 0]) *
                        (w0[:, 1] if dy == 0 else w1[:, 1]) *
                        (w0[:, 2] if dz == 0 else w1[:, 2]))
                ix = (p0[:, 0] + dx) % res
                iy = (p0[:, 1] + dy) % res
                iz = (p0[:, 2] + dz) % res
                yield (ix * res + iy) * res + iz, peso


def point_rasterize(pts, vals, res):
    """Splatting trilineal de vectores `vals` en las posiciones `pts`.

    pts  : (N,3) en [0,1)
    vals : (N,3)
    ->     (3,res,res,res)

    Derivable respecto de `vals` y de `pts`.
    """
    import torch
    grid = torch.zeros(3, res * res * res, device=pts.device, dtype=vals.dtype)
    for plano, peso in _pesos_trilineales(pts, res):
        for c in range(3):
            grid[c] = grid[c].index_add(0, plano, vals[:, c] * peso)
    return grid.view(3, res, res, res)


def grid_interp(grid, pts):
    """Interpolación trilineal de un campo escalar. grid (res,res,res),
    pts (N,3) en [0,1) → (N,). Derivable respecto de ambos."""
    import torch
    res  = grid.shape[0]
    liso = grid.reshape(-1)
    out  = torch.zeros(pts.shape[0], device=pts.device, dtype=grid.dtype)
    for plano, peso in _pesos_trilineales(pts, res):
        out = out + liso[plano] * peso
    return out


def solve_poisson(pts, normals, res: int, sigma: float = 2.0,
                  normalizar: bool = True):
    """
    Resuelve la ecuación de Poisson en el dominio espectral.

    pts       : (N,3) en [0,1)
    normals   : (N,3) normales orientadas
    res       : lado de la grilla
    sigma     : ancho del suavizado espectral, en celdas
    normalizar: divide χ por |∇χ| medido en los puntos, con lo que χ pasa a
                ser aproximadamente una distancia con signo en celdas.

    Devuelve el indicador χ como tensor (res,res,res). Su nivel cero es la
    superficie.

    Sobre `normalizar`
    ──────────────────
    Poisson deja χ definido salvo constante multiplicativa: dos resoluciones
    o dos bloques distintos producen campos con la misma raíz pero escalas
    diferentes. Mientras se extraiga el nivel cero de un solo campo eso da
    igual, pero al FUSIONAR bloques hay que promediar campos, y promediar
    magnitudes con escalas distintas desplaza el nivel cero y genera
    costuras. Dividir por el gradiente en la superficie lleva todos los
    bloques a la misma unidad (celdas de distancia) y los hace promediables.
    """
    import torch

    # 1. Campo de normales sobre la grilla
    v = point_rasterize(pts, normals, res)

    # 2. Al dominio espectral
    v_hat = torch.fft.rfftn(v, dim=(1, 2, 3))

    # 3. Frecuencias y atenuación gaussiana de alta frecuencia
    k  = _fftfreqs(res, pts.device)
    k2 = (k ** 2).sum(dim=-1)
    g  = torch.exp(-k2 * (sigma / res) ** 2 * (2 * np.pi ** 2))

    # 4. Divergencia dividida por el laplaciano
    k_dot_v = (k[..., 0] * v_hat[0] +
               k[..., 1] * v_hat[1] +
               k[..., 2] * v_hat[2])
    denom = 2.0 * np.pi * k2
    chi_hat = -1j * k_dot_v / torch.where(denom == 0,
                                          torch.ones_like(denom), denom)
    chi_hat = chi_hat * g
    # El indicador está definido salvo constante aditiva: se fija la
    # componente continua a cero y luego se recentra con los puntos.
    chi_hat[0, 0, 0] = 0.0

    # 5. Vuelta al dominio espacial
    chi = torch.fft.irfftn(chi_hat, s=(res, res, res), dim=(0, 1, 2))

    # 6. El nivel cero debe pasar por los puntos observados
    chi = chi - grid_interp(chi, pts).mean()

    # 7. Escala común: χ en unidades de distancia (celdas)
    if normalizar:
        gx, gy, gz = torch.gradient(chi)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        escala_grad = grid_interp(mag, pts).mean()
        if torch.isfinite(escala_grad) and escala_grad > 1e-12:
            chi = chi / escala_grad

    return chi


# ══════════════════════════════════════════════════════════════
#  PROCESAMIENTO POR BLOQUES
# ══════════════════════════════════════════════════════════════
# La grilla de DPSR es densa y uniforme: la memoria crece con res³, así que
# 1024³ son ~48 GB y no cabe en ninguna GPU de escritorio. La solución es
# resolver por bloques solapados en la GPU e ir acumulando el resultado en
# una grilla global que vive en RAM (1024³ en float32 son 4 GB).
#
# Tres cuidados que hacen que la fusión no deje costuras:
#
#   1. Halo. Cada bloque se resuelve sobre una región MAYOR que la que
#      aporta al resultado. La FFT impone condiciones de contorno
#      periódicas, así que el error se concentra en los bordes del bloque;
#      el halo los deja fuera de la zona que se conserva.
#   2. Escala común. Cada bloque normaliza su χ a unidades de distancia
#      (ver `solve_poisson`), sin lo cual promediar bloques desplazaría el
#      nivel cero.
#   3. Ventana de mezcla. En la zona de solape los bloques se promedian con
#      pesos que caen suavemente a cero, en vez de cortarse en seco.
#
# Es una aproximación: el Poisson exacto es un problema global y aquí se
# resuelve por trozos. Con solape suficiente la diferencia queda por debajo
# del propio suavizado, pero conviene decirlo.

# Bytes por celda de grilla que consume un solve (campo vectorial, su
# transformada, las frecuencias y los temporales). Medido empíricamente.
_BYTES_POR_CELDA = 45


def memoria_estimada_gb(res: int) -> float:
    """VRAM aproximada que necesita un solve de lado `res`."""
    return res ** 3 * _BYTES_POR_CELDA / 1e9


def bloque_automatico(resolution: int, device: str) -> int:
    """
    Mayor lado de bloque que cabe holgadamente en la memoria disponible.

    Devuelve `resolution` si el problema entero cabe (entonces no hace falta
    trocear y el resultado es el Poisson global exacto).
    """
    if device != "cuda":
        # En CPU la memoria no suele ser el límite, pero sí el tiempo.
        return min(resolution, 256)

    import torch
    try:
        libre = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        libre = 4.0
    presupuesto = libre * 0.55          # margen para fragmentación

    if memoria_estimada_gb(resolution) <= presupuesto:
        return resolution
    for r in (512, 448, 384, 320, 256, 192, 128):
        if r < resolution and memoria_estimada_gb(r) <= presupuesto:
            return r
    return 128


def _mascara_cerca_de_datos(pts_loc, res: int, radio_celdas: int, device):
    """
    Máscara de las celdas que están a menos de `radio_celdas` de algún punto
    del bloque.

    Sin esto, un bloque opina sobre TODO su volumen, incluidas las zonas de
    aire donde no tiene ningún dato. Al fusionar, esas opiniones sin
    fundamento se promedian con las de bloques vecinos que sí tienen datos y
    generan cruces por cero espurios: superficie inventada en el aire. Con
    la máscara, cada bloque solo aporta donde tiene con qué respaldarlo.
    """
    import torch
    import torch.nn.functional as F

    unos = torch.ones((pts_loc.shape[0], 3), device=device,
                      dtype=torch.float32)
    ocupacion = (point_rasterize(pts_loc, unos, res)[0] > 0).float()

    m = ocupacion[None, None]
    for _ in range(max(1, int(radio_celdas))):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m[0, 0]


def _signo_exterior(chi, pts_loc, normals_loc, res: int) -> float:
    """
    Determina de qué lado queda el exterior, muestreando χ en puntos
    desplazados a lo largo de la normal (hacia fuera).

    Sirve además para unificar el convenio entre bloques: si un bloque
    quedó con las normales invertidas, su χ sale con el signo contrario y
    promediarlo con sus vecinos destruiría el nivel cero. Con esto todos
    los bloques acaban con «exterior positivo».
    """
    import torch
    desplazados = pts_loc + normals_loc * (2.0 / res)
    desplazados = desplazados.clamp(0.0, 1.0 - 1e-6)
    valor = grid_interp(chi, desplazados).mean()
    if not torch.isfinite(valor) or valor == 0:
        return 1.0
    return 1.0 if valor > 0 else -1.0


def _ventana_mezcla(lado: int, halo: int, device, dtype):
    """
    Ventana separable 1D: vale 1 en el núcleo y cae a 0 con perfil coseno
    a lo largo del halo. El producto de tres de estas da la ventana 3D.
    """
    import torch
    w = torch.ones(lado, device=device, dtype=dtype)
    if halo > 0:
        t = torch.arange(halo, device=device, dtype=dtype)
        rampa = 0.5 * (1.0 - torch.cos(np.pi * (t + 0.5) / halo))
        w[:halo]  = rampa
        w[-halo:] = rampa.flip(0)
    return w


def _cerrar_en_el_borde(chi: np.ndarray, valor_exterior: float):
    """
    Fuerza el valor «exterior» en la capa más externa de la grilla.

    Sin esto, si la superficie llega al borde del dominio el marching cubes
    la deja abierta ahí: el nivel cero es cerrado solo si no toca el borde.
    Con el bunny nunca ocurría (objeto pequeño y centrado), pero en una
    escena abierta —un suelo que se extiende hasta el límite— sí ocurre, y
    la malla sale con frontera.
    """
    for eje in range(3):
        for idx in (0, -1):
            corte = [slice(None)] * 3
            corte[eje] = idx
            chi[tuple(corte)] = valor_exterior
    return chi


def _marching_por_bloques(chi: np.ndarray, level: float = 0.0,
                          paso: int = 256, log=print):
    """
    Marching cubes por trozos, para no materializar de golpe la malla de una
    grilla de 1024³.

    Los trozos se solapan una celda: dos trozos vecinos comparten el plano
    de voxels frontera, y como la interpolación de marching cubes depende
    solo de los dos valores del extremo de cada arista, los vértices que
    generan en ese plano coinciden exactamente. Basta fusionar duplicados al
    final para obtener una malla sin costuras.
    """
    from skimage.measure import marching_cubes

    R = chi.shape[0]
    if R <= paso:
        v, f, _, _ = marching_cubes(chi, level=level)
        return v, f

    verts_all, faces_all, desplazamiento = [], [], 0
    for i0 in range(0, R - 1, paso):
        i1 = min(i0 + paso + 1, R)
        for j0 in range(0, R - 1, paso):
            j1 = min(j0 + paso + 1, R)
            for k0 in range(0, R - 1, paso):
                k1 = min(k0 + paso + 1, R)
                sub = chi[i0:i1, j0:j1, k0:k1]
                # Sin cambio de signo dentro del trozo no hay superficie.
                if sub.min() > level or sub.max() < level:
                    continue
                try:
                    v, f, _, _ = marching_cubes(sub, level=level)
                except (ValueError, RuntimeError):
                    continue
                if len(v) == 0:
                    continue
                v = v + np.array([i0, j0, k0], dtype=v.dtype)
                verts_all.append(v)
                faces_all.append(f + desplazamiento)
                desplazamiento += len(v)

    if not verts_all:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts_all), np.vstack(faces_all)


def radio_datos_recomendado(pts_n, resolution: int) -> int:
    """
    Radio de la máscara de datos, en celdas, deducido del espaciado real de
    la nube.

    Es crítico y no puede ser una constante. La máscara debe formar una
    lámina CONEXA alrededor de la superficie: si el radio es menor que la
    separación entre puntos vecinos, la lámina queda agujereada, el interior
    del objeto se comunica con el aire de fuera y deja de reconocerse como
    cavidad encerrada — la salida sale hueca en vez de maciza.

    Ejemplo real: el bunny a 512³ tiene los puntos a ~14 celdas; con un
    radio fijo de 6 el volumen caía un 81 %.
    """
    from scipy.spatial import cKDTree

    n = min(len(pts_n), 20000)
    rng = np.random.default_rng(42)
    muestra = pts_n[rng.choice(len(pts_n), size=n, replace=False)]
    d = cKDTree(muestra).query(muestra, k=2)[0][:, 1]
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 6
    espaciado_celdas = float(np.median(d)) * resolution
    return int(max(4, min(64, np.ceil(espaciado_celdas * 1.5))))


def _resolver_por_bloques(pts_n, normals, resolution, sigma, block, overlap,
                          device, log, radio_datos: int = 6):
    """
    Resuelve χ sobre la grilla global troceando en bloques solapados.

    pts_n       : (N,3) normalizados a [0,1)
    radio_datos : hasta cuántas celdas de sus propios puntos aporta cada
                  bloque. Más allá, su opinión no está respaldada por datos.
    ->            (grilla global resolution³ en float32, valor de exterior)
    """
    import torch

    halo = max(1, int(round(block * overlap / 2.0)))
    nucleo = block - 2 * halo
    if nucleo <= 0:
        raise ValueError(
            f"El solape ({overlap:.0%}) no deja núcleo útil en bloques de "
            f"{block}³. Baja el solape o sube el tamaño de bloque."
        )

    n_por_eje = int(np.ceil(resolution / nucleo))
    total = n_por_eje ** 3
    log(f"    ▦ Troceando: {n_por_eje}×{n_por_eje}×{n_por_eje} = {total} "
        f"bloques de {block}³ (núcleo {nucleo}, halo {halo})")
    log(f"    ▦ Radio de la máscara de datos: {radio_datos} celdas")

    chi_global  = np.zeros((resolution,) * 3, dtype=np.float32)
    peso_global = np.zeros((resolution,) * 3, dtype=np.float32)

    # Índice espacial para seleccionar rápido los puntos de cada bloque
    celdas = np.floor(pts_n * resolution).astype(np.int64)
    celdas = np.clip(celdas, 0, resolution - 1)

    t_pts_todos = torch.as_tensor(pts_n,   dtype=torch.float32, device=device)
    t_nrm_todos = torch.as_tensor(normals, dtype=torch.float32, device=device)

    w1d = None
    procesados, vacios = 0, 0

    for bi in range(n_por_eje):
        for bj in range(n_por_eje):
            for bk in range(n_por_eje):
                # Región que este bloque RESUELVE (núcleo + halo a cada lado)
                ini = np.array([bi, bj, bk]) * nucleo - halo
                fin = ini + block

                # Puntos dentro de la región resuelta
                dentro = np.all((celdas >= ini) & (celdas < fin), axis=1)
                n_dentro = int(dentro.sum())
                if n_dentro < 32:
                    vacios += 1
                    continue

                idx = torch.as_tensor(np.flatnonzero(dentro), device=device)
                p_loc = (t_pts_todos[idx] * resolution -
                         torch.as_tensor(ini, dtype=torch.float32,
                                         device=device)) / block
                n_loc = t_nrm_todos[idx]

                chi_b = solve_poisson(p_loc, n_loc, res=block, sigma=sigma,
                                      normalizar=True)

                # Convenio común: exterior positivo en todos los bloques.
                chi_b = chi_b * _signo_exterior(chi_b, p_loc, n_loc, block)

                if w1d is None:
                    w1d = _ventana_mezcla(block, halo, device, chi_b.dtype)
                ventana = (w1d[:, None, None] *
                           w1d[None, :, None] *
                           w1d[None, None, :])

                # Cada bloque solo opina cerca de sus propios datos.
                if radio_datos > 0:
                    ventana = ventana * _mascara_cerca_de_datos(
                        p_loc, block, radio_datos, device)

                # Recorte a la parte del bloque que cae dentro de la grilla
                g_ini = np.maximum(ini, 0)
                g_fin = np.minimum(fin, resolution)
                if np.any(g_fin <= g_ini):
                    continue
                l_ini = g_ini - ini
                l_fin = l_ini + (g_fin - g_ini)

                trozo = (chi_b * ventana)[l_ini[0]:l_fin[0],
                                          l_ini[1]:l_fin[1],
                                          l_ini[2]:l_fin[2]]
                peso  = ventana[l_ini[0]:l_fin[0],
                                l_ini[1]:l_fin[1],
                                l_ini[2]:l_fin[2]]

                chi_global[g_ini[0]:g_fin[0], g_ini[1]:g_fin[1],
                           g_ini[2]:g_fin[2]] += trozo.cpu().numpy()
                peso_global[g_ini[0]:g_fin[0], g_ini[1]:g_fin[1],
                            g_ini[2]:g_fin[2]] += peso.cpu().numpy()

                procesados += 1
                del chi_b, trozo, peso, p_loc, n_loc, idx
                if device == "cuda":
                    torch.cuda.empty_cache()

                if procesados % 25 == 0:
                    log(f"      · {procesados} bloques con datos resueltos "
                        f"({vacios} vacíos omitidos)")

    log(f"    ▦ {procesados} bloques resueltos, {vacios} vacíos omitidos "
        f"(el aire no se calcula)")

    if procesados == 0:
        raise RuntimeError(
            "Ningún bloque contenía puntos suficientes. Prueba con una "
            "resolución menor o un tamaño de bloque mayor."
        )

    # Promedio ponderado sobre las celdas que algún bloque respaldó.
    cubierto = peso_global > 1e-6
    chi_global[cubierto] /= peso_global[cubierto]

    log(f"    ▦ Celdas con datos: {int(cubierto.sum()):,} de "
        f"{cubierto.size:,} ({cubierto.mean():.1%})")

    # ── Aire exterior vs. cavidad encerrada ─────────────────────────────
    # Las celdas que ningún bloque respaldó están lejos de todo dato, pero
    # no todas son «fuera»: pueden ser el interior profundo de un objeto
    # macizo. Distinguirlas es lo que decide si la salida es un SÓLIDO o
    # una CÁSCARA hueca.
    #
    # El criterio es topológico: se etiquetan los componentes conexos de la
    # región sin datos; el que toca el borde del dominio es el aire de
    # fuera, y cualquier otro está encerrado por superficie, luego es
    # interior.
    exterior = float(radio_datos)
    sin_datos = ~cubierto
    n_dentro = 0
    if sin_datos.any():
        try:
            from scipy import ndimage
            etiquetas, n = ndimage.label(sin_datos)
            if n > 0:
                # Etiquetas presentes en las seis caras del dominio
                caras = np.concatenate([
                    etiquetas[0].ravel(),  etiquetas[-1].ravel(),
                    etiquetas[:, 0].ravel(),  etiquetas[:, -1].ravel(),
                    etiquetas[:, :, 0].ravel(), etiquetas[:, :, -1].ravel(),
                ])
                fuera = np.unique(caras)
                fuera = fuera[fuera > 0]

                es_fuera = np.zeros(n + 1, dtype=bool)
                es_fuera[fuera] = True
                mapa_fuera = es_fuera[etiquetas]

                chi_global[sin_datos & mapa_fuera]  = exterior
                dentro = sin_datos & ~mapa_fuera
                n_dentro = int(dentro.sum())
                chi_global[dentro] = -exterior
                del etiquetas, mapa_fuera, dentro
            else:
                chi_global[sin_datos] = exterior
        except ImportError:
            chi_global[sin_datos] = exterior

    if n_dentro:
        log(f"    ▦ Cavidades encerradas rellenadas como interior: "
            f"{n_dentro:,} celdas (la salida es maciza)")
    else:
        log(f"    ▦ Sin cavidades encerradas: la salida es una lámina "
            f"(correcto en escenas abiertas)")

    del peso_global
    return chi_global, exterior


# ══════════════════════════════════════════════════════════════
#  API DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════

# Margen alrededor del objeto dentro de la grilla. La FFT es periódica: sin
# margen, el objeto se «pega» consigo mismo por los bordes opuestos.
_MARGEN = 0.20


def reconstruct(pcd: o3d.geometry.PointCloud,
                resolution: int = 128,
                sigma: float = 1.5,
                device: str | None = None,
                block_size: int = 0,
                overlap: float = 0.25,
                keep_largest: bool = True,
                close_border: bool = True,
                data_mask: str = "auto",
                log_callback=None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """
    Reconstruye una malla desde una nube con normales orientadas.

    Parameters
    ----------
    resolution   : lado de la grilla global. 512³ y 1024³ solo son viables
                   con troceado (ver `block_size`).
    sigma        : suavizado espectral en celdas.
    device       : "cuda" | "cpu" | None (automático).
    block_size   : lado del bloque que se resuelve de una vez en la GPU.
                   0 = automático: usa la resolución completa si cabe en
                   memoria, y si no elige el mayor bloque que quepa.
    overlap      : fracción de solape entre bloques (0.25 = 25 %).
    keep_largest : conservar solo el componente conexo mayor. Conviene
                   desactivarlo en escenas con varios objetos separados.
    close_border : forzar «exterior» en la capa externa de la grilla, para
                   que la superficie no quede abierta si toca el borde.
    data_mask    : "auto" | "on" | "off". Ver la nota de abajo; decide si el
                   resultado es una lámina fiel o un macizo extrapolado.

    El compromiso de la máscara de datos
    ────────────────────────────────────
    Con la máscara activa, cada bloque solo aporta cerca de sus propios
    puntos. Eso suprime la superficie inventada en el aire —imprescindible
    en una escena abierta, donde midiendo sobre un escaneo LiDAR industrial
    bajó el error de 6,5 % a 0,56 %— pero es EL MISMO mecanismo con el que
    Poisson rellena los huecos del muestreo.

    Es decir, no se puede tener las dos cosas:

    - máscara ON  → la superficie sigue a los datos y no inventa nada, pero
                    no cierra los huecos: el resultado es una LÁMINA. Correcto
                    para tuberías, paredes y escenas escaneadas de un lado.
    - máscara OFF → Poisson extrapola sobre los huecos y devuelve un MACIZO,
                    a costa de inventar envolvente donde no hay datos.
                    Correcto para un objeto compacto escaneado por completo.

    "auto" la activa solo al trocear, que es cuando se está reconstruyendo
    algo grande, típicamente una escena.

    Returns
    -------
    mesh  : TriangleMesh
    stats : dict con resolución, bloques, tiempos y diagnóstico
    """
    import time
    import torch

    log = log_callback or print

    pts = np.asarray(pcd.points)
    if not pcd.has_normals():
        raise ValueError(
            "DPSR necesita normales orientadas. Ejecuta antes el paso A "
            "(preprocesamiento), que las estima y las orienta."
        )
    nrm = np.asarray(pcd.normals)

    device = device or dispositivo_por_defecto()
    res    = int(resolution)

    # Normalización a [0,1)³ conservando la relación de aspecto
    lo, hi = pts.min(0), pts.max(0)
    centro = (lo + hi) / 2.0
    escala = float((hi - lo).max()) * (1.0 + _MARGEN)
    if escala <= 0:
        raise ValueError("La nube es degenerada (extensión nula).")
    pts_n = (pts - centro) / escala + 0.5

    # ── Decidir si hace falta trocear ───────────────────────────────────
    bloque = int(block_size) if block_size and block_size > 0 else \
        bloque_automatico(res, device)
    bloque = min(bloque, res)
    trocear = bloque < res

    celda_mundo = escala / res
    log(f"  ℹ️  DPSR (grilla {res}³, σ={sigma}, {device}) — "
        f"celda de {celda_mundo:.4f} unidades")
    if trocear:
        log(f"    ⚠️  {memoria_estimada_gb(res):.1f} GB no caben en la GPU: "
            f"se resuelve por bloques de {bloque}³ con {overlap:.0%} de "
            f"solape")

    # ── Resolver ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    exterior = None
    if trocear:
        usar_mascara = (str(data_mask).lower() != "off")
        radio = (radio_datos_recomendado(pts_n, res) if usar_mascara else 0)
        if usar_mascara:
            log("    ▦ Máscara de datos ACTIVA: la superficie seguirá a los "
                "puntos sin inventar envolvente, pero NO cerrará los huecos "
                "del muestreo (salida en lámina, no maciza).")
        else:
            log("    ▦ Máscara de datos DESACTIVADA: Poisson extrapolará "
                "sobre los huecos (salida maciza), a costa de inventar "
                "superficie donde no hay datos.")
        chi_np, exterior = _resolver_por_bloques(
            pts_n, nrm, res, float(sigma), bloque, float(overlap),
            device, log, radio_datos=radio)
    else:
        t_pts = torch.as_tensor(pts_n, dtype=torch.float32, device=device)
        t_nrm = torch.as_tensor(nrm,   dtype=torch.float32, device=device)
        chi = solve_poisson(t_pts, t_nrm, res=res, sigma=float(sigma))
        if device == "cuda":
            torch.cuda.synchronize()
        chi_np = chi.detach().cpu().numpy()
        del chi, t_pts, t_nrm
        if device == "cuda":
            torch.cuda.empty_cache()
    t_solve = time.perf_counter() - t0

    # ── Cierre en el borde del dominio ──────────────────────────────────
    # El nivel cero de una función continua es cerrado SOLO si no toca el
    # borde del dominio. Con un objeto pequeño y centrado nunca lo toca,
    # pero una escena abierta (un suelo que llega al límite) sí, y entonces
    # la malla sale con frontera.
    if close_border:
        if exterior is None:
            exterior = float(np.percentile(chi_np, 99.0))
        if not np.isfinite(exterior) or exterior <= 0:
            exterior = float(np.abs(chi_np).max()) or 1.0
        chi_np = _cerrar_en_el_borde(chi_np, exterior)

    # ── Extracción de la superficie ─────────────────────────────────────
    t0 = time.perf_counter()
    verts, faces = _marching_por_bloques(chi_np, level=0.0, log=log)
    t_mc = time.perf_counter() - t0

    if len(verts) == 0:
        raise RuntimeError(
            "El campo no cruza el nivel cero: no hay superficie que "
            "extraer. Suele indicar normales mal orientadas o una "
            "resolución demasiado baja para el detalle de la nube."
        )

    # De índices de grilla a coordenadas del mundo
    verts = (verts / res - 0.5) * escala + centro

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Imprescindible tras el marching por trozos: fusiona los vértices
    # duplicados del plano que comparten dos trozos vecinos.
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    idx, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    n_comp = len(counts)
    if keep_largest and n_comp > 1:
        idx = np.asarray(idx)
        mesh.remove_triangles_by_mask(idx != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        log(f"    🧹 Componentes: {n_comp} → 1 "
            f"({int(counts.max()):,} tri)")
    elif n_comp > 1:
        log(f"    ℹ️  {n_comp} componentes conservados "
            f"(mayor: {int(counts.max()):,} tri)")

    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    log(f"    ✓ Poisson espectral en {t_solve:.1f}s, "
        f"superficie en {t_mc:.1f}s")

    return mesh, {
        "dpsr_resolution" : res,
        "dpsr_sigma"      : float(sigma),
        "dpsr_device"     : device,
        "dpsr_block_size" : bloque,
        "dpsr_overlap"    : float(overlap) if trocear else 0.0,
        "dpsr_tiled"      : bool(trocear),
        "dpsr_data_mask"  : (str(data_mask).lower() != "off") and trocear,
        "dpsr_cell_size"  : round(celda_mundo, 6),
        "dpsr_solve_s"    : round(t_solve, 3),
        "dpsr_mc_s"       : round(t_mc, 3),
        "dpsr_components" : n_comp,
    }
