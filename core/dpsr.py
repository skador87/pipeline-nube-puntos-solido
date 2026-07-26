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
    try:
        import torch                                    # noqa: F401
    except ImportError:
        faltan.append("torch")
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

    import torch
    if torch.cuda.is_available():
        return True, (f"Método DPSR disponible en GPU "
                      f"({torch.cuda.get_device_name(0)})")
    return True, "Método DPSR disponible (solo CPU: será más lento)"


def dispositivo_por_defecto() -> str:
    """'cuda' si hay GPU utilizable, 'cpu' en otro caso."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


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


def solve_poisson(pts, normals, res: int, sigma: float = 2.0):
    """
    Resuelve la ecuación de Poisson en el dominio espectral.

    pts     : (N,3) en [0,1)
    normals : (N,3) normales orientadas
    res     : lado de la grilla
    sigma   : ancho del suavizado espectral, en celdas

    Devuelve el indicador χ como tensor (res,res,res). Es la función cuyo
    nivel cero es la superficie.
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
    return chi - grid_interp(chi, pts).mean()


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
                log_callback=None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """
    Reconstruye una malla cerrada desde una nube con normales orientadas.

    Returns
    -------
    mesh  : TriangleMesh (cerrada por construcción)
    stats : dict con resolución, sigma, dispositivo y tiempos
    """
    import torch
    from skimage.measure import marching_cubes

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

    log(f"  ℹ️  DPSR (grilla {res}³, σ={sigma}, {device})...")

    import time
    t0 = time.perf_counter()
    t_pts = torch.as_tensor(pts_n, dtype=torch.float32, device=device)
    t_nrm = torch.as_tensor(nrm,   dtype=torch.float32, device=device)
    chi = solve_poisson(t_pts, t_nrm, res=res, sigma=float(sigma))
    if device == "cuda":
        torch.cuda.synchronize()
    t_solve = time.perf_counter() - t0

    chi_np = chi.detach().cpu().numpy()
    del chi, t_pts, t_nrm
    if device == "cuda":
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(chi_np, level=0.0)
    t_mc = time.perf_counter() - t0

    # De índices de grilla a coordenadas del mundo
    verts = (verts / res - 0.5) * escala + centro

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # La grilla puede producir burbujas sueltas lejos del objeto cuando el
    # suavizado es bajo; se conserva solo el componente principal.
    idx, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    n_comp = len(counts)
    if n_comp > 1:
        idx = np.asarray(idx)
        mesh.remove_triangles_by_mask(idx != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
        log(f"    🧹 Componentes: {n_comp} → 1 "
            f"({int(counts.max()):,} tri)")

    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    log(f"    ✓ Poisson espectral en {t_solve:.2f}s, "
        f"marching cubes en {t_mc:.2f}s")

    stats = {
        "dpsr_resolution" : res,
        "dpsr_sigma"      : float(sigma),
        "dpsr_device"     : device,
        "dpsr_solve_s"    : round(t_solve, 3),
        "dpsr_mc_s"       : round(t_mc, 3),
        "dpsr_components" : n_comp,
    }
    return mesh, stats
