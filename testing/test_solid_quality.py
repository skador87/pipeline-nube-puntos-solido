# testing/test_solid_quality.py
# Harness de validación de calidad del sólido final.
#
# Corre el pipeline completo (preprocesamiento → reconstrucción → solidificación)
# sobre un dataset y reporta métricas de fidelidad geométrica calculadas de forma
# independiente al código de producción:
#
#   - watertight / orientable del sólido
#   - estrategia de solidificación usada
#   - error de Chamfer sólido↔malla reconstruida (media y p95), normalizado
#     por la diagonal del bounding box de la malla reconstruida
#   - volumen del sólido (solo si es watertight)
#
# Uso:
#   python testing/test_solid_quality.py <nube.ply> [--max-points N]
#       [--force-fallback] [--no-hull] [--quiet]
#
#   --force-fallback : desactiva la Estrategia 1 (reparación) para forzar el
#                      camino de fallback (voxel → hull) y medir cuánta
#                      fidelidad se pierde en ese camino.
#   --no-hull        : además desactiva el Convex Hull (aísla estrategias intermedias).

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import PointCloudPreprocessor
from core.reconstructor import MeshReconstructor
from core.solidifier import MeshSolidifier


def is_closed(mesh: o3d.geometry.TriangleMesh) -> bool:
    """
    Cierre topológico O(T) (manifold sin frontera). No se usa
    o3d.is_watertight() porque incluye un test de auto-intersecciones de
    coste cuadrático, inviable en mallas cerradas grandes.
    """
    return (len(mesh.triangles) > 0
            and mesh.is_edge_manifold(allow_boundary_edges=False)
            and mesh.is_vertex_manifold())


def signed_volume(mesh: o3d.geometry.TriangleMesh):
    """Volumen por suma de tetraedros firmados (requiere malla cerrada)."""
    if not is_closed(mesh):
        return None
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return abs(float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0))


def chamfer_metrics(mesh_ref: o3d.geometry.TriangleMesh,
                    mesh_eval: o3d.geometry.TriangleMesh,
                    n_samples: int = 30_000) -> dict:
    """
    Distancia de Chamfer simétrica entre dos mallas, por muestreo uniforme.
    Devuelve media, p95 y máximo, normalizados por la diagonal bbox de mesh_ref.
    """
    diag = np.linalg.norm(
        mesh_ref.get_max_bound() - mesh_ref.get_min_bound()
    )
    pts_ref  = np.asarray(mesh_ref.sample_points_uniformly(n_samples).points)
    pts_eval = np.asarray(mesh_eval.sample_points_uniformly(n_samples).points)

    d_re = cKDTree(pts_eval).query(pts_ref, k=1)[0]   # ref → eval
    d_er = cKDTree(pts_ref).query(pts_eval, k=1)[0]   # eval → ref
    d_all = np.concatenate([d_re, d_er])

    return {
        "chamfer_mean" : float(d_all.mean() / diag),
        "chamfer_p95"  : float(np.percentile(d_all, 95) / diag),
        "chamfer_max"  : float(d_all.max() / diag),
        "bbox_diag"    : float(diag),
    }


def run_pipeline(path: str, max_points: int, sol_params: dict,
                 verbose: bool = True) -> dict:
    log = print if verbose else (lambda *_: None)

    pcd = o3d.io.read_point_cloud(path)
    n0 = len(pcd.points)
    if max_points and n0 > max_points:
        every = int(np.ceil(n0 / max_points))
        pcd = pcd.uniform_down_sample(every)
        log(f"[harness] Nube reducida para el test: {n0:,} → {len(pcd.points):,} pts")

    t0 = time.perf_counter()
    pre = PointCloudPreprocessor(pcd, log_callback=log)
    pcd_clean, pre_stats = pre.run(None)
    t_pre = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec = MeshReconstructor(pcd_clean, log_callback=log)
    mesh_rec, rec_stats = rec.run(None)
    t_rec = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol = MeshSolidifier(mesh_rec, log_callback=log)
    mesh_solid, sol_stats = sol.run(sol_params)
    t_sol = time.perf_counter() - t0

    metrics = {
        "dataset"          : os.path.basename(path),
        "points_in"        : len(pcd.points),
        "rec_triangles"    : len(mesh_rec.triangles),
        "solid_triangles"  : len(mesh_solid.triangles),
        "strategy_used"    : sol_stats.get("strategy_used"),
        "is_closed"        : is_closed(mesh_solid),
        "is_orientable"    : bool(mesh_solid.is_orientable()),
        "fidelity_stats"   : sol_stats.get("fidelity_error"),
        "volume"           : signed_volume(mesh_solid),
        "volume_rec"       : signed_volume(mesh_rec),
        "t_pre_s"          : round(t_pre, 1),
        "t_rec_s"          : round(t_rec, 1),
        "t_sol_s"          : round(t_sol, 1),
    }
    if len(mesh_solid.triangles) > 0 and len(mesh_rec.triangles) > 0:
        metrics.update(chamfer_metrics(mesh_rec, mesh_solid))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--force-fallback", action="store_true")
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sol_params = {}
    if args.force_fallback:
        sol_params["strategy_repair"] = False
    if args.no_hull:
        sol_params["strategy_hull"] = False

    m = run_pipeline(args.path, args.max_points, sol_params,
                     verbose=not args.quiet)

    print("\n════════ RESUMEN MÉTRICAS ════════")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.6f}")
        else:
            print(f"  {k:<18}: {v}")


if __name__ == "__main__":
    main()
