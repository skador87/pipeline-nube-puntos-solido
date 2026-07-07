# core/preprocessor.py
# Preprocesamiento de nubes de puntos — v3
#
# Objetivo del flujo: nube de puntos (escáner láser) → nube limpia, de
# densidad uniforme y con normales bien orientadas, lista para
# reconstrucción de superficie (Poisson / Ball Pivoting / Alpha Shape).
#
# Principios de diseño
# ────────────────────
# 1. ESCALA RELATIVA: se calcula una escala característica de la nube
#    (mediana de la distancia al vecino más cercano, d̄) y TODOS los
#    umbrales espaciales se expresan como múltiplos de d̄. Así el mismo
#    preproceso funciona con el bunny (escala ≈ 1) y con un escaneo láser
#    en metros, sin tocar parámetros.
# 2. ORDEN ORIENTADO A CALIDAD: limpiar antes de bajar resolución.
#       validar → escala → dedup → ROR → SOR → voxel → denoise → normales
# 3. NO DESTRUIR GEOMETRÍA: el ruido se reduce reposicionando puntos
#    (MLS edge-aware), no eliminando zonas de alta curvatura (aristas).
# 4. REPRODUCIBILIDAD: todo muestreo aleatorio usa una semilla fija.
#
# Firma compatible con main_gui.py: run(params) -> (pcd, stats)

import numpy as np
import open3d as o3d
from copy import deepcopy
from scipy.spatial import cKDTree


class PointCloudPreprocessor:
    """Preprocesa una nube de puntos para reconstrucción de superficie."""

    # ── Valores por defecto (espejo del ParamPanel de main_gui.py) ─────────
    _DEFAULTS = {
        # Validación
        "remove_invalid"      : True,   # quitar NaN / Inf
        "remove_origin"       : True,   # quitar puntos exactamente en (0,0,0)

        # Deduplicación  (umbral = dedup_factor · d̄)
        "dedup_enabled"       : True,
        "dedup_factor"        : 0.2,

        # Radius Outlier Removal  (radio = ror_factor · d̄)
        "ror_enabled"         : True,
        "ror_factor"          : 2.5,
        "ror_min_neighbors"   : 4,

        # Statistical Outlier Removal
        "sor_enabled"         : True,
        "sor_k"               : 20,
        "sor_std_ratio"       : 2.0,

        # Voxelización  (voxel = voxel_factor · d̄)
        "voxel_enabled"       : True,
        "voxel_factor"        : 1.5,

        # Suavizado MLS edge-aware  (reemplaza al antiguo filtro de rugosidad)
        "denoise_enabled"     : True,
        "denoise_iterations"  : 1,
        "denoise_k"           : 16,     # vecinos por punto
        "preserve_edges"      : True,   # reduce suavizado donde hay curvatura

        # Normales  (radio = normal_factor · escala_local)
        "normal_factor"       : 4.0,
        "normal_max_nn"       : 30,
        "orient_k"            : 30,
        # Si se entrega, se orientan las normales hacia este punto (sensor).
        # Para PLY/bunny suele ser None → orientación geométrica robusta.
        "sensor_center"       : None,

        # Reproducibilidad
        "random_seed"         : 42,
    }

    def __init__(self, point_cloud: o3d.geometry.PointCloud, log_callback=None):
        self.pcd          = deepcopy(point_cloud)
        self.log_callback = log_callback or print
        self._rng         = np.random.default_rng(self._DEFAULTS["random_seed"])
        self.dbar         = None   # escala característica, se calcula en run()

    # ══════════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════════

    def run(self, params: dict | None = None
            ) -> tuple[o3d.geometry.PointCloud, dict]:
        p = {**self._DEFAULTS, **(params or {})}
        self._rng = np.random.default_rng(int(p["random_seed"]))

        stats = {
            "original_points" : len(self.pcd.points),
            "after_validate"  : None,
            "after_dedup"     : None,
            "after_ror"       : None,
            "after_sor"       : None,
            "after_voxel"     : None,
            "final_points"    : None,
            "total_removed"   : None,
            "d_bar"           : None,
            "voxel_size"      : None,
        }
        self._log(f"  Puntos iniciales: {stats['original_points']:,}")

        if stats["original_points"] < 4:
            self._log("  ⚠️  Nube demasiado pequeña, se omite el preproceso")
            stats["final_points"] = stats["original_points"]
            stats["total_removed"] = 0
            return self.pcd, stats

        # ── Paso 0: Validación / sanidad ───────────────────────────────────
        self._validate(p["remove_invalid"], p["remove_origin"])
        stats["after_validate"] = len(self.pcd.points)
        self._log(f"  ✓ Validación → {stats['after_validate']:,} puntos")

        # ── Paso 1: Escala característica ───────────────────────────────────
        self.dbar = self._characteristic_scale()
        stats["d_bar"] = self.dbar
        self._log(f"  ℹ️  Escala característica d̄ = {self.dbar:.6f}")

        # ── Paso 2: Deduplicación  (relativa a d̄) ──────────────────────────
        if p["dedup_enabled"]:
            thr = p["dedup_factor"] * self.dbar
            self._deduplicate(thr)
        stats["after_dedup"] = len(self.pcd.points)
        self._log(f"  ✓ Dedup ({'ON' if p['dedup_enabled'] else 'OFF'}) "
                  f"→ {stats['after_dedup']:,} puntos")

        # ── Paso 3: ROR — fliers aislados (antes de SOR) ────────────────────
        if p["ror_enabled"]:
            self._remove_radius_outliers(
                radius    = p["ror_factor"] * self.dbar,
                nb_points = int(p["ror_min_neighbors"]),
            )
        stats["after_ror"] = len(self.pcd.points)
        self._log(f"  ✓ ROR ({'ON' if p['ror_enabled'] else 'OFF'}) "
                  f"→ {stats['after_ror']:,} puntos")

        # ── Paso 4: SOR — ruido estadístico de superficie ───────────────────
        if p["sor_enabled"]:
            self._remove_statistical_outliers(
                nb_neighbors = int(p["sor_k"]),
                std_ratio    = float(p["sor_std_ratio"]),
            )
        stats["after_sor"] = len(self.pcd.points)
        self._log(f"  ✓ SOR ({'ON' if p['sor_enabled'] else 'OFF'}) "
                  f"→ {stats['after_sor']:,} puntos")

        # ── Paso 5: Voxelización — uniformar densidad ───────────────────────
        voxel_size = p["voxel_factor"] * self.dbar
        if p["voxel_enabled"]:
            self._voxelize(voxel_size)
            stats["voxel_size"] = voxel_size
        stats["after_voxel"] = len(self.pcd.points)
        self._log(f"  ✓ Voxel ({'ON' if p['voxel_enabled'] else 'OFF'}, "
                  f"{voxel_size:.6f}) → {stats['after_voxel']:,} puntos")

        # ── Paso 6: Suavizado MLS edge-aware (no destructivo) ───────────────
        if p["denoise_enabled"]:
            self._denoise_mls(
                k             = int(p["denoise_k"]),
                iterations    = int(p["denoise_iterations"]),
                preserve_edges= bool(p["preserve_edges"]),
            )
            self._log(f"  ✓ Suavizado MLS ({p['denoise_iterations']} iter, "
                      f"aristas={'sí' if p['preserve_edges'] else 'no'})")

        # ── Paso 7: Normales + orientación ──────────────────────────────────
        ref = voxel_size if p["voxel_enabled"] else self.dbar
        self._estimate_normals(
            radius        = p["normal_factor"] * ref,
            max_nn        = int(p["normal_max_nn"]),
            orient_k      = int(p["orient_k"]),
            sensor_center = p["sensor_center"],
        )

        # ── Totales ─────────────────────────────────────────────────────────
        stats["final_points"]  = len(self.pcd.points)
        stats["total_removed"] = stats["original_points"] - stats["final_points"]
        self._log(f"  Puntos finales   : {stats['final_points']:,}")
        self._log(f"  Puntos eliminados: {stats['total_removed']:,}")

        return self.pcd, stats

    # ══════════════════════════════════════════════════════════════
    #  PASOS
    # ══════════════════════════════════════════════════════════════

    def _validate(self, remove_invalid: bool, remove_origin: bool):
        """Quita puntos no finitos (NaN/Inf) y, opcionalmente, los del origen."""
        points = np.asarray(self.pcd.points)
        mask = np.ones(len(points), dtype=bool)

        if remove_invalid:
            mask &= np.isfinite(points).all(axis=1)

        if remove_origin:
            # Puntos exactamente en (0,0,0): returns inválidos típicos de láser
            eps = 1e-12
            at_origin = (np.abs(points) < eps).all(axis=1)
            mask &= ~at_origin

        if not mask.all():
            self._apply_mask(mask)

    def _characteristic_scale(self) -> float:
        """
        Escala característica d̄ = mediana de la distancia al vecino más
        cercano, sobre una muestra (semilla fija → reproducible).
        La mediana es robusta frente a outliers (mejor que la media).
        """
        points = np.asarray(self.pcd.points)
        n = min(len(points), 20_000)
        idx = self._rng.choice(len(points), size=n, replace=False)
        sample = points[idx]

        tree = cKDTree(sample)
        dists, _ = tree.query(sample, k=2)        # [:,0] es el propio punto
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        if len(nn) == 0:
            return 1.0
        return float(np.median(nn))

    def _deduplicate(self, threshold: float):
        """
        Deduplicación por rejilla (grid hashing): un punto por celda de
        lado `threshold`. O(N), determinista y de bajo consumo de memoria
        (mucho mejor que query_pairs en nubes densas).
        """
        if threshold <= 0:
            return
        points = np.asarray(self.pcd.points)
        keys = np.floor(points / threshold).astype(np.int64)
        # Índice del primer punto que cae en cada celda
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(first_idx)

        mask = np.zeros(len(points), dtype=bool)
        mask[keep] = True
        self._apply_mask(mask)

    def _remove_radius_outliers(self, radius: float, nb_points: int):
        self._log(f"    ROR: radio={radius:.6f}, min_pts={nb_points}")
        self.pcd, _ = self.pcd.remove_radius_outlier(
            nb_points=max(1, nb_points), radius=radius,
        )

    def _remove_statistical_outliers(self, nb_neighbors: int, std_ratio: float):
        self._log(f"    SOR: k={nb_neighbors}, σ={std_ratio}")
        self.pcd, _ = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio,
        )

    def _voxelize(self, voxel_size: float):
        if voxel_size <= 0:
            return
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)

    def _denoise_mls(self, k: int, iterations: int, preserve_edges: bool):
        """
        Suavizado tipo MLS (Moving Least Squares): cada punto se proyecta
        sobre el plano local estimado por PCA con sus k vecinos. Reduce el
        ruido SIN eliminar puntos.

        Edge-aware: la fuerza del desplazamiento se atenúa donde la
        curvatura local es alta (aristas, esquinas), preservando detalle.

        Totalmente vectorizado: usa np.linalg.eigh sobre el stack (N,3,3)
        de matrices de covarianza, evitando bucles en Python.
        """
        points = np.asarray(self.pcd.points)
        n = len(points)
        if n < k + 1:
            return

        for _ in range(max(1, iterations)):
            tree = cKDTree(points)
            _, idx = tree.query(points, k=k)            # (N, k)
            neigh = points[idx]                         # (N, k, 3)

            centroid = neigh.mean(axis=1)               # (N, 3)
            centered = neigh - centroid[:, None, :]     # (N, k, 3)

            # Covarianza por punto (N,3,3) y descomposición vectorizada
            cov = np.einsum("nki,nkj->nij", centered, centered) / k
            evals, evecs = np.linalg.eigh(cov)          # ascendente
            normal = evecs[:, :, 0]                     # menor autovalor

            # Proyección del punto sobre el plano local
            vec = points - centroid                     # (N, 3)
            d   = np.einsum("ni,ni->n", vec, normal)    # dist. firmada al plano

            if preserve_edges:
                # Curvatura ≈ λ0 / (λ0+λ1+λ2). Alta → arista → menos suavizado.
                curv = evals[:, 0] / (evals.sum(axis=1) + 1e-12)
                # Normalizar curvatura a [0,1] de forma robusta (percentil 90)
                cref = np.percentile(curv, 90) + 1e-12
                strength = 1.0 - np.clip(curv / cref, 0.0, 1.0)
            else:
                strength = np.ones(n)

            points = points - (strength * d)[:, None] * normal

        new = o3d.geometry.PointCloud()
        new.points = o3d.utility.Vector3dVector(points)
        if self.pcd.has_colors():
            new.colors = self.pcd.colors            # mismo orden, sin borrados
        self.pcd = new

    def _estimate_normals(self, radius: float, max_nn: int,
                          orient_k: int, sensor_center):
        """
        Estima normales con búsqueda híbrida y las orienta de forma
        consistente. Si se conoce la posición del sensor, orienta hacia
        ella (lo más fiable en datos láser); si no, usa propagación
        geométrica por plano tangente (MST interno de Open3D).
        """
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn,
            )
        )

        if sensor_center is not None:
            center = np.asarray(sensor_center, dtype=np.float64).reshape(3)
            self.pcd.orient_normals_towards_camera_location(center)
            self._log(f"  ✓ Normales orientadas hacia el sensor {center.tolist()}")
        else:
            try:
                self.pcd.orient_normals_consistent_tangent_plane(k=orient_k)
                self._log("  ✓ Normales orientadas (plano tangente consistente)")
            except Exception as e:
                self._log(f"  ⚠️  Orientación geométrica falló ({e}); "
                          f"se mantienen normales sin orientar")

    # ══════════════════════════════════════════════════════════════
    #  UTILIDADES
    # ══════════════════════════════════════════════════════════════

    def _apply_mask(self, mask: np.ndarray):
        """Reconstruye self.pcd conservando solo los puntos de `mask`,
        preservando colores y normales si existen."""
        points = np.asarray(self.pcd.points)
        new = o3d.geometry.PointCloud()
        new.points = o3d.utility.Vector3dVector(points[mask])
        if self.pcd.has_colors():
            new.colors = o3d.utility.Vector3dVector(np.asarray(self.pcd.colors)[mask])
        if self.pcd.has_normals():
            new.normals = o3d.utility.Vector3dVector(np.asarray(self.pcd.normals)[mask])
        self.pcd = new

    def _log(self, msg: str):
        self.log_callback(msg)
