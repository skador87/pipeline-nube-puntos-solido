# core/reconstructor.py
# Extraído de PointCloudProcessor.reconstruct_mesh() y métodos de limpieza
# Actualizado: firma run(params) compatible con main_gui.py WorkerThread (Bloque B)

import numpy as np
import open3d as o3d
from copy import deepcopy
from scipy.spatial import cKDTree


class MeshReconstructor:
    """
    Reconstruye una malla desde una nube de puntos preprocesada.

    Firma run(params) compatible con WorkerThread de main_gui.py

    Parámetros esperados en el dict params
    ──────────────────────────────────────
    method               : str   — "poisson" | "ball_pivoting" | "alpha_shape"
    poisson_depth        : int   — profundidad octree Poisson
    bp_radius            : float — radio base Ball Pivoting
    alpha                : float — valor alpha para Alpha Shape
    alpha_downsample     : int   — factor submuestreo Alpha Shape
    remove_webbing       : bool  — activar filtro phantom webbing
    remove_hollow        : bool  — activar filtro zonas huecas
    smooth_method        : str   — "none" | "taubin" | "laplacian"
    smooth_iterations    : int   — iteraciones de suavizado
    taubin_lambda        : float — λ para Taubin
    taubin_mu            : float — μ para Taubin
    laplacian_lambda     : float — λ para Laplacian
    """

    # ── Valores por defecto (espejo de ParamPanel en main_gui.py) ──────────
    _DEFAULTS = {
        "method"            : "poisson",
        "poisson_depth"     : 10,
        "bp_radius"         : 1.0,
        "alpha"             : 0.1,
        "alpha_downsample"  : 1,
        "remove_webbing"    : True,
        "remove_hollow"     : True,
        "smooth_method"     : "taubin",
        "smooth_iterations" : 5,
        "taubin_lambda"     : 0.5,
        "taubin_mu"         : -0.53,
        "laplacian_lambda"  : 0.5,
    }

    def __init__(self, point_cloud: o3d.geometry.PointCloud,
                 log_callback=None):
        """
        Parameters
        ----------
        point_cloud  : o3d.geometry.PointCloud
            Nube preprocesada (se trabaja sobre una copia interna).
        log_callback : callable | None
            Función que recibe strings de log. Si es None usa print().
        """
        self.pcd          = deepcopy(point_cloud)
        self.log_callback = log_callback or print
        self.mesh         = None

    # ══════════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════════

    def run(self, params: dict | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
        """
        Ejecuta el pipeline completo de reconstrucción.

        Parameters
        ----------
        params : dict | None
            Diccionario de parámetros (ver _DEFAULTS).
            Las claves faltantes se completan con _DEFAULTS.

        Returns
        -------
        mesh  : o3d.geometry.TriangleMesh
            Malla reconstruida y limpiada.
        stats : dict
            Estadísticas con las claves:
            - method
            - initial_vertices
            - initial_triangles
            - after_webbing_triangles
            - after_hollow_triangles
            - after_smooth_vertices
            - final_vertices
            - final_triangles
        """
        # ── Combinar params con defaults ───────────────────────────────────
        p = {**self._DEFAULTS, **(params or {})}

        # ── Estadísticas acumuladas ────────────────────────────────────────
        stats = {
            "method"                 : p["method"],
            "initial_vertices"       : 0,
            "initial_triangles"      : 0,
            "after_webbing_triangles": None,
            "after_hollow_triangles" : None,
            "after_smooth_vertices"  : None,
            "final_vertices"         : 0,
            "final_triangles"        : 0,
        }

        # ── Paso 1: Reconstrucción ─────────────────────────────────────────
        densities = None

        if p["method"] == "poisson":
            self.mesh, densities = self._poisson(int(p["poisson_depth"]))

        elif p["method"] == "ball_pivoting":
            self.mesh = self._ball_pivoting(float(p["bp_radius"]))

        elif p["method"] == "alpha_shape":
            self.mesh = self._alpha_shape(
                float(p["alpha"]),
                int(p["alpha_downsample"]),
            )
        else:
            raise ValueError(f"Método desconocido: {p['method']}")

        stats["initial_vertices"]  = len(self.mesh.vertices)
        stats["initial_triangles"] = len(self.mesh.triangles)
        self._log(f"  ✓ Reconstrucción inicial: "
                  f"{stats['initial_vertices']:,} vért, "
                  f"{stats['initial_triangles']:,} tri")

        # ── Paso 2: Filtro phantom webbing ────────────────────────────────
        if p["remove_webbing"] and densities is not None:
            self._log("  🧹 Eliminando phantom webbing...")
            self.mesh = self._filter_phantom_webbing(self.mesh, densities)
        stats["after_webbing_triangles"] = len(self.mesh.triangles)

        # ── Paso 3: Filtro zonas huecas ────────────────────────────────────
        if p["remove_hollow"]:
            self._log("  🧹 Eliminando zonas huecas...")
            self.mesh = self._filter_by_mesh_thickness(self.mesh)
        stats["after_hollow_triangles"] = len(self.mesh.triangles)

        # ── Paso 4: Suavizado ──────────────────────────────────────────────
        self.mesh = self._smooth(
            self.mesh,
            method     = p["smooth_method"],
            iterations = int(p["smooth_iterations"]),
            t_lambda   = float(p["taubin_lambda"]),
            t_mu       = float(p["taubin_mu"]),
            l_lambda   = float(p["laplacian_lambda"]),
        )
        stats["after_smooth_vertices"] = len(self.mesh.vertices)

        # ── Paso 5: Limpieza final común ──────────────────────────────────
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        self.mesh.compute_vertex_normals()
        self.mesh.orient_triangles()

        # ── Totales ────────────────────────────────────────────────────────
        stats["final_vertices"]  = len(self.mesh.vertices)
        stats["final_triangles"] = len(self.mesh.triangles)

        self._log(f"  ✓ Malla final: "
                  f"{stats['final_vertices']:,} vértices, "
                  f"{stats['final_triangles']:,} triángulos")

        return self.mesh, stats

    # ══════════════════════════════════════════════════════════════
    #  MÉTODOS DE RECONSTRUCCIÓN
    # ══════════════════════════════════════════════════════════════

    def _poisson(
        self, depth: int
    ) -> tuple[o3d.geometry.TriangleMesh, o3d.utility.DoubleVector]:
        """
        Reconstrucción Poisson.

        Returns
        -------
        mesh      : TriangleMesh
        densities : DoubleVector  — requerido por _filter_phantom_webbing
        """
        self._log(f"  ℹ️  Poisson (depth={depth})...")
        mesh, densities = \
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.pcd, depth=depth
            )
        self._log(f"    Inicial: {len(mesh.vertices):,} vért, "
                  f"{len(mesh.triangles):,} tri")
        return mesh, densities

    def _ball_pivoting(self, radius: float) -> o3d.geometry.TriangleMesh:
        """Ball Pivoting con dos radios (radio y radio×2)."""
        self._log(f"  ℹ️  Ball Pivoting (radius={radius})...")
        radii = o3d.utility.DoubleVector([radius, radius * 2])
        mesh  = o3d.geometry.TriangleMesh \
                    .create_from_point_cloud_ball_pivoting(self.pcd, radii)
        self._log(f"    Inicial: {len(mesh.vertices):,} vért, "
                  f"{len(mesh.triangles):,} tri")
        return mesh

    def _alpha_shape(
        self, alpha: float, downsample_factor: int
    ) -> o3d.geometry.TriangleMesh:
        """Alpha Shape con submuestreo opcional."""
        self._log(f"  ℹ️  Alpha Shape (alpha={alpha}, "
                  f"downsample={downsample_factor})...")
        pcd_work = (self.pcd.uniform_down_sample(downsample_factor)
                    if downsample_factor > 1 else self.pcd)
        mesh = o3d.geometry.TriangleMesh \
                   .create_from_point_cloud_alpha_shape(pcd_work, alpha=alpha)
        self._log(f"    Inicial: {len(mesh.vertices):,} vért, "
                  f"{len(mesh.triangles):,} tri")
        return mesh

    # ══════════════════════════════════════════════════════════════
    #  LIMPIEZA DE ARTEFACTOS
    # ══════════════════════════════════════════════════════════════

    def _filter_phantom_webbing(
        self,
        mesh: o3d.geometry.TriangleMesh,
        densities: o3d.utility.DoubleVector,
    ) -> o3d.geometry.TriangleMesh:
        """
        Elimina tela fantasma y zonas de baja densidad.
        Lógica idéntica al original.
        """
        vertices  = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(densities) == 0 or len(triangles) == 0:
            return mesh

        densities_np   = np.asarray(densities)
        density_q10    = np.percentile(densities_np, 10)
        density_median = np.median(densities_np)

        self._log(f"    📊 Densidad — mediana: {density_median:.6f}, "
                  f"Q10: {density_q10:.6f}")

        cloud_points  = np.asarray(self.pcd.points)
        tree          = cKDTree(cloud_points)

        distances     = self.pcd.compute_nearest_neighbor_distance()
        avg_pt_dist   = np.asarray(distances).mean()

        bbox_vol = float(np.prod(
            np.max(cloud_points, axis=0) -
            np.min(cloud_points, axis=0)
        ))
        local_density_ref = len(cloud_points) / max(bbox_vol, 1e-6)
        dist_threshold    = 3.0 * avg_pt_dist

        valid_tris        = []
        low_density_count = 0
        far_count         = 0

        for idx, tri in enumerate(triangles):
            try:
                center      = vertices[tri].mean(axis=0)
                avg_density = np.mean(densities_np[tri])

                # Radio local = 2× tamaño promedio de arista
                edge_len = np.mean([
                    np.linalg.norm(
                        vertices[tri[i]] - vertices[tri[(i + 1) % 3]]
                    )
                    for i in range(3)
                ])
                radius = edge_len * 2

                nearby    = tree.query_ball_point(center, radius)
                local_den = len(nearby) / max(
                    4 / 3 * np.pi * radius ** 3, 1e-6
                )

                dist_arr = tree.query(center, k=1)[0]
                min_dist = float(dist_arr)

                remove = False
                if avg_density < density_q10 * 0.5:
                    if local_den < local_density_ref * 0.3:
                        remove = True
                        low_density_count += 1
                if min_dist > dist_threshold:
                    remove = True
                    far_count += 1

                if not remove:
                    valid_tris.append(idx)

            except Exception:
                valid_tris.append(idx)     # conservar si hay error

        if not valid_tris:
            self._log("    ⚠️  Filtrado muy agresivo — sin cambios")
            return mesh

        filtered = triangles[valid_tris]
        m2 = o3d.geometry.TriangleMesh()
        m2.vertices  = o3d.utility.Vector3dVector(vertices)
        m2.triangles = o3d.utility.Vector3iVector(filtered)
        m2.remove_unreferenced_vertices()

        removed = len(triangles) - len(valid_tris)
        self._log(f"    🗑️  Phantom webbing: {removed:,} tri eliminados "
                  f"(densidad={low_density_count}, lejanía={far_count})")
        return m2

    def _filter_by_mesh_thickness(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """
        Elimina triángulos cuyo entorno sugiere zona hueca interna.
        El radio de búsqueda es relativo a la escala de la nube (antes era
        un valor fijo 0.1 que a escala del objeto podía cubrir gran parte
        de la geometría y eliminar triángulos válidos en masa).
        """
        vertices     = np.asarray(mesh.vertices)
        triangles    = np.asarray(mesh.triangles)
        cloud_points = np.asarray(self.pcd.points)
        tree         = cKDTree(cloud_points)

        # Radio local: 4× distancia media al vecino más cercano de la nube,
        # de modo que el test frente/detrás mire solo una lámina local.
        # La banda muerta (gap) descarta los puntos de la propia superficie:
        # solo cuenta como evidencia lo que está claramente separado del
        # plano local (una segunda pared), no la curvatura de la misma capa.
        avg_pt_dist = float(np.asarray(
            self.pcd.compute_nearest_neighbor_distance()
        ).mean())
        radius = 4.0 * avg_pt_dist
        gap    = 2.0 * avg_pt_dist
        self._log(f"    📏 Radio de análisis (4·d̄): {radius:.6f}, "
                  f"banda muerta ±{gap:.6f}")

        valid_tris = []
        removed    = 0

        for idx, tri in enumerate(triangles):
            try:
                center     = vertices[tri].mean(axis=0)
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                normal     = np.cross(v1 - v0, v2 - v0)
                norm_len   = np.linalg.norm(normal)

                if norm_len < 1e-10:
                    valid_tris.append(idx)
                    continue

                normal /= norm_len
                nearby  = tree.query_ball_point(center, radius)

                # Sin vecinos = triángulo lejos de la nube: eso lo maneja el
                # filtro de webbing (criterio de lejanía), no este.
                if len(nearby) == 0:
                    valid_tris.append(idx)
                    continue

                projs   = np.dot(cloud_points[nearby] - center, normal)
                n_front = int(np.sum(projs > gap))
                n_back  = int(np.sum(projs < -gap))

                # Zona hueca solo si hay evidencia sólida a AMBOS lados:
                # con pocos vecinos el ratio es puro ruido de muestreo.
                if n_front >= 5 and n_back >= 5:
                    ratio = min(n_front, n_back) / max(n_front, n_back)
                    if ratio > 0.3:
                        removed += 1
                        continue

                valid_tris.append(idx)

            except Exception:
                valid_tris.append(idx)

        if not valid_tris:
            return mesh

        # Guard de seguridad: si el filtro eliminaría una fracción muy alta
        # de la malla, es señal de falso positivo masivo — se omite.
        frac = removed / max(len(triangles), 1)
        if frac > 0.4:
            self._log(f"    ⚠️  Filtro de huecos marcó {frac:.0%} de la malla "
                      f"— omitido por seguridad")
            return mesh

        filtered = triangles[valid_tris]
        m2 = o3d.geometry.TriangleMesh()
        m2.vertices  = o3d.utility.Vector3dVector(vertices)
        m2.triangles = o3d.utility.Vector3iVector(filtered)
        m2.remove_unreferenced_vertices()

        self._log(f"    🗑️  Interior hueco: {removed:,} tri eliminados")
        return m2

    # ══════════════════════════════════════════════════════════════
    #  SUAVIZADO
    # ══════════════════════════════════════════════════════════════

    def _smooth(
        self,
        mesh: o3d.geometry.TriangleMesh,
        method: str,
        iterations: int,
        t_lambda: float,
        t_mu: float,
        l_lambda: float,
    ) -> o3d.geometry.TriangleMesh:
        """
        Aplica suavizado Taubin o Laplacian según params.

        Parameters
        ----------
        method     : "none" | "taubin" | "laplacian"
        iterations : número de iteraciones
        t_lambda   : λ Taubin (expansión)
        t_mu       : μ Taubin (contracción)
        l_lambda   : λ Laplacian
        """
        if method == "none" or iterations <= 0:
            self._log("  ℹ️  Suavizado: desactivado")
            return mesh

        if method == "taubin":
            self._log(f"  ℹ️  Taubin (iter={iterations}, "
                      f"λ={t_lambda}, μ={t_mu})...")
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations = iterations,
                lambda_filter        = t_lambda,
                mu                   = t_mu,
            )

        elif method == "laplacian":
            self._log(f"  ℹ️  Laplacian (iter={iterations}, "
                      f"λ={l_lambda})...")
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations = iterations,
                lambda_filter        = l_lambda,
            )

        else:
            self._log(f"  ⚠️  Método de suavizado desconocido: '{method}' "
                      f"— omitido")

        self._log(f"  ✓ Suavizado completado")
        return mesh

    # ══════════════════════════════════════════════════════════════
    #  UTILIDAD
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.log_callback(msg)

