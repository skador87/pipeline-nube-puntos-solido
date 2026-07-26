# core/solidifier.py
# Extraído de PointCloudProcessor.run_solidify() / solidify_mesh()
# Actualizado: firma run(params) compatible con main_gui.py WorkerThread (Bloque C)
#
# Rediseño de la cascada de solidificación:
#   E1 repair  → reparación topológica + cierre real de agujeros (peel & fill)
#   E2 poisson → envolvente Poisson implícita (preserva concavidades)
#   E3 voxel   → voxelización + Ball Pivoting + cierre de agujeros
#   E4 hull    → Convex Hull (último recurso; destruye concavidades)
#
# Cada candidata se evalúa con dos criterios:
#   1. estanqueidad (is_watertight)
#   2. fidelidad geométrica: distancia de Chamfer simétrica contra la malla
#      de entrada, normalizada por la diagonal del bounding box.
# Se acepta la primera estrategia que cumpla ambos; si ninguna cumple, se
# devuelve la mejor candidata (estanca primero, menor error después).

import numpy as np
import open3d as o3d
from copy import deepcopy
from scipy.spatial import cKDTree


class MeshSolidifier:
    """
    Convierte una malla en un sólido cerrado.

    Firma run(params) compatible con WorkerThread de main_gui.py

    Parámetros esperados en el dict params
    ──────────────────────────────────────
    voxel_size_auto        : bool  — calcular voxel size automáticamente
    voxel_size             : float — voxel size manual (ignorado si auto=True)
    strategy_repair        : bool  — activar Estrategia 1 (reparación)
    strategy_poisson       : bool  — activar Estrategia 2 (envolvente Poisson)
    strategy_voxel         : bool  — activar Estrategia 3 (voxelización)
    strategy_hull          : bool  — activar Estrategia 4 (convex hull)
    fallback_poisson_depth : int   — profundidad octree de la envolvente Poisson
    merge_eps_factor       : float — factor × voxel_size para merge_close_vertices
    fill_holes             : bool  — cerrar agujeros en la Estrategia 1
    quality_check          : bool  — validar fidelidad geométrica (Chamfer)
    fidelity_max_error     : float — error medio máximo aceptado
                                     (fracción de la diagonal del bbox)
    """

    # ── Valores por defecto (espejo de ParamPanel en main_gui.py) ──────────
    _DEFAULTS = {
        "voxel_size_auto"        : True,
        "voxel_size"             : 0.01,
        "strategy_repair"        : True,
        "strategy_poisson"       : True,
        "strategy_voxel"         : True,
        "strategy_hull"          : True,
        "fallback_poisson_depth" : 8,
        "merge_eps_factor"       : 0.5,
        "fill_holes"             : True,
        "quality_check"          : True,
        "fidelity_max_error"     : 0.02,
    }

    # Muestras usadas para estimar la distancia de Chamfer
    _FIDELITY_SAMPLES = 15_000

    def __init__(self, mesh: o3d.geometry.TriangleMesh,
                 log_callback=None):
        """
        Parameters
        ----------
        mesh         : o3d.geometry.TriangleMesh
            Malla reconstruida (se trabaja sobre una copia interna).
        log_callback : callable | None
            Función que recibe strings de log. Si es None usa print().
        """
        self.mesh         = deepcopy(mesh)
        self.log_callback = log_callback or print
        self._ref_points  = None    # cache de muestras de la malla de entrada
        self._ref_diag    = None

    # ══════════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════════

    def run(self, params: dict | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
        """
        Ejecuta el pipeline completo de solidificación.

        Parameters
        ----------
        params : dict | None
            Diccionario de parámetros (ver _DEFAULTS).
            Las claves faltantes se completan con _DEFAULTS.

        Returns
        -------
        mesh  : o3d.geometry.TriangleMesh
            Malla solidificada.
        stats : dict
            Estadísticas con las claves:
            - voxel_size_used     : float
            - strategy_used       : str   ("repair" | "poisson" | "voxel" |
                                           "hull" | "passthrough")
            - input_vertices      : int
            - input_triangles     : int
            - output_vertices     : int
            - output_triangles    : int
            - is_watertight       : bool
            - is_orientable       : bool
            - holes_found         : int | None  (aristas frontera en la entrada)
            - fidelity_error      : float | None (Chamfer medio / diagonal bbox)
            - volume              : float | None (solo si la malla es cerrada)
            - strategies_tried    : dict  (por estrategia: watertight y error)
        """
        # ── Combinar params con defaults ───────────────────────────────────
        p = {**self._DEFAULTS, **(params or {})}

        # ── Estadísticas iniciales ─────────────────────────────────────────
        stats = {
            "voxel_size_used"  : 0.0,
            "strategy_used"    : "passthrough",
            "input_vertices"   : len(self.mesh.vertices),
            "input_triangles"  : len(self.mesh.triangles),
            "output_vertices"  : 0,
            "output_triangles" : 0,
            "is_watertight"    : False,
            "is_orientable"    : False,
            "holes_found"      : None,
            "fidelity_error"   : None,
            "volume"           : None,
            "strategies_tried" : {},
        }

        self._log(f"  ℹ️  Entrada: {stats['input_vertices']:,} vért, "
                  f"{stats['input_triangles']:,} tri")

        stats["holes_found"] = self._boundary_edge_count(self.mesh)
        if stats["holes_found"]:
            self._log(f"  ℹ️  Aristas frontera (agujeros) en la entrada: "
                      f"{stats['holes_found']:,}")

        # ── Calcular voxel size ────────────────────────────────────────────
        if p["voxel_size_auto"]:
            voxel_size = self._optimal_voxel_size()
        else:
            voxel_size = float(p["voxel_size"])
            self._log(f"  ℹ️  Voxel size manual: {voxel_size:.6f}")

        stats["voxel_size_used"] = voxel_size

        # ── Pipeline de solidificación en cascada ──────────────────────────
        result_mesh, strategy_used, fidelity = self._solidify(self.mesh, p,
                                                              voxel_size, stats)

        # ── Estadísticas finales ───────────────────────────────────────────
        stats["strategy_used"]   = strategy_used
        stats["fidelity_error"]  = fidelity
        stats["output_vertices"] = len(result_mesh.vertices)
        stats["output_triangles"]= len(result_mesh.triangles)
        stats["is_watertight"]   = self.is_topologically_closed(result_mesh)
        stats["is_orientable"]   = result_mesh.is_orientable()
        stats["volume"]          = self.signed_volume(result_mesh)

        self._log(f"  ✓ Solidificación completada "
                  f"[estrategia: {strategy_used}]")
        self._log(f"  ✓ Salida: {stats['output_vertices']:,} vért, "
                  f"{stats['output_triangles']:,} tri")
        if fidelity is not None:
            self._log(f"  📐 Error de fidelidad (Chamfer/diag): "
                      f"{fidelity:.4%}")
        if stats["volume"] is not None:
            self._log(f"  📦 Volumen encerrado: {stats['volume']:.6f}")
        self._log(f"  {'✓' if stats['is_watertight'] else '⚠️'} "
                  f"Watertight: {stats['is_watertight']} | "
                  f"Orientable: {stats['is_orientable']}")

        return result_mesh, stats

    # ══════════════════════════════════════════════════════════════
    #  CÁLCULO DE PARÁMETROS
    # ══════════════════════════════════════════════════════════════

    def _optimal_voxel_size(self) -> float:
        """
        Calcula voxel size óptimo basado en la escala de la malla.
        Lógica idéntica al original.
        """
        verts = np.asarray(self.mesh.vertices)
        size  = np.max(verts, axis=0) - np.min(verts, axis=0)
        scale = float(np.mean(size))
        voxel = scale * 0.015

        self._log(f"  📊 Tamaño objeto: "
                  f"{size[0]:.4f} × {size[1]:.4f} × {size[2]:.4f}")
        self._log(f"  📊 Voxel óptimo calculado: {voxel:.6f}")
        return voxel

    # ══════════════════════════════════════════════════════════════
    #  SOLIDIFICACIÓN CON CASCADA
    # ══════════════════════════════════════════════════════════════

    def _solidify(
        self,
        mesh: o3d.geometry.TriangleMesh,
        p: dict,
        voxel_size: float,
        stats: dict,
    ) -> tuple[o3d.geometry.TriangleMesh, str, float | None]:
        """
        Ejecuta las estrategias en cascada evaluando cada candidata con
        estanqueidad + fidelidad. Devuelve (mesh, estrategia, error_fidelidad).
        """
        quality_check = bool(p["quality_check"])
        max_error     = float(p["fidelity_max_error"])

        estrategias = [
            ("repair", bool(p["strategy_repair"]),
             lambda: self._strategy_repair(mesh, voxel_size,
                                           float(p["merge_eps_factor"]),
                                           bool(p["fill_holes"]))),
            ("poisson", bool(p["strategy_poisson"]),
             lambda: self._strategy_poisson(mesh,
                                            int(p["fallback_poisson_depth"]))),
            ("voxel", bool(p["strategy_voxel"]),
             lambda: self._strategy_voxel(mesh, voxel_size)),
            ("hull", bool(p["strategy_hull"]),
             lambda: self._strategy_hull(mesh)),
        ]

        best = None    # (mesh, nombre, watertight, error)

        for nombre, activa, fn in estrategias:
            if not activa:
                self._log(f"  ℹ️  Estrategia '{nombre}' desactivada")
                continue

            cand = fn()
            if cand is None or len(cand.triangles) == 0:
                continue

            wt  = self.is_topologically_closed(cand)
            err = self._fidelity_error(cand) if quality_check else None
            stats["strategies_tried"][nombre] = {
                "watertight"     : wt,
                "fidelity_error" : err,
            }

            err_txt = f"{err:.4%}" if err is not None else "—"
            self._log(f"    📋 Candidata '{nombre}': watertight={wt}, "
                      f"error={err_txt}")

            # Aceptación directa: estanca y dentro de la tolerancia
            if wt and (err is None or err <= max_error):
                return cand, nombre, err

            if err is not None and err > max_error:
                self._log(f"    ⚠️  '{nombre}' excede tolerancia de "
                          f"fidelidad ({max_error:.2%}) — probando siguiente")

            # Conservar la mejor candidata: estanca primero, menor error después
            key = (not wt, err if err is not None else float("inf"))
            if best is None or key < (not best[2],
                                      best[3] if best[3] is not None
                                      else float("inf")):
                best = (cand, nombre, wt, err)

        if best is not None:
            cand, nombre, wt, err = best
            self._log(f"  ⚠️  Ninguna estrategia cumplió ambos criterios — "
                      f"usando la mejor candidata: '{nombre}' "
                      f"(watertight={wt})")
            return cand, nombre, err

        # ── Fallback: devolver malla de entrada sin modificar ──────────────
        self._log("  ⚠️  Todas las estrategias fallaron — devolviendo "
                  "malla de entrada")
        return deepcopy(mesh), "passthrough", None

    # ══════════════════════════════════════════════════════════════
    #  ESTRATEGIAS INDIVIDUALES
    # ══════════════════════════════════════════════════════════════

    def _strategy_repair(
        self,
        mesh            : o3d.geometry.TriangleMesh,
        voxel_size      : float,
        merge_eps_factor: float,
        fill_holes      : bool,
    ) -> o3d.geometry.TriangleMesh | None:
        """
        Estrategia 1: limpieza y reparación topológica con cierre real
        de agujeros. Devuelve la malla reparada o None si falla.
        """
        try:
            self._log("  ℹ️  Estrategia 1: Reparación de malla...")

            m = deepcopy(mesh)
            self._basic_clean(m)

            # merge_close_vertices con clamp de seguridad: un eps mayor que
            # la arista mediana colapsa la malla y destruye el detalle.
            med_edge = self._median_edge_length(m)
            eps_req  = voxel_size * merge_eps_factor
            eps      = min(eps_req, med_edge * 0.5)
            if eps < eps_req:
                self._log(f"    ⚠️  merge eps {eps_req:.6f} limitado a "
                          f"{eps:.6f} (50% arista mediana) para no perder "
                          f"detalle")
            if eps > 0:
                m.merge_close_vertices(eps=eps)
                self._basic_clean(m)

            self._remove_small_components(m)

            self._log(f"    Reparada: {len(m.vertices):,} vért, "
                      f"{len(m.triangles):,} tri")

            if not self.is_topologically_closed(m):
                if fill_holes:
                    self._log("    ⚠️  No estanca — cerrando agujeros...")
                    m = self._peel_and_fill(m, rounds=10)
                    # Reintento: el estado de la malla cambió, un segundo
                    # pase suele destrabar cierres que quedaron al borde.
                    if not self.is_topologically_closed(m):
                        self._log("    🔁 Reintento de cierre...")
                        m = self._peel_and_fill(m, rounds=5)
                else:
                    self._log("    ⚠️  No estanca (cierre de agujeros "
                              "desactivado)")
            else:
                self._log("    ✓ Malla estanca (watertight)")

            m.orient_triangles()
            m.compute_vertex_normals()
            return m

        except Exception as e:
            self._log(f"  ⚠️  Estrategia 1 falló: {e}")
            return None

    def _strategy_poisson(
        self,
        mesh : o3d.geometry.TriangleMesh,
        depth: int,
    ) -> o3d.geometry.TriangleMesh | None:
        """
        Estrategia 2: envolvente Poisson. Muestrea la superficie de la malla
        (con normales interpoladas) y reconstruye una superficie implícita
        cerrada que preserva las concavidades, a diferencia del Convex Hull.
        """
        try:
            self._log(f"  ℹ️  Estrategia 2: Envolvente Poisson "
                      f"(depth={depth})...")

            m = deepcopy(mesh)
            m.compute_vertex_normals()

            n_pts = int(np.clip(3 * len(m.vertices), 30_000, 300_000))
            pcd   = m.sample_points_uniformly(n_pts)

            env, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            if len(env.triangles) == 0:
                self._log("    ⚠️  Poisson produjo malla vacía")
                return None

            # La superficie implícita puede generar burbujas desconectadas
            # lejos del objeto: conservar solo el componente principal.
            self._keep_largest_component(env)

            # Recorte de seguridad si la envolvente se extiende mucho más
            # allá del objeto (burbuja adherida al componente principal).
            diag_in  = np.linalg.norm(m.get_max_bound() - m.get_min_bound())
            diag_env = np.linalg.norm(env.get_max_bound() -
                                      env.get_min_bound())
            if diag_env > 1.5 * diag_in:
                self._log("    ⚠️  Envolvente excede bbox de entrada — "
                          "recortando")
                aabb = m.get_axis_aligned_bounding_box()
                aabb = aabb.scale(1.1, aabb.get_center())
                env  = env.crop(aabb)
                self._keep_largest_component(env)

            env = self._peel_and_fill(env, rounds=5)
            env.orient_triangles()
            env.compute_vertex_normals()

            self._log(f"    ✓ Envolvente Poisson: {len(env.vertices):,} vért, "
                      f"{len(env.triangles):,} tri")
            return env

        except Exception as e:
            self._log(f"  ⚠️  Estrategia 2 falló: {e}")
            return None

    def _strategy_voxel(
        self,
        mesh      : o3d.geometry.TriangleMesh,
        voxel_size: float,
    ) -> o3d.geometry.TriangleMesh | None:
        """
        Estrategia 3: voxelización + Ball Pivoting + cierre de agujeros.
        Devuelve la malla resultante o None si falla.
        """
        try:
            self._log("  ℹ️  Estrategia 3: Voxelización de malla...")

            vg     = o3d.geometry.VoxelGrid \
                         .create_from_triangle_mesh(mesh,
                                                    voxel_size=voxel_size)
            voxels = vg.get_voxels()
            points = np.array([
                vg.get_voxel_center_coordinate(v.grid_index) for v in voxels
            ])

            if len(points) == 0:
                self._log("    ⚠️  Voxelización produjo 0 puntos")
                return None

            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(points)
            tmp.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

            radii = o3d.utility.DoubleVector([voxel_size * 1.5,
                                              voxel_size * 2.5])
            solid = o3d.geometry.TriangleMesh \
                        .create_from_point_cloud_ball_pivoting(tmp, radii)
            if len(solid.triangles) == 0:
                self._log("    ⚠️  Ball Pivoting produjo malla vacía")
                return None

            solid = self._peel_and_fill(solid, rounds=3)
            solid.orient_triangles()
            solid.compute_vertex_normals()

            self._log(f"    ✓ Voxelización: {len(solid.vertices):,} vért, "
                      f"{len(solid.triangles):,} tri")
            return solid

        except Exception as e:
            self._log(f"  ⚠️  Estrategia 3 falló: {e}")
            return None

    def _strategy_hull(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh | None:
        """
        Estrategia 4: Convex Hull como último recurso.
        Destruye las concavidades: solo se usa si todo lo demás falla.
        """
        try:
            self._log("  ℹ️  Estrategia 4: Convex Hull (último recurso)...")
            hull, _ = mesh.compute_convex_hull()
            hull.compute_vertex_normals()
            hull.orient_triangles()
            self._log(f"    ✓ Convex Hull: {len(hull.vertices):,} vért, "
                      f"{len(hull.triangles):,} tri")
            return hull

        except Exception as e:
            self._log(f"  ⚠️  Estrategia 4 falló: {e}")
            return None

    # ══════════════════════════════════════════════════════════════
    #  CIERRE DE AGUJEROS  (peel & fill)
    # ══════════════════════════════════════════════════════════════

    def _peel_and_fill(
        self,
        mesh  : o3d.geometry.TriangleMesh,
        rounds: int = 4,
    ) -> o3d.geometry.TriangleMesh:
        """
        Cierra agujeros de forma iterativa:
          1. eliminar defectos no-manifold (aristas y vértices pinch),
          2. cerrar loops frontera con abanico al centroide (manifold-seguro),
          3. fill_holes (API tensor) para las cadenas no simples restantes,
          4. merge fino de ranuras (solo primera ronda).
        Si una ronda no progresa y queda poca frontera, pela localmente la
        ranura degenerada y reintenta. Conserva el mejor estado alcanzado.
        Repite hasta lograr el cierre topológico o agotar las rondas.
        """
        m = mesh
        self._basic_clean(m)
        med_edge = self._median_edge_length(m)
        prev_boundary = None
        best_state    = None   # (n_frontera, copia) — nunca devolver algo peor

        for i in range(rounds):
            if self.is_topologically_closed(m):
                break

            m.remove_non_manifold_edges()
            nmv = np.asarray(m.get_non_manifold_vertices())
            if len(nmv) > 0:
                mask = np.zeros(len(m.vertices), dtype=bool)
                mask[nmv] = True
                m.remove_vertices_by_mask(mask)
            self._basic_clean(m)
            if self.is_topologically_closed(m):
                break

            # Abanico primero: es manifold-seguro por construcción (un
            # triángulo nuevo por arista frontera, centroide nuevo).
            n_fan = self._fan_fill_boundary_loops(m)
            if n_fan:
                self._log(f"    🪭 Cierre por abanico: {n_fan} loops")
                self._basic_clean(m)
                if self.is_topologically_closed(m):
                    break

            # fill_holes solo para los bordes que el abanico no pudo
            # cerrar (cadenas no simples). Puede introducir defectos
            # no-manifold, que se eliminan en la ronda siguiente.
            if self._boundary_edge_count(m) > 0:
                try:
                    tm = o3d.t.geometry.TriangleMesh.from_legacy(m)
                    m  = tm.fill_holes(hole_size=1e6).to_legacy()
                except Exception as e:
                    self._log(f"    ⚠️  fill_holes falló: {e}")
                    break
                self._basic_clean(m)

            # El merge fino solo en la primera ronda: cierra ranuras del
            # borde original, pero repetido sobre los parches nuevos
            # genera aristas no-manifold en cada iteración.
            if i == 0 and med_edge > 0:
                m.merge_close_vertices(eps=med_edge * 0.3)
                self._basic_clean(m)

            n_boundary = self._boundary_edge_count(m)
            self._log(f"    🔧 Ronda {i + 1}: {n_boundary:,} aristas "
                      f"frontera restantes")

            if best_state is None or n_boundary < best_state[0]:
                best_state = (n_boundary, deepcopy(m))

            # Escalada si fill_holes se estancó:
            #  - frontera pequeña (ranuras degeneradas): pelar localmente
            #    esos triángulos para que la siguiente ronda los rellene.
            #  - frontera grande: cerrar los loops con abanico al centroide
            #    (funciona con los loops no simples que fill_holes no
            #    puede triangular).
            if (prev_boundary is not None and 0 < n_boundary
                    and n_boundary >= prev_boundary):
                if n_boundary <= 24:
                    self._peel_boundary_triangles(m)
                    self._log("    🔪 Ranura degenerada pelada localmente")
                else:
                    n_fan = self._fan_fill_boundary_loops(m)
                    if n_fan:
                        self._log(f"    🪭 Cierre por abanico: {n_fan} loops")
                self._basic_clean(m)
            prev_boundary = n_boundary

        # Pasada final: eliminar defectos no-manifold residuales (que
        # fill_holes puede haber introducido) y cerrar con abanico.
        if not self.is_topologically_closed(m):
            m.remove_non_manifold_edges()
            self._basic_clean(m)
            self._fan_fill_boundary_loops(m)
            self._basic_clean(m)

        # Si el bucle terminó en un estado peor que una ronda anterior
        # (p. ej. una escalada que abrió más borde), recuperar el mejor.
        if (not self.is_topologically_closed(m) and best_state is not None
                and self._boundary_edge_count(m) > best_state[0]):
            m = best_state[1]

        if self.is_topologically_closed(m):
            self._log("    ✓ Agujeros cerrados — malla estanca")
        else:
            self._log(f"    ⚠️  Persisten "
                      f"{self._boundary_edge_count(m):,} aristas frontera")
        return m

    # ══════════════════════════════════════════════════════════════
    #  MÉTRICA DE FIDELIDAD
    # ══════════════════════════════════════════════════════════════

    def _fidelity_error(
        self,
        candidate: o3d.geometry.TriangleMesh,
    ) -> float | None:
        """
        Distancia de Chamfer simétrica (media) entre la malla de entrada y la
        candidata, por muestreo uniforme, normalizada por la diagonal del
        bounding box de la entrada. Valores bajos = mayor fidelidad.
        """
        try:
            if len(self.mesh.triangles) == 0 or len(candidate.triangles) == 0:
                return None

            if self._ref_points is None:
                self._ref_diag = float(np.linalg.norm(
                    self.mesh.get_max_bound() - self.mesh.get_min_bound()
                ))
                self._ref_points = np.asarray(
                    self.mesh.sample_points_uniformly(
                        self._FIDELITY_SAMPLES).points
                )
            if self._ref_diag <= 0:
                return None

            cand_pts = np.asarray(
                candidate.sample_points_uniformly(
                    self._FIDELITY_SAMPLES).points
            )

            d_rc = cKDTree(cand_pts).query(self._ref_points, k=1)[0]
            d_cr = cKDTree(self._ref_points).query(cand_pts, k=1)[0]
            return float(np.concatenate([d_rc, d_cr]).mean() / self._ref_diag)

        except Exception as e:
            self._log(f"    ⚠️  No se pudo medir fidelidad: {e}")
            return None

    # ══════════════════════════════════════════════════════════════
    #  UTILIDADES DE MALLA
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def is_topologically_closed(mesh: o3d.geometry.TriangleMesh) -> bool:
        """
        Cierre topológico en O(T): toda arista tiene exactamente dos
        triángulos adyacentes (manifold sin frontera) y todos los vértices
        son manifold. Equivale al criterio de is_watertight() de Open3D
        pero sin su test de auto-intersecciones, cuyo coste cuadrático lo
        hace inviable en mallas cerradas grandes.
        """
        return (len(mesh.triangles) > 0
                and mesh.is_edge_manifold(allow_boundary_edges=False)
                and mesh.is_vertex_manifold())

    @staticmethod
    def signed_volume(mesh: o3d.geometry.TriangleMesh) -> float | None:
        """
        Volumen encerrado por suma de tetraedros firmados, en O(T).

        Devuelve None si la malla no es cerrada (en ese caso el valor no
        significa nada). No se usa `get_volume()` de Open3D: incluye un test
        de auto-intersecciones cuadrático que tarda horas en mallas grandes.
        """
        if not MeshSolidifier.is_topologically_closed(mesh):
            return None
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        if len(t) == 0:
            return None
        p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        return abs(float(
            np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0
        ))

    @staticmethod
    def _basic_clean(mesh: o3d.geometry.TriangleMesh):
        """Limpieza estándar in-place: duplicados, degenerados, huérfanos."""
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    @staticmethod
    def _median_edge_length(mesh: o3d.geometry.TriangleMesh) -> float:
        """Longitud mediana de arista (escala local de la malla)."""
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            return 0.0
        verts = np.asarray(mesh.vertices)
        edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        return float(np.median(
            np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        ))

    @staticmethod
    def _boundary_edge_count(mesh: o3d.geometry.TriangleMesh) -> int:
        """Número de aristas frontera (con un solo triángulo adyacente)."""
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            return 0
        edges = np.sort(
            np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
            axis=1,
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int((counts == 1).sum())

    @staticmethod
    def _peel_boundary_triangles(mesh: o3d.geometry.TriangleMesh):
        """Elimina los triángulos que tocan una arista frontera (in-place).
        Solo para fronteras pequeñas: en bordes grandes agranda el agujero."""
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            return
        edges = np.sort(
            np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
            axis=1,
        )
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        border_verts = np.unique(uniq[counts == 1])
        if len(border_verts) == 0:
            return
        mask = np.isin(tris, border_verts).any(axis=1)
        mesh.remove_triangles_by_mask(mask)
        mesh.remove_unreferenced_vertices()

    @staticmethod
    def _fan_fill_boundary_loops(mesh: o3d.geometry.TriangleMesh,
                                 max_loop: int = 10_000) -> int:
        """
        Cierra cada loop frontera con un abanico de triángulos hacia su
        centroide (in-place). A diferencia de fill_holes, funciona también
        con loops largos o no simples. El winding de los triángulos nuevos
        se toma opuesto al de la arista frontera para mantener la
        orientación coherente. Devuelve el número de loops cerrados.
        """
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            return 0

        # Frontera = arista con un único triángulo dueño (conteo no
        # dirigido). La dirección se toma del triángulo dueño, así el
        # criterio no depende de que el winding global sea consistente.
        directed  = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]],
                               tris[:, [2, 0]]])
        undirected = np.sort(directed, axis=1)
        _, inv, counts = np.unique(undirected, axis=0,
                                   return_inverse=True, return_counts=True)
        b_edges = directed[counts[inv] == 1]
        if len(b_edges) == 0:
            return 0

        nxt = {int(a): int(b) for a, b in b_edges}
        # si un vértice tiene varias salidas, queda una (mejor esfuerzo)

        verts     = list(np.asarray(mesh.vertices))
        new_tris  = []
        visited   = set()
        n_loops   = 0

        for start in list(nxt.keys()):
            if start in visited or start not in nxt:
                continue
            loop, v = [start], nxt[start]
            while (v not in visited and v != start
                   and v in nxt and len(loop) < max_loop):
                loop.append(v)
                v = nxt[v]
            visited.update(loop)
            if v != start or len(loop) < 3:
                continue               # cadena abierta o degenerada

            centroid = np.mean([verts[i] for i in loop], axis=0)
            c_idx    = len(verts)
            verts.append(centroid)
            for k in range(len(loop)):
                a, b = loop[k], loop[(k + 1) % len(loop)]
                new_tris.append([b, a, c_idx])
            n_loops += 1

        if n_loops == 0:
            return 0

        all_tris = np.vstack([np.asarray(mesh.triangles),
                              np.asarray(new_tris, dtype=np.int64)])
        mesh.vertices  = o3d.utility.Vector3dVector(np.asarray(verts))
        mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        return n_loops

    def _keep_largest_component(self, mesh: o3d.geometry.TriangleMesh):
        """Conserva solo el componente conexo con más triángulos (in-place)."""
        idx, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) > 1:
            idx = np.asarray(idx)
            mesh.remove_triangles_by_mask(idx != int(np.argmax(counts)))
            mesh.remove_unreferenced_vertices()
            self._log(f"    🧹 Componentes: {len(counts)} → 1 "
                      f"({int(counts.max()):,} tri)")

    def _remove_small_components(self, mesh: o3d.geometry.TriangleMesh,
                                 min_frac: float = 0.01):
        """Elimina componentes conexos con menos del min_frac de triángulos."""
        idx, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) <= 1:
            return
        idx   = np.asarray(idx)
        small = np.flatnonzero(counts < min_frac * len(idx))
        if len(small) == 0:
            return
        mesh.remove_triangles_by_mask(np.isin(idx, small))
        mesh.remove_unreferenced_vertices()
        self._log(f"    🧹 Eliminados {len(small)} componentes pequeños")

    # ══════════════════════════════════════════════════════════════
    #  UTILIDAD
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.log_callback(msg)
