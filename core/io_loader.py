# core/io_loader.py
# Cargador multi-formato para Point Cloud Processor v2.1
#
# Formatos soportados:
#   .ply  → o3d.io.read_point_cloud()  (nativo)
#   .pcd  → o3d.io.read_point_cloud()  (nativo)
#   .xyz  → parser manual con detección de columnas
#   .e57  → pye57 con fallback informativo
#
# Uso:
#   from core.io_loader import PointCloudLoader
#   pcd, meta = PointCloudLoader.load(path, log_callback=print)

import os
import numpy as np
import open3d as o3d


class PointCloudLoader:
    """
    Carga nubes de puntos desde múltiples formatos.
    Siempre devuelve (o3d.geometry.PointCloud, dict_metadata).
    """

    # Extensiones que maneja Open3D de forma nativa sin problemas
    _O3D_NATIVE = {".ply", ".pcd"}

    # ══════════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def load(
        cls,
        path        : str,
        log_callback: callable = None,
    ) -> tuple[o3d.geometry.PointCloud, dict]:
        """
        Carga una nube de puntos desde disco.

        Parameters
        ----------
        path         : str
            Ruta al archivo.
        log_callback : callable | None
            Función de log. Si es None usa print().

        Returns
        -------
        pcd  : o3d.geometry.PointCloud
        meta : dict  con claves:
            - format        : str   extensión detectada
            - n_points      : int   puntos cargados
            - has_colors    : bool
            - has_normals   : bool
            - e57_scans     : int | None   (solo .e57)
            - xyz_columns   : list | None  (solo .xyz)
            - warnings      : list[str]    advertencias no fatales
        """
        log = log_callback or print
        ext = os.path.splitext(path)[1].lower()

        meta = {
            "format"      : ext,
            "n_points"    : 0,
            "has_colors"  : False,
            "has_normals" : False,
            "e57_scans"   : None,
            "xyz_columns" : None,
            "warnings"    : [],
        }

        # ── Despacho por extensión ─────────────────────────────────────
        if ext in cls._O3D_NATIVE:
            pcd = cls._load_native(path, log)

        elif ext == ".xyz":
            pcd = cls._load_xyz(path, meta, log)

        elif ext == ".e57":
            pcd = cls._load_e57(path, meta, log)

        else:
            # Intentar con Open3D de todos modos
            log(f"  ⚠️  Formato '{ext}' no reconocido explícitamente, "
                f"intentando con Open3D...")
            pcd = cls._load_native(path, log)

        # ── Validación final ───────────────────────────────────────────
        if pcd is None or len(pcd.points) == 0:
            raise ValueError(
                f"No se pudieron cargar puntos desde '{os.path.basename(path)}'"
            )

        meta["n_points"]   = len(pcd.points)
        meta["has_colors"] = pcd.has_colors()
        meta["has_normals"]= pcd.has_normals()

        log(f"  ✓ {meta['n_points']:,} puntos cargados "
            f"[colores={'sí' if meta['has_colors'] else 'no'}, "
            f"normales={'sí' if meta['has_normals'] else 'no'}]")

        if meta["warnings"]:
            for w in meta["warnings"]:
                log(f"  ⚠️  {w}")

        return pcd, meta

    # ══════════════════════════════════════════════════════════════
    #  LOADERS INTERNOS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _load_native(path: str, log: callable) -> o3d.geometry.PointCloud:
        """Usa Open3D directamente para .ply y .pcd."""
        log(f"  ℹ️  Cargando con Open3D nativo...")
        pcd = o3d.io.read_point_cloud(path)
        return pcd

    # ──────────────────────────────────────────────────────────────
    #  .XYZ
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _load_xyz(
        cls,
        path : str,
        meta : dict,
        log  : callable,
    ) -> o3d.geometry.PointCloud:
        """
        Carga .xyz con detección automática de columnas.

        Formatos detectados (por número de columnas):
          3 cols → X Y Z
          4 cols → X Y Z Intensity
          6 cols → X Y Z R G B
          7 cols → X Y Z Intensity R G B   (orden alternativo)
          9 cols → X Y Z R G B Nx Ny Nz
        """
        log(f"  ℹ️  Parseando .xyz con detección de columnas...")

        # ── Detectar separador y saltarse líneas de cabecera ──────────
        data, col_names = cls._parse_xyz_file(path, meta, log)

        if data is None or len(data) == 0:
            raise ValueError("Archivo .xyz vacío o ilegible")

        meta["xyz_columns"] = col_names
        log(f"  ℹ️  Columnas detectadas: {col_names}")

        # ── Construir PointCloud ───────────────────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])

        n_cols = data.shape[1]

        if n_cols >= 6 and col_names[3] in ("r", "red"):
            # Columnas 3,4,5 son RGB
            rgb = data[:, 3:6]
            # Normalizar si están en rango 0-255
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(rgb, 0, 1).astype(np.float64)
            )

        if n_cols >= 9 and col_names[6] in ("nx", "normal_x"):
            # Columnas 6,7,8 son normales
            pcd.normals = o3d.utility.Vector3dVector(
                data[:, 6:9].astype(np.float64)
            )

        return pcd

    @staticmethod
    def _parse_xyz_file(
        path : str,
        meta : dict,
        log  : callable,
    ) -> tuple[np.ndarray | None, list[str]]:
        """
        Lee el archivo .xyz y devuelve (array_float, nombres_columnas).
        Detecta automáticamente:
          - separador (espacio / coma / tabulador)
          - líneas de cabecera
          - número de columnas
        """
        rows       = []
        sep        = None
        skip_lines = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            return None, []

        # ── Detectar cuántas líneas iniciales son cabecera ────────────
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                skip_lines = i + 1
                continue
            # Intentar parsear como float
            for s in (" ", "\t", ",", ";"):
                parts = line.split(s)
                parts = [p.strip() for p in parts if p.strip()]
                try:
                    [float(p) for p in parts if p]
                    sep = s
                    break
                except ValueError:
                    continue
            if sep is not None:
                # Primera línea numérica encontrada en índice i
                # Todo lo anterior es cabecera
                skip_lines = i
                break
            else:
                skip_lines = i + 1  # línea no numérica → cabecera

        if sep is None:
            sep = " "   # fallback

        # ── Leer datos ────────────────────────────────────────────────
        for line in lines[skip_lines:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(sep)
            parts = [p.strip() for p in parts if p.strip()]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue    # saltar líneas malformadas

        if not rows:
            return None, []

        # Homogenizar longitud de filas (usar la moda)
        lengths = [len(r) for r in rows]
        n_cols  = max(set(lengths), key=lengths.count)
        rows    = [r for r in rows if len(r) == n_cols]

        data = np.array(rows, dtype=np.float64)

        # ── Asignar nombres de columna por convención ─────────────────
        col_map = {
            3: ["x", "y", "z"],
            4: ["x", "y", "z", "intensity"],
            6: ["x", "y", "z", "r", "g", "b"],
            7: ["x", "y", "z", "intensity", "r", "g", "b"],
            9: ["x", "y", "z", "r", "g", "b", "nx", "ny", "nz"],
        }
        col_names = col_map.get(n_cols, [f"col{i}" for i in range(n_cols)])

        return data, col_names

    # ──────────────────────────────────────────────────────────────
    #  .E57
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _load_e57(
        cls,
        path : str,
        meta : dict,
        log  : callable,
    ) -> o3d.geometry.PointCloud:
        """
        Carga .e57 usando pye57.
        Combina todos los scans del archivo en una sola PointCloud.

        Requiere: pip install pye57
        """
        try:
            import pye57
        except ImportError:
            raise ImportError(
                "El formato .e57 requiere la librería 'pye57'.\n"
                "Instálala con:  pip install pye57"
            )

        log(f"  ℹ️  Abriendo .e57 con pye57...")
        e57_file  = pye57.E57(path)
        n_scans   = e57_file.scan_count
        meta["e57_scans"] = n_scans
        log(f"  ℹ️  Scans encontrados: {n_scans}")

        all_xyz    = []
        all_rgb    = []
        has_colors = False

        for scan_idx in range(n_scans):
            log(f"  ℹ️  Leyendo scan {scan_idx + 1}/{n_scans}...")

            try:
                data = e57_file.read_scan(
                    scan_idx,
                    ignore_missing_fields=True,
                )
            except Exception as e:
                meta["warnings"].append(
                    f"Scan {scan_idx} omitido por error: {e}"
                )
                continue

            # ── Coordenadas ───────────────────────────────────────────
            xyz = cls._e57_extract_xyz(data)
            if xyz is None or len(xyz) == 0:
                meta["warnings"].append(
                    f"Scan {scan_idx} no contiene coordenadas válidas"
                )
                continue

            all_xyz.append(xyz)
            log(f"    {len(xyz):,} puntos en scan {scan_idx}")

            # ── Color ─────────────────────────────────────────────────
            rgb = cls._e57_extract_rgb(data)
            if rgb is not None:
                all_rgb.append(rgb)
                has_colors = True
            elif has_colors:
                # Scans anteriores tenían color, este no → relleno gris
                all_rgb.append(
                    np.full((len(xyz), 3), 0.5, dtype=np.float64)
                )

        e57_file.close()

        if not all_xyz:
            raise ValueError("No se encontraron puntos válidos en el .e57")

        # ── Combinar todos los scans ───────────────────────────────────
        combined_xyz = np.vstack(all_xyz)
        log(f"  ℹ️  Total combinado: {len(combined_xyz):,} puntos")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_xyz)

        if has_colors and all_rgb:
            combined_rgb = np.vstack(all_rgb)
            # Normalizar a [0,1] si es necesario
            if combined_rgb.max() > 1.0:
                combined_rgb = combined_rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(combined_rgb, 0, 1)
            )

        return pcd

    @staticmethod
    def _e57_extract_xyz(data: dict) -> np.ndarray | None:
        """
        Extrae coordenadas XYZ del dict devuelto por pye57.read_scan().
        Maneja tanto formato cartesiano como esférico.
        """
        # ── Cartesiano (más común) ─────────────────────────────────────
        keys_cart = [
            ("cartesianX", "cartesianY", "cartesianZ"),
            ("x", "y", "z"),
            ("X", "Y", "Z"),
        ]
        for kx, ky, kz in keys_cart:
            if kx in data and ky in data and kz in data:
                x = np.asarray(data[kx], dtype=np.float64).ravel()
                y = np.asarray(data[ky], dtype=np.float64).ravel()
                z = np.asarray(data[kz], dtype=np.float64).ravel()
                xyz = np.column_stack([x, y, z])
                # Filtrar NaN / Inf
                mask = np.isfinite(xyz).all(axis=1)
                return xyz[mask]

        # ── Esférico → convertir a cartesiano ─────────────────────────
        keys_sph = [
            ("sphericalRange", "sphericalAzimuth", "sphericalElevation"),
        ]
        for kr, ka, ke in keys_sph:
            if kr in data and ka in data and ke in data:
                r   = np.asarray(data[kr], dtype=np.float64).ravel()
                az  = np.asarray(data[ka], dtype=np.float64).ravel()
                el  = np.asarray(data[ke], dtype=np.float64).ravel()
                x   = r * np.cos(el) * np.cos(az)
                y   = r * np.cos(el) * np.sin(az)
                z   = r * np.sin(el)
                xyz = np.column_stack([x, y, z])
                mask = np.isfinite(xyz).all(axis=1)
                return xyz[mask]

        return None

    @staticmethod
    def _e57_extract_rgb(data: dict) -> np.ndarray | None:
        """
        Extrae colores RGB del dict de pye57.
        Devuelve array (N, 3) float64 normalizado [0,1] o None.
        """
        key_sets = [
            ("colorRed",   "colorGreen",  "colorBlue"),
            ("red",        "green",       "blue"),
            ("r",          "g",           "b"),
        ]
        for kr, kg, kb in key_sets:
            if kr in data and kg in data and kb in data:
                r = np.asarray(data[kr], dtype=np.float64).ravel()
                g = np.asarray(data[kg], dtype=np.float64).ravel()
                b = np.asarray(data[kb], dtype=np.float64).ravel()
                rgb = np.column_stack([r, g, b])
                # Normalizar si está en rango 0-255
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                return np.clip(rgb, 0, 1)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDAD: verificar dependencias disponibles
# ══════════════════════════════════════════════════════════════════════════════

def check_e57_support() -> tuple[bool, str]:
    """
    Verifica si pye57 está instalado.

    Returns
    -------
    (available, message)
    """
    try:
        import pye57  # noqa: F401
        return True, "pye57 disponible ✓"
    except ImportError:
        return False, (
            "pye57 no instalado — .e57 no disponible\n"
            "Instalar: pip install pye57"
        )
