# core/io_loader.py
# Cargador multi-formato para Point Cloud Processor v2.1
#
# Formatos soportados:
#   .ply  → o3d.io.read_point_cloud()  (nativo)
#   .pcd  → o3d.io.read_point_cloud()  (nativo)
#   .xyz  → parser rápido numpy + fallback tolerante con detección de columnas
#   .txt  → ídem .xyz (p. ej. S3DIS / Stanford3dDataset: "X Y Z R G B"
#           con RGB en 0–255)
#   .e57  → pye57 con fallback informativo
#   .las  → laspy (LiDAR); RGB 16-bit si existe, o intensidad como gris
#   .laz  → ídem .las (requiere backend lazrs/laszip)
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
            "las_offset"  : None,   # traslación aplicada a coords geo (LAS)
            "warnings"    : [],
        }

        # ── Despacho por extensión ─────────────────────────────────────
        if ext in cls._O3D_NATIVE:
            pcd = cls._load_native(path, log)

        elif ext in (".xyz", ".txt"):
            pcd = cls._load_xyz(path, meta, log)

        elif ext == ".e57":
            pcd = cls._load_e57(path, meta, log)

        elif ext in (".las", ".laz"):
            pcd = cls._load_las(path, meta, log)

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

    # Nombres de columna por convención según su número
    _COL_MAP = {
        3: ["x", "y", "z"],
        4: ["x", "y", "z", "intensity"],
        6: ["x", "y", "z", "r", "g", "b"],
        7: ["x", "y", "z", "intensity", "r", "g", "b"],
        9: ["x", "y", "z", "r", "g", "b", "nx", "ny", "nz"],
    }

    @classmethod
    def _load_xyz(
        cls,
        path : str,
        meta : dict,
        log  : callable,
    ) -> o3d.geometry.PointCloud:
        """
        Carga .xyz / .txt con detección automática de columnas.
        Compatible con S3DIS (Stanford3dDataset): "X Y Z R G B", RGB 0–255.

        Formatos detectados (por número de columnas):
          3 cols → X Y Z
          4 cols → X Y Z Intensity
          6 cols → X Y Z R G B
          7 cols → X Y Z Intensity R G B   (orden alternativo)
          9 cols → X Y Z R G B Nx Ny Nz
        """
        log(f"  ℹ️  Parseando texto con detección de columnas...")

        # ── Vía rápida: parser C de numpy (archivos grandes tipo S3DIS) ─
        data, col_names = cls._parse_text_fast(path, meta, log)

        # ── Fallback tolerante: cabeceras, separadores raros o líneas
        #    malformadas (p. ej. los .txt corruptos conocidos de S3DIS) ──
        if data is None:
            log("  ℹ️  Vía rápida falló — usando parser tolerante "
                "(se omiten líneas malformadas)...")
            data, col_names = cls._parse_xyz_file(path, meta, log)

        if data is None or len(data) == 0:
            raise ValueError("Archivo de texto vacío o ilegible")

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

    @classmethod
    def _parse_text_fast(
        cls,
        path : str,
        meta : dict,
        log  : callable,
    ) -> tuple[np.ndarray | None, list[str]]:
        """
        Intento de carga con np.loadtxt (parser en C, órdenes de magnitud
        más rápido que el parser línea a línea). Devuelve (None, []) si el
        archivo tiene cabecera no comentada, separadores inusuales o
        líneas malformadas — en ese caso se usa el parser tolerante.
        """
        data = None
        for delim in (None, ",", ";"):        # None = espacios/tabs
            try:
                data = np.loadtxt(path, dtype=np.float64,
                                  delimiter=delim, ndmin=2)
                break
            except Exception:
                data = None
        if data is None or data.size == 0 or data.shape[1] < 3:
            return None, []

        # Filtrar filas no finitas
        finite = np.isfinite(data).all(axis=1)
        if not finite.all():
            n_bad = int((~finite).sum())
            meta["warnings"].append(
                f"{n_bad:,} filas con valores no finitos omitidas")
            data = data[finite]

        n_cols = data.shape[1]
        col_names = cls._COL_MAP.get(
            n_cols, [f"col{i}" for i in range(n_cols)])
        log(f"  ℹ️  Carga rápida: {len(data):,} filas × {n_cols} columnas")
        return data, col_names

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
        n_malformed = 0
        for line in lines[skip_lines:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(sep)
            parts = [p.strip() for p in parts if p.strip()]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                n_malformed += 1    # saltar líneas malformadas

        if not rows:
            return None, []

        # Homogenizar longitud de filas (usar la moda)
        lengths = [len(r) for r in rows]
        n_cols  = max(set(lengths), key=lengths.count)
        n_before = len(rows)
        rows    = [r for r in rows if len(r) == n_cols]
        n_malformed += n_before - len(rows)

        if n_malformed:
            meta["warnings"].append(
                f"{n_malformed:,} líneas malformadas omitidas")

        data = np.array(rows, dtype=np.float64)

        # ── Asignar nombres de columna por convención ─────────────────
        col_names = PointCloudLoader._COL_MAP.get(
            n_cols, [f"col{i}" for i in range(n_cols)])

        return data, col_names

    # ──────────────────────────────────────────────────────────────
    #  .LAS / .LAZ
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _load_las(
        cls,
        path : str,
        meta : dict,
        log  : callable,
    ) -> o3d.geometry.PointCloud:
        """
        Carga .las/.laz (LiDAR) usando laspy.

        - RGB: si el point format lo trae (formatos 2,3,5,7,8,10), se
          normaliza desde 16-bit (o 8-bit si el archivo lo guarda así).
        - Sin RGB: la intensidad se mapea a escala de grises con
          normalización robusta por percentiles.
        - Coordenadas georreferenciadas (UTM, ~10⁵–10⁷): se trasladan a
          un origen local para evitar pérdida de precisión float32 en el
          visor; el offset queda en meta["las_offset"] y en el log.

        Requiere: pip install laspy   (para .laz: laspy[lazrs])
        """
        try:
            import laspy
        except ImportError:
            raise ImportError(
                "El formato .las/.laz requiere la librería 'laspy'.\n"
                "Instálala con:  pip install laspy[lazrs]"
            )

        log(f"  ℹ️  Abriendo {meta['format']} con laspy...")
        las = laspy.read(path)

        fmt = las.point_format
        log(f"  ℹ️  LAS {las.header.version} · point format {fmt.id} · "
            f"{las.header.point_count:,} puntos")

        pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

        # ── Filtrar no finitos ────────────────────────────────────────
        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            meta["warnings"].append(
                f"{int((~finite).sum()):,} puntos no finitos omitidos")
            pts = pts[finite]

        # ── Recentrar coordenadas georreferenciadas ───────────────────
        # Coordenadas UTM (~10⁵–10⁷ m) exceden la precisión de float32 y
        # producen "jitter" al renderizar; el pipeline trabaja igual de
        # bien en un origen local.
        if len(pts) and np.abs(pts).max() > 1e5:
            offset = np.floor(pts.min(axis=0))
            pts    = pts - offset
            meta["las_offset"] = offset.tolist()
            log(f"  ℹ️  Coordenadas georreferenciadas: trasladadas a "
                f"origen local (offset aplicado: "
                f"{offset[0]:.0f}, {offset[1]:.0f}, {offset[2]:.0f})")
            meta["warnings"].append(
                "Nube trasladada a origen local; las exportaciones "
                "quedan en coordenadas locales (offset en el log)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # ── Color: RGB del archivo o intensidad como gris ─────────────
        dims = set(fmt.dimension_names)
        if {"red", "green", "blue"} <= dims:
            rgb = np.column_stack([las.red, las.green, las.blue]) \
                    .astype(np.float64)
            if not finite.all():
                rgb = rgb[finite]
            escala = 65535.0 if rgb.max() > 255 else 255.0
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(rgb / escala, 0, 1))
            log("  ℹ️  Color RGB del archivo aplicado")

        elif "intensity" in dims:
            inten = np.asarray(las.intensity, dtype=np.float64)
            if not finite.all():
                inten = inten[finite]
            lo, hi = np.percentile(inten, [2, 98])
            if hi > lo:
                g = np.clip((inten - lo) / (hi - lo), 0, 1)
            else:
                g = np.full(len(inten), 0.6)
            pcd.colors = o3d.utility.Vector3dVector(
                np.column_stack([g, g, g]))
            log("  ℹ️  Sin RGB — intensidad LiDAR mapeada a "
                "escala de grises")

        return pcd

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


def check_las_support() -> tuple[bool, bool, str]:
    """
    Verifica si laspy está instalado y si hay backend LAZ.

    Returns
    -------
    (las_available, laz_available, message)
    """
    try:
        import laspy
    except ImportError:
        return False, False, (
            "laspy no instalado — .las/.laz no disponibles\n"
            "Instalar: pip install laspy[lazrs]"
        )
    try:
        laz_ok = len(laspy.LazBackend.detect_available()) > 0
    except Exception:
        laz_ok = False
    if laz_ok:
        return True, True, "laspy disponible ✓ (.las y .laz)"
    return True, False, (
        "laspy disponible ✓ (.las) — para .laz instalar: pip install lazrs"
    )
