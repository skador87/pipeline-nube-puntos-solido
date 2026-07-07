# utils/helpers.py
import logging
import os
import sys

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging(level=logging.DEBUG):
    """Configura el sistema de logging con formato detallado."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_point_cloud(filepath: str):
    """
    Carga una nube de puntos desde archivo .ply / .pcd / .xyz
    Retorna un objeto open3d.geometry.PointCloud o None si falla.
    """
    if not OPEN3D_AVAILABLE:
        logger.error("open3d no está instalado. Ejecuta: pip install open3d")
        return None

    if not os.path.exists(filepath):
        logger.error(f"Archivo no encontrado: {filepath}")
        return None

    try:
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            logger.error(f"La nube de puntos está vacía: {filepath}")
            return None
        logger.info(f"Cargado: {filepath} — {len(pcd.points)} puntos")
        return pcd
    except Exception as e:
        logger.error(f"Error al cargar {filepath}: {e}")
        return None


def save_mesh(mesh, filepath: str) -> bool:
    """
    Guarda una malla 3D en formato .ply / .obj / .stl
    Retorna True si fue exitoso.
    """
    if not OPEN3D_AVAILABLE:
        logger.error("open3d no está instalado.")
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        success = o3d.io.write_triangle_mesh(filepath, mesh)
        if success:
            logger.info(f"Malla guardada: {filepath}")
        else:
            logger.error(f"No se pudo guardar: {filepath}")
        return success
    except Exception as e:
        logger.error(f"Error al guardar {filepath}: {e}")
        return False

