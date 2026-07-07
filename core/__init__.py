# core/__init__.py
# Expone los módulos del core para importación directa
from .preprocessor  import PointCloudPreprocessor
from .reconstructor import MeshReconstructor
from .solidifier    import MeshSolidifier

__all__ = [
    'PointCloudPreprocessor',
    'MeshReconstructor', 
    'MeshSolidifier'
]

