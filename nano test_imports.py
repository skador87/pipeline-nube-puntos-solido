import sys, os, traceback
sys.path.insert(0, os.getcwd())
print('PASO 1: imports base OK')

try:
    from utils.helpers import setup_logging, load_point_cloud, save_mesh
    print('PASO 2: utils.helpers OK')
except Exception as e:
    print(f'PASO 2 ERROR: {e}')
    traceback.print_exc()
    sys.exit()

try:
    from core.preprocessor import PointCloudPreprocessor
    print('PASO 3: core.preprocessing OK')
except Exception as e:
    print(f'PASO 3 ERROR: {e}')
    traceback.print_exc()
    sys.exit()

try:
    from core.reconstructor import MeshReconstructor
    print('PASO 4: core.reconstruction OK')
except Exception as e:
    print(f'PASO 4 ERROR: {e}')
    traceback.print_exc()
    sys.exit()

try:
    from core.solidifier import MeshSolidifier
    print('PASO 5: core.solidification OK')
except Exception as e:
    print(f'PASO 5 ERROR: {e}')
    traceback.print_exc()
    sys.exit()

print('')
print('=== TODOS LOS IMPORTS OK ===')
print('=== EL PROBLEMA ESTA EN main() ===')
