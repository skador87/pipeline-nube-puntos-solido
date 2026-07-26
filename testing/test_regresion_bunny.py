# testing/test_regresion_bunny.py
# Red de seguridad del pipeline: convierte las métricas que imprime
# test_solid_quality.py en aserciones que fallan solas.
#
# Antes de esto la validación era "correr el harness y comparar los números a
# ojo con los de la memoria". Eso no detecta una regresión pequeña y no deja
# constancia de cuál era la referencia.
#
# Uso (no requiere pytest):
#
#   & $PY testing\test_regresion_bunny.py
#
# Devuelve 0 si todo pasa y 1 si algo falla, así que sirve tal cual en
# cualquier automatización. Si en algún momento se instala pytest, las
# funciones test_* se recogen sin tocar nada.

import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import PointCloudPreprocessor
from core.reconstructor import MeshReconstructor
from testing.test_solid_quality import run_pipeline

# ── Referencia medida sobre el commit 1ccde2a ──────────────────────────────
# Cualquier cambio que empeore estos números es una regresión.
BUNNY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "conejo", "bunny", "reconstruction", "bun_zipper_res2.ply")

REF_FIDELIDAD = 0.003968      # 0.3968 % — error Chamfer/diagonal del sólido
REF_VOLUMEN   = 0.000746
REF_ESTRATEGIA = "repair"

# Tolerancias: la fidelidad no puede empeorar más de un 10 % relativo sobre la
# referencia, y el volumen no puede moverse más de un 5 %. El muestreo de la
# distancia de Chamfer es aleatorio, de ahí que no se exija igualdad exacta.
TOL_FIDELIDAD = 1.10
TOL_VOLUMEN   = 0.05

_cache = {}
mute = lambda *_: None


def _metricas():
    """Corre el pipeline completo una sola vez y cachea el resultado."""
    if "m" not in _cache:
        _cache["m"] = run_pipeline(BUNNY, 0, {}, verbose=False)
    return _cache["m"]


def _nube_limpia():
    if "pcd" not in _cache:
        pcd = o3d.io.read_point_cloud(BUNNY)
        limpia, _ = PointCloudPreprocessor(pcd, log_callback=mute).run(None)
        _cache["pcd"] = limpia
    return _cache["pcd"]


# ══════════════════════════════════════════════════════════════
#  CALIDAD DEL SÓLIDO
# ══════════════════════════════════════════════════════════════

def test_solido_cerrado():
    """El criterio no negociable: el pipeline produce un sólido cerrado."""
    m = _metricas()
    assert m["is_closed"], "el sólido dejó de ser topológicamente cerrado"
    assert m["is_orientable"], "el sólido dejó de ser orientable"


def test_fidelidad_no_empeora():
    """La fidelidad geométrica no puede degradarse respecto a la referencia."""
    m = _metricas()
    fid = m["fidelity_stats"]
    assert fid is not None, "no se calculó el error de fidelidad"
    limite = REF_FIDELIDAD * TOL_FIDELIDAD
    assert fid <= limite, (
        f"fidelidad degradada: {fid:.6f} ({fid:.4%}) supera el límite "
        f"{limite:.6f} ({limite:.4%}); referencia {REF_FIDELIDAD:.4%}"
    )


def test_estrategia_es_reparacion():
    """
    Con el bunny debe ganar 'repair'. Si gana otra, es que la reparación
    dejó de cerrar la malla y se está cayendo a un respaldo que deforma más.
    """
    m = _metricas()
    assert m["strategy_used"] == REF_ESTRATEGIA, (
        f"estrategia inesperada: '{m['strategy_used']}' "
        f"(se esperaba '{REF_ESTRATEGIA}')"
    )


def test_volumen_estable():
    """El volumen encerrado no debe moverse: detecta colapsos y burbujas."""
    m = _metricas()
    vol = m["volume"]
    assert vol is not None, "no se pudo calcular el volumen (¿malla abierta?)"
    desvio = abs(vol - REF_VOLUMEN) / REF_VOLUMEN
    assert desvio <= TOL_VOLUMEN, (
        f"volumen fuera de rango: {vol:.6f} se desvía {desvio:.1%} de "
        f"la referencia {REF_VOLUMEN:.6f}"
    )


# ══════════════════════════════════════════════════════════════
#  ESCALA DE LOS PARÁMETROS DE RECONSTRUCCIÓN
# ══════════════════════════════════════════════════════════════

def test_parametros_relativos_a_la_escala():
    """
    Guarda contra la regresión que motivó el pendiente 1.1: con radios en
    unidades absolutas, Ball Pivoting usaba esferas 4× más grandes que el
    objeto entero y no terminaba nunca.

    Se comprueba el valor RESUELTO antes de reconstruir, no el tiempo de
    ejecución: si el radio vuelve a ser absurdo, este test falla en
    milisegundos en vez de colgarse.
    """
    pcd  = _nube_limpia()
    rec  = MeshReconstructor(pcd, log_callback=mute)
    dbar = rec.characteristic_scale()
    diag = float(np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()))

    assert dbar > 0, "escala característica no válida"

    radio = rec.resolve_bp_radius(rec._DEFAULTS)
    alpha = rec.resolve_alpha(rec._DEFAULTS)

    # Ball Pivoting usa el par (r, 2r): el mayor tiene que seguir siendo
    # pequeño frente al objeto, o el algoritmo degenera.
    assert radio * 2 <= 0.10 * diag, (
        f"bp_radius desproporcionado: el radio mayor {radio * 2:.6f} es "
        f"{radio * 2 / diag:.2f}× la diagonal del objeto ({diag:.6f}). "
        f"Debe ser relativo a d̄ = {dbar:.6f}."
    )
    assert radio >= dbar, (
        f"bp_radius {radio:.6f} menor que d̄ = {dbar:.6f}: la esfera no "
        f"alcanza a los vecinos y no se forma ningún triángulo"
    )
    assert alpha <= 0.10 * diag, (
        f"alpha desproporcionado: {alpha:.6f} es {alpha / diag:.2f}× la "
        f"diagonal del objeto; la envolvente pierde todo el detalle"
    )


def test_metodos_alternativos_terminan():
    """
    ball_pivoting y alpha_shape con sus valores por defecto deben producir
    una malla en un tiempo razonable. Cubre los dos métodos que el modo
    Básico no usa y que por eso nadie ejercita a diario.
    """
    pcd = _nube_limpia()
    for metodo in ("ball_pivoting", "alpha_shape"):
        t0 = time.perf_counter()
        mesh, st = MeshReconstructor(pcd, log_callback=mute).run(
            {"method": metodo}
        )
        dt = time.perf_counter() - t0
        assert st["final_triangles"] > 1000, (
            f"{metodo} produjo solo {st['final_triangles']} triángulos"
        )
        assert dt < 60, f"{metodo} tardó {dt:.0f}s (límite 60s)"


# ══════════════════════════════════════════════════════════════
#  RUNNER SIN PYTEST
# ══════════════════════════════════════════════════════════════

def main() -> int:
    if not os.path.exists(BUNNY):
        print(f"✗ No se encuentra el dataset de referencia: {BUNNY}")
        return 1

    pruebas = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)]

    fallos = []
    for fn in pruebas:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ✓ {fn.__name__}  ({time.perf_counter() - t0:.1f}s)")
        except AssertionError as e:
            fallos.append((fn.__name__, str(e)))
            print(f"  ✗ {fn.__name__}  ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            fallos.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {fn.__name__} — error inesperado")

    print()
    m = _cache.get("m")
    if m:
        print(f"  estrategia = {m['strategy_used']}, "
              f"cerrado = {m['is_closed']}, "
              f"fidelidad = {m['fidelity_stats']:.4%}, "
              f"volumen = {m['volume']:.6f}")

    if fallos:
        print(f"\n✗ {len(fallos)}/{len(pruebas)} pruebas fallaron:\n")
        for nombre, msg in fallos:
            print(f"  [{nombre}]\n    {msg}\n")
        return 1

    print(f"\n✓ {len(pruebas)}/{len(pruebas)} pruebas pasaron")
    return 0


if __name__ == "__main__":
    sys.exit(main())
