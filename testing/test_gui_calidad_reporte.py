# testing/test_gui_calidad_reporte.py
# Smoke test sin pantalla del panel de calidad y del reporte de sesión.
#
# Corre el pipeline A→B→C completo contra la GUI real (sin mostrarla) y
# comprueba que las métricas que produce el núcleo llegan efectivamente a las
# etiquetas y al reporte exportable. Es la parte de la suite de la sección 6
# que cubre lo añadido en la Fase 1.
#
# Requiere QT_QPA_PLATFORM=offscreen (lo fija este propio archivo) y sustituye
# Viewport3D por un doble: crear el canvas de Vispy sin pantalla falla, y
# además el visor no es lo que se está probando aquí.
#
# Uso:
#   & $PY testing\test_gui_calidad_reporte.py

import json
import os
import sys
import tempfile

# Debe fijarse ANTES de importar cualquier cosa de Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import pyqtSignal

BUNNY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "conejo", "bunny", "reconstruction", "bun_zipper_res2.ply")


class ViewportDoble(QWidget):
    """Doble del visor 3D: misma interfaz, sin OpenGL."""

    transform_committed = pyqtSignal(object, str)
    point_picked        = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ultima_nube  = None
        self.ultima_malla = None

    def show_point_cloud(self, pcd, fit: bool = True):
        self.ultima_nube = pcd

    def show_mesh(self, mesh, fit: bool = True):
        self.ultima_malla = mesh

    def set_edit_mode(self, mode):        pass
    def set_measure_points(self, pts):    pass
    def clear_measure(self):              pass


def _construir_ventana():
    import main_gui
    main_gui.Viewport3D = ViewportDoble          # antes de instanciar
    win = main_gui.MainWindow()
    return main_gui, win


def _ejecutar_bloque(main_gui, win, block, data, params):
    """Corre un bloque de forma síncrona y entrega el resultado a la GUI."""
    worker = main_gui.WorkerThread(block, data, params)
    capturado = {}
    worker.result_signal.connect(lambda r: capturado.update(r))
    worker.error_signal.connect(
        lambda m: (_ for _ in ()).throw(AssertionError(f"worker falló: {m}"))
    )
    worker.run()                                  # síncrono, sin hilo
    assert capturado, f"el bloque {block} no emitió resultado"
    win._on_result(capturado)
    return capturado


def main() -> int:
    if not os.path.exists(BUNNY):
        print(f"✗ No se encuentra el dataset: {BUNNY}")
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    main_gui, win = _construir_ventana()

    fallos = []

    def check(cond, msg):
        if cond:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
            fallos.append(msg)

    # ── Estado inicial ──────────────────────────────────────────────────
    check(not win.act_export_report.isEnabled(),
          "el reporte arranca deshabilitado (no hay nada que reportar)")
    check(win.lbl_q_strategy.text() == "Estrategia: —",
          "el panel de calidad arranca vacío")
    check(all(p.estado() == "pendiente"
              for p in (win.paso_a, win.paso_b, win.paso_c)),
          "los tres pasos arrancan en «pendiente»")
    # isHidden() y no isVisible(): la ventana no se llega a mostrar en el
    # test, y isVisible() es False para cualquier hijo de una ventana oculta.
    check(not win.empty_state.isHidden(),
          "el visor muestra el estado vacío al arrancar")
    check(not win.edit_toolbar.isEnabled(),
          "la barra de edición arranca desactivada (no hay nada que editar)")
    check(not win.btn_run_all.isEnabled(),
          "«Ejecutar todo» arranca desactivado")

    # ── Cargar la nube (sin diálogo de archivo) ─────────────────────────
    win.raw_pcd      = o3d.io.read_point_cloud(BUNNY)
    win._source_path = BUNNY
    win._mostrar_estado_vacio(False)
    win._refresh_pipeline_state()

    check(win.empty_state.isHidden(),
          "el estado vacío desaparece al haber datos")
    check(win.paso_a.estado() == "listo" and win.paso_c.estado() == "pendiente",
          "con solo la nube cargada, A está listo y C sigue pendiente")
    check(win.edit_toolbar.isEnabled(),
          "la barra de edición se activa al haber una nube")

    # ── A → B → C ───────────────────────────────────────────────────────
    p = win.param_panel
    r_a = _ejecutar_bloque(main_gui, win, main_gui.WorkerThread.BLOCK_PREPROCESS,
                           {"point_cloud": win.raw_pcd}, p.get_pre_params())
    r_b = _ejecutar_bloque(main_gui, win, main_gui.WorkerThread.BLOCK_RECONSTRUCT,
                           {"point_cloud": win.clean_pcd}, p.get_rec_params())
    r_c = _ejecutar_bloque(main_gui, win, main_gui.WorkerThread.BLOCK_SOLIDIFY,
                           {"mesh": win.recon_mesh}, p.get_sol_params())

    stats_c = r_c["stats"]

    # ── 1.2 · El panel de calidad refleja las métricas reales ───────────
    legible = win._ESTRATEGIA_NOMBRE.get(stats_c["strategy_used"],
                                         stats_c["strategy_used"])
    check(legible in win.lbl_q_strategy.text(),
          f"la estrategia «{legible}» aparece en el panel")

    fid_txt = win.lbl_q_fidelity.text()
    check("%" in fid_txt and "—" not in fid_txt,
          f"el error de fidelidad se muestra ({fid_txt})")

    check("✓ sí" in win.lbl_q_closed.text(),
          f"el cierre topológico se muestra ({win.lbl_q_closed.text()})")

    check(stats_c["volume"] is not None
          and "no aplica" not in win.lbl_q_volume.text(),
          f"el volumen se muestra ({win.lbl_q_volume.text()})")

    corta = win._ESTRATEGIA_CORTA.get(stats_c["strategy_used"],
                                      stats_c["strategy_used"])
    check(win.lbl_q_cascade.text() != "—"
          and corta in win.lbl_q_cascade.text()
          and "✓" in win.lbl_q_cascade.text(),
          "la cascada marca con ✓ la estrategia que ganó")

    # Ningún identificador interno debe llegar a la interfaz
    textos_ui = " ".join([
        win.lbl_q_strategy.text(), win.lbl_q_cascade.text(),
        win.paso_b.lbl_resumen.text(),
    ])
    check(not any(t in textos_ui for t in ("repair", "ball_pivoting",
                                           "alpha_shape", "passthrough")),
          "la interfaz usa nombres legibles, no los identificadores internos")

    # ── Cada paso conserva su propio resumen ────────────────────────────
    # Antes la malla y el sólido compartían una etiqueta y al terminar C se
    # perdía de vista el resultado de B.
    check(win.paso_a.lbl_resumen.text() != "—"
          and win.paso_b.lbl_resumen.text() != "—"
          and win.paso_c.lbl_resumen.text() != "—",
          "los tres pasos muestran su resumen a la vez")
    check(all(p.estado() == "hecho"
              for p in (win.paso_a, win.paso_b, win.paso_c)),
          "los tres pasos quedan marcados como completados")
    check(all(p.btn_ver.isEnabled()
              for p in (win.paso_a, win.paso_b, win.paso_c)),
          "los tres productos se pueden visualizar")

    # ── Formato numérico en español ─────────────────────────────────────
    resumen_b = win.paso_b.lbl_resumen.text()
    check("." in resumen_b and ",000" not in resumen_b,
          f"los millares usan punto, no coma ({resumen_b.splitlines()[0]})")

    # ── 1.1 · los parámetros relativos llegaron al núcleo ───────────────
    stats_b = r_b["stats"]
    check(stats_b.get("d_bar") and stats_b["d_bar"] > 0,
          f"la reconstrucción reporta d̄ = {stats_b.get('d_bar'):.6f}")
    check(p.get_rec_params()["bp_radius_mode"] == "relative",
          "el modo por defecto de bp_radius es relativo a d̄")

    # ── 1.3 · El reporte de sesión ──────────────────────────────────────
    check(win.act_export_report.isEnabled(),
          "el reporte queda habilitado tras ejecutar")
    check(len(win._runs) == 3,
          f"la bitácora tiene las 3 ejecuciones (tiene {len(win._runs)})")
    check(all(r["duracion_s"] is not None and r["duracion_s"] >= 0
              for r in win._runs),
          "todas las ejecuciones traen su duración")
    check(all(r["parametros"] for r in win._runs),
          "todas las ejecuciones traen sus parámetros")

    rep = win._session_report()
    check(rep["dataset"] == os.path.basename(BUNNY),
          "el reporte identifica el dataset")
    check(rep["tiempo_total_s"] > 0,
          f"el reporte suma el tiempo total ({rep['tiempo_total_s']} s)")

    # JSON: el punto delicado son los escalares de numpy en las stats
    try:
        texto = json.dumps(rep, indent=2, ensure_ascii=False)
        recargado = json.loads(texto)
        check(len(recargado["ejecuciones"]) == 3,
              "el reporte serializa a JSON y se puede releer")
    except (TypeError, ValueError) as e:
        check(False, f"el reporte serializa a JSON — falló: {e}")

    md = win._report_markdown(rep)
    check("# Reporte de sesión" in md
          and "C · Solidificación" in md
          and "strategies_tried" in md,
          "el Markdown incluye los tres bloques y la cascada")

    # Escritura real a disco, en ambos formatos
    with tempfile.TemporaryDirectory() as tmp:
        for nombre, contenido in (("r.json", texto), ("r.md", md)):
            destino = os.path.join(tmp, nombre)
            with open(destino, "w", encoding="utf-8") as f:
                f.write(contenido)
            check(os.path.getsize(destino) > 200,
                  f"{nombre} se escribe en disco con contenido")

    # ── Invalidación: re-ejecutar B debe limpiar el sólido anterior ─────
    _ejecutar_bloque(main_gui, win, main_gui.WorkerThread.BLOCK_RECONSTRUCT,
                     {"point_cloud": win.clean_pcd}, p.get_rec_params())
    check(win.lbl_q_strategy.text() == "Estrategia: —",
          "re-ejecutar B limpia el panel de calidad del sólido anterior")
    # C pierde su producto pero queda «listo», no «pendiente»: la malla de B
    # sigue existiendo, así que se puede volver a solidificar.
    check(win.paso_c.lbl_resumen.text() == "—"
          and win.paso_c.estado() == "listo"
          and not win.paso_c.btn_ver.isEnabled(),
          "re-ejecutar B invalida el producto de C pero lo deja ejecutable")
    check(win.paso_b.estado() == "hecho"
          and win.paso_a.estado() == "hecho",
          "re-ejecutar B conserva A y B")
    check(len(win._runs) == 4,
          "la bitácora conserva también la re-ejecución")

    print()
    if fallos:
        print(f"✗ {len(fallos)} comprobaciones fallaron")
        return 1
    print("✓ Todas las comprobaciones pasaron")
    return 0


if __name__ == "__main__":
    sys.exit(main())
