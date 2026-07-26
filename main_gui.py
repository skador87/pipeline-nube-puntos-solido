# main_gui.py
# 3D Point Cloud Processor v2.1
# GUI principal — PyQt5 + Vispy
# Compatible con:
#   core/preprocessor.py  → PointCloudPreprocessor.run(params)
#   core/reconstructor.py → MeshReconstructor.run(params)
#   core/solidifier.py    → MeshSolidifier.run(params)
#   core/io_loader.py     → PointCloudLoader.load(path, log_callback)

import sys
import os
import json
import time
import traceback
import numpy as np
import open3d as o3d

from copy import deepcopy
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QScrollArea, QTextEdit,
    QFileDialog, QProgressBar, QTabWidget,
    QSizePolicy, QFrame, QAction, QToolBar,
    QStatusBar, QMessageBox, QMenu, QStackedWidget,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui   import QFont, QColor, QPalette, QIcon, QSurfaceFormat

import vispy
vispy.use(app="pyqt5")           # fija el backend antes de crear cualquier canvas
from vispy         import scene
from vispy.scene   import visuals
from vispy.visuals.transforms import MatrixTransform

# ── Core modules ──────────────────────────────────────────────────────────────
from core.preprocessor import PointCloudPreprocessor
from core.reconstructor import MeshReconstructor
from core.solidifier    import MeshSolidifier

# ── CAMBIO 1: nuevo import ─────────────────────────────────────────────────────
from core.io_loader import (PointCloudLoader, check_e57_support,
                            check_las_support)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMA OSCURO
# ══════════════════════════════════════════════════════════════════════════════

def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()

    dark   = QColor(30,  30,  35)
    mid    = QColor(45,  45,  52)
    light  = QColor(60,  60,  68)
    text   = QColor(220, 220, 220)
    accent = QColor(70,  130, 180)
    hilite = QColor(80,  150, 200)

    pal.setColor(QPalette.Window,          dark)
    pal.setColor(QPalette.WindowText,      text)
    pal.setColor(QPalette.Base,            mid)
    pal.setColor(QPalette.AlternateBase,   light)
    pal.setColor(QPalette.ToolTipBase,     dark)
    pal.setColor(QPalette.ToolTipText,     text)
    pal.setColor(QPalette.Text,            text)
    pal.setColor(QPalette.Button,          mid)
    pal.setColor(QPalette.ButtonText,      text)
    pal.setColor(QPalette.BrightText,      Qt.white)
    pal.setColor(QPalette.Link,            accent)
    pal.setColor(QPalette.Highlight,       hilite)
    pal.setColor(QPalette.HighlightedText, Qt.white)
    pal.setColor(QPalette.Disabled, QPalette.Text,       QColor(120, 120, 120))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))

    app.setPalette(pal)

    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #3a3a44;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 4px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            color: #82b4d0;
        }
        QPushButton {
            background-color: #2d2d36;
            border: 1px solid #4a4a58;
            border-radius: 4px;
            padding: 4px 10px;
            min-height: 24px;
        }
        QPushButton:hover  { background-color: #3a3a48; border-color: #6688aa; }
        QPushButton:pressed{ background-color: #1e1e28; }
        QPushButton:disabled{ color: #666; border-color: #333; }
        QTabWidget::pane   { border: 1px solid #3a3a44; }
        QTabBar::tab {
            background: #2a2a32;
            padding: 5px 14px;
            border: 1px solid #3a3a44;
        }
        QTabBar::tab:selected { background: #3c3c50; color: #aad0f0; }
        QScrollBar:vertical {
            background: #2a2a32; width: 8px;
        }
        QScrollBar::handle:vertical {
            background: #4a4a5a; border-radius: 4px;
        }
        QTextEdit {
            background-color: #1a1a22;
            color: #b0d0b0;
            font-family: monospace;
            font-size: 11px;
            border: 1px solid #3a3a44;
        }
        QProgressBar {
            border: 1px solid #3a3a44;
            border-radius: 3px;
            text-align: center;
            background: #1e1e28;
        }
        QProgressBar::chunk { background-color: #4682b4; border-radius: 2px; }
        QComboBox { background: #2d2d36; border: 1px solid #4a4a58;
                    border-radius: 3px; padding: 2px 6px; }
        QSpinBox, QDoubleSpinBox {
            background: #2d2d36; border: 1px solid #4a4a58;
            border-radius: 3px; padding: 2px 4px;
        }
    """)


def _tt(texto: str) -> str:
    """Envuelve el texto en rich-text para que el tooltip haga word-wrap."""
    return f"<qt>{texto}</qt>"


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal   = pyqtSignal(dict)
    error_signal    = pyqtSignal(str)

    BLOCK_PREPROCESS  = "preprocess"
    BLOCK_RECONSTRUCT = "reconstruct"
    BLOCK_SOLIDIFY    = "solidify"

    def __init__(self, block: str, data: dict, params: dict):
        super().__init__()
        self.block  = block
        self.data   = data
        self.params = params
        self._t0    = None

    def run(self):
        try:
            # Se cronometra aquí, alrededor del bloque completo: es el tiempo
            # que el usuario percibe y el que hay que poder citar en la
            # memoria junto a los parámetros que lo produjeron.
            self._t0 = time.perf_counter()
            if self.block == self.BLOCK_PREPROCESS:
                self._run_preprocess()
            elif self.block == self.BLOCK_RECONSTRUCT:
                self._run_reconstruct()
            elif self.block == self.BLOCK_SOLIDIFY:
                self._run_solidify()
            else:
                raise ValueError(f"Bloque desconocido: {self.block}")
        except Exception as e:
            self.error_signal.emit(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self._t0, 2) if self._t0 else 0.0

    def _run_preprocess(self):
        self.log_signal.emit("━━━ BLOQUE A: Preprocesamiento ━━━")
        self.progress_signal.emit(5)
        pcd = self.data["point_cloud"]
        pre = PointCloudPreprocessor(pcd, log_callback=self.log_signal.emit)
        self.progress_signal.emit(15)
        pcd_clean, stats = pre.run(self.params)
        self.progress_signal.emit(100)
        self.log_signal.emit(
            f"  📊 Resumen: {stats['original_points']:,} → "
            f"{stats['final_points']:,} puntos "
            f"(-{stats['total_removed']:,})"
        )
        self.result_signal.emit({
            "block"       : self.BLOCK_PREPROCESS,
            "point_cloud" : pcd_clean,
            "stats"       : stats,
            "params"      : self.params,
            "elapsed_s"   : self._elapsed(),
        })

    def _run_reconstruct(self):
        self.log_signal.emit("━━━ BLOQUE B: Reconstrucción ━━━")
        self.progress_signal.emit(5)
        pcd = self.data["point_cloud"]
        rec = MeshReconstructor(pcd, log_callback=self.log_signal.emit)
        self.progress_signal.emit(20)
        mesh, stats = rec.run(self.params)
        # Cierre topológico calculado aquí (en el hilo worker) para que la
        # etiqueta "Watertight" de la GUI muestre el valor real sin
        # congelar la interfaz.
        stats["is_closed"] = MeshSolidifier.is_topologically_closed(mesh)
        self.progress_signal.emit(100)
        self.log_signal.emit(
            f"  📊 Malla: {stats['final_vertices']:,} vért, "
            f"{stats['final_triangles']:,} tri "
            f"[método: {stats['method']}]"
        )
        self.result_signal.emit({
            "block"     : self.BLOCK_RECONSTRUCT,
            "mesh"      : mesh,
            "stats"     : stats,
            "params"    : self.params,
            "elapsed_s" : self._elapsed(),
        })

    def _run_solidify(self):
        self.log_signal.emit("━━━ BLOQUE C: Solidificación ━━━")
        self.progress_signal.emit(5)
        mesh = self.data["mesh"]
        sol  = MeshSolidifier(mesh, log_callback=self.log_signal.emit)
        self.progress_signal.emit(20)
        solid_mesh, stats = sol.run(self.params)
        self.progress_signal.emit(100)
        wt = "✓ Watertight" if stats["is_watertight"] else "⚠️ No watertight"
        self.log_signal.emit(
            f"  📊 {wt} | {stats['output_vertices']:,} vért, "
            f"{stats['output_triangles']:,} tri "
            f"[{stats['strategy_used']}]"
        )
        self.result_signal.emit({
            "block"     : self.BLOCK_SOLIDIFY,
            "mesh"      : solid_mesh,
            "stats"     : stats,
            "params"    : self.params,
            "elapsed_s" : self._elapsed(),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL DE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

class ParamPanel(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(290)
        self.setMaximumWidth(340)

        container = QWidget()
        self.setWidget(container)
        root = QVBoxLayout(container)
        root.setSpacing(6)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Selector de modo de uso ─────────────────────────────────────
        g_mode = QGroupBox("Modo de uso")
        gml    = QHBoxLayout(g_mode)
        gml.addWidget(QLabel("Modo:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Básico", "Avanzado"])
        self.mode_combo.setToolTip(_tt(
            "<b>Básico</b>: controles por categorías (sin números); los "
            "parámetros críticos quedan siempre en valores seguros."
            "<br><b>Avanzado</b>: acceso numérico a todos los parámetros, "
            "para especialistas. Muestra los valores que el modo básico "
            "está aplicando."
        ))
        gml.addWidget(self.mode_combo, stretch=1)
        root.addWidget(g_mode)

        # ── Páginas: básico / avanzado ──────────────────────────────────
        self.tabs = QTabWidget()
        self._build_pre_tab()
        self._build_rec_tab()
        self._build_sol_tab()

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_basic_page())   # índice 0
        self.stack.addWidget(self.tabs)                  # índice 1
        root.addWidget(self.stack)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Estado inicial coherente: aplicar los presets del modo básico
        # (equivalen a los valores por defecto validados).
        self._apply_basic_presets()

    # ══════════════════════════════════════════════════════════════
    #  MODO BÁSICO
    # ══════════════════════════════════════════════════════════════

    _SMOOTH_LEVELS = ["Ninguno", "Leve", "Medio", "Alto"]

    def _build_basic_page(self) -> QWidget:
        """
        Página del modo básico: categorías en lugar de números, agrupadas
        por proceso (A, B y C). Cada control se traduce a los parámetros
        numéricos del modo avanzado (que sigue siendo la única fuente de
        verdad para get_*_params), y los parámetros críticos para la
        robustez del sólido quedan siempre fijos en configuración segura.
        """
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setSpacing(6)

        intro = QLabel(
            "Configura cada proceso con categorías simples. Los valores "
            "numéricos que se aplican pueden verse en el modo Avanzado."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9ab0c0;")
        lay.addWidget(intro)

        # ── A · Preprocesamiento ────────────────────────────────────────
        g_a = QGroupBox("A · Preprocesamiento")
        ga  = QGridLayout(g_a)

        ga.addWidget(QLabel("Limpieza de ruido:"), 0, 0)
        self.basic_clean = QComboBox()
        self.basic_clean.addItems(["Suave", "Media", "Agresiva"])
        self.basic_clean.setCurrentIndex(1)
        self.basic_clean.setToolTip(_tt(
            "Cuánto ruido se elimina de la nube (filtros ROR y SOR)."
            "<br><b>Suave</b>: conserva casi todo (escaneos limpios o con "
            "detalle fino que no se quiere perder)."
            "<br><b>Media</b>: equilibrio recomendado."
            "<br><b>Agresiva</b>: escaneos con mucho ruido o puntos "
            "«voladores»; puede comerse bordes y zonas poco densas."
        ))
        ga.addWidget(self.basic_clean, 0, 1)

        ga.addWidget(QLabel("Densidad de trabajo:"), 1, 0)
        self.basic_density = QComboBox()
        self.basic_density.addItems(
            ["Baja (rápida)", "Media", "Alta (detalle)"])
        self.basic_density.setCurrentIndex(1)
        self.basic_density.setToolTip(_tt(
            "Cuántos puntos se conservan para trabajar (voxelización)."
            "<br><b>Baja</b>: procesos rápidos, ideal para probar "
            "parámetros."
            "<br><b>Media</b>: equilibrio recomendado."
            "<br><b>Alta</b>: conserva el máximo de puntos; todo el "
            "pipeline se vuelve más lento."
        ))
        ga.addWidget(self.basic_density, 1, 1)

        ga.addWidget(QLabel("Reducción de ruido fino:"), 2, 0)
        self.basic_mls = QComboBox()
        self.basic_mls.addItems(["Ninguna", "Leve", "Media"])
        self.basic_mls.setCurrentIndex(1)
        self.basic_mls.setToolTip(_tt(
            "Suavizado MLS de la nube: reduce la «rugosidad» de "
            "medición proyectando cada punto sobre la superficie local."
            "<br><b>Ninguna</b>: conserva los puntos tal cual."
            "<br><b>Leve</b>: 1 pasada (recomendado)."
            "<br><b>Media</b>: 2 pasadas, para escaneos muy rugosos."
        ))
        ga.addWidget(self.basic_mls, 2, 1)

        lay.addWidget(g_a)

        # ── B · Reconstrucción ──────────────────────────────────────────
        g_b = QGroupBox("B · Reconstrucción")
        gb  = QGridLayout(g_b)

        gb.addWidget(QLabel("Nivel de detalle:"), 0, 0)
        self.basic_detail = QComboBox()
        self.basic_detail.addItems(["Bajo", "Medio", "Alto"])
        self.basic_detail.setCurrentIndex(1)
        self.basic_detail.setToolTip(_tt(
            "Resolución de la malla reconstruida (profundidad Poisson)."
            "<br><b>Bajo</b>: rápido, para pruebas o piezas simples."
            "<br><b>Medio</b>: equilibrio recomendado."
            "<br><b>Alto</b>: máximo detalle; más lento y produce mallas "
            "mucho más pesadas."
        ))
        gb.addWidget(self.basic_detail, 0, 1)

        gb.addWidget(QLabel("Suavizado de superficie:"), 1, 0)
        sm_box = QVBoxLayout()
        self.basic_smooth = QSlider(Qt.Horizontal)
        self.basic_smooth.setRange(0, 3)
        self.basic_smooth.setValue(2)
        self.basic_smooth.setTickPosition(QSlider.TicksBelow)
        self.basic_smooth.setTickInterval(1)
        self.basic_smooth.setToolTip(_tt(
            "Cuánto se alisa la superficie reconstruida (método Taubin, "
            "que no encoge el objeto). Más suavizado = superficie más "
            "limpia pero menos detalle fino."
        ))
        sm_box.addWidget(self.basic_smooth)
        self.basic_smooth_label = QLabel(self._SMOOTH_LEVELS[2])
        self.basic_smooth_label.setAlignment(Qt.AlignCenter)
        self.basic_smooth_label.setStyleSheet("color: #9ab0c0;")
        sm_box.addWidget(self.basic_smooth_label)
        gb.addLayout(sm_box, 1, 1)

        self.basic_artifacts = QCheckBox("Eliminar artefactos")
        self.basic_artifacts.setChecked(True)
        self.basic_artifacts.setToolTip(_tt(
            "Elimina las «telas fantasma» (superficie inventada al "
            "puentear zonas sin datos) y las paredes internas espurias. "
            "Recomendado dejarlo activado; desactivar solo si el filtro "
            "se está comiendo geometría real."
        ))
        gb.addWidget(self.basic_artifacts, 2, 0, 1, 2)

        lay.addWidget(g_b)

        # ── C · Solidificación ──────────────────────────────────────────
        g_c = QGroupBox("C · Solidificación")
        gc  = QGridLayout(g_c)

        gc.addWidget(QLabel("Fidelidad del sólido:"), 0, 0)
        self.basic_fidelity = QComboBox()
        self.basic_fidelity.addItems(
            ["Estricta", "Equilibrada", "Flexible"])
        self.basic_fidelity.setCurrentIndex(1)
        self.basic_fidelity.setToolTip(_tt(
            "Cuánta deformación se acepta al cerrar el sólido."
            "<br><b>Estricta</b> (1%): objetos con concavidades "
            "importantes que no se pueden perder."
            "<br><b>Equilibrada</b> (2%): recomendada."
            "<br><b>Flexible</b> (5%): escaneos muy incompletos, donde "
            "hay que «inventar» superficie para lograr el cierre."
        ))
        gc.addWidget(self.basic_fidelity, 0, 1)

        gc.addWidget(QLabel("Resolución del cierre:"), 1, 0)
        self.basic_close_res = QComboBox()
        self.basic_close_res.addItems(["Ligera", "Media", "Alta"])
        self.basic_close_res.setCurrentIndex(1)
        self.basic_close_res.setToolTip(_tt(
            "Resolución de la envolvente Poisson usada como respaldo "
            "cuando la reparación directa no logra cerrar la malla."
            "<br><b>Ligera</b>: sólido liviano, cierre más grueso."
            "<br><b>Media</b>: recomendada."
            "<br><b>Alta</b>: cierre más fino; el sólido resultante pesa "
            "bastante más."
        ))
        gc.addWidget(self.basic_close_res, 1, 1)

        lay.addWidget(g_c)

        nota = QLabel(
            "ℹ️ En modo básico la reconstrucción usa siempre Poisson y la "
            "solidificación mantiene la cascada completa de estrategias "
            "con control de calidad: la configuración validada más robusta."
        )
        nota.setWordWrap(True)
        nota.setStyleSheet("color: #7a8a7a; font-size: 11px;")
        lay.addWidget(nota)
        lay.addStretch()

        # Cualquier cambio en las categorías se traduce de inmediato a los
        # parámetros numéricos del modo avanzado.
        for combo in (self.basic_clean, self.basic_density, self.basic_mls,
                      self.basic_detail, self.basic_fidelity,
                      self.basic_close_res):
            combo.currentIndexChanged.connect(self._apply_basic_presets)
        self.basic_smooth.valueChanged.connect(self._apply_basic_presets)
        self.basic_artifacts.toggled.connect(self._apply_basic_presets)

        return page

    def _on_mode_changed(self, index: int):
        self.stack.setCurrentIndex(index)
        # Al volver al modo básico se re-aplican los presets: garantiza un
        # estado predecible aunque se haya experimentado en avanzado.
        if index == 0:
            self._apply_basic_presets()

    def _apply_basic_presets(self):
        """
        Traduce las categorías del modo básico (por proceso A/B/C) a los
        parámetros numéricos del modo avanzado y fija los resguardos de
        robustez del sólido.
        """
        # ══ A · Preprocesamiento ════════════════════════════════════════
        # Limpieza de ruido: (ror_factor, ror_min, sor_k, sor_std)
        ror_f, ror_min, sor_k, sor_std = [
            (3.0, 3, 20, 3.0),      # Suave
            (2.5, 4, 20, 2.0),      # Media  (= defaults validados)
            (2.0, 6, 30, 1.5),      # Agresiva
        ][self.basic_clean.currentIndex()]
        self.ror_enabled.setChecked(True)
        self.sor_enabled.setChecked(True)
        self.pre_dedup_enabled.setChecked(True)
        self.ror_factor.setValue(ror_f)
        self.ror_min_neighbors.setValue(ror_min)
        self.sor_k.setValue(sor_k)
        self.sor_std.setValue(sor_std)

        # Densidad de trabajo: voxel_factor (más grande = menos puntos)
        self.pre_voxel_enabled.setChecked(True)
        self.pre_voxel_factor.setValue(
            [2.5, 1.5, 1.0][self.basic_density.currentIndex()])

        # Reducción de ruido fino: suavizado MLS
        mls_iters = [0, 1, 2][self.basic_mls.currentIndex()]
        self.pre_denoise_enabled.setChecked(mls_iters > 0)
        if mls_iters > 0:
            self.pre_denoise_iter.setValue(mls_iters)
        self.pre_preserve_edges.setChecked(True)

        # ══ B · Reconstrucción ══════════════════════════════════════════
        # Nivel de detalle: profundidad Poisson
        self.rec_method.setCurrentText("poisson")
        self.poisson_depth.setValue(
            [8, 10, 11][self.basic_detail.currentIndex()])

        # Suavizado de superficie: iteraciones Taubin (0 = desactivado)
        nivel = self.basic_smooth.value()
        if hasattr(self, "basic_smooth_label"):
            self.basic_smooth_label.setText(self._SMOOTH_LEVELS[nivel])
        iters = [0, 3, 5, 10][nivel]
        if iters == 0:
            self.smooth_method.setCurrentText("none")
        else:
            self.smooth_method.setCurrentText("taubin")
            self.smooth_iter.setValue(iters)
            self.taubin_lambda.setValue(0.5)
            self.taubin_mu.setValue(-0.53)

        # Filtros de artefactos (un solo interruptor en modo básico)
        artefactos = self.basic_artifacts.isChecked()
        self.remove_webbing.setChecked(artefactos)
        self.remove_hollow.setChecked(artefactos)

        # ══ C · Solidificación ══════════════════════════════════════════
        # Fidelidad del sólido: tolerancia Chamfer (%)
        self.sol_fidelity_max.setValue(
            [1.0, 2.0, 5.0][self.basic_fidelity.currentIndex()])

        # Resolución del cierre: profundidad de la envolvente Poisson
        self.sol_poisson_depth.setValue(
            [7, 8, 9][self.basic_close_res.currentIndex()])

        # ── Resguardos de robustez (siempre fijos en modo básico) ──────
        self.pre_remove_invalid.setChecked(True)
        # Escala relativa a d̄: en modo absoluto un radio mal elegido puede
        # dejar Ball Pivoting corriendo indefinidamente.
        self.bp_radius_mode.setCurrentIndex(0)
        self.alpha_mode.setCurrentIndex(0)
        self.bp_radius_factor.setValue(2.0)
        self.alpha_factor.setValue(5.0)
        self.sol_voxel_auto.setChecked(True)
        self.sol_merge_eps.setValue(0.5)
        self.sol_strategy_repair.setChecked(True)
        self.sol_strategy_poisson.setChecked(True)
        self.sol_strategy_voxel.setChecked(True)
        self.sol_strategy_hull.setChecked(True)
        self.sol_quality_check.setChecked(True)
        self.sol_fill_holes.setChecked(True)

    def _build_pre_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        # ── Validación ──────────────────────────────────────────────────
        g_val = QGroupBox("Validación")
        gv    = QVBoxLayout(g_val)
        self.pre_remove_invalid = QCheckBox("Quitar NaN / Inf")
        self.pre_remove_invalid.setChecked(True)
        gv.addWidget(self.pre_remove_invalid)
        self.pre_remove_origin = QCheckBox("Quitar puntos en el origen (0,0,0)")
        self.pre_remove_origin.setChecked(True)
        gv.addWidget(self.pre_remove_origin)
        lay.addWidget(g_val)

        # ── Deduplicación ───────────────────────────────────────────────
        g_ded = QGroupBox("Deduplicación (× d̄)")
        gd    = QGridLayout(g_ded)
        self.pre_dedup_enabled = QCheckBox("Activar")
        self.pre_dedup_enabled.setChecked(True)
        gd.addWidget(self.pre_dedup_enabled, 0, 0, 1, 2)
        gd.addWidget(QLabel("Factor:"), 1, 0)
        self.pre_dedup_factor = QDoubleSpinBox()
        self.pre_dedup_factor.setRange(0.01, 1.0)
        self.pre_dedup_factor.setSingleStep(0.05)
        self.pre_dedup_factor.setValue(0.2)
        gd.addWidget(self.pre_dedup_factor, 1, 1)
        lay.addWidget(g_ded)

        # ── Radius Outlier Removal ──────────────────────────────────────
        g_ror = QGroupBox("Radius Outlier Removal (ROR)")
        gl2   = QGridLayout(g_ror)
        self.ror_enabled = QCheckBox("Activar ROR")
        self.ror_enabled.setChecked(True)
        gl2.addWidget(self.ror_enabled, 0, 0, 1, 2)
        gl2.addWidget(QLabel("Factor radio (× d̄):"), 1, 0)
        self.ror_factor = QDoubleSpinBox()
        self.ror_factor.setRange(0.5, 20.0)
        self.ror_factor.setSingleStep(0.5)
        self.ror_factor.setValue(2.5)
        gl2.addWidget(self.ror_factor, 1, 1)
        gl2.addWidget(QLabel("Mín. vecinos:"), 2, 0)
        self.ror_min_neighbors = QSpinBox()
        self.ror_min_neighbors.setRange(1, 30)
        self.ror_min_neighbors.setValue(4)
        gl2.addWidget(self.ror_min_neighbors, 2, 1)
        lay.addWidget(g_ror)

        # ── Statistical Outlier Removal ─────────────────────────────────
        g_sor = QGroupBox("Statistical Outlier Removal (SOR)")
        gl    = QGridLayout(g_sor)
        self.sor_enabled = QCheckBox("Activar SOR")
        self.sor_enabled.setChecked(True)
        gl.addWidget(self.sor_enabled, 0, 0, 1, 2)
        gl.addWidget(QLabel("k vecinos:"), 1, 0)
        self.sor_k = QSpinBox()
        self.sor_k.setRange(5, 100)
        self.sor_k.setValue(20)
        gl.addWidget(self.sor_k, 1, 1)
        gl.addWidget(QLabel("σ ratio:"), 2, 0)
        self.sor_std = QDoubleSpinBox()
        self.sor_std.setRange(0.1, 10.0)
        self.sor_std.setSingleStep(0.1)
        self.sor_std.setValue(2.0)
        gl.addWidget(self.sor_std, 2, 1)
        lay.addWidget(g_sor)

        # ── Voxelización ────────────────────────────────────────────────
        g_vox = QGroupBox("Voxelización (× d̄)")
        gvx   = QGridLayout(g_vox)
        self.pre_voxel_enabled = QCheckBox("Activar (uniformar densidad)")
        self.pre_voxel_enabled.setChecked(True)
        gvx.addWidget(self.pre_voxel_enabled, 0, 0, 1, 2)
        gvx.addWidget(QLabel("Factor:"), 1, 0)
        self.pre_voxel_factor = QDoubleSpinBox()
        self.pre_voxel_factor.setRange(0.5, 5.0)
        self.pre_voxel_factor.setSingleStep(0.1)
        self.pre_voxel_factor.setValue(1.5)
        gvx.addWidget(self.pre_voxel_factor, 1, 1)
        lay.addWidget(g_vox)

        # ── Suavizado MLS edge-aware ────────────────────────────────────
        g_mls = QGroupBox("Suavizado MLS (no destructivo)")
        gm    = QGridLayout(g_mls)
        self.pre_denoise_enabled = QCheckBox("Activar")
        self.pre_denoise_enabled.setChecked(True)
        gm.addWidget(self.pre_denoise_enabled, 0, 0, 1, 2)
        gm.addWidget(QLabel("Iteraciones:"), 1, 0)
        self.pre_denoise_iter = QSpinBox()
        self.pre_denoise_iter.setRange(1, 10)
        self.pre_denoise_iter.setValue(1)
        gm.addWidget(self.pre_denoise_iter, 1, 1)
        gm.addWidget(QLabel("Vecinos (k):"), 2, 0)
        self.pre_denoise_k = QSpinBox()
        self.pre_denoise_k.setRange(6, 60)
        self.pre_denoise_k.setValue(16)
        gm.addWidget(self.pre_denoise_k, 2, 1)
        self.pre_preserve_edges = QCheckBox("Preservar aristas")
        self.pre_preserve_edges.setChecked(True)
        gm.addWidget(self.pre_preserve_edges, 3, 0, 1, 2)
        lay.addWidget(g_mls)

        # ── Normales ────────────────────────────────────────────────────
        g_nrm = QGroupBox("Normales")
        gn    = QGridLayout(g_nrm)
        gn.addWidget(QLabel("Factor radio (× voxel):"), 0, 0)
        self.pre_normal_factor = QDoubleSpinBox()
        self.pre_normal_factor.setRange(1.0, 10.0)
        self.pre_normal_factor.setSingleStep(0.5)
        self.pre_normal_factor.setValue(4.0)
        gn.addWidget(self.pre_normal_factor, 0, 1)
        gn.addWidget(QLabel("Máx. vecinos:"), 1, 0)
        self.pre_normal_max_nn = QSpinBox()
        self.pre_normal_max_nn.setRange(5, 100)
        self.pre_normal_max_nn.setValue(30)
        gn.addWidget(self.pre_normal_max_nn, 1, 1)
        gn.addWidget(QLabel("Orientación k:"), 2, 0)
        self.pre_orient_k = QSpinBox()
        self.pre_orient_k.setRange(5, 100)
        self.pre_orient_k.setValue(30)
        gn.addWidget(self.pre_orient_k, 2, 1)
        lay.addWidget(g_nrm)

        lay.addStretch()

        # ── Tooltips explicativos ───────────────────────────────────────
        # d̄ = mediana de la distancia al vecino más cercano (la resolución
        # real de la nube); los umbrales espaciales se expresan como ×d̄.
        tips = {
            self.pre_remove_invalid:
                "Elimina puntos con coordenadas NaN o infinitas (mediciones "
                "fallidas del escáner). No cuesta nada: dejar siempre activado.",
            self.pre_remove_origin:
                "Elimina puntos exactamente en (0,0,0), donde muchos escáneres "
                "láser registran los retornos fallidos. Desactivar solo si el "
                "objeto realmente tiene puntos en el origen.",
            self.pre_dedup_enabled:
                "Fusiona puntos duplicados o casi coincidentes. Útil cuando se "
                "combinan varios escaneos de la misma zona.",
            self.pre_dedup_factor:
                "Distancia (× d̄) bajo la cual dos puntos se consideran el "
                "mismo. 0.1 = solo duplicados casi exactos; 0.3+ ya reduce "
                "densidad. Típico: 0.2.",
            self.ror_enabled:
                "Radius Outlier Removal: elimina puntos con pocos vecinos "
                "dentro de un radio — ruido disperso y puntos «voladores».",
            self.ror_factor:
                "Radio de búsqueda (× d̄). Más grande = más tolerante "
                "(elimina menos). Típico: 2–3.",
            self.ror_min_neighbors:
                "Mínimo de vecinos dentro del radio para conservar el punto. "
                "Subirlo limpia más ruido, pero puede comerse bordes y zonas "
                "poco densas.",
            self.sor_enabled:
                "Statistical Outlier Removal: elimina puntos cuya distancia "
                "media a sus k vecinos se aleja estadísticamente del resto. "
                "Bueno contra ruido difuso general.",
            self.sor_k:
                "Vecinos usados para calcular la distancia media de cada "
                "punto. Típico: 20.",
            self.sor_std:
                "Umbral en desviaciones estándar (σ): menor = más agresivo. "
                "2.0 es equilibrado; 1.0 elimina bastante más.",
            self.pre_voxel_enabled:
                "Uniforma la densidad conservando un punto por celda de "
                "rejilla. Acelera el pipeline y estabiliza la reconstrucción.",
            self.pre_voxel_factor:
                "Tamaño de celda (× d̄). 1.0–1.5 casi no pierde detalle; 3+ "
                "reduce fuerte la nube. Si la reconstrucción pierde detalle "
                "fino, bajar este factor.",
            self.pre_denoise_enabled:
                "Suavizado Moving Least Squares: proyecta cada punto sobre "
                "una superficie local ajustada. Reduce rugosidad sin encoger "
                "el objeto.",
            self.pre_denoise_iter:
                "Más iteraciones = superficie más lisa, pero puede borrar "
                "detalle fino. 1–2 suele bastar.",
            self.pre_denoise_k:
                "Vecinos usados para ajustar la superficie local. Más "
                "vecinos = suavizado más amplio.",
            self.pre_preserve_edges:
                "Reduce el suavizado cerca de aristas y esquinas detectadas "
                "para no redondearlas.",
            self.pre_normal_factor:
                "Radio de búsqueda para estimar normales (× voxel). Muy "
                "chico = normales ruidosas; muy grande = normales «lavadas» "
                "en los detalles.",
            self.pre_normal_max_nn:
                "Tope de vecinos por punto al estimar la normal (limita el "
                "costo computacional).",
            self.pre_orient_k:
                "Vecinos del grafo usado para orientar las normales de forma "
                "consistente (hacia afuera). Crucial para Poisson: normales "
                "mal orientadas producen burbujas y superficies dobles.",
        }
        for w, tip in tips.items():
            w.setToolTip(_tt(tip))

        self.tabs.addTab(tab, "A · Pre")

    def _build_rec_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        g_method = QGroupBox("Método de reconstrucción")
        gml      = QGridLayout(g_method)

        gml.addWidget(QLabel("Método:"), 0, 0)
        self.rec_method = QComboBox()
        self.rec_method.addItems(["poisson", "ball_pivoting", "alpha_shape"])
        gml.addWidget(self.rec_method, 0, 1)

        gml.addWidget(QLabel("Poisson depth:"), 1, 0)
        self.poisson_depth = QSpinBox()
        self.poisson_depth.setRange(4, 16)
        self.poisson_depth.setValue(10)
        gml.addWidget(self.poisson_depth, 1, 1)

        gml.addWidget(QLabel("BP radio — modo:"), 2, 0)
        self.bp_radius_mode = QComboBox()
        self.bp_radius_mode.addItems(["relativo a d̄", "absoluto"])
        gml.addWidget(self.bp_radius_mode, 2, 1)

        gml.addWidget(QLabel("    factor × d̄:"), 3, 0)
        self.bp_radius_factor = QDoubleSpinBox()
        self.bp_radius_factor.setRange(0.5, 20.0)
        self.bp_radius_factor.setSingleStep(0.5)
        self.bp_radius_factor.setValue(2.0)
        gml.addWidget(self.bp_radius_factor, 3, 1)

        gml.addWidget(QLabel("    valor absoluto:"), 4, 0)
        self.bp_radius = QDoubleSpinBox()
        self.bp_radius.setRange(0.0001, 10.0)
        self.bp_radius.setSingleStep(0.001)
        self.bp_radius.setDecimals(4)
        self.bp_radius.setValue(1.0)
        self.bp_radius.setEnabled(False)
        gml.addWidget(self.bp_radius, 4, 1)

        gml.addWidget(QLabel("Alpha — modo:"), 5, 0)
        self.alpha_mode = QComboBox()
        self.alpha_mode.addItems(["relativo a d̄", "absoluto"])
        gml.addWidget(self.alpha_mode, 5, 1)

        gml.addWidget(QLabel("    factor × d̄:"), 6, 0)
        self.alpha_factor = QDoubleSpinBox()
        self.alpha_factor.setRange(0.5, 50.0)
        self.alpha_factor.setSingleStep(0.5)
        self.alpha_factor.setValue(5.0)
        gml.addWidget(self.alpha_factor, 6, 1)

        gml.addWidget(QLabel("    valor absoluto:"), 7, 0)
        self.alpha_val = QDoubleSpinBox()
        self.alpha_val.setRange(0.0001, 5.0)
        self.alpha_val.setSingleStep(0.001)
        self.alpha_val.setDecimals(4)
        self.alpha_val.setValue(0.1)
        self.alpha_val.setEnabled(False)
        gml.addWidget(self.alpha_val, 7, 1)

        # El campo activo depende del modo: así queda claro cuál de los dos
        # valores se está aplicando de verdad.
        self.bp_radius_mode.currentIndexChanged.connect(
            lambda i: (self.bp_radius_factor.setEnabled(i == 0),
                       self.bp_radius.setEnabled(i == 1))
        )
        self.alpha_mode.currentIndexChanged.connect(
            lambda i: (self.alpha_factor.setEnabled(i == 0),
                       self.alpha_val.setEnabled(i == 1))
        )

        gml.addWidget(QLabel("Alpha downsample:"), 8, 0)
        self.alpha_ds = QSpinBox()
        self.alpha_ds.setRange(1, 20)
        self.alpha_ds.setValue(1)
        gml.addWidget(self.alpha_ds, 8, 1)

        lay.addWidget(g_method)

        g_clean = QGroupBox("Limpieza de artefactos")
        gcl     = QVBoxLayout(g_clean)

        self.remove_webbing = QCheckBox("Eliminar phantom webbing")
        self.remove_webbing.setChecked(True)
        gcl.addWidget(self.remove_webbing)

        self.remove_hollow = QCheckBox("Eliminar zonas huecas")
        self.remove_hollow.setChecked(True)
        gcl.addWidget(self.remove_hollow)

        lay.addWidget(g_clean)

        g_smooth = QGroupBox("Suavizado")
        gsl      = QGridLayout(g_smooth)

        gsl.addWidget(QLabel("Método:"), 0, 0)
        self.smooth_method = QComboBox()
        self.smooth_method.addItems(["none", "taubin", "laplacian"])
        self.smooth_method.setCurrentIndex(1)
        gsl.addWidget(self.smooth_method, 0, 1)

        gsl.addWidget(QLabel("Iteraciones:"), 1, 0)
        self.smooth_iter = QSpinBox()
        self.smooth_iter.setRange(1, 50)
        self.smooth_iter.setValue(5)
        gsl.addWidget(self.smooth_iter, 1, 1)

        gsl.addWidget(QLabel("Taubin λ:"), 2, 0)
        self.taubin_lambda = QDoubleSpinBox()
        self.taubin_lambda.setRange(0.01, 1.0)
        self.taubin_lambda.setSingleStep(0.01)
        self.taubin_lambda.setValue(0.5)
        gsl.addWidget(self.taubin_lambda, 2, 1)

        gsl.addWidget(QLabel("Taubin μ:"), 3, 0)
        self.taubin_mu = QDoubleSpinBox()
        self.taubin_mu.setRange(-1.0, -0.01)
        self.taubin_mu.setSingleStep(0.01)
        self.taubin_mu.setValue(-0.53)
        gsl.addWidget(self.taubin_mu, 3, 1)

        gsl.addWidget(QLabel("Laplacian λ:"), 4, 0)
        self.laplacian_lambda = QDoubleSpinBox()
        self.laplacian_lambda.setRange(0.01, 1.0)
        self.laplacian_lambda.setSingleStep(0.01)
        self.laplacian_lambda.setValue(0.5)
        gsl.addWidget(self.laplacian_lambda, 4, 1)

        lay.addWidget(g_smooth)
        lay.addStretch()

        # ── Tooltips explicativos ───────────────────────────────────────
        tips = {
            self.rec_method:
                "<b>poisson</b>: superficie implícita, cerrada y suave — la "
                "mejor opción general.<br><b>ball_pivoting</b>: triangula los "
                "puntos reales (fiel a los datos, pero deja agujeros)."
                "<br><b>alpha_shape</b>: envolvente ajustada, útil para "
                "formas simples.",
            self.poisson_depth:
                "Profundidad del octree = resolución de la superficie. 8–9 "
                "para objetos simples, 10–11 para detalle fino. Cada +1 "
                "duplica la resolución y multiplica triángulos y tiempo.",
            self.bp_radius_mode:
                "<b>relativo a d̄</b> (recomendado): el radio se calcula "
                "sobre la escala de la nube, así el mismo valor sirve para "
                "el bunny y para un escaneo en metros."
                "<br><b>absoluto</b>: radio fijo en unidades de la nube. "
                "Un valor mayor que el objeto hace que Ball Pivoting no "
                "termine nunca.",
            self.bp_radius_factor:
                "Radio de la esfera pivotante como múltiplo de d̄ (la "
                "distancia típica entre puntos vecinos). También se usa "
                "radio×2. Orientativo: 1.5–3. Por debajo de 1 la esfera no "
                "alcanza a los vecinos y no se forma ningún triángulo.",
            self.bp_radius:
                "Radio de la esfera pivotante en unidades de la nube. Solo "
                "se aplica con el modo «absoluto». Debe ser mucho menor que "
                "el objeto: como referencia, el bunny mide 0.25 de diagonal.",
            self.alpha_mode:
                "<b>relativo a d̄</b> (recomendado): alpha proporcional al "
                "espaciado real de la nube."
                "<br><b>absoluto</b>: valor fijo en unidades de la nube.",
            self.alpha_factor:
                "Alpha como múltiplo de d̄. Menor = envolvente más ajustada "
                "al detalle (puede fragmentarse); mayor = más gruesa y "
                "cerrada. Orientativo: 3–8.",
            self.alpha_val:
                "Alpha en unidades de la nube. Solo se aplica con el modo "
                "«absoluto».",
            self.alpha_ds:
                "Submuestreo previo para Alpha Shape (1 = usar todos los "
                "puntos). Subirlo acelera a costa de detalle.",
            self.remove_webbing:
                "Elimina las «telas fantasma» que Poisson genera al puentear "
                "zonas sin datos (triángulos de baja densidad o lejos de la "
                "nube). Recomendado con Poisson.",
            self.remove_hollow:
                "Elimina paredes internas espurias: triángulos con puntos de "
                "la nube claramente a ambos lados (señal de doble pared). El "
                "radio de análisis se adapta a la escala de la nube.",
            self.smooth_method:
                "<b>taubin</b>: suaviza SIN encoger el objeto (recomendado)."
                "<br><b>laplacian</b>: más simple pero encoge y redondea."
                "<br><b>none</b>: sin suavizado.",
            self.smooth_iter:
                "Iteraciones de suavizado. 5–10 típico con Taubin; muchas "
                "iteraciones con Laplacian «derriten» el detalle.",
            self.taubin_lambda:
                "Paso de suavizado (expansión) de Taubin. Típico 0.5.",
            self.taubin_mu:
                "Paso de contracción negativo que compensa el encogimiento "
                "del suavizado. Típico -0.53 (magnitud levemente mayor a λ).",
            self.laplacian_lambda:
                "Peso del suavizado Laplacian: mayor = más agresivo (y más "
                "encogimiento del objeto).",
        }
        for w, tip in tips.items():
            w.setToolTip(_tt(tip))

        self.tabs.addTab(tab, "B · Rec")

    def _build_sol_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(6)

        g_vox = QGroupBox("Voxel Size")
        gvl   = QGridLayout(g_vox)

        self.sol_voxel_auto = QCheckBox("Calcular automáticamente")
        self.sol_voxel_auto.setChecked(True)
        gvl.addWidget(self.sol_voxel_auto, 0, 0, 1, 2)

        gvl.addWidget(QLabel("Voxel size:"), 1, 0)
        self.sol_voxel_size = QDoubleSpinBox()
        self.sol_voxel_size.setRange(0.0001, 1.0)
        self.sol_voxel_size.setSingleStep(0.001)
        self.sol_voxel_size.setDecimals(5)
        self.sol_voxel_size.setValue(0.01)
        gvl.addWidget(self.sol_voxel_size, 1, 1)

        self.sol_voxel_auto.toggled.connect(
            lambda checked: self.sol_voxel_size.setEnabled(not checked)
        )
        self.sol_voxel_size.setEnabled(False)

        lay.addWidget(g_vox)

        g_strat = QGroupBox("Estrategias de solidificación")
        gstl    = QVBoxLayout(g_strat)

        self.sol_strategy_repair = QCheckBox(
            "Estrategia 1: Reparación topológica")
        self.sol_strategy_repair.setChecked(True)
        gstl.addWidget(self.sol_strategy_repair)

        self.sol_strategy_poisson = QCheckBox(
            "Estrategia 2: Envolvente Poisson\n(preserva concavidades)")
        self.sol_strategy_poisson.setChecked(True)
        gstl.addWidget(self.sol_strategy_poisson)

        hp = QHBoxLayout()
        hp.addWidget(QLabel("    Profundidad Poisson:"))
        self.sol_poisson_depth = QSpinBox()
        self.sol_poisson_depth.setRange(6, 12)
        self.sol_poisson_depth.setValue(8)
        hp.addWidget(self.sol_poisson_depth)
        gstl.addLayout(hp)

        self.sol_strategy_voxel = QCheckBox(
            "Estrategia 3: Voxelización + Ball Pivoting")
        self.sol_strategy_voxel.setChecked(True)
        gstl.addWidget(self.sol_strategy_voxel)

        self.sol_strategy_hull = QCheckBox(
            "Estrategia 4: Convex Hull (último recurso)")
        self.sol_strategy_hull.setChecked(True)
        gstl.addWidget(self.sol_strategy_hull)

        lay.addWidget(g_strat)

        g_qc = QGroupBox("Control de calidad")
        gql  = QGridLayout(g_qc)

        self.sol_quality_check = QCheckBox(
            "Validar fidelidad geométrica (Chamfer)")
        self.sol_quality_check.setChecked(True)
        gql.addWidget(self.sol_quality_check, 0, 0, 1, 2)

        gql.addWidget(QLabel("Error máx. (% diagonal):"), 1, 0)
        self.sol_fidelity_max = QDoubleSpinBox()
        self.sol_fidelity_max.setRange(0.1, 20.0)
        self.sol_fidelity_max.setSingleStep(0.5)
        self.sol_fidelity_max.setDecimals(1)
        self.sol_fidelity_max.setValue(2.0)
        gql.addWidget(self.sol_fidelity_max, 1, 1)

        lay.addWidget(g_qc)

        g_rep = QGroupBox("Parámetros de reparación")
        grl   = QGridLayout(g_rep)

        grl.addWidget(QLabel("Merge eps factor:"), 0, 0)
        self.sol_merge_eps = QDoubleSpinBox()
        self.sol_merge_eps.setRange(0.01, 2.0)
        self.sol_merge_eps.setSingleStep(0.05)
        self.sol_merge_eps.setValue(0.5)
        grl.addWidget(self.sol_merge_eps, 0, 1)

        self.sol_fill_holes = QCheckBox("Intentar cerrar agujeros")
        self.sol_fill_holes.setChecked(True)
        grl.addWidget(self.sol_fill_holes, 1, 0, 1, 2)

        lay.addWidget(g_rep)
        lay.addStretch()

        # ── Tooltips explicativos ───────────────────────────────────────
        tips = {
            self.sol_voxel_auto:
                "Calcula el tamaño de voxel según la escala del objeto "
                "(1.5% de su tamaño medio). Es la referencia para el merge "
                "de vértices y la voxelización.",
            self.sol_voxel_size:
                "Tamaño de voxel manual, en las unidades de la nube. Solo "
                "se usa si el cálculo automático está desactivado.",
            self.sol_strategy_repair:
                "Repara la malla existente: limpieza topológica + cierre "
                "real de agujeros. Máxima fidelidad porque conserva la "
                "geometría original. Casi siempre es la estrategia ganadora.",
            self.sol_strategy_poisson:
                "Fallback si la reparación no logra cerrar: muestrea la "
                "malla y reconstruye una envolvente Poisson, cerrada por "
                "construcción. Preserva las concavidades (a diferencia del "
                "Convex Hull).",
            self.sol_poisson_depth:
                "Resolución de la envolvente Poisson. 8 da la misma "
                "fidelidad que 9 con la mitad de triángulos; subir solo "
                "para objetos con detalle muy fino.",
            self.sol_strategy_voxel:
                "Voxeliza la malla y re-triangula con Ball Pivoting. "
                "Respaldo intermedio; rara vez logra el cierre completo.",
            self.sol_strategy_hull:
                "Envolvente convexa: siempre produce un sólido cerrado pero "
                "destruye toda concavidad del objeto. Último recurso — "
                "normalmente la rechaza el control de calidad.",
            self.sol_quality_check:
                "Mide cuánto se aleja cada candidata de la malla original "
                "(distancia de Chamfer) y rechaza las que deforman "
                "demasiado. Sin esto se acepta la primera cerrada, aunque "
                "sea el Convex Hull.",
            self.sol_fidelity_max:
                "Error medio máximo aceptado, como % de la diagonal del "
                "objeto. 2% equilibrado; 1% más estricto (objetos con "
                "concavidades importantes); subir si el escaneo está muy "
                "incompleto y hay que «inventar» superficie para cerrar.",
            self.sol_merge_eps:
                "Distancia de fusión de vértices cercanos, como factor del "
                "voxel. Se limita automáticamente al 50% de la arista "
                "mediana para no destruir el detalle de la malla.",
            self.sol_fill_holes:
                "Cierre real de agujeros: abanico al centroide de cada "
                "borde + triangulación + eliminación de defectos "
                "no-manifold, en rondas iterativas.",
        }
        for w, tip in tips.items():
            w.setToolTip(_tt(tip))

        self.tabs.addTab(tab, "C · Sol")

    def get_pre_params(self) -> dict:
        return {
            # Validación
            "remove_invalid"     : self.pre_remove_invalid.isChecked(),
            "remove_origin"      : self.pre_remove_origin.isChecked(),
            # Deduplicación
            "dedup_enabled"      : self.pre_dedup_enabled.isChecked(),
            "dedup_factor"       : self.pre_dedup_factor.value(),
            # ROR
            "ror_enabled"        : self.ror_enabled.isChecked(),
            "ror_factor"         : self.ror_factor.value(),
            "ror_min_neighbors"  : self.ror_min_neighbors.value(),
            # SOR
            "sor_enabled"        : self.sor_enabled.isChecked(),
            "sor_k"              : self.sor_k.value(),
            "sor_std_ratio"      : self.sor_std.value(),
            # Voxel
            "voxel_enabled"      : self.pre_voxel_enabled.isChecked(),
            "voxel_factor"       : self.pre_voxel_factor.value(),
            # Suavizado MLS
            "denoise_enabled"    : self.pre_denoise_enabled.isChecked(),
            "denoise_iterations" : self.pre_denoise_iter.value(),
            "denoise_k"          : self.pre_denoise_k.value(),
            "preserve_edges"     : self.pre_preserve_edges.isChecked(),
            # Normales
            "normal_factor"      : self.pre_normal_factor.value(),
            "normal_max_nn"      : self.pre_normal_max_nn.value(),
            "orient_k"           : self.pre_orient_k.value(),
        }

    def get_rec_params(self) -> dict:
        return {
            "method"            : self.rec_method.currentText(),
            "poisson_depth"     : self.poisson_depth.value(),
            # Índice 0 = "relativo a d̄", 1 = "absoluto"
            "bp_radius_mode"    : ("relative"
                                   if self.bp_radius_mode.currentIndex() == 0
                                   else "absolute"),
            "bp_radius_factor"  : self.bp_radius_factor.value(),
            "bp_radius"         : self.bp_radius.value(),
            "alpha_mode"        : ("relative"
                                   if self.alpha_mode.currentIndex() == 0
                                   else "absolute"),
            "alpha_factor"      : self.alpha_factor.value(),
            "alpha"             : self.alpha_val.value(),
            "alpha_downsample"  : self.alpha_ds.value(),
            "remove_webbing"    : self.remove_webbing.isChecked(),
            "remove_hollow"     : self.remove_hollow.isChecked(),
            "smooth_method"     : self.smooth_method.currentText(),
            "smooth_iterations" : self.smooth_iter.value(),
            "taubin_lambda"     : self.taubin_lambda.value(),
            "taubin_mu"         : self.taubin_mu.value(),
            "laplacian_lambda"  : self.laplacian_lambda.value(),
        }

    def get_sol_params(self) -> dict:
        return {
            "voxel_size_auto"        : self.sol_voxel_auto.isChecked(),
            "voxel_size"             : self.sol_voxel_size.value(),
            "strategy_repair"        : self.sol_strategy_repair.isChecked(),
            "strategy_poisson"       : self.sol_strategy_poisson.isChecked(),
            "strategy_voxel"         : self.sol_strategy_voxel.isChecked(),
            "strategy_hull"          : self.sol_strategy_hull.isChecked(),
            "fallback_poisson_depth" : self.sol_poisson_depth.value(),
            "merge_eps_factor"       : self.sol_merge_eps.value(),
            "fill_holes"             : self.sol_fill_holes.isChecked(),
            "quality_check"          : self.sol_quality_check.isChecked(),
            "fidelity_max_error"     : self.sol_fidelity_max.value() / 100.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  VISPY VIEWPORT
# ══════════════════════════════════════════════════════════════════════════════

class _RobustSceneCanvas(scene.SceneCanvas):
    """
    SceneCanvas sin render de picking en GPU.

    Al hacer clic, Vispy re-renderiza toda la escena a un framebuffer
    oculto (picking) para saber qué visual está bajo el cursor; ese
    render extra es la fuente de los access violations nativos al rotar
    ("OSError: access violation" en glDrawArrays). La cámara solo
    necesita saber si el clic cayó dentro del ViewBox, cosa que el
    fallback por bounding-box en CPU (el mismo que Vispy usa cuando no
    hay soporte de read_pixels) resuelve sin tocar la GPU.
    """

    def visual_at(self, pos):
        try:
            return self._visual_bounds_at(pos)
        except Exception:
            return None


class Viewport3D(QWidget):

    # Señales hacia MainWindow
    transform_committed = pyqtSignal(object, str)   # (matriz 4×4, modo)
    point_picked        = pyqtSignal(object)        # punto 3D medido

    _DRAG_DEG_PER_PX   = 0.4     # sensibilidad de rotación con cursor
    _DRAG_SCALE_PER_PX = 0.005   # sensibilidad de escala con cursor
    _PICK_RADIUS_PX    = 20      # radio de captura para medición
    _MAX_PICK_POINTS   = 150_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = _RobustSceneCanvas(
            keys="interactive",
            bgcolor="#1a1a22",
            show=False,
        )
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, distance=5, elevation=30, azimuth=45
        )

        # Visuales persistentes: se crean una sola vez y se actualizan con
        # set_data(). Crear/desconectar visuales en cada ejecución deja
        # buffers GL huérfanos en el contexto compartido, que terminan en
        # crashes nativos al redibujar.
        self._pcd_visual  = None
        self._mesh_visual = None

        # Estado de edición interactiva (mover/rotar/escalar con cursor)
        self._edit_mode   = None      # None|'move'|'rotate'|'scale'|'measure'
        self.edit_axis    = 0         # eje activo para rotación/paramétrico
        self._dragging    = False
        self._drag_last   = None
        self._drag_T      = np.eye(4)
        self._drag_center = None
        self._press_pos   = None

        # Puntos para picking de medición (submuestreo de lo visible)
        self._pick_pts = None

        # Modo de color de la nube: "rgb" (colores del archivo) o
        # "solid" (un solo color uniforme)
        self.color_mode   = "rgb"
        self._SOLID_COLOR = (0.62, 0.68, 0.75)

        # Visuales de medición (marcadores + línea)
        self._measure_markers = None
        self._measure_line    = None

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

        scene.visuals.XYZAxis(parent=self.view.scene)

    def show_point_cloud(self, pcd: o3d.geometry.PointCloud, fit: bool = True):
        try:
            pts = np.asarray(pcd.points, dtype=np.float32)
            finite = np.isfinite(pts).all(axis=1)
            if not finite.all():
                pts = pts[finite]
            if len(pts) == 0:
                return

            if pcd.has_colors() and self.color_mode == "rgb":
                colors = np.asarray(pcd.colors, dtype=np.float32)
                if not finite.all():
                    colors = colors[finite]
            else:
                colors = np.full((len(pts), 3), self._SOLID_COLOR,
                                 dtype=np.float32)

            if self._pcd_visual is None:
                self._pcd_visual = visuals.Markers(parent=self.view.scene)
                self._pcd_visual.interactive = False

            self._pcd_visual.set_data(
                pts,
                face_color=colors,
                size=2,
                edge_width=0,
            )
            self._pcd_visual.visible = True
            if self._mesh_visual is not None:
                self._mesh_visual.visible = False

            self._set_pick_points(pts)
            if fit:
                self._fit_camera(pts)
            self.canvas.update()

        except Exception as e:
            print(f"[Viewport3D] Error mostrando nube: {e}")

    def show_mesh(self, mesh: o3d.geometry.TriangleMesh, fit: bool = True):
        try:
            verts = np.asarray(mesh.vertices,  dtype=np.float32)
            tris  = np.asarray(mesh.triangles, dtype=np.uint32)

            if len(verts) == 0 or len(tris) == 0:
                return

            # Vértices no finitos o índices fuera de rango corrompen los
            # buffers GL: se sanea antes de subir a la GPU.
            if not np.isfinite(verts).all() or tris.max() >= len(verts):
                m2 = o3d.geometry.TriangleMesh(mesh)
                m2.remove_vertices_by_mask(
                    ~np.isfinite(np.asarray(m2.vertices)).all(axis=1)
                )
                m2.remove_degenerate_triangles()
                m2.remove_unreferenced_vertices()
                verts = np.asarray(m2.vertices,  dtype=np.float32)
                tris  = np.asarray(m2.triangles, dtype=np.uint32)
                mesh  = m2
                if len(verts) == 0 or len(tris) == 0:
                    return

            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            else:
                colors = np.full((len(verts), 3), 0.65, dtype=np.float32)

            if self._mesh_visual is None:
                self._mesh_visual = visuals.Mesh(
                    vertices      = verts,
                    faces         = tris,
                    vertex_colors = colors,
                    shading       = "smooth",
                    parent        = self.view.scene,
                )
                self._mesh_visual.interactive = False
            else:
                self._mesh_visual.set_data(
                    vertices      = verts,
                    faces         = tris,
                    vertex_colors = colors,
                )

            self._mesh_visual.visible = True
            if self._pcd_visual is not None:
                self._pcd_visual.visible = False

            self._set_pick_points(verts)
            if fit:
                self._fit_camera(verts)
            self.canvas.update()

        except Exception as e:
            print(f"[Viewport3D] Error mostrando malla: {e}")

    def _clear_visuals(self):
        # Solo ocultar: los visuales se reutilizan (ver __init__).
        for v in (self._pcd_visual, self._mesh_visual):
            if v is not None:
                v.visible = False

    def _fit_camera(self, points: np.ndarray):
        center   = points.mean(axis=0)
        max_span = float(np.max(points.max(axis=0) - points.min(axis=0)))
        if not np.isfinite(center).all() or not np.isfinite(max_span):
            return
        self.view.camera.center   = center
        self.view.camera.distance = max(max_span, 1e-6) * 1.8

    # ══════════════════════════════════════════════════════════════
    #  EDICIÓN INTERACTIVA  (mover / rotar / escalar con el cursor)
    # ══════════════════════════════════════════════════════════════

    def set_edit_mode(self, mode: str | None):
        """
        Activa una herramienta de edición: 'move' | 'rotate' | 'scale' |
        'measure' | None. En los modos de arrastre se desactiva la cámara
        (el arrastre transforma la pieza, no la vista); en medición y sin
        herramienta la cámara queda libre.
        """
        self._edit_mode = mode
        self._dragging  = False
        self.view.camera.interactive = mode not in ("move", "rotate", "scale")
        if mode != "measure":
            self.clear_measure()

    def _set_pick_points(self, pts: np.ndarray):
        """Guarda un submuestreo de lo visible para picking de medición."""
        step = max(1, len(pts) // self._MAX_PICK_POINTS)
        self._pick_pts = np.asarray(pts[::step], dtype=np.float64)

    def _visible_visual(self):
        if self._mesh_visual is not None and self._mesh_visual.visible:
            return self._mesh_visual
        if self._pcd_visual is not None and self._pcd_visual.visible:
            return self._pcd_visual
        return None

    def _data_center(self):
        if self._pick_pts is None or len(self._pick_pts) == 0:
            return None
        return (self._pick_pts.max(axis=0) + self._pick_pts.min(axis=0)) / 2.0

    def _camera_axes_and_scale(self):
        """Ejes derecha/arriba de la cámara en mundo y unidades por píxel."""
        cam = self.view.camera
        tr  = cam.transform
        right = np.asarray(tr.map(np.array([1.0, 0.0, 0.0, 0.0]))[:3], float)
        up    = np.asarray(tr.map(np.array([0.0, 1.0, 0.0, 0.0]))[:3], float)
        right /= max(np.linalg.norm(right), 1e-12)
        up    /= max(np.linalg.norm(up),    1e-12)
        dist = cam.distance
        if dist is None:
            dist = getattr(cam, "_actual_distance", None) or 1.0
        alto = max(self.canvas.size[1], 1)
        wpp  = 2.0 * float(dist) * np.tan(np.radians(cam.fov or 45.0) / 2.0) / alto
        return right, up, wpp

    def _on_mouse_press(self, ev):
        if ev.button != 1:
            return
        self._press_pos = np.array(ev.pos, dtype=float)
        if self._edit_mode in ("move", "rotate", "scale"):
            if self._pick_pts is None or len(self._pick_pts) == 0:
                return
            self._dragging    = True
            self._drag_last   = np.array(ev.pos, dtype=float)
            self._drag_T      = np.eye(4)
            self._drag_center = self._data_center()

    def _on_mouse_move(self, ev):
        if not self._dragging:
            return
        pos = np.array(ev.pos, dtype=float)
        d   = pos - self._drag_last
        if not np.isfinite(d).all() or (d == 0).all():
            return
        self._drag_last = pos

        T = np.eye(4)
        if self._edit_mode == "move":
            right, up, wpp = self._camera_axes_and_scale()
            T[:3, 3] = (d[0] * right - d[1] * up) * wpp
        elif self._edit_mode == "rotate":
            c   = self._drag_center
            eje = np.zeros(3)
            eje[self.edit_axis] = 1.0
            ang = np.radians(d[0] * self._DRAG_DEG_PER_PX)
            R   = o3d.geometry.get_rotation_matrix_from_axis_angle(eje * ang)
            T[:3, :3] = R
            T[:3, 3]  = c - R @ c
        elif self._edit_mode == "scale":
            c = self._drag_center
            s = float(np.exp(-d[1] * self._DRAG_SCALE_PER_PX))
            T[:3, :3] *= s
            T[:3, 3]   = c - s * c
        else:
            return

        self._drag_T = T @ self._drag_T

        # Vista previa barata: se transforma el visual (GPU), no los datos.
        # Los datos se transforman una sola vez al soltar el botón.
        for v in (self._pcd_visual, self._mesh_visual):
            if v is not None and v.visible:
                v.transform = MatrixTransform(self._drag_T.T)
        self.canvas.update()

    def _on_mouse_release(self, ev):
        if ev.button != 1:
            return

        # Medición: un clic (sin arrastre) captura el punto más cercano
        if self._edit_mode == "measure":
            if (self._press_pos is not None and
                    np.linalg.norm(np.array(ev.pos, float) -
                                   self._press_pos) < 4.0):
                p = self._pick_point_at(np.array(ev.pos, dtype=float))
                if p is not None:
                    self.point_picked.emit(p)
            self._press_pos = None
            return

        if not self._dragging:
            return
        self._dragging = False
        T = self._drag_T
        self._drag_T = np.eye(4)
        for v in (self._pcd_visual, self._mesh_visual):
            if v is not None:
                v.transform = MatrixTransform()
        if not np.allclose(T, np.eye(4), atol=1e-12):
            self.transform_committed.emit(T, self._edit_mode)

    def _pick_point_at(self, pos: np.ndarray):
        """Punto 3D visible más cercano al clic (proyección a pantalla)."""
        if self._pick_pts is None or len(self._pick_pts) == 0:
            return None
        vis = self._visible_visual()
        if vis is None:
            return None
        try:
            tr = vis.get_transform("visual", "canvas")
            m  = tr.map(self._pick_pts)
            w  = m[:, 3:4].copy()
            w[w == 0] = 1.0
            scr = m[:, :2] / w
            d2  = ((scr - pos) ** 2).sum(axis=1)
            i   = int(np.nanargmin(d2))
            if d2[i] > self._PICK_RADIUS_PX ** 2:
                return None
            return self._pick_pts[i].copy()
        except Exception as e:
            print(f"[Viewport3D] Error en picking: {e}")
            return None

    # ── Visuales de medición ────────────────────────────────────────────

    def set_measure_points(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if self._measure_markers is None:
            self._measure_markers = visuals.Markers(parent=self.view.scene)
            self._measure_markers.interactive = False
            self._measure_line = visuals.Line(parent=self.view.scene,
                                              color="yellow", width=2)
            self._measure_line.interactive = False
        self._measure_markers.set_data(
            pts, face_color="yellow", size=10, edge_width=0)
        self._measure_markers.visible = True
        if len(pts) == 2:
            self._measure_line.set_data(pos=pts)
            self._measure_line.visible = True
        else:
            self._measure_line.visible = False
        self.canvas.update()

    def clear_measure(self):
        for v in (self._measure_markers, self._measure_line):
            if v is not None:
                v.visible = False
        self.canvas.update()


# ══════════════════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    # ── Constantes base (los formatos de import se arman en __init__
    #    según las librerías opcionales disponibles: pye57, laspy) ──────
    _IMPORT_EXTS_BASE       = ["*.ply", "*.xyz", "*.txt", "*.pcd"]
    EXPORT_FORMATS          = ("PLY (*.ply);;STL (*.stl);;OBJ (*.obj);;"
                               "OFF (*.off);;GLTF (*.gltf)")
    CLOUD_EXPORT_FORMATS    = "PLY (*.ply);;PCD (*.pcd);;XYZ (*.xyz)"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Point Cloud Processor v2.1")
        self.resize(1400, 860)

        # ── Estado de la aplicación ────────────────────────────────────
        self.raw_pcd      : o3d.geometry.PointCloud    | None = None
        self.clean_pcd    : o3d.geometry.PointCloud    | None = None
        self.recon_mesh   : o3d.geometry.TriangleMesh  | None = None
        self.solid_mesh   : o3d.geometry.TriangleMesh  | None = None

        self._undo_stack  : dict | None = None
        self._worker      : WorkerThread | None = None

        # Historial de transformaciones (matrices 4×4) para poder
        # deshacerlas; se limpia cuando el pipeline genera productos
        # nuevos (que ya nacen en las coordenadas actuales).
        self._transform_history : list = []
        self._current_view      : str | None = None
        self._measure_p1        : np.ndarray | None = None

        # ── Bitácora de la sesión ──────────────────────────────────────
        # Cada ejecución de un bloque deja aquí sus parámetros, métricas y
        # tiempo. Antes las stats se usaban para pintar una etiqueta y se
        # descartaban, así que la trazabilidad "qué parámetros produjeron
        # este sólido" vivía solo en el log, que no se puede guardar.
        self._session_start = datetime.now()
        self._source_path   : str | None = None
        self._runs          : list = []

        # ── Formatos de importación según librerías disponibles ───────
        exts   = list(self._IMPORT_EXTS_BASE)
        avisos = []

        e57_ok, e57_msg = check_e57_support()
        if e57_ok:
            exts.append("*.e57")
        else:
            avisos.append(e57_msg)

        las_ok, laz_ok, las_msg = check_las_support()
        if las_ok:
            exts.append("*.las")
            if laz_ok:
                exts.append("*.laz")
        if not (las_ok and laz_ok):
            avisos.append(las_msg)

        self.IMPORT_FORMATS = f"Point Cloud ({' '.join(exts)})"

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()

        # Emitir avisos de formatos no disponibles (log_box ya existe)
        for aviso in avisos:
            self._log(f"  ℹ️  {aviso}")

        self._status("Listo — carga una nube de puntos para comenzar")

    # ══════════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN DE LA UI  (sin cambios respecto al original)
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        self.param_panel = ParamPanel()
        splitter.addWidget(self.param_panel)

        center_widget = QWidget()
        cv = QVBoxLayout(center_widget)
        cv.setSpacing(4)
        cv.setContentsMargins(0, 0, 0, 0)

        self.viewport = Viewport3D()

        # Barra de edición adosada al borde superior del visor
        self.edit_toolbar = self._build_edit_toolbar()
        cv.addWidget(self.edit_toolbar)
        cv.addWidget(self.viewport, stretch=7)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        cv.addWidget(self.log_box, stretch=2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(14)
        cv.addWidget(self.progress_bar)

        splitter.addWidget(center_widget)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([290, 870, 240])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_right_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 4, 4, 4)
        w.setFixedWidth(240)

        g_io = QGroupBox("Entrada / Salida")
        gio  = QVBoxLayout(g_io)

        self.btn_load = QPushButton("📂 Cargar nube")
        gio.addWidget(self.btn_load)

        # Acciones de exportación por producto: se comparten entre el menú
        # desplegable de este botón y el menú Archivo.
        self.act_export_cloud = QAction("Exportar nube de puntos…", self)
        self.act_export_cloud.triggered.connect(self._export_cloud)
        self.act_export_cloud.setEnabled(False)

        self.act_export_mesh = QAction("Exportar malla reconstruida…", self)
        self.act_export_mesh.triggered.connect(self._export_recon_mesh)
        self.act_export_mesh.setEnabled(False)

        self.act_export_solid = QAction("Exportar sólido…", self)
        self.act_export_solid.triggered.connect(self._export_solid_mesh)
        self.act_export_solid.setEnabled(False)

        self.act_export_report = QAction("Exportar reporte de sesión…", self)
        self.act_export_report.triggered.connect(self._export_session_report)
        self.act_export_report.setEnabled(False)
        self.act_export_report.setToolTip(_tt(
            "Guarda en un archivo los parámetros, las métricas y los "
            "tiempos de cada bloque ejecutado en esta sesión. Es la "
            "trazabilidad de qué configuración produjo qué resultado."))

        self.btn_export = QPushButton("💾 Exportar…")
        self.btn_export.setEnabled(False)
        menu_export = QMenu(self.btn_export)
        menu_export.addAction(self.act_export_cloud)
        menu_export.addAction(self.act_export_mesh)
        menu_export.addAction(self.act_export_solid)
        menu_export.addSeparator()
        menu_export.addAction(self.act_export_report)
        self.btn_export.setMenu(menu_export)
        gio.addWidget(self.btn_export)

        lay.addWidget(g_io)

        g_pipe = QGroupBox("Pipeline")
        gpipe  = QVBoxLayout(g_pipe)

        self.btn_preprocess  = QPushButton("A · Preprocesar")
        self.btn_preprocess.setEnabled(False)
        gpipe.addWidget(self.btn_preprocess)

        self.btn_reconstruct = QPushButton("B · Reconstruir")
        self.btn_reconstruct.setEnabled(False)
        gpipe.addWidget(self.btn_reconstruct)

        self.btn_solidify    = QPushButton("C · Solidificar")
        self.btn_solidify.setEnabled(False)
        gpipe.addWidget(self.btn_solidify)

        self.btn_run_all     = QPushButton("▶ Ejecutar todo")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setStyleSheet(
            "background-color: #2a4a6a; border-color: #4a7aaa;"
        )
        gpipe.addWidget(self.btn_run_all)

        lay.addWidget(g_pipe)

        g_view = QGroupBox("Visualización")
        gview  = QVBoxLayout(g_view)

        self.btn_show_pcd  = QPushButton("👁 Ver nube limpia")
        self.btn_show_pcd.setEnabled(False)
        gview.addWidget(self.btn_show_pcd)

        self.btn_show_mesh = QPushButton("👁 Ver malla reconstruida")
        self.btn_show_mesh.setEnabled(False)
        gview.addWidget(self.btn_show_mesh)

        self.btn_show_solid = QPushButton("👁 Ver sólido")
        self.btn_show_solid.setEnabled(False)
        gview.addWidget(self.btn_show_solid)

        self.chk_rgb = QCheckBox("Color RGB de la nube")
        self.chk_rgb.setChecked(True)
        self.chk_rgb.setToolTip(_tt(
            "Muestra la nube con los colores RGB del archivo (si los "
            "tiene) o con un color uniforme. El color uniforme ayuda a "
            "evaluar la geometría sin la distracción de la textura."))
        gview.addWidget(self.chk_rgb)

        lay.addWidget(g_view)

        g_edit = QGroupBox("Edición")
        gedit  = QVBoxLayout(g_edit)

        self.btn_undo = QPushButton("↩ Deshacer (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        gedit.addWidget(self.btn_undo)

        lay.addWidget(g_edit)

        # ── Productos del pipeline ──────────────────────────────────────
        # Una etiqueta por producto: antes la malla y el sólido compartían
        # una sola, así que al terminar C se perdía de vista el resultado
        # de B sin forma de recuperarlo salvo re-ejecutando.
        g_info = QGroupBox("Productos")
        ginf   = QVBoxLayout(g_info)

        self.lbl_pcd_info  = QLabel("Nube: —")
        self.lbl_pcd_info.setWordWrap(True)
        ginf.addWidget(self.lbl_pcd_info)

        self.lbl_mesh_info = QLabel("Malla: —")
        self.lbl_mesh_info.setWordWrap(True)
        ginf.addWidget(self.lbl_mesh_info)

        self.lbl_solid_info = QLabel("Sólido: —")
        self.lbl_solid_info.setWordWrap(True)
        ginf.addWidget(self.lbl_solid_info)

        lay.addWidget(g_info)

        # ── Panel de calidad del sólido ─────────────────────────────────
        # La cascada de solidificación ya calculaba todo esto; hasta ahora
        # solo llegaba al log, donde se pierde entre el resto de mensajes.
        g_qual = QGroupBox("Calidad del sólido")
        gq     = QVBoxLayout(g_qual)
        gq.setSpacing(3)

        self.lbl_q_strategy  = QLabel("Estrategia: —")
        self.lbl_q_fidelity  = QLabel("Fidelidad: —")
        self.lbl_q_closed    = QLabel("Cerrado: —")
        self.lbl_q_volume    = QLabel("Volumen: —")
        self.lbl_q_holes     = QLabel("Agujeros entrada: —")

        for lbl in (self.lbl_q_strategy, self.lbl_q_fidelity,
                    self.lbl_q_closed, self.lbl_q_volume,
                    self.lbl_q_holes):
            lbl.setWordWrap(True)
            gq.addWidget(lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a44;")
        gq.addWidget(sep)

        cascada_hdr = QLabel("Cascada de estrategias")
        cascada_hdr.setStyleSheet("color: #82b4d0; font-size: 10px;")
        gq.addWidget(cascada_hdr)

        self.lbl_q_cascade = QLabel("—")
        self.lbl_q_cascade.setWordWrap(True)
        self.lbl_q_cascade.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #9ab0c0;")
        gq.addWidget(self.lbl_q_cascade)

        g_qual.setToolTip(_tt(
            "Resultado del control de calidad de la solidificación."
            "<br><b>Estrategia</b>: cuál de las cuatro logró cerrar el "
            "sólido cumpliendo la tolerancia de fidelidad."
            "<br><b>Fidelidad</b>: distancia de Chamfer media entre el "
            "sólido y la malla de entrada, como % de la diagonal del "
            "objeto. Más bajo es mejor."
            "<br><b>Cascada</b>: qué se probó y con qué resultado, en "
            "orden. Sirve para entender por qué ganó la estrategia que "
            "ganó."))

        lay.addWidget(g_qual)
        lay.addStretch()

        return w

    # ══════════════════════════════════════════════════════════════
    #  BARRA DE EDICIÓN  (adosada al borde superior del visor 3D)
    # ══════════════════════════════════════════════════════════════

    def _build_edit_toolbar(self) -> QToolBar:
        """
        Barra de herramientas de edición estilo AutoCAD, adosada al borde
        superior del visor. Las herramientas Mover/Rotar/Escalar operan
        con el cursor (arrastrando sobre el visor) y también en forma
        paramétrica (botones −/+ con paso configurable). Medir entrega
        distancias entre dos puntos clicados. Las transformaciones se
        aplican a todos los productos cargados para mantenerlos alineados.
        """
        tb = QToolBar("Edición")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { spacing: 4px; }")

        # ── Herramientas interactivas (cursor) ─────────────────────────
        self.act_tool_move    = QAction("⬄ Mover",   self)
        self.act_tool_rotate  = QAction("⟳ Rotar",   self)
        self.act_tool_scale   = QAction("⤢ Escalar", self)
        self.act_tool_measure = QAction("📏 Medir",  self)

        tips = {
            self.act_tool_move:
                "Mover con el cursor: activa la herramienta y arrastra "
                "sobre el visor para desplazar la pieza en el plano de la "
                "cámara. También −/+ para mover por paso en el eje activo.",
            self.act_tool_rotate:
                "Rotar con el cursor: arrastra horizontalmente para girar "
                "la pieza en torno al eje seleccionado (por su centro). "
                "También −/+ para rotar por paso.",
            self.act_tool_scale:
                "Escalar con el cursor: arrastra verticalmente para "
                "agrandar/achicar la pieza respecto a su centro. "
                "También −/+ para escalar por factor.",
            self.act_tool_measure:
                "Medir distancias: haz clic en dos puntos de la pieza; la "
                "distancia y sus componentes ΔX/ΔY/ΔZ aparecen en el "
                "terminal. La cámara sigue libre para orbitar entre clics.",
        }
        for act in (self.act_tool_move, self.act_tool_rotate,
                    self.act_tool_scale, self.act_tool_measure):
            act.setCheckable(True)
            act.setToolTip(_tt(tips[act]))
            act.triggered.connect(
                lambda _, a=act: self._on_edit_tool(a))
            tb.addAction(act)

        tb.addSeparator()

        # ── Eje activo (rotación y movimiento paramétrico) ─────────────
        tb.addWidget(QLabel(" Eje: "))
        self.edit_axis_combo = QComboBox()
        self.edit_axis_combo.addItems(["X", "Y", "Z"])
        self.edit_axis_combo.setToolTip(_tt(
            "Eje sobre el que actúan la rotación (con cursor o "
            "paramétrica) y el movimiento paramétrico −/+."))
        self.edit_axis_combo.currentIndexChanged.connect(
            lambda i: setattr(self.viewport, "edit_axis", i))
        tb.addWidget(self.edit_axis_combo)

        tb.addSeparator()

        # ── Pasos paramétricos ──────────────────────────────────────────
        tb.addWidget(QLabel(" Paso: "))
        self.step_move = QDoubleSpinBox()
        self.step_move.setRange(0.0001, 1e6)
        self.step_move.setDecimals(4)
        self.step_move.setValue(0.1)
        self.step_move.setMaximumWidth(90)
        self.step_move.setToolTip(_tt(
            "Distancia del movimiento paramétrico (−/+), en unidades de "
            "la nube. Se ajusta automáticamente al 5% del tamaño del "
            "modelo al cargar."))
        tb.addWidget(self.step_move)

        tb.addWidget(QLabel(" ∠°: "))
        self.step_rot = QDoubleSpinBox()
        self.step_rot.setRange(0.1, 180.0)
        self.step_rot.setDecimals(1)
        self.step_rot.setValue(15.0)
        self.step_rot.setMaximumWidth(70)
        self.step_rot.setToolTip(_tt(
            "Grados de la rotación paramétrica (−/+), en torno al centro "
            "del modelo."))
        tb.addWidget(self.step_rot)

        tb.addWidget(QLabel(" ×: "))
        self.step_scale = QDoubleSpinBox()
        self.step_scale.setRange(1.01, 10.0)
        self.step_scale.setDecimals(2)
        self.step_scale.setSingleStep(0.05)
        self.step_scale.setValue(1.10)
        self.step_scale.setMaximumWidth(70)
        self.step_scale.setToolTip(_tt(
            "Factor de la escala paramétrica: «+» multiplica, «−» divide. "
            "Útil también para convertir unidades (p. ej. factor 10)."))
        tb.addWidget(self.step_scale)

        b_menos = QPushButton("−")
        b_mas   = QPushButton("+")
        for b in (b_menos, b_mas):
            b.setFixedWidth(28)
        b_menos.setToolTip(_tt(
            "Aplica la herramienta activa en forma paramétrica, en "
            "sentido negativo (sin herramienta activa: mueve)."))
        b_mas.setToolTip(_tt(
            "Aplica la herramienta activa en forma paramétrica, en "
            "sentido positivo (sin herramienta activa: mueve)."))
        b_menos.clicked.connect(lambda: self._parametric_apply(-1))
        b_mas.clicked.connect(lambda: self._parametric_apply(+1))
        tb.addWidget(b_menos)
        tb.addWidget(b_mas)

        tb.addSeparator()

        a_undo = QAction("↩ Deshacer", self)
        a_undo.setToolTip(_tt("Deshace la última transformación aplicada "
                              "(cursor o paramétrica)."))
        a_undo.triggered.connect(self._undo_transform)
        tb.addAction(a_undo)

        return tb

    def _on_edit_tool(self, act: QAction):
        """Activación exclusiva de herramientas (clic en la activa la
        desactiva y libera la cámara)."""
        herramientas = {
            self.act_tool_move   : "move",
            self.act_tool_rotate : "rotate",
            self.act_tool_scale  : "scale",
            self.act_tool_measure: "measure",
        }
        if act.isChecked():
            for otra in herramientas:
                if otra is not act:
                    otra.setChecked(False)
            modo = herramientas[act]
        else:
            modo = None

        self._measure_p1 = None
        self.viewport.set_edit_mode(modo)

        ayudas = {
            "move"   : "🛠 Mover: arrastra sobre el visor para desplazar "
                       "la pieza (la cámara queda fija)",
            "rotate" : "🛠 Rotar: arrastra horizontalmente para girar en "
                       "torno al eje "
                       f"{self.edit_axis_combo.currentText()}",
            "scale"  : "🛠 Escalar: arrastra verticalmente para cambiar "
                       "el tamaño",
            "measure": "📏 Medir: haz clic en dos puntos de la pieza",
            None     : "Herramienta desactivada — cámara libre",
        }
        self._log(f"  {ayudas[modo]}")
        self._status(ayudas[modo])

    def _parametric_apply(self, sign: int):
        """Botones −/+: aplican la herramienta activa por paso numérico."""
        axis = self.edit_axis_combo.currentIndex()
        if self.act_tool_rotate.isChecked():
            self._transform_rotate(axis, sign)
        elif self.act_tool_scale.isChecked():
            self._transform_scale(sign)
        else:
            self._transform_move(axis, sign)

    def _on_drag_transform(self, T, modo: str):
        """Transformación confirmada por arrastre de cursor en el visor."""
        T = np.asarray(T, dtype=float)
        if modo == "move":
            d = T[:3, 3]
            desc = (f"Mover (cursor) Δ=({d[0]:+.4f}, {d[1]:+.4f}, "
                    f"{d[2]:+.4f})")
        elif modo == "rotate":
            ang = np.degrees(np.arccos(
                np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
            desc = (f"Rotar (cursor) {ang:.1f}° en eje "
                    f"{self.edit_axis_combo.currentText()}")
        else:
            s = float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))
            desc = f"Escalar (cursor) ×{s:.3f}"
        self._apply_transform(T, desc, renormalize=(modo == "scale"))

    def _on_measure_point(self, p):
        """Puntos de medición clicados en el visor → distancia en el log."""
        p = np.asarray(p, dtype=float)
        if self._measure_p1 is None:
            self._measure_p1 = p
            self.viewport.set_measure_points([p])
            self._log(f"  📍 Punto 1: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            self._status("Medir: selecciona el segundo punto")
        else:
            p1, self._measure_p1 = self._measure_p1, None
            d    = p - p1
            dist = float(np.linalg.norm(d))
            self.viewport.set_measure_points([p1, p])
            self._log(f"  📍 Punto 2: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            self._log(f"  📏 Distancia: {dist:.4f}  |  ΔX={d[0]:+.4f}  "
                      f"ΔY={d[1]:+.4f}  ΔZ={d[2]:+.4f}")
            self._status(f"Distancia: {dist:.4f}")

    # ── Operaciones de transformación ──────────────────────────────────

    def _all_products(self) -> list:
        return [g for g in (self.raw_pcd, self.clean_pcd,
                            self.recon_mesh, self.solid_mesh)
                if g is not None]

    def _reference_center(self) -> np.ndarray | None:
        """Centro del bounding box del producto más avanzado disponible."""
        for g in (self.solid_mesh, self.recon_mesh,
                  self.clean_pcd, self.raw_pcd):
            if g is not None:
                return (np.asarray(g.get_max_bound()) +
                        np.asarray(g.get_min_bound())) / 2.0
        return None

    def _apply_transform(self, T: np.ndarray, desc: str,
                         renormalize: bool = False):
        productos = self._all_products()
        if not productos:
            self._show_error("Carga una nube antes de usar las "
                             "herramientas de edición.")
            return
        for g in productos:
            g.transform(T)
            if renormalize:
                if isinstance(g, o3d.geometry.TriangleMesh):
                    if g.has_vertex_normals():
                        g.compute_vertex_normals()
                elif g.has_normals():
                    n = np.asarray(g.normals)
                    mag = np.linalg.norm(n, axis=1, keepdims=True)
                    mag[mag == 0] = 1.0
                    g.normals = o3d.utility.Vector3dVector(n / mag)
        self._transform_history.append(T)
        self._log(f"  🛠 {desc}")
        self._status(desc)
        self._refresh_view()

    def _transform_move(self, axis: int, sign: int):
        paso = self.step_move.value() * sign
        T = np.eye(4)
        T[axis, 3] = paso
        self._apply_transform(T, f"Mover {'XYZ'[axis]} {paso:+.4f}")

    def _transform_rotate(self, axis: int, sign: int):
        c = self._reference_center()
        if c is None:
            self._show_error("Carga una nube antes de usar las "
                             "herramientas de edición.")
            return
        grados = self.step_rot.value() * sign
        eje = np.zeros(3)
        eje[axis] = 1.0
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            eje * np.radians(grados))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = c - R @ c          # rotación en torno al centro
        self._apply_transform(T, f"Rotar {'XYZ'[axis]} {grados:+.1f}°")

    def _transform_scale(self, sign: int):
        c = self._reference_center()
        if c is None:
            self._show_error("Carga una nube antes de usar las "
                             "herramientas de edición.")
            return
        f = self.step_scale.value()
        s = f if sign > 0 else 1.0 / f
        T = np.eye(4)
        T[:3, :3] *= s
        T[:3, 3]   = c - s * c         # escala en torno al centro
        self._apply_transform(T, f"Escalar ×{s:.3f}", renormalize=True)

    def _undo_transform(self):
        if not self._transform_history:
            self._log("  ℹ️  No hay transformaciones que deshacer")
            return
        T = self._transform_history.pop()
        Ti = np.linalg.inv(T)
        for g in self._all_products():
            g.transform(Ti)
        self._log("  ↩ Transformación deshecha")
        self._refresh_view()

    def _on_rgb_toggled(self, checked: bool):
        """Alterna entre colores RGB del archivo y color uniforme."""
        self.viewport.color_mode = "rgb" if checked else "solid"
        self._refresh_view()
        self._status("Nube en color RGB" if checked
                     else "Nube en color uniforme")

    def _refresh_view(self):
        """
        Redibuja el producto que se estaba mostrando SIN re-encuadrar la
        cámara (fit=False): así se ve que es la figura la que se mueve,
        rota o cambia de tamaño, no el eje de coordenadas.
        """
        if self._current_view == "solid" and self.solid_mesh:
            self.viewport.show_mesh(self.solid_mesh, fit=False)
        elif self._current_view == "mesh" and self.recon_mesh:
            self.viewport.show_mesh(self.recon_mesh, fit=False)
        elif self._current_view == "clean" and self.clean_pcd:
            self.viewport.show_point_cloud(self.clean_pcd, fit=False)
        elif self.raw_pcd:
            self.viewport.show_point_cloud(self.raw_pcd, fit=False)

    # ══════════════════════════════════════════════════════════════
    #  MENÚ  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _build_menu(self):
        mb = self.menuBar()

        m_file = mb.addMenu("Archivo")

        a_load = QAction("Cargar nube…", self)
        a_load.setShortcut("Ctrl+O")
        a_load.triggered.connect(self._load_point_cloud)
        m_file.addAction(a_load)

        # Exportación por producto (mismas acciones que el botón Exportar)
        self.act_export_solid.setShortcut("Ctrl+S")
        m_file.addAction(self.act_export_cloud)
        m_file.addAction(self.act_export_mesh)
        m_file.addAction(self.act_export_solid)

        m_file.addSeparator()

        self.act_export_report.setShortcut("Ctrl+R")
        m_file.addAction(self.act_export_report)

        m_file.addSeparator()

        a_quit = QAction("Salir", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_edit = mb.addMenu("Edición")

        a_undo = QAction("Deshacer", self)
        a_undo.setShortcut("Ctrl+Z")
        a_undo.triggered.connect(self._undo)
        m_edit.addAction(a_undo)

        m_pipe = mb.addMenu("Pipeline")

        a_pre = QAction("A · Preprocesar", self)
        a_pre.triggered.connect(self._run_preprocess)
        m_pipe.addAction(a_pre)

        a_rec = QAction("B · Reconstruir", self)
        a_rec.triggered.connect(self._run_reconstruct)
        m_pipe.addAction(a_rec)

        a_sol = QAction("C · Solidificar", self)
        a_sol.triggered.connect(self._run_solidify)
        m_pipe.addAction(a_sol)

        m_pipe.addSeparator()

        a_all = QAction("▶ Ejecutar todo", self)
        a_all.triggered.connect(self._run_all)
        m_pipe.addAction(a_all)

        m_view = mb.addMenu("Vista")

        a_pcd = QAction("Ver nube limpia", self)
        a_pcd.triggered.connect(self._show_clean_pcd)
        m_view.addAction(a_pcd)

        a_mesh = QAction("Ver malla reconstruida", self)
        a_mesh.triggered.connect(self._show_recon_mesh)
        m_view.addAction(a_mesh)

        a_solid = QAction("Ver sólido", self)
        a_solid.triggered.connect(self._show_solid_mesh)
        m_view.addAction(a_solid)

        m_view.addSeparator()

        a_raw = QAction("Ver nube original", self)
        a_raw.triggered.connect(self._show_raw_pcd)
        m_view.addAction(a_raw)

        m_view.addSeparator()

        a_editbar = QAction("Barra de edición", self)
        a_editbar.setCheckable(True)
        a_editbar.setChecked(True)
        a_editbar.toggled.connect(self.edit_toolbar.setVisible)
        m_view.addAction(a_editbar)

    # ══════════════════════════════════════════════════════════════
    #  TOOLBAR  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _build_toolbar(self):
        tb = QToolBar("Principal")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("📂 Cargar",   self._load_point_cloud)
        tb.addAction("💾 Exportar", self._export_mesh)
        tb.addSeparator()
        tb.addAction("A · Pre",    self._run_preprocess)
        tb.addAction("B · Rec",    self._run_reconstruct)
        tb.addAction("C · Sol",    self._run_solidify)
        tb.addAction("▶ Todo",     self._run_all)
        tb.addSeparator()
        tb.addAction("↩ Undo",     self._undo)

    # ══════════════════════════════════════════════════════════════
    #  CONEXIÓN DE SEÑALES  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _connect_signals(self):
        self.btn_load.clicked.connect(self._load_point_cloud)
        self.btn_export.clicked.connect(self._export_mesh)

        self.btn_preprocess.clicked.connect(self._run_preprocess)
        self.btn_reconstruct.clicked.connect(self._run_reconstruct)
        self.btn_solidify.clicked.connect(self._run_solidify)
        self.btn_run_all.clicked.connect(self._run_all)

        self.btn_show_pcd.clicked.connect(self._show_clean_pcd)
        self.btn_show_mesh.clicked.connect(self._show_recon_mesh)
        self.btn_show_solid.clicked.connect(self._show_solid_mesh)
        self.chk_rgb.toggled.connect(self._on_rgb_toggled)

        self.btn_undo.clicked.connect(self._undo)

        # Señales del visor (edición interactiva y medición)
        self.viewport.transform_committed.connect(self._on_drag_transform)
        self.viewport.point_picked.connect(self._on_measure_point)

    # ══════════════════════════════════════════════════════════════
    #  CARGA / EXPORTACIÓN
    # ══════════════════════════════════════════════════════════════

    def _load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar nube de puntos", "", self.IMPORT_FORMATS
        )
        if not path:
            return

        try:
            self._log(f"  📂 Cargando: {os.path.basename(path)}")

            # ── CAMBIO 3: reemplaza o3d.io.read_point_cloud(path) ─────
            pcd, meta = PointCloudLoader.load(path, log_callback=self._log)

            # Metadata extra en log
            if meta.get("xyz_columns"):
                self._log(f"  ℹ️  Columnas detectadas: {meta['xyz_columns']}")
            if meta.get("e57_scans") is not None:
                self._log(
                    f"  ℹ️  Scans E57 combinados: {meta['e57_scans']}"
                )
            if meta.get("warnings"):
                for w in meta["warnings"]:
                    self._log(f"  ⚠️  {w}")
            # ─────────────────────────────────────────────────────────

            if len(pcd.points) == 0:
                raise ValueError("La nube de puntos está vacía")

            self.raw_pcd     = pcd
            self.clean_pcd   = None
            self.recon_mesh  = None
            self.solid_mesh  = None
            self._undo_stack = None
            self._transform_history.clear()
            self._current_view = "raw"

            # Nube nueva = sesión nueva: las ejecuciones anteriores son de
            # otro dataset y mezclarlas en el reporte sería engañoso.
            self._source_path = path
            self._runs.clear()
            self.lbl_mesh_info.setText("Malla: —")
            self.lbl_solid_info.setText("Sólido: —")
            self._reset_quality_panel()

            # Paso de movimiento proporcional al tamaño del modelo
            diag = float(np.linalg.norm(
                np.asarray(pcd.get_max_bound()) -
                np.asarray(pcd.get_min_bound())
            ))
            if np.isfinite(diag) and diag > 0:
                self.step_move.setValue(round(diag * 0.05, 4))

            self.viewport.show_point_cloud(pcd)
            self._update_info_labels()

            n = len(pcd.points)
            self._log(
                f"  ✓ {n:,} puntos cargados desde "
                f"{os.path.basename(path)}"
            )
            self._status(f"Nube cargada: {n:,} puntos")

            self.btn_preprocess.setEnabled(True)
            self.btn_run_all.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.act_export_cloud.setEnabled(True)
            self.act_export_mesh.setEnabled(False)
            self.act_export_solid.setEnabled(False)

        except Exception as e:
            self._show_error(f"Error al cargar archivo:\n{e}")

    def _ask_save_path(self, titulo: str, formatos: str) -> str | None:
        """Diálogo de guardado; asegura la extensión del filtro elegido."""
        path, fmt = QFileDialog.getSaveFileName(self, titulo, "", formatos)
        if not path:
            return None
        if not os.path.splitext(path)[1] and "*" in fmt:
            ext = fmt[fmt.find("*") + 1 : fmt.find(")")].strip()
            path += ext
        return path

    def _export_cloud(self):
        pcd = self.clean_pcd or self.raw_pcd
        if pcd is None:
            self._show_error("No hay nube de puntos para exportar.")
            return
        origen = "preprocesada" if self.clean_pcd is not None else "original"
        path = self._ask_save_path(f"Exportar nube ({origen})",
                                   self.CLOUD_EXPORT_FORMATS)
        if not path:
            return
        try:
            if not o3d.io.write_point_cloud(path, pcd):
                raise RuntimeError("Open3D no pudo escribir el archivo")
            self._log(f"  ✓ Nube {origen} exportada: "
                      f"{os.path.basename(path)}")
            self._status(f"Exportada: {os.path.basename(path)}")
        except Exception as e:
            self._show_error(f"Error al exportar:\n{e}")

    def _export_recon_mesh(self):
        self._export_mesh_product(self.recon_mesh, "malla reconstruida")

    def _export_solid_mesh(self):
        self._export_mesh_product(self.solid_mesh, "sólido")

    def _export_mesh(self):
        """Exporta el producto más avanzado disponible (toolbar)."""
        if self.solid_mesh is not None:
            self._export_solid_mesh()
        elif self.recon_mesh is not None:
            self._export_recon_mesh()
        else:
            self._export_cloud()

    def _export_mesh_product(self, mesh, nombre: str):
        if mesh is None:
            self._show_error(f"No hay {nombre} para exportar.")
            return

        path = self._ask_save_path(f"Exportar {nombre}",
                                   self.EXPORT_FORMATS)
        if not path:
            return

        try:
            # Chequeo topológico O(T); is_watertight() de Open3D incluye un
            # test de auto-intersecciones cuadrático que congela la GUI en
            # mallas cerradas grandes.
            if not MeshSolidifier.is_topologically_closed(mesh):
                reply = QMessageBox.question(
                    self,
                    "Malla no watertight",
                    f"La {nombre} no es watertight.\n"
                    "¿Exportar de todas formas?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            # STL exige normales de triángulo calculadas
            if path.lower().endswith(".stl"):
                mesh.compute_triangle_normals()

            if not o3d.io.write_triangle_mesh(path, mesh):
                raise RuntimeError("Open3D no pudo escribir el archivo")
            self._log(f"  ✓ Exportada ({nombre}): {os.path.basename(path)}")
            self._status(f"Exportada: {os.path.basename(path)}")

        except Exception as e:
            self._show_error(f"Error al exportar:\n{e}")

    # ══════════════════════════════════════════════════════════════
    #  REPORTE DE SESIÓN
    # ══════════════════════════════════════════════════════════════

    _BLOQUE_NOMBRE = {
        WorkerThread.BLOCK_PREPROCESS  : "A · Preprocesamiento",
        WorkerThread.BLOCK_RECONSTRUCT : "B · Reconstrucción",
        WorkerThread.BLOCK_SOLIDIFY    : "C · Solidificación",
    }

    def _record_run(self, result: dict):
        """Anota una ejecución de bloque en la bitácora de la sesión."""
        self._runs.append({
            "bloque"     : result["block"],
            "nombre"     : self._BLOQUE_NOMBRE.get(result["block"],
                                                   result["block"]),
            "hora"       : datetime.now().isoformat(timespec="seconds"),
            "duracion_s" : result.get("elapsed_s"),
            "parametros" : self._jsonable(result.get("params") or {}),
            "metricas"   : self._jsonable(result.get("stats") or {}),
        })
        self.act_export_report.setEnabled(True)

    @staticmethod
    def _jsonable(obj):
        """
        Convierte a tipos serializables. Las stats traen escalares de numpy
        (np.float64, np.bool_) que `json.dump` rechaza, y diccionarios
        anidados como `strategies_tried`.
        """
        if isinstance(obj, dict):
            return {str(k): MainWindow._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MainWindow._jsonable(v) for v in obj]
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return MainWindow._jsonable(obj.tolist())
        if obj is None or isinstance(obj, (str, int, float)):
            return obj
        return str(obj)

    def _session_report(self) -> dict:
        """Arma el reporte completo de la sesión."""
        return {
            "generado"   : datetime.now().isoformat(timespec="seconds"),
            "inicio"     : self._session_start.isoformat(timespec="seconds"),
            "dataset"    : (os.path.basename(self._source_path)
                            if self._source_path else None),
            "ruta"       : self._source_path,
            "aplicacion" : "3D Point Cloud Processor v2.1",
            "entorno"    : {
                "python" : sys.version.split()[0],
                "open3d" : o3d.__version__,
                "numpy"  : np.__version__,
            },
            "puntos_originales" : (len(self.raw_pcd.points)
                                   if self.raw_pcd else None),
            "ejecuciones"       : self._runs,
            "tiempo_total_s"    : round(sum(
                r["duracion_s"] or 0 for r in self._runs), 2),
        }

    def _report_markdown(self, rep: dict) -> str:
        """Versión legible del reporte, para pegar en la memoria."""
        L = []
        L.append("# Reporte de sesión — Point Cloud Processor")
        L.append("")
        L.append(f"- **Dataset:** `{rep['dataset'] or '—'}`")
        L.append(f"- **Generado:** {rep['generado']}")
        n_pts = rep["puntos_originales"]
        L.append(f"- **Puntos originales:** "
                 f"{n_pts:,}" if n_pts else "- **Puntos originales:** —")
        L.append(f"- **Tiempo total:** {rep['tiempo_total_s']} s")
        e = rep["entorno"]
        L.append(f"- **Entorno:** Python {e['python']}, "
                 f"Open3D {e['open3d']}, numpy {e['numpy']}")
        L.append("")

        if not rep["ejecuciones"]:
            L.append("_No se ejecutó ningún bloque en esta sesión._")
            return "\n".join(L)

        L.append("## Resumen de tiempos")
        L.append("")
        L.append("| Bloque | Hora | Duración (s) |")
        L.append("|---|---|---|")
        for r in rep["ejecuciones"]:
            L.append(f"| {r['nombre']} | {r['hora'][11:]} "
                     f"| {r['duracion_s']} |")
        L.append("")

        for i, r in enumerate(rep["ejecuciones"], 1):
            L.append(f"## {i}. {r['nombre']}")
            L.append("")
            L.append(f"Duración: **{r['duracion_s']} s** — {r['hora']}")
            L.append("")
            L.append("### Parámetros")
            L.append("")
            L.append("| Parámetro | Valor |")
            L.append("|---|---|")
            for k, v in sorted(r["parametros"].items()):
                L.append(f"| `{k}` | {v} |")
            L.append("")
            L.append("### Métricas")
            L.append("")
            L.append("| Métrica | Valor |")
            L.append("|---|---|")
            for k, v in r["metricas"].items():
                if isinstance(v, dict):
                    # strategies_tried: una fila por estrategia probada
                    for sub, det in v.items():
                        err = det.get("fidelity_error")
                        err = "—" if err is None else f"{err:.4%}"
                        L.append(f"| `{k}.{sub}` | "
                                 f"cerrada={det.get('watertight')}, "
                                 f"error={err} |")
                elif isinstance(v, float):
                    L.append(f"| `{k}` | {v:.6g} |")
                else:
                    L.append(f"| `{k}` | {v} |")
            L.append("")

        return "\n".join(L)

    def _export_session_report(self):
        """Exporta la bitácora de la sesión a JSON o Markdown."""
        if not self._runs:
            self._show_error(
                "Todavía no hay nada que reportar.\n\n"
                "Ejecuta al menos un bloque del pipeline (A, B o C) y "
                "vuelve a intentarlo."
            )
            return

        base = "reporte_sesion"
        if self._source_path:
            base = (f"reporte_{os.path.splitext(os.path.basename(self._source_path))[0]}"
                    f"_{datetime.now():%Y%m%d_%H%M}")

        path, fmt = QFileDialog.getSaveFileName(
            self, "Exportar reporte de sesión", base,
            "Markdown (*.md);;JSON (*.json)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".json" if "json" in fmt.lower() else ".md"

        try:
            rep = self._session_report()
            if path.lower().endswith(".json"):
                contenido = json.dumps(rep, indent=2, ensure_ascii=False)
            else:
                contenido = self._report_markdown(rep)

            with open(path, "w", encoding="utf-8") as f:
                f.write(contenido)

            self._log(f"  ✓ Reporte de sesión exportado: "
                      f"{os.path.basename(path)} "
                      f"({len(self._runs)} ejecuciones)")
            self._status(f"Reporte exportado: {os.path.basename(path)}")

        except OSError as e:
            self._show_error(
                f"No se pudo guardar el reporte en:\n{path}\n\n"
                f"Comprueba que la carpeta existe y que tienes permiso "
                f"de escritura.\n\nDetalle: {e}"
            )

    # ══════════════════════════════════════════════════════════════
    #  PIPELINE  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _run_preprocess(self):
        if self.raw_pcd is None:
            self._show_error("Carga una nube de puntos primero.")
            return
        if self._worker_busy():
            return
        self._save_undo_state()
        self._start_worker(
            block  = WorkerThread.BLOCK_PREPROCESS,
            data   = {"point_cloud": self.raw_pcd},
            params = self.param_panel.get_pre_params(),
        )

    def _run_reconstruct(self):
        pcd = self.clean_pcd or self.raw_pcd
        if pcd is None:
            self._show_error("Carga y preprocesa una nube primero.")
            return
        if self._worker_busy():
            return
        self._save_undo_state()
        self._start_worker(
            block  = WorkerThread.BLOCK_RECONSTRUCT,
            data   = {"point_cloud": pcd},
            params = self.param_panel.get_rec_params(),
        )

    def _run_solidify(self):
        mesh = self.recon_mesh
        if mesh is None:
            self._show_error("Reconstruye la malla primero.")
            return
        if self._worker_busy():
            return
        self._save_undo_state()
        self._start_worker(
            block  = WorkerThread.BLOCK_SOLIDIFY,
            data   = {"mesh": mesh},
            params = self.param_panel.get_sol_params(),
        )

    def _run_all(self):
        if self.raw_pcd is None:
            self._show_error("Carga una nube de puntos primero.")
            return
        if self._worker_busy():
            return
        self._save_undo_state()
        self._log("━━━ PIPELINE COMPLETO A→B→C ━━━")
        self._run_all_step_a()

    def _run_all_step_a(self):
        self._start_worker(
            block   = WorkerThread.BLOCK_PREPROCESS,
            data    = {"point_cloud": self.raw_pcd},
            params  = self.param_panel.get_pre_params(),
            on_done = self._run_all_step_b,
        )

    def _run_all_step_b(self):
        pcd = self.clean_pcd or self.raw_pcd
        self._start_worker(
            block   = WorkerThread.BLOCK_RECONSTRUCT,
            data    = {"point_cloud": pcd},
            params  = self.param_panel.get_rec_params(),
            on_done = self._run_all_step_c,
        )

    def _run_all_step_c(self):
        self._start_worker(
            block   = WorkerThread.BLOCK_SOLIDIFY,
            data    = {"mesh": self.recon_mesh},
            params  = self.param_panel.get_sol_params(),
            on_done = self._run_all_finished,
        )

    def _run_all_finished(self):
        self._log("━━━ PIPELINE COMPLETO — LISTO ━━━")
        self._show_solid_mesh()

    # ══════════════════════════════════════════════════════════════
    #  VISUALIZACIÓN  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _show_raw_pcd(self):
        if self.raw_pcd:
            self._current_view = "raw"
            self.viewport.show_point_cloud(self.raw_pcd)
            self._status("Vista: nube original")

    def _show_clean_pcd(self):
        if self.clean_pcd:
            self._current_view = "clean"
            self.viewport.show_point_cloud(self.clean_pcd)
            self._status("Vista: nube preprocesada")
        else:
            self._log("  ⚠️  Nube preprocesada no disponible")

    def _show_recon_mesh(self):
        if self.recon_mesh:
            self._current_view = "mesh"
            self.viewport.show_mesh(self.recon_mesh)
            self._status("Vista: malla reconstruida")
        else:
            self._log("  ⚠️  Malla reconstruida no disponible")

    def _show_solid_mesh(self):
        if self.solid_mesh:
            self._current_view = "solid"
            self.viewport.show_mesh(self.solid_mesh)
            self._status("Vista: sólido final")
        else:
            self._log("  ⚠️  Sólido no disponible")

    # ══════════════════════════════════════════════════════════════
    #  WORKER — ARRANQUE Y RECEPCIÓN DE RESULTADOS  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _start_worker(
        self,
        block   : str,
        data    : dict,
        params  : dict,
        on_done : callable = None,
    ):
        # Esperar a que el hilo anterior termine del todo antes de soltar
        # su referencia: destruir un QThread cuya limpieza interna no ha
        # concluido aborta el proceso completo (crash intermitente al
        # re-ejecutar un bloque).
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

        self._worker = WorkerThread(block, data, params)
        self._worker._on_done_callback = on_done

        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_pipeline_buttons(False)

        self._worker.start()

    def _on_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _on_result(self, result: dict):
        block = result["block"]

        # Los productos nuevos nacen en las coordenadas actuales: las
        # transformaciones anteriores ya no se pueden deshacer sobre
        # ellos sin desalinearlos.
        self._transform_history.clear()

        # Registrar la ejecución antes de tocar la UI: es lo que después
        # alimenta el reporte de sesión.
        self._record_run(result)

        if block == WorkerThread.BLOCK_PREPROCESS:
            # Sobrescribe la nube limpia e invalida los productos aguas
            # abajo: la malla/sólido anteriores ya no corresponden.
            self.clean_pcd  = result["point_cloud"]
            self.recon_mesh = None
            self.solid_mesh = None
            self.btn_show_mesh.setEnabled(False)
            self.btn_show_solid.setEnabled(False)
            self.btn_solidify.setEnabled(False)
            self.act_export_mesh.setEnabled(False)
            self.act_export_solid.setEnabled(False)
            self.lbl_mesh_info.setText("Malla: —")
            self.lbl_solid_info.setText("Sólido: —")
            self._reset_quality_panel()
            self._current_view = "clean"
            self.viewport.show_point_cloud(self.clean_pcd)
            self.btn_show_pcd.setEnabled(True)
            self.btn_reconstruct.setEnabled(True)
            self._update_pcd_label(result["stats"])

        elif block == WorkerThread.BLOCK_RECONSTRUCT:
            # Sobrescribe la malla e invalida el sólido anterior.
            self.recon_mesh = result["mesh"]
            self.solid_mesh = None
            self.btn_show_solid.setEnabled(False)
            self.act_export_solid.setEnabled(False)
            self.lbl_solid_info.setText("Sólido: —")
            self._reset_quality_panel()
            self._current_view = "mesh"
            self.viewport.show_mesh(self.recon_mesh)
            self.btn_show_mesh.setEnabled(True)
            self.btn_solidify.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.act_export_mesh.setEnabled(True)
            self._update_mesh_label(result["stats"])

        elif block == WorkerThread.BLOCK_SOLIDIFY:
            self.solid_mesh = result["mesh"]
            self._current_view = "solid"
            self.viewport.show_mesh(self.solid_mesh)
            self.btn_show_solid.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.act_export_solid.setEnabled(True)
            self._update_solid_label(result["stats"])

    def _on_error(self, msg: str):
        self._log(f"  ❌ ERROR: {msg}")
        self._show_error(msg)

    def _on_worker_finished(self):
        self.progress_bar.setVisible(False)
        self._set_pipeline_buttons(True)

        # No soltar la referencia al worker aquí: el hilo puede seguir en
        # su limpieza interna. Se libera en el próximo _start_worker
        # (tras wait()), lo que evita que Qt lo destruya en ejecución.
        cb = getattr(self._worker, "_on_done_callback", None)
        if cb is not None:
            self._worker._on_done_callback = None
            cb()

    def _worker_busy(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            self._log("  ⚠️  Hay una operación en curso, espera...")
            return True
        return False

    def closeEvent(self, event):
        # Esperar al worker antes de cerrar: destruir la ventana con un
        # QThread vivo aborta el proceso.
        if self._worker is not None:
            self._worker.wait()
        super().closeEvent(event)

    def _set_pipeline_buttons(self, enabled: bool):
        if self.raw_pcd is not None:
            self.btn_preprocess.setEnabled(enabled)
            self.btn_run_all.setEnabled(enabled)
        if self.clean_pcd is not None or self.raw_pcd is not None:
            self.btn_reconstruct.setEnabled(enabled)
        if self.recon_mesh is not None:
            self.btn_solidify.setEnabled(enabled)

    # ══════════════════════════════════════════════════════════════
    #  UNDO  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _save_undo_state(self):
        self._undo_stack = {
            "clean_pcd"  : deepcopy(self.clean_pcd)  if self.clean_pcd  else None,
            "recon_mesh" : deepcopy(self.recon_mesh) if self.recon_mesh else None,
            "solid_mesh" : deepcopy(self.solid_mesh) if self.solid_mesh else None,
        }
        self.btn_undo.setEnabled(True)

    def _undo(self):
        if self._undo_stack is None:
            self._log("  ℹ️  Nada que deshacer")
            return

        self.clean_pcd  = self._undo_stack["clean_pcd"]
        self.recon_mesh = self._undo_stack["recon_mesh"]
        self.solid_mesh = self._undo_stack["solid_mesh"]
        self._undo_stack = None
        self.btn_undo.setEnabled(False)

        if self.solid_mesh:
            self.viewport.show_mesh(self.solid_mesh)
        elif self.recon_mesh:
            self.viewport.show_mesh(self.recon_mesh)
        elif self.clean_pcd:
            self.viewport.show_point_cloud(self.clean_pcd)

        self._update_info_labels()
        self._log("  ↩ Deshacer aplicado")
        self._status("Estado anterior restaurado")

    # ══════════════════════════════════════════════════════════════
    #  ETIQUETAS DE INFORMACIÓN  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _update_info_labels(self):
        if self.raw_pcd:
            n = len(self.raw_pcd.points)
            self.lbl_pcd_info.setText(f"Nube orig: {n:,} pts")

    def _update_pcd_label(self, stats: dict):
        self.lbl_pcd_info.setText(
            f"Nube: {stats['final_points']:,} pts\n"
            f"(orig: {stats['original_points']:,})"
        )

    def _update_mesh_label(self, stats: dict):
        wt = stats.get("is_closed")
        cierre = "—" if wt is None else ("cerrada" if wt else "abierta")
        self.lbl_mesh_info.setText(
            f"Malla: {stats['final_vertices']:,}v "
            f"{stats['final_triangles']:,}t\n"
            f"[{stats['method']}, {cierre}]"
        )

    def _update_solid_label(self, stats: dict):
        self.lbl_solid_info.setText(
            f"Sólido: {stats['output_vertices']:,}v "
            f"{stats['output_triangles']:,}t"
        )
        self._update_quality_panel(stats)

    # ══════════════════════════════════════════════════════════════
    #  PANEL DE CALIDAD
    # ══════════════════════════════════════════════════════════════

    _OK    = "#8fd08f"
    _WARN  = "#d0b070"
    _BAD   = "#d08080"
    _MUTED = "#9ab0c0"

    def _update_quality_panel(self, stats: dict):
        """
        Vuelca en el panel el resultado del control de calidad de la
        cascada. Todo esto ya lo calculaba `MeshSolidifier`; hasta ahora
        solo se podía leer en el log.
        """
        def pintar(lbl, texto, color):
            lbl.setText(texto)
            lbl.setStyleSheet(f"color: {color};")

        # ── Estrategia ganadora ─────────────────────────────────────────
        estrategia = stats.get("strategy_used", "—")
        # 'passthrough' significa que ninguna estrategia produjo nada y se
        # devolvió la malla de entrada tal cual: no hay sólido de verdad.
        # 'hull' cierra siempre, pero destruyendo las concavidades.
        color_est = {
            "repair"      : self._OK,
            "poisson"     : self._OK,
            "voxel"       : self._WARN,
            "hull"        : self._BAD,
            "passthrough" : self._BAD,
        }.get(estrategia, self._MUTED)
        pintar(self.lbl_q_strategy, f"Estrategia: {estrategia}", color_est)

        # ── Error de fidelidad ──────────────────────────────────────────
        fid = stats.get("fidelity_error")
        if fid is None:
            pintar(self.lbl_q_fidelity,
                   "Fidelidad: no medida", self._MUTED)
        else:
            tol = self.param_panel.sol_fidelity_max.value() / 100.0
            color = self._OK if fid <= tol else self._BAD
            pintar(self.lbl_q_fidelity,
                   f"Fidelidad: {fid:.4%}  (tol. {tol:.2%})", color)

        # ── Cierre topológico ───────────────────────────────────────────
        cerrado    = bool(stats.get("is_watertight"))
        orientable = bool(stats.get("is_orientable"))
        pintar(self.lbl_q_closed,
               f"Cerrado: {'✓ sí' if cerrado else '✗ no'}   "
               f"Orientable: {'✓' if orientable else '✗'}",
               self._OK if cerrado else self._BAD)

        # ── Volumen ─────────────────────────────────────────────────────
        vol = stats.get("volume")
        if vol is None:
            pintar(self.lbl_q_volume,
                   "Volumen: n/d (malla abierta)", self._MUTED)
        else:
            pintar(self.lbl_q_volume, f"Volumen: {vol:.6g}", self._MUTED)

        # ── Agujeros de la malla de entrada ─────────────────────────────
        holes = stats.get("holes_found")
        pintar(self.lbl_q_holes,
               f"Agujeros entrada: "
               f"{'—' if holes is None else f'{holes:,} aristas'}",
               self._MUTED)

        # ── Cascada de estrategias probadas ─────────────────────────────
        tried = stats.get("strategies_tried") or {}
        if not tried:
            self.lbl_q_cascade.setText("—")
            self.lbl_q_cascade.setToolTip("")
            return

        lineas = []
        for nombre, r in tried.items():
            err = r.get("fidelity_error")
            err_txt = "—" if err is None else f"{err:.3%}"
            marca = "✓" if nombre == estrategia else " "
            cierra = "cerr" if r.get("watertight") else "abre"
            lineas.append(f"{marca} {nombre:<8} {cierra}  {err_txt}")

        self.lbl_q_cascade.setText("\n".join(lineas))
        self.lbl_q_cascade.setToolTip(_tt(
            "Cada estrategia que se probó, en orden, con si logró cerrar "
            "la malla y cuánto se alejó de la geometría original. La "
            "marcada con ✓ es la que se aceptó."))

    def _reset_quality_panel(self):
        """Limpia el panel cuando el sólido deja de ser válido."""
        for lbl, texto in (
            (self.lbl_q_strategy, "Estrategia: —"),
            (self.lbl_q_fidelity, "Fidelidad: —"),
            (self.lbl_q_closed,   "Cerrado: —"),
            (self.lbl_q_volume,   "Volumen: —"),
            (self.lbl_q_holes,    "Agujeros entrada: —"),
            (self.lbl_q_cascade,  "—"),
        ):
            lbl.setText(texto)
            lbl.setStyleSheet("")
        self.lbl_q_cascade.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #9ab0c0;")

    # ══════════════════════════════════════════════════════════════
    #  UTILIDADES  (sin cambios)
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.log_box.append(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _status(self, msg: str):
        self.status_bar.showMessage(msg)

    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Formato OpenGL por defecto ─────────────────────────────────────
    # Vispy crea internamente QOpenGLWidget. Sin un QSurfaceFormat por
    # defecto explícito, esos widgets no obtienen un contexto válido y
    # fallan con "Failed to make context current". Hay que fijarlo ANTES
    # de construir QApplication.
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setVersion(2, 1)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Contextos compartidos: imprescindible para que el canvas de vispy
    # comparta el contexto GL con el resto de la app.
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
