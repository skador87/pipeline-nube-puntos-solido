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
import traceback
import numpy as np
import open3d as o3d

from copy import deepcopy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QScrollArea, QTextEdit,
    QFileDialog, QProgressBar, QTabWidget,
    QSizePolicy, QFrame, QAction, QToolBar,
    QStatusBar, QMessageBox, QMenu,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui   import QFont, QColor, QPalette, QIcon, QSurfaceFormat

import vispy
vispy.use(app="pyqt5")           # fija el backend antes de crear cualquier canvas
from vispy         import scene
from vispy.scene   import visuals

# ── Core modules ──────────────────────────────────────────────────────────────
from core.preprocessor import PointCloudPreprocessor
from core.reconstructor import MeshReconstructor
from core.solidifier    import MeshSolidifier

# ── CAMBIO 1: nuevo import ─────────────────────────────────────────────────────
from core.io_loader import PointCloudLoader, check_e57_support


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

    def run(self):
        try:
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
            "block" : self.BLOCK_RECONSTRUCT,
            "mesh"  : mesh,
            "stats" : stats,
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

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_pre_tab()
        self._build_rec_tab()
        self._build_sol_tab()

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

        gml.addWidget(QLabel("BP radius:"), 2, 0)
        self.bp_radius = QDoubleSpinBox()
        self.bp_radius.setRange(0.001, 10.0)
        self.bp_radius.setSingleStep(0.1)
        self.bp_radius.setValue(1.0)
        gml.addWidget(self.bp_radius, 2, 1)

        gml.addWidget(QLabel("Alpha:"), 3, 0)
        self.alpha_val = QDoubleSpinBox()
        self.alpha_val.setRange(0.001, 5.0)
        self.alpha_val.setSingleStep(0.01)
        self.alpha_val.setValue(0.1)
        gml.addWidget(self.alpha_val, 3, 1)

        gml.addWidget(QLabel("Alpha downsample:"), 4, 0)
        self.alpha_ds = QSpinBox()
        self.alpha_ds.setRange(1, 20)
        self.alpha_ds.setValue(1)
        gml.addWidget(self.alpha_ds, 4, 1)

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
            self.bp_radius:
                "Radio de la esfera pivotante de Ball Pivoting (en unidades "
                "de la nube; también se usa radio×2). Orientativo: 2–4× la "
                "distancia media entre puntos.",
            self.alpha_val:
                "Alpha de la envolvente: menor = más ajustada al detalle "
                "(puede fragmentarse); mayor = más gruesa y cerrada.",
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
            "bp_radius"         : self.bp_radius.value(),
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

class Viewport3D(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#1a1a22",
            show=False,
        )
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, distance=5, elevation=30, azimuth=45
        )

        self._pcd_visual  = None
        self._mesh_visual = None

        scene.visuals.XYZAxis(parent=self.view.scene)

    def show_point_cloud(self, pcd: o3d.geometry.PointCloud):
        self._clear_visuals()

        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return

        if pcd.has_colors():
            colors = np.asarray(pcd.colors).astype(np.float32)
        else:
            colors = np.full((len(pts), 3), 0.6, dtype=np.float32)

        self._pcd_visual = visuals.Markers()
        self._pcd_visual.set_data(
            pts.astype(np.float32),
            face_color=colors,
            size=2,
            edge_width=0,
        )
        self.view.add(self._pcd_visual)
        self._fit_camera(pts)

    def show_mesh(self, mesh: o3d.geometry.TriangleMesh):
        self._clear_visuals()

        verts = np.asarray(mesh.vertices,  dtype=np.float32)
        tris  = np.asarray(mesh.triangles, dtype=np.uint32)

        if len(verts) == 0 or len(tris) == 0:
            return

        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        else:
            colors = np.full((len(verts), 3), 0.65, dtype=np.float32)

        self._mesh_visual = visuals.Mesh(
            vertices      = verts,
            faces         = tris,
            vertex_colors = colors,
            shading       = "smooth",
        )
        self.view.add(self._mesh_visual)
        self._fit_camera(verts)

    def _clear_visuals(self):
        for v in (self._pcd_visual, self._mesh_visual):
            if v is not None:
                v.parent = None
        self._pcd_visual  = None
        self._mesh_visual = None

    def _fit_camera(self, points: np.ndarray):
        center   = points.mean(axis=0)
        max_span = np.max(points.max(axis=0) - points.min(axis=0))
        self.view.camera.center   = center
        self.view.camera.distance = max_span * 1.8


# ══════════════════════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    # ── Constantes base (se ajustan en __init__ según disponibilidad e57) ──
    _IMPORT_FORMATS_FULL    = "Point Cloud (*.ply *.xyz *.pcd *.e57)"
    _IMPORT_FORMATS_NO_E57  = "Point Cloud (*.ply *.xyz *.pcd)"
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

        # ── CAMBIO 2: verificar e57 y fijar IMPORT_FORMATS ────────────
        e57_ok, e57_msg = check_e57_support()
        if e57_ok:
            self.IMPORT_FORMATS = self._IMPORT_FORMATS_FULL
        else:
            self.IMPORT_FORMATS = self._IMPORT_FORMATS_NO_E57
            # El log todavía no existe; guardamos el aviso para emitirlo
            # después de que _build_ui() construya self.log_box
            self._pending_e57_warn = e57_msg

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()

        # Emitir aviso e57 si corresponde (log_box ya existe aquí)
        if hasattr(self, "_pending_e57_warn"):
            self._log(f"  ℹ️  {self._pending_e57_warn}")
            del self._pending_e57_warn

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

        splitter.setSizes([290, 900, 210])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_right_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 4, 4, 4)
        w.setFixedWidth(210)

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

        self.btn_export = QPushButton("💾 Exportar…")
        self.btn_export.setEnabled(False)
        menu_export = QMenu(self.btn_export)
        menu_export.addAction(self.act_export_cloud)
        menu_export.addAction(self.act_export_mesh)
        menu_export.addAction(self.act_export_solid)
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

        lay.addWidget(g_view)

        g_edit = QGroupBox("Edición")
        gedit  = QVBoxLayout(g_edit)

        self.btn_undo = QPushButton("↩ Deshacer (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        gedit.addWidget(self.btn_undo)

        lay.addWidget(g_edit)

        g_info = QGroupBox("Info")
        ginf   = QVBoxLayout(g_info)

        self.lbl_pcd_info  = QLabel("Nube: —")
        self.lbl_pcd_info.setWordWrap(True)
        ginf.addWidget(self.lbl_pcd_info)

        self.lbl_mesh_info = QLabel("Malla: —")
        self.lbl_mesh_info.setWordWrap(True)
        ginf.addWidget(self.lbl_mesh_info)

        self.lbl_watertight = QLabel("Watertight: —")
        ginf.addWidget(self.lbl_watertight)

        lay.addWidget(g_info)
        lay.addStretch()

        return w

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

        self.btn_undo.clicked.connect(self._undo)

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
            self.viewport.show_point_cloud(self.raw_pcd)
            self._status("Vista: nube original")

    def _show_clean_pcd(self):
        if self.clean_pcd:
            self.viewport.show_point_cloud(self.clean_pcd)
            self._status("Vista: nube preprocesada")
        else:
            self._log("  ⚠️  Nube preprocesada no disponible")

    def _show_recon_mesh(self):
        if self.recon_mesh:
            self.viewport.show_mesh(self.recon_mesh)
            self._status("Vista: malla reconstruida")
        else:
            self._log("  ⚠️  Malla reconstruida no disponible")

    def _show_solid_mesh(self):
        if self.solid_mesh:
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
            self.lbl_watertight.setText("Watertight: —")
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
            self.viewport.show_mesh(self.recon_mesh)
            self.btn_show_mesh.setEnabled(True)
            self.btn_solidify.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.act_export_mesh.setEnabled(True)
            self._update_mesh_label(result["stats"])

        elif block == WorkerThread.BLOCK_SOLIDIFY:
            self.solid_mesh = result["mesh"]
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
        self.lbl_mesh_info.setText(
            f"Malla: {stats['final_vertices']:,}v "
            f"{stats['final_triangles']:,}t\n"
            f"[{stats['method']}]"
        )
        wt = stats.get("is_closed")
        if wt is None:
            self.lbl_watertight.setText("Watertight: —")
        else:
            self.lbl_watertight.setText(
                f"Watertight: {'✓ Sí' if wt else '✗ No'}"
            )

    def _update_solid_label(self, stats: dict):
        wt = "✓ Sí" if stats["is_watertight"] else "✗ No"
        self.lbl_mesh_info.setText(
            f"Sólido: {stats['output_vertices']:,}v "
            f"{stats['output_triangles']:,}t\n"
            f"[{stats['strategy_used']}]"
        )
        self.lbl_watertight.setText(f"Watertight: {wt}")

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
