# ui/widgets.py
# Componentes de interfaz propios del pipeline.

from PyQt5.QtCore    import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
)

from . import theme


def _tt(texto: str) -> str:
    """Envuelve el texto para que el tooltip haga word-wrap."""
    return f"<qt>{texto}</qt>"


class PasoPipeline(QWidget):
    """
    Un paso del pipeline (A, B o C) como unidad completa: su estado, el
    resumen de lo que produjo y las acciones para ejecutarlo y verlo.

    Antes esta información estaba repartida en cuatro grupos distintos del
    panel derecho — «Pipeline» tenía el botón de ejecutar, «Visualización» el
    de ver, «Productos» el resumen, y el estado no se mostraba en ninguna
    parte. Juntarlos hace visible la secuencia A→B→C, que es el modelo mental
    de toda la aplicación.

    Estados
    ───────
    pendiente : falta el producto de la etapa anterior, no se puede ejecutar
    listo     : se puede ejecutar
    corriendo : en ejecución
    hecho     : produjo su resultado

    El estado se comunica con símbolo Y color a la vez, nunca solo con color.
    """

    ejecutar = pyqtSignal()
    ver      = pyqtSignal()

    _ESTADOS = {
        "pendiente": ("○", theme.TEXT_DISABLED, "pendiente"),
        "listo":     ("▸", theme.INFO,          "listo para ejecutar"),
        "corriendo": ("⟳", theme.WARN,          "en ejecución"),
        "hecho":     ("✓", theme.OK,            "completado"),
    }

    def __init__(self, letra: str, titulo: str, descripcion: str,
                 parent=None):
        super().__init__(parent)
        self.letra   = letra
        self.titulo  = titulo
        self._estado = "pendiente"

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.SPACE_3, theme.SPACE_3,
                               theme.SPACE_3, theme.SPACE_3)
        lay.setSpacing(theme.SPACE_2)

        # ── Cabecera: marca de estado + nombre de la etapa ──────────────
        fila = QHBoxLayout()
        fila.setSpacing(theme.SPACE_3)

        self.lbl_marca = QLabel("○")
        self.lbl_marca.setFixedWidth(14)
        self.lbl_marca.setAlignment(Qt.AlignCenter)
        fila.addWidget(self.lbl_marca)

        self.lbl_titulo = QLabel(f"{letra} · {titulo}")
        self.lbl_titulo.setStyleSheet(
            theme.label_style(theme.TEXT, theme.FONT_MD, bold=True))
        fila.addWidget(self.lbl_titulo, stretch=1)
        lay.addLayout(fila)

        # ── Resumen del producto ────────────────────────────────────────
        self.lbl_resumen = QLabel("—")
        self.lbl_resumen.setWordWrap(True)
        self.lbl_resumen.setStyleSheet(
            theme.label_style(theme.TEXT_SUBTLE, theme.FONT_SM))
        self.lbl_resumen.setContentsMargins(14 + theme.SPACE_3, 0, 0, 0)
        lay.addWidget(self.lbl_resumen)

        # ── Acciones ────────────────────────────────────────────────────
        acciones = QHBoxLayout()
        acciones.setSpacing(theme.SPACE_2)
        acciones.setContentsMargins(14 + theme.SPACE_3, 0, 0, 0)

        self.btn_ejecutar = QPushButton("Ejecutar")
        self.btn_ejecutar.setToolTip(_tt(descripcion))
        self.btn_ejecutar.clicked.connect(self.ejecutar.emit)
        acciones.addWidget(self.btn_ejecutar, stretch=2)

        self.btn_ver = QPushButton("Ver")
        self.btn_ver.setEnabled(False)
        self.btn_ver.setToolTip(_tt(
            f"Muestra en el visor el resultado de {letra} · {titulo}."))
        self.btn_ver.clicked.connect(self.ver.emit)
        acciones.addWidget(self.btn_ver, stretch=1)

        lay.addLayout(acciones)

        self.set_estado("pendiente")

    # ── API ─────────────────────────────────────────────────────────────

    def set_estado(self, estado: str):
        """Cambia el estado del paso y con él su marca, color y tooltip."""
        if estado not in self._ESTADOS:
            return
        self._estado = estado
        marca, color, texto = self._ESTADOS[estado]

        self.lbl_marca.setText(marca)
        self.lbl_marca.setStyleSheet(
            theme.label_style(color, theme.FONT_MD, bold=True))
        # El estado también va en texto accesible, no solo en el color.
        self.lbl_marca.setToolTip(_tt(f"{self.letra} · {self.titulo}: {texto}"))

        titulo_color = (theme.TEXT if estado in ("hecho", "corriendo", "listo")
                        else theme.TEXT_SUBTLE)
        self.lbl_titulo.setStyleSheet(
            theme.label_style(titulo_color, theme.FONT_MD, bold=True))

        self.btn_ejecutar.setEnabled(estado in ("listo", "hecho"))
        self.btn_ejecutar.setText(
            "Re-ejecutar" if estado == "hecho" else "Ejecutar")

    def estado(self) -> str:
        return self._estado

    def set_resumen(self, texto: str, disponible: bool = True):
        """Actualiza la línea de resumen y habilita el botón «Ver»."""
        self.lbl_resumen.setText(texto or "—")
        self.lbl_resumen.setStyleSheet(theme.label_style(
            theme.TEXT_MUTED if disponible else theme.TEXT_SUBTLE,
            theme.FONT_SM))
        # Sin producto no hay nada que resumir: se oculta la línea en vez de
        # dejar un guion suelto bajo cada título.
        self.lbl_resumen.setVisible(disponible)
        self.btn_ver.setEnabled(disponible)

    def limpiar(self):
        """Devuelve el paso a «sin producto»."""
        self.set_resumen("—", disponible=False)


class SeparadorPaso(QFrame):
    """Línea entre pasos: sugiere la secuencia A→B→C."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setStyleSheet(theme.separator_style())
        self.setFixedHeight(1)


class EstadoVacio(QWidget):
    """
    Mensaje que ocupa el visor mientras no hay nada que mostrar.

    Se superpone al visor en vez de sustituirlo: ocultar el widget OpenGL de
    Vispy puede provocar que el driver destruya y recree el contexto, y el
    historial de este proyecto con los buffers GL desaconseja tocar eso.
    """

    def __init__(self, titulo: str, detalle: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(theme.SPACE_4)

        self.lbl_titulo = QLabel(titulo)
        self.lbl_titulo.setAlignment(Qt.AlignCenter)
        self.lbl_titulo.setStyleSheet(
            theme.label_style(theme.TEXT_MUTED, theme.FONT_XL, bold=True))
        lay.addWidget(self.lbl_titulo)

        self.lbl_detalle = QLabel(detalle)
        self.lbl_detalle.setAlignment(Qt.AlignCenter)
        self.lbl_detalle.setWordWrap(True)
        self.lbl_detalle.setMaximumWidth(420)
        self.lbl_detalle.setStyleSheet(
            theme.label_style(theme.TEXT_SUBTLE, theme.FONT_MD))
        lay.addWidget(self.lbl_detalle, alignment=Qt.AlignCenter)

    def set_texto(self, titulo: str, detalle: str):
        self.lbl_titulo.setText(titulo)
        self.lbl_detalle.setText(detalle)
