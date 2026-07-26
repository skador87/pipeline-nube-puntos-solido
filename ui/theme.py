# ui/theme.py
# Tokens del tema oscuro y construcción del stylesheet.
#
# Antes los colores estaban repartidos entre `apply_dark_theme()` y una docena
# de `setStyleSheet` sueltos por `main_gui.py`, así que cambiar un gris
# implicaba buscarlo en varios sitios y los tonos se habían ido separando
# entre sí. Aquí viven una sola vez.
#
# Reglas de la paleta
# ───────────────────
# 1. Las superficies forman una escala de profundidad: cuanto más «arriba»
#    está un elemento (un input dentro de un panel dentro de la ventana),
#    más claro es su fondo.
# 2. Todo par texto/fondo previsto cumple WCAG AA (4.5:1 para texto normal).
#    `testing/test_tema_contraste.py` lo verifica y falla si alguien
#    introduce una combinación que no cumple.
# 3. Los colores semánticos (ok / aviso / error) no se usan nunca solos:
#    siempre acompañan a un texto que dice lo mismo, para no depender del
#    color como único canal de información.

# ══════════════════════════════════════════════════════════════
#  COLOR
# ══════════════════════════════════════════════════════════════

# Superficies, de más profunda a más elevada
BG_BASE     = "#16161a"   # fondo de la aplicación, visor, terminal
BG_SURFACE  = "#1e1e23"   # paneles
BG_RAISED   = "#2a2a31"   # inputs, botones
BG_HOVER    = "#353540"
BG_PRESSED  = "#131318"
BG_SELECTED = "#2f3a46"   # fila/elemento seleccionado

# Bordes
BORDER        = "#33333d"
BORDER_STRONG = "#4a4a58"
BORDER_FOCUS  = "#5a96c8"

# Texto
TEXT          = "#dcdcdc"   # texto normal
TEXT_MUTED    = "#a8bcc8"   # secundario, valores de apoyo
TEXT_SUBTLE   = "#8b98a4"   # notas y ayuda
TEXT_DISABLED = "#6a6a74"   # exento de contraste por WCAG 1.4.3

# Acento. La rampa va de oscuro a claro para que el hover se perciba como
# «se enciende», y los tres estados mantienen texto blanco legible: el azul
# original (#4682b4) solo daba 4.11:1 con blanco y no llegaba a AA.
ACCENT         = "#356a95"
ACCENT_HOVER   = "#3d76a4"
ACCENT_PRESSED = "#2b5c80"
ACCENT_TEXT    = "#ffffff"

# Semánticos
OK    = "#8fd08f"
WARN  = "#e0bc7a"
ERROR = "#e09090"
INFO  = "#82b4d0"

# Terminal
TERMINAL_TEXT = "#b0d0b0"

# ══════════════════════════════════════════════════════════════
#  TIPOGRAFÍA
# ══════════════════════════════════════════════════════════════
# Escala explícita: antes todo vivía entre 10 y 12 px sin que el tamaño
# indicara jerarquía. Ahora cada nivel tiene un uso definido.

FONT_XS = 10   # datos densos y monoespaciados (cascada de estrategias)
FONT_SM = 11   # notas, texto de ayuda
FONT_MD = 12   # cuerpo: etiquetas, botones, valores
FONT_LG = 13   # títulos de grupo
FONT_XL = 15   # títulos de sección destacados

FONT_UI   = '"Segoe UI", "Noto Sans", sans-serif'
FONT_MONO = '"Cascadia Mono", "Consolas", monospace'

# ══════════════════════════════════════════════════════════════
#  ESPACIADO Y FORMA
# ══════════════════════════════════════════════════════════════

SPACE_1 = 2
SPACE_2 = 4
SPACE_3 = 6
SPACE_4 = 8
SPACE_5 = 12
SPACE_6 = 16

RADIUS_SM = 3
RADIUS_MD = 4
RADIUS_LG = 6

# Alto mínimo de un control clicable. 28 px es el mínimo cómodo con ratón
# en escritorio; los botones primarios usan 32.
CONTROL_H         = 28
CONTROL_H_PRIMARY = 32


# ══════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL TEMA
# ══════════════════════════════════════════════════════════════

def build_palette():
    """QPalette base. Importa Qt de forma perezosa para poder testear los
    tokens (contraste) sin necesitar una QApplication."""
    from PyQt5.QtGui  import QPalette, QColor
    from PyQt5.QtCore import Qt

    def c(h):
        return QColor(h)

    pal = QPalette()
    pal.setColor(QPalette.Window,          c(BG_SURFACE))
    pal.setColor(QPalette.WindowText,      c(TEXT))
    pal.setColor(QPalette.Base,            c(BG_RAISED))
    pal.setColor(QPalette.AlternateBase,   c(BG_HOVER))
    pal.setColor(QPalette.ToolTipBase,     c(BG_BASE))
    pal.setColor(QPalette.ToolTipText,     c(TEXT))
    pal.setColor(QPalette.Text,            c(TEXT))
    pal.setColor(QPalette.Button,          c(BG_RAISED))
    pal.setColor(QPalette.ButtonText,      c(TEXT))
    pal.setColor(QPalette.BrightText,      Qt.white)
    # Los enlaces usan INFO y no ACCENT: el acento está calibrado para llevar
    # texto blanco encima, no para ser texto sobre el fondo del panel.
    pal.setColor(QPalette.Link,            c(INFO))
    pal.setColor(QPalette.Highlight,       c(ACCENT))
    pal.setColor(QPalette.HighlightedText, c(ACCENT_TEXT))
    pal.setColor(QPalette.Disabled, QPalette.Text,       c(TEXT_DISABLED))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, c(TEXT_DISABLED))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, c(TEXT_DISABLED))
    return pal


def build_stylesheet() -> str:
    """Stylesheet global, generado a partir de los tokens."""
    return f"""
        QWidget {{
            font-family: {FONT_UI};
            font-size: {FONT_MD}px;
        }}

        QMainWindow, QDialog {{ background: {BG_SURFACE}; }}

        /* ── Agrupadores ───────────────────────────────────────── */
        QGroupBox {{
            border: 1px solid {BORDER};
            border-radius: {RADIUS_MD}px;
            margin-top: {SPACE_5}px;
            padding: {SPACE_4}px {SPACE_3}px {SPACE_3}px {SPACE_3}px;
            font-size: {FONT_LG}px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: {SPACE_3}px;
            padding: 0 {SPACE_2}px;
            color: {INFO};
        }}

        /* ── Botones ───────────────────────────────────────────── */
        QPushButton {{
            background-color: {BG_RAISED};
            border: 1px solid {BORDER_STRONG};
            border-radius: {RADIUS_MD}px;
            padding: {SPACE_2}px {SPACE_5}px;
            min-height: {CONTROL_H}px;
            text-align: center;
        }}
        QPushButton:hover  {{
            background-color: {BG_HOVER};
            border-color: {BORDER_FOCUS};
        }}
        QPushButton:pressed {{ background-color: {BG_PRESSED}; }}
        QPushButton:disabled {{
            color: {TEXT_DISABLED};
            border-color: {BORDER};
            background-color: {BG_SURFACE};
        }}
        QPushButton[primary="true"] {{
            background-color: {ACCENT};
            border-color: {ACCENT_HOVER};
            color: {ACCENT_TEXT};
            font-weight: 600;
            min-height: {CONTROL_H_PRIMARY}px;
        }}
        QPushButton[primary="true"]:hover {{
            background-color: {ACCENT_HOVER};
        }}
        QPushButton[primary="true"]:pressed {{
            background-color: {ACCENT_PRESSED};
        }}
        QPushButton[primary="true"]:disabled {{
            background-color: {BG_SURFACE};
            border-color: {BORDER};
            color: {TEXT_DISABLED};
        }}

        /* ── Pestañas ──────────────────────────────────────────── */
        QTabWidget::pane {{
            border: 1px solid {BORDER};
            border-radius: {RADIUS_MD}px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: {BG_SURFACE};
            color: {TEXT_SUBTLE};
            padding: {SPACE_3}px {SPACE_5}px;
            border: 1px solid {BORDER};
            border-bottom: none;
            border-top-left-radius: {RADIUS_MD}px;
            border-top-right-radius: {RADIUS_MD}px;
        }}
        QTabBar::tab:selected {{
            background: {BG_RAISED};
            color: {INFO};
            font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

        /* ── Campos ────────────────────────────────────────────── */
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
            background: {BG_RAISED};
            border: 1px solid {BORDER_STRONG};
            border-radius: {RADIUS_SM}px;
            padding: {SPACE_1}px {SPACE_3}px;
            min-height: {CONTROL_H - 8}px;
            color: {TEXT};
        }}
        QComboBox:focus, QSpinBox:focus,
        QDoubleSpinBox:focus, QLineEdit:focus {{
            border-color: {BORDER_FOCUS};
        }}
        QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled {{
            color: {TEXT_DISABLED};
            background: {BG_SURFACE};
            border-color: {BORDER};
        }}
        QComboBox QAbstractItemView {{
            background: {BG_RAISED};
            border: 1px solid {BORDER_STRONG};
            selection-background-color: {ACCENT};
            selection-color: {ACCENT_TEXT};
        }}

        QCheckBox {{ spacing: {SPACE_3}px; min-height: {CONTROL_H - 6}px; }}
        QCheckBox:disabled {{ color: {TEXT_DISABLED}; }}

        /* ── Terminal ──────────────────────────────────────────── */
        QTextEdit {{
            background-color: {BG_BASE};
            color: {TERMINAL_TEXT};
            font-family: {FONT_MONO};
            font-size: {FONT_SM}px;
            border: 1px solid {BORDER};
            border-radius: {RADIUS_MD}px;
        }}

        /* ── Progreso ──────────────────────────────────────────── */
        QProgressBar {{
            border: 1px solid {BORDER};
            border-radius: {RADIUS_SM}px;
            text-align: center;
            background: {BG_BASE};
            /* Blanco y no TEXT: el porcentaje se lee tanto sobre el fondo
               vacío como sobre la parte ya rellenada con el acento. */
            color: {ACCENT_TEXT};
        }}
        QProgressBar::chunk {{
            background-color: {ACCENT};
            border-radius: {RADIUS_SM - 1}px;
        }}

        /* ── Barras y separadores ──────────────────────────────── */
        QToolBar {{
            background: {BG_SURFACE};
            border: none;
            border-bottom: 1px solid {BORDER};
            spacing: {SPACE_2}px;
            padding: {SPACE_2}px;
        }}
        QToolBar QToolButton {{
            padding: {SPACE_2}px {SPACE_4}px;
            border-radius: {RADIUS_SM}px;
            color: {TEXT};
        }}
        QToolBar QToolButton:hover {{ background: {BG_HOVER}; }}
        QToolBar QToolButton:checked {{
            background: {BG_SELECTED};
            color: {INFO};
        }}
        QToolBar QToolButton:disabled {{ color: {TEXT_DISABLED}; }}
        QToolBar::separator {{
            background: {BORDER};
            width: 1px;
            margin: {SPACE_2}px {SPACE_3}px;
        }}

        QMenuBar {{ background: {BG_SURFACE}; border-bottom: 1px solid {BORDER}; }}
        QMenuBar::item {{ padding: {SPACE_2}px {SPACE_4}px; }}
        QMenuBar::item:selected {{ background: {BG_HOVER}; }}
        QMenu {{
            background: {BG_RAISED};
            border: 1px solid {BORDER_STRONG};
            padding: {SPACE_2}px;
        }}
        QMenu::item {{ padding: {SPACE_3}px {SPACE_5}px; border-radius: {RADIUS_SM}px; }}
        QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
        QMenu::separator {{ height: 1px; background: {BORDER}; margin: {SPACE_2}px 0; }}

        QStatusBar {{ background: {BG_SURFACE}; border-top: 1px solid {BORDER}; }}
        QStatusBar::item {{ border: none; }}

        QSplitter::handle {{ background: {BORDER}; }}
        QSplitter::handle:horizontal {{ width: {SPACE_2}px; }}
        QSplitter::handle:vertical   {{ height: {SPACE_2}px; }}
        QSplitter::handle:hover {{ background: {BORDER_FOCUS}; }}

        /* ── Scrollbars ────────────────────────────────────────── */
        QScrollBar:vertical {{
            background: transparent; width: 10px; margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER_STRONG};
            border-radius: {RADIUS_SM}px;
            min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {BORDER_FOCUS}; }}
        QScrollBar:horizontal {{
            background: transparent; height: 10px; margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {BORDER_STRONG};
            border-radius: {RADIUS_SM}px;
            min-width: 24px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

        QScrollArea {{ border: none; background: {BG_SURFACE}; }}

        QToolTip {{
            background: {BG_BASE};
            color: {TEXT};
            border: 1px solid {BORDER_STRONG};
            padding: {SPACE_3}px;
        }}
    """


def apply(app):
    """Aplica el tema completo a la QApplication."""
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(build_stylesheet())


# ══════════════════════════════════════════════════════════════
#  ESTILOS PUNTUALES
# ══════════════════════════════════════════════════════════════
# Para los pocos casos en que un widget necesita un tratamiento propio.
# Se exponen como funciones para que ningún color quede escrito a mano
# en main_gui.py.

def label_style(color: str = TEXT, size: int = FONT_MD,
                mono: bool = False, bold: bool = False) -> str:
    familia = f"font-family: {FONT_MONO};" if mono else ""
    peso    = "font-weight: 600;" if bold else ""
    return f"color: {color}; font-size: {size}px; {familia} {peso}".strip()


def separator_style() -> str:
    return f"color: {BORDER}; background: {BORDER}; max-height: 1px;"
