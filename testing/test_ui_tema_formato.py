# testing/test_ui_tema_formato.py
# Garantías de la capa de presentación: contraste de la paleta y formato
# numérico en español.
#
# No necesita Qt ni pantalla: los tokens son constantes y el formato es
# aritmética de cadenas, así que corre en milisegundos.
#
# Uso:
#   & $PY testing\test_ui_tema_formato.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import theme
from ui import formato as F


# ══════════════════════════════════════════════════════════════
#  CONTRASTE (WCAG 2.1)
# ══════════════════════════════════════════════════════════════

def _luminancia(hexcolor: str) -> float:
    h = hexcolor.lstrip("#")
    canales = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    canales = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
               for c in canales]
    return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]


def contraste(fg: str, bg: str) -> float:
    a, b = _luminancia(fg), _luminancia(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


# Cada combinación texto/fondo que la interfaz usa de verdad.
# TEXT_DISABLED queda fuera a propósito: WCAG 1.4.3 exime explícitamente los
# componentes de interfaz inactivos.
COMBINACIONES = [
    ("texto sobre panel",        theme.TEXT,          theme.BG_SURFACE),
    ("texto sobre control",      theme.TEXT,          theme.BG_RAISED),
    ("texto sobre fondo base",   theme.TEXT,          theme.BG_BASE),
    ("secundario sobre panel",   theme.TEXT_MUTED,    theme.BG_SURFACE),
    ("nota sobre panel",         theme.TEXT_SUBTLE,   theme.BG_SURFACE),
    ("título de grupo",          theme.INFO,          theme.BG_SURFACE),
    ("terminal",                 theme.TERMINAL_TEXT, theme.BG_BASE),
    ("estado correcto",          theme.OK,            theme.BG_SURFACE),
    ("estado de aviso",          theme.WARN,          theme.BG_SURFACE),
    ("estado de error",          theme.ERROR,         theme.BG_SURFACE),
    ("texto sobre acento",       theme.ACCENT_TEXT,   theme.ACCENT),
    ("texto sobre acento hover", theme.ACCENT_TEXT,   theme.ACCENT_HOVER),
    ("texto sobre acento pulsado", theme.ACCENT_TEXT, theme.ACCENT_PRESSED),
    ("texto sobre selección",    theme.TEXT,          theme.BG_SELECTED),
    ("texto sobre hover",        theme.TEXT,          theme.BG_HOVER),
]

# WCAG 1.4.11 (contraste no textual) pide 3:1 para «la información visual
# necesaria para identificar un componente y su estado». Aquí eso se traduce
# en el indicador de foco: es lo que permite a quien navega con teclado saber
# dónde está, y no tiene ningún otro canal que lo sustituya.
#
# Deliberadamente NO se exige 3:1 al borde en reposo contra el panel. Un
# borde con esa relación en un tema oscuro sale casi blanco y convierte un
# formulario denso en una reja; los controles ya se identifican por su relleno
# (más claro que el panel) y por su etiqueta. Exigirlo empeoraría la interfaz
# en nombre de un criterio que la norma no impone a ese elemento.
FOCO = [
    ("foco sobre panel",   theme.BORDER_FOCUS, theme.BG_SURFACE),
    ("foco sobre control", theme.BORDER_FOCUS, theme.BG_RAISED),
    ("foco sobre base",    theme.BORDER_FOCUS, theme.BG_BASE),
]
AA_COMPONENTE = 3.0

AA_NORMAL = 4.5


def test_contraste_aa():
    """Todo par texto/fondo de la paleta cumple WCAG AA."""
    fallos = []
    for nombre, fg, bg in COMBINACIONES:
        r = contraste(fg, bg)
        if r < AA_NORMAL:
            fallos.append(f"{nombre}: {r:.2f}:1 (mínimo {AA_NORMAL})")
    assert not fallos, "combinaciones que no cumplen AA:\n  " + \
                       "\n  ".join(fallos)


def test_indicador_de_foco():
    """El foco de teclado cumple el 3:1 de WCAG 1.4.11 sobre las tres
    superficies en las que puede aparecer."""
    fallos = []
    for nombre, fg, bg in FOCO:
        r = contraste(fg, bg)
        if r < AA_COMPONENTE:
            fallos.append(f"{nombre}: {r:.2f}:1 (mínimo {AA_COMPONENTE})")
    assert not fallos, "el indicador de foco no destaca lo suficiente:\n  " + \
                       "\n  ".join(fallos)


def test_control_se_distingue_del_panel():
    """
    Un control debe distinguirse de la superficie que lo contiene por su
    relleno. No se mide con la razón de contraste (a estos niveles de
    oscuridad da valores diminutos que no reflejan lo que se percibe) sino
    con la diferencia de luminancia absoluta.
    """
    delta = _luminancia(theme.BG_RAISED) - _luminancia(theme.BG_SURFACE)
    assert delta >= 0.008, (
        f"el relleno del control apenas se separa del panel (Δ={delta:.4f}); "
        f"los campos se pierden sobre el fondo"
    )


def test_superficies_son_escala():
    """Las superficies deben ir de más profunda a más elevada, sin empates:
    es lo que da sensación de profundidad al panel."""
    escala = [theme.BG_BASE, theme.BG_SURFACE, theme.BG_RAISED, theme.BG_HOVER]
    lums = [_luminancia(c) for c in escala]
    assert lums == sorted(lums), (
        f"las superficies no forman una escala creciente: {lums}"
    )
    assert len(set(lums)) == len(lums), "hay dos superficies con igual luminancia"


def test_stylesheet_sin_colores_sueltos():
    """
    El stylesheet debe salir íntegramente de los tokens. Si aparece un color
    hexadecimal que no está declarado como token, es que alguien lo escribió
    a mano y el tema vuelve a estar repartido.
    """
    import re
    qss = theme.build_stylesheet()
    tokens = {v.lower() for k, v in vars(theme).items()
              if isinstance(v, str) and v.startswith("#")}
    encontrados = {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}", qss)}
    huerfanos = encontrados - tokens
    assert not huerfanos, f"colores fuera de los tokens: {sorted(huerfanos)}"


# ══════════════════════════════════════════════════════════════
#  FORMATO NUMÉRICO
# ══════════════════════════════════════════════════════════════

def test_enteros_con_punto_de_millares():
    assert F.entero(8171) == "8.171"
    assert F.entero(23126) == "23.126"
    assert F.entero(5838) == "5.838"
    assert F.entero(999) == "999"
    assert F.entero(1234567) == "1.234.567"
    assert F.entero(None) == "—"


def test_decimales_con_coma():
    assert F.decimal(1.5) == "1,50"
    assert F.decimal(-0.53, 2) == "-0,53"
    assert F.decimal(1234.5, 2) == "1.234,50"
    assert F.decimal(None) == "—"


def test_porcentaje():
    # El caso real: el error de fidelidad del bunny.
    assert F.porcentaje(0.003968, 2) == "0,40 %"
    assert F.porcentaje(0.02, 2) == "2,00 %"
    assert F.porcentaje(None) == "—"


def test_significativo():
    assert F.significativo(0.000746066) == "0,000746066"
    assert F.significativo(None) == "—"


def test_segundos():
    assert F.segundos(0.23) == "0,2 s"
    assert F.segundos(3.69) == "3,7 s"
    assert F.segundos(65.4) == "1 min 5 s"
    assert F.segundos(None) == "—"


def test_vert_tri():
    assert F.vert_tri(11674, 23126) == "11.674 vért · 23.126 tri"


def test_nunca_coma_de_millares():
    """
    La regresión concreta que se está arreglando: ninguna salida puede usar
    la coma como separador de millares, porque en la misma pantalla la coma
    ya significa decimal.
    """
    casos = [F.entero(1234567), F.decimal(1234.5), F.vert_tri(11674, 23126)]
    for s in casos:
        # Una coma solo es válida si va seguida de dígitos decimales, nunca
        # separando grupos de tres al estilo inglés.
        import re
        assert not re.search(r"\d,\d{3}(?!\d)", s.replace(" ", "")) or "," not in s, \
            f"'{s}' parece usar coma de millares"


# ══════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════

def main() -> int:
    pruebas = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)]
    fallos = []
    for fn in pruebas:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            fallos.append((fn.__name__, str(e)))
            print(f"  ✗ {fn.__name__}")

    if fallos:
        print(f"\n✗ {len(fallos)}/{len(pruebas)} fallaron:\n")
        for nombre, msg in fallos:
            print(f"  [{nombre}]\n    {msg}\n")
        return 1

    print(f"\n✓ {len(pruebas)}/{len(pruebas)} pruebas pasaron")
    print("\n  Contraste de la paleta:")
    for nombre, fg, bg in COMBINACIONES + FOCO:
        print(f"    {nombre:<28} {contraste(fg, bg):5.2f}:1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
