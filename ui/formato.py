# ui/formato.py
# Formato numérico en convención española.
#
# El problema que resuelve: la interfaz mezclaba dos significados de la coma
# en la misma pantalla. Las etiquetas usaban el `:,` de Python, que separa
# millares con coma ("5,838 pts"), mientras que los campos numéricos usan el
# locale español de Qt, que separa decimales con coma ("0,1000"). Leído en
# español, "5,838 pts" son cinco puntos y pico.
#
# Convención adoptada (la de Qt, que no se puede cambiar sin pelearse con los
# spinboxes): millares con punto, decimales con coma.
#
#     8171      → "8.171"
#     0.003968  → "0,003968"
#     0.003968  → "0,40 %"  (como porcentaje)


def entero(n) -> str:
    """Entero con punto como separador de millares. 8171 → '8.171'."""
    if n is None:
        return "—"
    return f"{int(n):,}".replace(",", ".")


def decimal(x, decimales: int = 2) -> str:
    """Decimal con coma. 1.5 → '1,50'."""
    if x is None:
        return "—"
    return f"{float(x):,.{decimales}f}".replace(",", "\x00") \
                                       .replace(".", ",") \
                                       .replace("\x00", ".")


def significativo(x, cifras: int = 6) -> str:
    """Decimal con cifras significativas, para magnitudes de escala
    desconocida (volúmenes, tamaños de voxel). 0.000746066 → '0,000746066'."""
    if x is None:
        return "—"
    return f"{float(x):.{cifras}g}".replace(".", ",")


def porcentaje(x, decimales: int = 2) -> str:
    """Fracción como porcentaje. 0.003968 → '0,40 %'.
    Se separa el símbolo con espacio, como manda la norma en español."""
    if x is None:
        return "—"
    return decimal(float(x) * 100.0, decimales) + " %"


def segundos(x) -> str:
    """Duración legible. 0.23 → '0,2 s'; 65.4 → '1 min 5 s'."""
    if x is None:
        return "—"
    x = float(x)
    if x < 60:
        return f"{decimal(x, 1)} s"
    minutos, resto = divmod(x, 60)
    return f"{int(minutos)} min {int(resto)} s"


def vert_tri(vertices, triangulos) -> str:
    """Resumen de una malla: '11.674 vért · 23.126 tri'."""
    return f"{entero(vertices)} vért · {entero(triangulos)} tri"
