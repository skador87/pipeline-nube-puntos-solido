# Modos de reconstrucción basados en IA

> Evaluación de modelos y resultados del primer método implementado.
> Julio 2026.

---

## 1. Advertencia terminológica (importante para la memoria)

El método implementado **no es un modelo de IA preentrenado**. No hay pesos,
ni checkpoint, ni dataset de entrenamiento.

Lo que hay es **la maquinaria del aprendizaje profundo aplicada a la
geometría**: la ecuación de Poisson resuelta de forma *derivable* en PyTorch,
lo que permite propagar gradientes desde la malla resultante hasta los puntos
y normales de entrada.

En una defensa, la descripción precisa es:

> reconstrucción de superficie por Poisson diferenciable, resuelta en el
> dominio espectral sobre GPU, sin modelo preentrenado

Decir «usamos un modelo de IA» sin más sería impreciso y expone a una
repregunta incómoda. La ausencia de pesos entrenados es, además, una
**fortaleza**: no hay nada que pueda generalizar mal a datos fuera de
distribución.

---

## 2. Modelos evaluados y por qué se descartaron

Criterios fijados: repositorio público y reutilizable, Windows nativo, sin
pesos preentrenados.

| Modelo | Calidad | Licencia | Veredicto |
|---|---|---|---|
| **Shape As Points / DPSR** | Alta | MIT | ✅ **Implementado** |
| NKSR (CVPR'23) | La mejor | Nvidia, solo investigación | ❌ Solo Linux + licencia incompatible con repo público |
| POCO (CVPR'22) | Alta | No declarada | ❌ Python 3.7 / CUDA 11.1 / extensión en C |
| NumGrad-Pull / Neural-Pull | Media-alta | Variable | 🔸 Plan B, PyTorch puro |
| SHS-Net / OCMG-Net | — | — | 🔸 Pendiente, para el paso A (normales) |

**NKSR** era el mejor candidato en calidad y escala, pero acumula dos
bloqueantes duros: solo corre en Linux, y su licencia restringe el uso a
investigación y evaluación, lo que choca con un repositorio público
reutilizable. Además exige normales orientadas de entrada, así que no
resuelve el problema de las normales: lo presupone.

---

## 3. Por qué DPSR y no otro

El núcleo de Shape As Points importa únicamente `torch`, `torch.nn`,
`torch.fft` y `numpy`. **No necesita PyTorch3D ni torch-scatter ni kernels
CUDA propios** — esas dependencias las arrastra el pipeline que lo rodea
(pérdida de Chamfer, utilidades de malla), y en este proyecto ya existen
equivalentes: `cKDTree` de scipy para Chamfer y Open3D para mallas.

Eso reduce el coste de adopción a `torch` y `scikit-image`, ambos con wheels
oficiales para Windows. Cero compilación.

Además, conceptualmente **no se abandona Poisson: se hace derivable**. Es
continuidad con lo ya validado, no un cambio de paradigma.

---

## 4. Resultados medidos (bunny, RTX 3080 Ti)

Métrica: distancia de Chamfer simétrica **contra la nube de entrada**
normalizada por su diagonal, que es la comparación honesta (medir contra el
Poisson de Open3D sería medir contra otra aproximación, no contra la verdad).

| Camino | Triángulos | Cerrada | Volumen | Fidelidad | p95 | Tiempo |
|---|---|---|---|---|---|---|
| **Pipeline actual** (B+C) | 16.748 | ✅ | 0,000746 | **0,5861 %** | 1,5029 % | 4,6 s |
| DPSR 64³ σ=1,0 | 17.708 | ✅ | 0,000750 | 0,6623 % | 1,1129 % | 0,3 s |
| DPSR 128³ σ=1,5 | 71.612 | ✅ | 0,000747 | 0,6248 % | 1,0133 % | 0,2 s |
| DPSR 192³ σ=2,0 | 161.492 | ✅ | 0,000745 | 0,6136 % | 1,0037 % | 0,7 s |
| DPSR 256³ σ=2,5 | 287.372 | ✅ | 0,000745 | **0,6095 %** | 0,9911 % | 1,0 s |

### Lectura honesta de estos números

**El pipeline clásico gana en fidelidad media**: 0,5861 % frente a 0,6095 %
del mejor DPSR, un 4 % relativo mejor. Y lo hace con **17 veces menos
triángulos**. A igualdad de peso de malla (64³, ~17k triángulos) la
diferencia se amplía a un 13 %.

**DPSR gana en tres cosas:**

1. **Cierre garantizado en un solo paso.** Todas las configuraciones dan
   `borde = 0`. Verificado a través del software: la malla llega al paso C
   con `holes_found = 0`.
2. **Velocidad**: 0,2–1,0 s frente a 4,6 s del camino B+C.
3. **Error en la cola** (p95): 0,99 % frente a 1,50 % del pipeline actual.
   El pipeline clásico es mejor *en promedio* pero tiene peores casos
   extremos, probablemente en las zonas que la cascada de solidificación
   tuvo que inventar para cerrar.

**Por qué el clásico gana en la media, y qué significa.** El Poisson de
Open3D es *cribado* (screened), con restricciones sobre los puntos de
entrada; el DPSR implementado es el Poisson clásico **no cribado**. La
ventaja del trabajo original de Shape As Points no está en la pasada hacia
adelante, sino en el **bucle de optimización** que ajusta posiciones y
normales — que todavía no está implementado. Lo medido aquí es el cimiento,
no el método completo.

**Y el bunny es el mejor caso posible para el pipeline clásico**: objeto
limpio, denso, cerrado y sin huecos. La encuesta comparativa de referencia
concluye que los métodos aprendidos ganan en datos consistentes pero los
tradicionales resisten mejor las anomalías del escaneo real. La comparación
que falta —y que sí puede favorecer a DPSR— es sobre escaneos incompletos
tipo S3DIS, donde los filtros de artefactos fragmentan la malla.

### Dato lateral relevante

Se midió de dónde salen los agujeros que el paso C repara:

| Etapa | Aristas frontera | Componentes |
|---|---|---|
| Poisson crudo | 18 | 1 |
| tras filtro de webbing | 118 | 1 |
| tras filtro de zonas huecas | **249** | **6** |

Poisson sale casi cerrado; son los propios filtros de artefactos los que
multiplican los agujeros por 14 y fragmentan la malla. Buena parte del
trabajo del paso C consiste en deshacer lo que hace el paso B.

---

## 5. Dónde se ve en el software

**Modo Avanzado → pestaña `B · Rec` → Método → «Poisson diferenciable (GPU)»**

Al seleccionarlo aparece su propio grupo de parámetros (resolución, σ,
dispositivo) y se ocultan los de los otros métodos. El filtro de zonas huecas
se desmarca automáticamente, porque eliminaría triángulos y abriría la malla
cerrada — y si se reactiva, el registro lo advierte.

El método **solo aparece si `torch` y `scikit-image` están instalados**,
siguiendo el mismo patrón que los formatos `.e57` y `.las`. Si faltan, el
registro lo indica al arrancar con el comando de instalación.

Los parámetros y los tiempos internos (`dpsr_solve_s`, `dpsr_mc_s`,
`dpsr_device`) quedan registrados en el reporte de sesión exportable, así que
las mediciones para la memoria salen del propio software.

### Instalación de las dependencias opcionales

```powershell
& $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu121
& $PY -m pip install scikit-image
```

Verificado que **no altera `numpy` 2.4.6** ni el resto del entorno validado.

---

## 5 bis. Escenas grandes: troceado por bloques

### El problema

La grilla de DPSR es **densa y uniforme**, no un octree adaptativo como el
Poisson de Open3D: la memoria crece con res³. Sobre una GPU de 12 GB:

| Resolución | VRAM de un solo tirón | Viable |
|---|---|---|
| 256³ | 0,75 GB | ✅ |
| 512³ | 6,0 GB | ✅ |
| 768³ | 20,4 GB | ❌ solo troceado |
| 1024³ | 48,3 GB | ❌ solo troceado |

Y para una escena hace falta resolución: en el escaneo industrial medido
(5,02 m de lado mayor), 128³ da celdas de 4,7 cm — una tubería de 10 cm
tendría 2 celdas de ancho y desaparecería. La regla es **3 celdas mínimo a
lo ancho del detalle más fino**.

### La solución implementada

Bloques solapados resueltos en GPU y acumulados en una grilla global en RAM
(1024³ en float32 son 4 GB, asumibles). Cuatro cuidados sin los cuales la
fusión no funciona:

1. **Halo.** Cada bloque resuelve una región mayor de la que aporta: la FFT
   impone contorno periódico y el error se concentra en los bordes.
2. **Escala común.** Cada bloque normaliza χ dividiéndolo por |∇χ| en la
   superficie, con lo que pasa a ser una distancia con signo. Sin esto,
   promediar bloques con escalas distintas desplaza el nivel cero.
3. **Convenio de signo.** Se detecta de qué lado queda el exterior
   muestreando χ a lo largo de las normales, y se unifica. Corrige de paso
   los bloques que quedaron con las normales invertidas.
4. **Máscara de datos.** Cada bloque solo aporta cerca de sus propios
   puntos. Ver el apartado siguiente: es lo más importante.

Los bloques sin puntos ni se calculan — en una escena, la mayor parte del
volumen es aire.

### El compromiso que hay que entender

La máscara de datos suprime la superficie inventada en el aire, pero es
**el mismo mecanismo** con el que Poisson rellena los huecos del muestreo.
No se pueden tener las dos cosas:

| | Máscara ON | Máscara OFF |
|---|---|---|
| Salida | **lámina** fiel a los datos | **macizo** extrapolado |
| Huecos del escaneo | quedan abiertos | se rellenan |
| Superficie inventada | ninguna | bastante |
| Caso correcto | tuberías, paredes, escaneo de un lado | objeto compacto escaneado entero |

Medido sobre el escaneo industrial, activar la máscara bajó el error de
**6,510 % a 0,558 %** y la superficie inventada de **1,02 m a 6,4 cm**.

Medido sobre el bunny (objeto cerrado, con huecos en la base), la misma
máscara le quita el 59 % del volumen: se vuelve hueco. Por eso el bunny
debe reconstruirse **de una sola pasada**, que le sobra.

En la interfaz esto es el desplegable **«Tipo de dato»**, y en `auto` la
máscara se activa solo al trocear.

### Resultados sobre el escaneo LiDAR industrial

Escena de 4,73 × 5,02 × 3,78 m, 1.213.990 puntos originales (64.841 tras el
paso A), d̄ = 0,0176 m, 22 agrupaciones separadas.

| Configuración | Triángulos | Cerrada | Fidelidad | Malla→nube | Tiempo |
|---|---|---|---|---|---|
| Poisson clásico (bloque B) | 234.535 | ❌ (124 piezas) | **0,174 %** | 1,4 cm | 36 s |
| DPSR 256³ una pasada | 522.388 | ✅ | 3,128 % | 44,9 cm | 3 s |
| DPSR 512³ una pasada | 2.126.944 | ✅ | 2,771 % | 40,9 cm | 16 s |
| DPSR 768³ bloques 256³ | 3.569.154 | ❌ | 0,562 % | 6,5 cm | 42 s |
| **DPSR 1024³ bloques 256³** | 6.397.936 | ✅ | **0,535 %** | **6,4 cm** | 79 s |

El troceado a 1024³ mejora la pasada única **5×** en fidelidad. El Poisson
clásico sigue siendo 3× mejor en la media, pero devuelve 124 fragmentos
abiertos, no un sólido.

### Parámetros recomendados para un escaneo LiDAR industrial

| Parámetro | Valor | Motivo |
|---|---|---|
| Resolución | **1024³** (768³ si hay prisa) | celda de 0,6 cm: 3+ celdas en una tubería de 5 cm |
| Tamaño de bloque | **Automático** (elige 256³) | mayor que quepa en la VRAM libre |
| Solape | **25 %** | suficiente para que no se vean costuras |
| Tipo de dato | **Escena abierta (lámina)** | 12× mejor que sin máscara |
| Conservar solo la pieza mayor | **desactivado** | hay 260 objetos separados; si no, se queda con uno |
| Suavizado σ | **1,5** | subir solo si aparecen burbujas |

### Límite conocido

Trocear un objeto compacto y cerrado no funciona bien en ninguno de los dos
modos: con máscara sale hueco (−59 % de volumen), sin máscara sale inflado
(+369 %). No es un problema en la práctica —un objeto suelto cabe de una
pasada, donde el resultado es correcto (+0,1 % de volumen)— pero conviene
saberlo. El modo `auto` evita el caso al no trocear resoluciones pequeñas.

---

## 6. Qué falta

1. **El bucle de optimización por forma.** Es donde está el valor real de que
   el solucionador sea derivable, y donde DPSR debería superar al Poisson
   cribado. Requiere resolver la propagación de gradientes a través del
   marching cubes.
2. **Comparación sobre datos difíciles** (S3DIS, LiDAR, escaneos de una sola
   cara). Es la prueba que puede invertir el veredicto, y da el capítulo de
   resultados.
3. **Normales aprendidas en el paso A** (SHS-Net u OCMG-Net). Probablemente
   la mejora de mayor impacto real: tanto Poisson como DPSR dependen por
   completo de que las normales estén bien orientadas.

---

## 7. Referencias

- Peng, Jiang, Liao, Niemeyer, Pollefeys, Geiger. *Shape As Points: A
  Differentiable Poisson Solver*. NeurIPS 2021.
  <https://arxiv.org/abs/2106.03452>
- Huang, Gojcic, Atzmon et al. *Neural Kernel Surface Reconstruction*.
  CVPR 2023. <https://github.com/nv-tlabs/NKSR>
- Boulch, Marlet. *POCO: Point Convolution for Surface Reconstruction*.
  CVPR 2022. <https://github.com/valeoai/POCO>
- Sulzer et al. *A Survey and Benchmark of Automatic Surface Reconstruction
  from Point Clouds*. <https://arxiv.org/abs/2301.13656>
- Li et al. *SHS-Net: Learning Signed Hyper Surfaces for Oriented Normal
  Estimation*. CVPR 2023. <https://github.com/LeoQLi/SHS-Net>

La implementación de `core/dpsr.py` está escrita desde las ecuaciones del
artículo de Shape As Points, no copiada de su código.
