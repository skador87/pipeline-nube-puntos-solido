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
