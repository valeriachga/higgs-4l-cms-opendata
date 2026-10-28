# Análisis H → ZZ → 4ℓ con CMS Open Data

Análisis educativo del canal Higgs → ZZ → 4 leptones (electrones y muones) con
datos abiertos de CMS de 2012 (√s = 8 TeV, 11.58 fb⁻¹). La selección de eventos
sigue la del artículo de CMS [arXiv:1207.7235](https://arxiv.org/abs/1207.7235).

## Contenido

```
├── scripts/
│   ├── 02_cutflow.py          # aplica los filtros uno por uno: tablas cutflow y un histograma por filtro
│   └── 03_grafica_final.py    # aplica todos los filtros y dibuja la gráfica final (datos vs. simulación)
├── resultados/                # tablas cutflow (CSV) y gráficas (PNG)
├── data/                      # aquí se colocan los CSV de entrada (ver data/README.md)
├── requirements.txt
└── README.md
```

## Datos
Los CSV pesan demasiado para GitHub. Descárgalos desde el enlace indicado en
[`data/README.md`](data/README.md) y colócalos en la carpeta `data/`.

## Cómo reproducir los resultados
1. Instala Python 3.10 o superior y las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloca los CSV en `data/`.
3. Desde la carpeta principal del repositorio ejecuta:
   ```bash
   python scripts/02_cutflow.py
   python scripts/03_grafica_final.py
   ```
Los resultados se guardan en `resultados/`.

## Selección de eventos (cortes)

| Corte | Valor |
|---|---|
| pT mínimo | electrones > 7 GeV, muones > 5 GeV |
| \|η\| máximo | electrones < 2.5, muones < 2.4 |
| \|dxy\|, \|dz\| | < 0.5 cm, < 1.0 cm |
| SIP3D | < 4 |
| Aislamiento relativo | < 0.4 |
| Carga total | 0 |
| pT de los 2 leptones más duros | > 20 GeV y > 10 GeV |
| Emparejamiento | dos pares del mismo sabor y carga opuesta (Z1 es el más cercano a la masa de la Z) |
| m(Z1) | 40 a 120 GeV |
| m(Z2) | 12 a 120 GeV |
| m(ℓℓ) mínima (pares de carga opuesta) | > 4 GeV |
| m(4ℓ) | > 70 GeV (se grafica de 70 a 180 GeV) |

## Normalización de las simulaciones
Cada evento simulado se pondera con `peso = L · σ · k / N_generados`, donde
L = 11 580 pb⁻¹, σ es la sección eficaz, k el factor de corrección a orden
superior y N_generados los eventos producidos en la simulación.

## Resultados
Aquí puedes insertar la gráfica final:

```markdown
![Gráfica final](resultados/grafica_final.png)
```

## Licencia y créditos
- Código: licencia MIT.
- Datos: CMS Open Data, CC0.
- Datos y simulaciones cortesía de la colaboración CMS, a través del portal
  CERN Open Data (opendata.cern.ch).
