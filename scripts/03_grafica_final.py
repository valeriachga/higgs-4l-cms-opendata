"""
03_grafica_final.py
===================

QUÉ HACE ESTE SCRIPT
--------------------
Aplica TODOS los cortes de selección del canal H -> ZZ -> 4 leptones de una sola
vez (a diferencia de 02_cutflow.py, que los va aplicando paso a paso) y dibuja
la gráfica final de la masa invariante de los 4 leptones (m4l), al estilo de
CMS Open Data:

  * Datos reales (CMS 2012, 8 TeV, 11.58 fb^-1): puntos negros con barras de
    incertidumbre estadística de Poisson, sqrt(N).
  * Fondo ZZ (suma de los canales 4mu, 4e y 2e2mu): histograma azul relleno.
  * Señal del Higgs (m_H = 125 GeV): línea roja, dibujada SOBRE el fondo
    (es decir, muestra ZZ + Higgs).

Las simulaciones se normalizan a la luminosidad de los datos con el peso
    peso = LUMI * xsec * k / nevt.

QUÉ NECESITA (ver data/README.md)
---------------------------------
Los mismos CSV de 4 leptones que 02_cutflow.py: 4 datasets de datos reales y
4 simulaciones, en la carpeta data/.

QUÉ PRODUCE
-----------
  * resultados/grafica_final.png
  * En pantalla: número de eventos de datos en la gráfica y los eventos
    esperados de ZZ y de Higgs.

CÓMO EJECUTARLO (desde la carpeta principal del repositorio)
------------------------------------------------------------
    python scripts/03_grafica_final.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# Rutas RELATIVAS a la carpeta principal del repositorio (funcionan en
# cualquier computadora sin editar nada).
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(RAIZ, "data")
SALIDA = os.path.join(RAIZ, "resultados")

MZ = 91.1876          # masa del bosón Z en GeV (PDG)
CHUNKSIZE = 200_000   # filas por bloque; baja a 100_000 si falta memoria

DATOS_REALES = [
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleMuParked_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleMuParked_Data_Aplanada.csv"),
]

RUTAS = {   # CSV de 4 leptones de las simulaciones
    "SMHiggsToZZTo4L": os.path.join(CARPETA_DATOS, "SMHiggs_4lep.csv"),
    "ZZTo4mu":         os.path.join(CARPETA_DATOS, "ZZTo4mu_4lep.csv"),
    "ZZTo4e":          os.path.join(CARPETA_DATOS, "ZZTo4e_4lep.csv"),
    "ZZTo2e2mu":       os.path.join(CARPETA_DATOS, "ZZTo2e2mu_4lep.csv"),
}
MOSTRAR = False   # True = además de guardar el PNG, lo muestra en pantalla

# Normalización de las simulaciones: peso = LUMI * xsec * k / nevt
#   LUMI : luminosidad en pb^-1 (11.58 fb^-1 = 11580 pb^-1)
#   xsec : sección eficaz del proceso (pb)
#   k    : factor de corrección a orden superior (K-factor)
#   nevt : eventos GENERADOS en la simulación (no las filas del CSV)
LUMI = 11580.0
MUESTRAS = {
    "SMHiggsToZZTo4L": {"xsec": 0.0065, "nevt": 299973.0,  "k": 1.0,   "tipo": "senal"},
    "ZZTo4mu":         {"xsec": 0.077,  "nevt": 1499064.0, "k": 1.386, "tipo": "fondo"},
    "ZZTo4e":          {"xsec": 0.077,  "nevt": 1499093.0, "k": 1.386, "tipo": "fondo"},
    "ZZTo2e2mu":       {"xsec": 0.18,   "nevt": 1497445.0, "k": 1.386, "tipo": "fondo"},
}

# Cortes de la selección "estricta" del artículo de CMS (arXiv:1207.7235).
CUTS = {
    "pt_e_min": 7,   "pt_mu_min": 5,      # pT mínimo (GeV): electrones / muones
    "eta_e_max": 2.5, "eta_mu_max": 2.4,  # |eta| máximo: electrones / muones
    "dxy_max": 0.5,  "dz_max": 1.0,       # parámetros de impacto máximos (cm)
    "sip_max": 4.0,                       # SIP3D máximo
    "iso_max": 0.4,                       # aislamiento relativo máximo
    "pt1_min": 20,   "pt2_min": 10,       # pT mínimo de los 2 leptones más duros (GeV)
    "mz1_min": 40,   "mz1_max": 120,      # ventana de masa de Z1 (GeV)
    "mz2_min": 12,   "mz2_max": 120,      # ventana de masa de Z2 (GeV)
    "mll_min": 4,                         # masa mínima de cualquier par de carga opuesta (GeV)
    "m4l_min": 70,                        # masa mínima de los 4 leptones (GeV)
}

# Intervalos del histograma: 36 bins de 70 a 180 GeV (37 bordes), como en el artículo.
BINS = np.linspace(70, 180, 37)

# Columnas que se leen de cada CSV (se ignoran las demás para ahorrar memoria).
COLS_LEP = ["PID", "Q", "pt", "eta", "phi", "mass", "dxy", "dxyErr", "dz", "dzErr",
            "pfRelIso03_all", "pfRelIso04_all"]
NECESARIAS = {f"{c}{i}" for c in COLS_LEP for i in range(1, 5)} | {"run", "luminosityBlock", "event"}
# Columnas que no pueden faltar para considerar "válido" un evento de 4 leptones.
COLS_VALIDAS = [f"{v}{i}" for v in ["PID", "Q", "pt", "eta", "phi", "mass"] for i in range(1, 5)]


# =============================================================================
# 2. FUNCIONES DE FÍSICA
# =============================================================================

def p4(df, i):
    """
    Cuadrivector (E, px, py, pz) del leptón número i (1 a 4).
        px = pT cos(phi),  py = pT sin(phi),  pz = pT sinh(eta)
        E  = sqrt(px^2 + py^2 + pz^2 + m^2)
    Devuelve un arreglo de forma (n_eventos, 4).
    """
    pt = df[f"pt{i}"].to_numpy()
    eta = df[f"eta{i}"].to_numpy()
    phi = df[f"phi{i}"].to_numpy()
    m = df[f"mass{i}"].to_numpy()
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return np.stack([E, px, py, pz], axis=1)


def inv_mass(*vecs):
    """
    Masa invariante de un sistema de partículas: se suman sus cuadrivectores y
        m^2 = E^2 - px^2 - py^2 - pz^2.
    Se recorta m^2 a >= 0 para evitar raíces de números negativos por errores
    numéricos de redondeo.
    """
    s = sum(vecs)
    m2 = s[:, 0]**2 - s[:, 1]**2 - s[:, 2]**2 - s[:, 3]**2
    return np.sqrt(np.clip(m2, 0, None))


def sip3d(dxy, dxyErr, dz, dzErr):
    """
    Aproximación de SIP3D (significancia del parámetro de impacto en 3D).

    IP3D = sqrt(dxy^2 + dz^2) mide qué tan lejos pasa la traza del vértice
    primario. Su incertidumbre se obtiene por propagación de errores, SIN
    considerar correlaciones entre dxy y dz (por eso es una aproximación):
        sigma_IP = sqrt( (dxy^2 * sigma_dxy^2 + dz^2 * sigma_dz^2) / IP^2 )
        SIP3D    = IP / sigma_IP
    Un SIP3D grande indica que el leptón probablemente NO viene del vértice
    primario.
    """
    dxy = np.asarray(dxy, float)
    dxyErr = np.asarray(dxyErr, float)
    dz = np.asarray(dz, float)
    dzErr = np.asarray(dzErr, float)
    ip = np.sqrt(dxy**2 + dz**2)
    # np.clip evita dividir entre cero cuando el impacto es exactamente 0.
    sig = np.sqrt((dxy**2 * dxyErr**2 + dz**2 * dzErr**2) / np.clip(ip**2, 1e-12, None))
    return ip / np.clip(sig, 1e-12, None)


# =============================================================================
# 3. SELECCIÓN DE EVENTOS (todos los cortes de una vez)
# =============================================================================

def seleccion_final(df, C, con_claves=False):
    """
    Aplica todos los cortes a un bloque de filas del CSV.

    Parámetros
    ----------
    df : DataFrame con los eventos del bloque (una fila = un evento de 4 leptones).
    C : diccionario con los valores de los cortes (ver CUTS).
    con_claves : si es True (datos reales) también se devuelve un identificador
        único por evento (hash de run, luminosityBlock y event) para poder
        eliminar duplicados entre archivos.

    Devuelve
    --------
    (m4l, claves) de los eventos que superan TODOS los cortes.
    """
    claves = None
    if con_claves:
        claves = pd.util.hash_pandas_object(
            df[["run", "luminosityBlock", "event"]], index=False).to_numpy()

    # --- 4 leptones válidos ---------------------------------------------------
    # En los CSV, -999 significa "este leptón no existe en el evento".
    # Se convierten a NaN y se descartan los eventos con algún dato faltante.
    df = df.replace(-999, np.nan)
    valido = df[COLS_VALIDAS].notna().all(axis=1).to_numpy()
    df = df[valido]
    if claves is not None:
        claves = claves[valido]
    n = len(df)
    if n == 0:
        return np.array([], np.float32), (np.array([], np.uint64) if con_claves else None)

    v = [p4(df, i) for i in range(1, 5)]
    m4l = inv_mass(*v)
    pid = [df[f"PID{i}"].abs().to_numpy() for i in range(1, 5)]   # 11 = electrón, 13 = muón
    q = [df[f"Q{i}"].to_numpy() for i in range(1, 5)]             # carga eléctrica

    # "ok" marca los eventos que cumplen TODOS los cortes. Cada corte nuevo
    # puede solo quitar eventos (operador &=), nunca recuperarlos.
    ok = np.ones(n, bool)

    # --- Cortes que se exigen a CADA uno de los 4 leptones --------------------
    for i in range(1, 5):
        es_e = pid[i - 1] == 11
        es_mu = pid[i - 1] == 13
        pt = df[f"pt{i}"].to_numpy()
        eta = np.abs(df[f"eta{i}"].to_numpy())
        dxy = df[f"dxy{i}"].to_numpy(float)
        dz = df[f"dz{i}"].to_numpy(float)

        # pT mínimo (umbral distinto para electrones y muones).
        ok &= (es_e & (pt > C["pt_e_min"])) | (es_mu & (pt > C["pt_mu_min"]))
        # |eta| máximo: región angular donde el detector mide bien.
        ok &= (es_e & (eta < C["eta_e_max"])) | (es_mu & (eta < C["eta_mu_max"]))
        # Parámetros de impacto transversal (dxy) y longitudinal (dz).
        ok &= (np.abs(dxy) < C["dxy_max"]) & (np.abs(dz) < C["dz_max"])

        # SIP3D. Si no se puede calcular (NaN) no se rechaza el leptón.
        sip = sip3d(dxy, df[f"dxyErr{i}"], dz, df[f"dzErr{i}"])
        ok &= (sip < C["sip_max"]) | np.isnan(sip)

        # Aislamiento relativo: los muones usan cono 0.4; los electrones no
        # tienen esa variable y usan la de cono 0.3 (fillna la toma como respaldo).
        iso = df[f"pfRelIso04_all{i}"].fillna(df[f"pfRelIso03_all{i}"]).to_numpy()
        ok &= iso < C["iso_max"]

    # --- Carga total cero -----------------------------------------------------
    # Dos Z neutras dan 2 leptones positivos y 2 negativos: suma de cargas = 0.
    ok &= (q[0] + q[1] + q[2] + q[3]) == 0

    # --- pT de los dos leptones más duros -------------------------------------
    # Se ordenan los 4 pT de mayor a menor y se exige que los dos primeros
    # superen un umbral (el evento tuvo que activar el trigger).
    pts = np.sort(np.stack([df[f"pt{i}"].to_numpy() for i in range(1, 5)], axis=1), axis=1)[:, ::-1]
    ok &= (pts[:, 0] > C["pt1_min"]) & (pts[:, 1] > C["pt2_min"])

    # --- mll mínima sobre TODOS los pares de carga opuesta (cualquier sabor) --
    # Sirve para eliminar resonancias de baja masa (J/psi, etc.).
    mll_min = np.full(n, np.inf)
    for a in range(4):
        for b in range(a + 1, 4):
            os_ = (q[a] + q[b]) == 0                       # par de carga opuesta (OS)
            mab = inv_mass(v[a], v[b])
            mll_min = np.where(os_, np.minimum(mll_min, mab), mll_min)

    # --- Emparejamiento en dos Z (mismo sabor y carga opuesta) ----------------
    # Con 4 leptones hay 3 formas de dividirlos en 2 pares.
    combos = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

    def par_valido(a, b):
        """Un par puede venir de una Z si es del mismo sabor y de carga opuesta."""
        return (pid[a] == pid[b]) & (q[a] + q[b] == 0)

    # De las combinaciones válidas se elige la que tiene Z1 más cerca de MZ.
    mejor_score = np.full(n, np.inf)
    mZ1 = np.full(n, np.nan)
    mZ2 = np.full(n, np.nan)
    for (a, b), (c, d) in combos:
        valido_c = par_valido(a, b) & par_valido(c, d)
        mab = inv_mass(v[a], v[b])
        mcd = inv_mass(v[c], v[d])
        ab_es_z1 = np.abs(mab - MZ) < np.abs(mcd - MZ)     # Z1 = el par más cercano a MZ
        z1 = np.where(ab_es_z1, mab, mcd)
        z2 = np.where(ab_es_z1, mcd, mab)
        score = np.abs(z1 - MZ)
        mejor = valido_c & (score < mejor_score)
        mejor_score = np.where(mejor, score, mejor_score)
        mZ1 = np.where(mejor, z1, mZ1)
        mZ2 = np.where(mejor, z2, mZ2)

    ok &= np.isfinite(mZ1)                                  # existe algún emparejamiento válido
    ok &= (mZ1 > C["mz1_min"]) & (mZ1 < C["mz1_max"])       # Z1 en ventana de masa
    ok &= (mZ2 > C["mz2_min"]) & (mZ2 < C["mz2_max"])       # Z2 en ventana de masa
    ok &= mll_min > C["mll_min"]                            # sin resonancias de baja masa
    ok &= m4l > C["m4l_min"]                                # m4l mínima

    return m4l[ok].astype(np.float32), (None if claves is None else claves[ok])


def procesar(rutas, nombre, con_claves):
    """
    Lee uno o varios CSV por bloques y devuelve el arreglo de m4l de los
    eventos que pasan todos los cortes.
    Con con_claves=True (datos reales) se eliminan los eventos duplicados entre
    archivos, identificándolos por (run, luminosityBlock, event).
    """
    m4l_lista, claves_lista = [], []
    leidos = 0
    for ruta in rutas:
        # usecols lee solo las columnas necesarias; chunksize lee por bloques.
        for bloque in pd.read_csv(ruta, chunksize=CHUNKSIZE,
                                  usecols=lambda c: c in NECESARIAS):
            m, cl = seleccion_final(bloque, CUTS, con_claves)
            m4l_lista.append(m)
            if cl is not None:
                claves_lista.append(cl)
            leidos += len(bloque)
            print(f"  {nombre}: {leidos} filas leidas", end="\r")
    print()

    m4l = np.concatenate(m4l_lista) if m4l_lista else np.array([], np.float32)
    if con_claves and claves_lista:
        cl = np.concatenate(claves_lista)
        _, idx = np.unique(cl, return_index=True)   # se queda con una copia de cada evento
        m4l = m4l[idx]
    return m4l


def peso_por_evento(nombre):
    """Peso de cada evento simulado: LUMI * xsec * k / nevt (ver sección 1)."""
    m = MUESTRAS[nombre]
    return LUMI * m["xsec"] * m["k"] / m["nevt"]


# =============================================================================
# 4. GRÁFICA FINAL
# =============================================================================

def grafica_final(m4l_sim, m4l_datos, archivo, bins=BINS):
    """
    Dibuja m4l al estilo CMS Open Data: datos como puntos con incertidumbre
    estadística (raíz de N) sobre el fondo ZZ (azul) y la señal del Higgs (rojo).
    """
    # --- Simulaciones ponderadas por luminosidad y sección eficaz -------------
    h_zz = np.zeros(len(bins) - 1)     # fondo ZZ (suma de los 3 canales)
    h_h = np.zeros(len(bins) - 1)      # señal del Higgs
    for nombre, m4l in m4l_sim.items():
        h, _ = np.histogram(m4l, bins=bins, weights=np.full(len(m4l), peso_por_evento(nombre)))
        if MUESTRAS[nombre]["tipo"] == "senal":
            h_h += h
        else:
            h_zz += h
    h_tot = h_zz + h_h                 # ZZ + Higgs (la línea roja)

    # --- Datos reales: conteo por bin y error de Poisson ----------------------
    h_dat, _ = np.histogram(m4l_datos, bins=bins)
    err = np.sqrt(h_dat)                                   # incertidumbre = raíz de N
    centros = 0.5 * (bins[:-1] + bins[1:])
    con_datos = h_dat > 0          # solo se dibujan puntos en bins con eventos

    print(f"\nGrafica final ({bins[0]:.0f}-{bins[-1]:.0f} GeV): datos = {int(h_dat.sum())}, "
          f"ZZ esperado = {h_zz.sum():.2f}, Higgs esperado = {h_h.sum():.2f}")

    # --- Dibujo ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.stairs(h_tot, bins, color="red", linewidth=2, label="m$_H$ = 125 GeV")
    ax.stairs(h_zz, bins, fill=True, facecolor="#9ecae1", edgecolor="black",
              linewidth=1.5, label="ZZ")
    ax.errorbar(centros[con_datos], h_dat[con_datos], yerr=err[con_datos],
                fmt="o", color="black", markersize=5, elinewidth=1.3, capsize=0,
                label="Data", zorder=5)

    # El eje Y llega al mayor valor entre datos (con su error) y simulación, + 25 %.
    tope = max((h_dat + err).max(), h_tot.max())
    ax.set_ylim(0, (tope if tope > 0 else 1) * 1.25)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Mass 4 leptons / GeV", fontsize=12)
    ax.set_ylabel("N_Events", fontsize=12)
    ax.text(0.02, 1.02, "CMS Open Data", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(0.98, 1.02, r"$\sqrt{s}$ = 8 TeV, L$_{int}$ = 11.6 fb$^{-1}$",
            transform=ax.transAxes, fontsize=11, ha="right")

    # Leyenda en el orden del artículo: Data, ZZ, Higgs.
    handles, labels = ax.get_legend_handles_labels()
    d = dict(zip(labels, handles))
    orden = ["Data", "ZZ", "m$_H$ = 125 GeV"]
    ax.legend([d[k] for k in orden], orden, frameon=False,
              loc="upper right", bbox_to_anchor=(0.97, 0.95))

    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    if MOSTRAR:
        plt.show()
    plt.close(fig)


# =============================================================================
# 5. PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)

    # 1) Aplicar los cortes a cada simulación por separado.
    m4l_sim = {}
    for nombre, ruta in RUTAS.items():
        print(f"Procesando {nombre}...")
        m4l_sim[nombre] = procesar([ruta], nombre, con_claves=False)

    # 2) Aplicar los cortes a los 4 datasets de datos reales, unidos y sin duplicados.
    print("Procesando datos reales (4 datasets unidos)...")
    m4l_datos = procesar(DATOS_REALES, "Datos reales", con_claves=True)

    # 3) Dibujar y guardar la gráfica final.
    grafica_final(m4l_sim, m4l_datos, os.path.join(SALIDA, "grafica_final.png"))
    print("Listo. Gráfica en:", os.path.join(SALIDA, "grafica_final.png"))
