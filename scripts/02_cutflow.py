"""
02_cutflow.py
=============

QUÉ HACE ESTE SCRIPT
--------------------
Aplica, uno por uno y de forma ACUMULATIVA, los filtros de selección del canal
Higgs -> ZZ -> 4 leptones (H -> ZZ* -> 4l) a los datos abiertos de CMS de 2012
(sqrt(s) = 8 TeV, luminosidad de 11.58 fb^-1) y a las simulaciones Monte Carlo.

Después de cada filtro registra:
  * cuántos eventos sobreviven (tabla "cutflow"), y
  * la masa invariante de los 4 leptones, m4l, de los eventos que sobreviven
    (para dibujar un histograma por cada filtro).

QUÉ NECESITA (ver data/README.md)
---------------------------------
Archivos CSV de 4 leptones, con una fila por evento y las columnas
PID, Q, pt, eta, phi, mass, dxy, dxyErr, dz, dzErr, pfRelIso03_all, pfRelIso04_all
repetidas para los leptones 1 a 4 (PID1, PID2, ..., pfRelIso04_all4),
más las columnas run, luminosityBlock y event.
  * Datos reales: Run2012B y Run2012C, DoubleElectron y DoubleMuParked.
  * Simulaciones: Higgs -> ZZ -> 4l (señal) y ZZ -> 4mu, 4e, 2e2mu (fondo).

QUÉ PRODUCE (carpeta resultados/)
---------------------------------
  * cutflow_crudo.csv       -> eventos que pasan cada filtro, sin ponderar.
  * cutflow_ponderado.csv   -> eventos ESPERADOS a 11.58 fb^-1 para las
                               simulaciones (datos reales sin ponderar).
  * datos_etapaNN.png       -> histograma de m4l de los datos tras el filtro NN.
  * mc_etapaNN.png          -> histograma de m4l de las simulaciones tras el filtro NN.
  * datos_panel.png y mc_panel.png -> todas las etapas juntas en una figura.

CÓMO EJECUTARLO (desde la carpeta principal del repositorio)
------------------------------------------------------------
    python scripts/02_cutflow.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# Las rutas son RELATIVAS a la carpeta principal del repositorio, de modo que
# el script funciona en cualquier computadora sin editar nada.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(RAIZ, "data")        # aquí van los CSV de entrada
SALIDA = os.path.join(RAIZ, "resultados")         # aquí se guardan tablas y gráficas

# Masa del bosón Z en GeV (valor del PDG). Se usa para decidir cuál par de
# leptones es "Z1" (el más cercano a la masa de la Z).
MZ = 91.1876

# Los CSV son muy grandes, así que se leen por bloques de filas para no agotar
# la memoria RAM. Si aun así falta memoria, baja este valor a 100_000.
CHUNKSIZE = 200_000

# Archivos de datos reales (se procesan juntos, como un solo conjunto).
DATOS_REALES = [
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleMuParked_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleMuParked_Data_Aplanada.csv"),
]

# Archivos de las simulaciones Monte Carlo (cada una por separado).
RUTAS = {
    "SMHiggsToZZTo4L": os.path.join(CARPETA_DATOS, "SMHiggs_4lep.csv"),
    "ZZTo4mu":         os.path.join(CARPETA_DATOS, "ZZTo4mu_4lep.csv"),
    "ZZTo4e":          os.path.join(CARPETA_DATOS, "ZZTo4e_4lep.csv"),
    "ZZTo2e2mu":       os.path.join(CARPETA_DATOS, "ZZTo2e2mu_4lep.csv"),
}

# True = además de guardar los PNG, los muestra en pantalla.
MOSTRAR = False

# ---- Rango de m4l que se cuenta y se grafica --------------------------------
# VENTANA = (70.0, 180.0) -> solo se cuentan/grafican eventos con m4l en ese rango.
# VENTANA = None          -> sin corte en m4l (se usan todos los eventos).
VENTANA = (70.0, 180.0)
ANCHO_BIN = 5.0    # GeV por bin (solo se usa si VENTANA = None)
XMAX = None        # None = el eje X llega al m4l máximo; o un número (p. ej. 500)

# ---- Normalización de las simulaciones --------------------------------------
# Para comparar simulación con datos hay que darle a cada evento simulado un
# "peso":   peso = LUMI * xsec * k / nevt
#   LUMI : luminosidad de los datos, en pb^-1 (11.58 fb^-1 = 11580 pb^-1)
#   xsec : sección eficaz del proceso, en pb
#   k    : factor de corrección a orden superior (K-factor)
#   nevt : eventos GENERADOS en la simulación (no las filas del CSV)
LUMI = 11580.0
MUESTRAS = {
    "SMHiggsToZZTo4L": {"xsec": 0.0065, "nevt": 299973.0,  "k": 1.0,   "tipo": "senal"},
    "ZZTo4mu":         {"xsec": 0.077,  "nevt": 1499064.0, "k": 1.386, "tipo": "fondo"},
    "ZZTo4e":          {"xsec": 0.077,  "nevt": 1499093.0, "k": 1.386, "tipo": "fondo"},
    "ZZTo2e2mu":       {"xsec": 0.18,   "nevt": 1497445.0, "k": 1.386, "tipo": "fondo"},
}

# Nombres de las columnas en la tabla de salida.
NOMBRES_COLUMNA = {
    "SMHiggsToZZTo4L": "Higgs",
    "ZZTo4mu": "ZZ->4mu",
    "ZZTo4e": "ZZ->4e",
    "ZZTo2e2mu": "ZZ->2e2mu",
}

# ---- Valores de los cortes (selección "estricta" del artículo de CMS) -------
CUTS = {
    "pt_e_min": 7,   "pt_mu_min": 5,      # pT mínimo (GeV): electrones / muones
    "eta_e_max": 2.5, "eta_mu_max": 2.4,  # |eta| máximo: electrones / muones
    "dxy_max": 0.5,  "dz_max": 1.0,       # parámetros de impacto máximos (cm)
    "sip_max": 4.0,                       # SIP3D máximo (significancia del impacto 3D)
    "iso_max": 0.4,                       # aislamiento relativo máximo
    "pt1_min": 20,   "pt2_min": 10,       # pT mínimo de los dos leptones más duros (GeV)
    "mz1_min": 40,   "mz1_max": 120,      # ventana de masa de Z1 (GeV)
    "mz2_min": 12,   "mz2_max": 120,      # ventana de masa de Z2 (GeV)
    "mll_min": 4,                         # masa mínima de cualquier par de carga opuesta (GeV)
    "m4l_min": 70,                        # masa mínima de los 4 leptones (GeV)
}

# Nombres de las etapas (filtros), EN ORDEN. Cada una incluye a las anteriores.
ETAPAS = [
    "Eventos de entrada",
    "4 leptones validos (sin -999/NaN)",
    "pT de los leptones",
    "|eta| de los leptones",
    "|dxy| y |dz|",
    "SIP3D",
    "Aislamiento relativo",
    "Carga total = 0",
    "pT1 y pT2 minimos",
    "Existe emparejamiento Z+Z (SF, OS)",
    "mZ1 en ventana",
    "mZ2 en ventana",
    "mll minimo (todos los pares OS)",
    "m4l minimo",
]

# Columnas que se leen de cada CSV (se ignoran las demás para ahorrar memoria).
COLS_LEP = ["PID", "Q", "pt", "eta", "phi", "mass", "dxy", "dxyErr", "dz", "dzErr",
            "pfRelIso03_all", "pfRelIso04_all"]
NECESARIAS = {f"{c}{i}" for c in COLS_LEP for i in range(1, 5)} | {"run", "luminosityBlock", "event"}
# Columnas que no pueden faltar para considerar "válido" un evento de 4 leptones.
COLS_VALIDAS = [f"{v}{i}" for v in ["PID", "Q", "pt", "eta", "phi", "mass"] for i in range(1, 5)]

# Se define en el bloque principal (depende de VENTANA y XMAX).
BINS = None


# =============================================================================
# 2. FUNCIONES DE FÍSICA
# =============================================================================

def p4(df, i):
    """
    Cuadrivector (E, px, py, pz) del leptón número i (1 a 4).

    Se pasa de las variables medidas (pT, eta, phi, masa) a componentes:
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
    primario (p. ej. viene de la desintegración de un quark b).
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
# 3. SELECCIÓN DE EVENTOS (los filtros)
# =============================================================================

def evaluar(df, C, con_claves=False):
    """
    Aplica los cortes de forma acumulativa a UN bloque de filas del CSV.

    Parámetros
    ----------
    df : DataFrame con los eventos del bloque (una fila = un evento de 4 leptones).
    C : diccionario con los valores de los cortes (ver CUTS).
    con_claves : si es True (datos reales) también se devuelve un identificador
        único por evento (hash de run, luminosityBlock y event), que sirve para
        eliminar eventos duplicados entre los archivos de datos.

    Devuelve
    --------
    Diccionario {nombre de etapa: (m4l de los sobrevivientes, claves, n)}.
      * m4l: arreglo con la masa de 4 leptones de los eventos que pasan hasta esa etapa.
      * claves: identificadores de esos eventos (o None).
      * n: cuántos eventos pasan. Si VENTANA no es None, solo se cuentan los
        que además tienen m4l dentro de la ventana.
    """
    res = {}
    claves = None
    if con_claves:
        claves = pd.util.hash_pandas_object(
            df[["run", "luminosityBlock", "event"]], index=False).to_numpy()

    # Etapa 0: todos los eventos del bloque, sin ningún filtro.
    res[ETAPAS[0]] = (None, claves, len(df))

    # --- Etapa 1: 4 leptones válidos -----------------------------------------
    # En los CSV, -999 significa "este leptón no existe en el evento".
    # Se convierten a NaN y se descartan los eventos con algún dato faltante.
    df = df.replace(-999, np.nan)
    valido = df[COLS_VALIDAS].notna().all(axis=1).to_numpy()
    df = df[valido]
    if claves is not None:
        claves = claves[valido]
    n = len(df)

    # Cuadrivectores y m4l de TODOS los eventos válidos. Se calculan una sola
    # vez y así se puede contar y graficar desde la primera etapa.
    v = [p4(df, i) for i in range(1, 5)]
    m4l = inv_mass(*v)
    pid = [df[f"PID{i}"].abs().to_numpy() for i in range(1, 5)]   # 11 = electrón, 13 = muón
    q = [df[f"Q{i}"].to_numpy() for i in range(1, 5)]             # carga eléctrica

    # "vivo" marca los eventos que han superado TODOS los filtros hasta ahora.
    vivo = np.ones(n, bool)
    if VENTANA is None:
        en_ventana = np.ones(n, bool)
    else:
        en_ventana = (m4l >= VENTANA[0]) & (m4l <= VENTANA[1])

    def cerrar(nombre, mascara):
        """Aplica el filtro 'mascara' sobre los eventos vivos y guarda el resultado de la etapa."""
        nonlocal vivo
        vivo = vivo & mascara                 # los cortes son acumulativos
        sel = vivo & en_ventana
        res[nombre] = (m4l[sel].astype(np.float32),
                       None if claves is None else claves[sel],
                       int(sel.sum()))

    cerrar(ETAPAS[1], np.ones(n, bool))       # solo cuenta los eventos válidos

    # --- Etapas 2 a 6: cortes que se exigen a CADA uno de los 4 leptones -----
    pt_ok = np.ones(n, bool)
    eta_ok = np.ones(n, bool)
    ip_ok = np.ones(n, bool)
    sip_ok = np.ones(n, bool)
    iso_ok = np.ones(n, bool)
    for i in range(1, 5):
        es_e = pid[i - 1] == 11
        es_mu = pid[i - 1] == 13
        pt = df[f"pt{i}"].to_numpy()
        eta = np.abs(df[f"eta{i}"].to_numpy())
        dxy = df[f"dxy{i}"].to_numpy(float)
        dz = df[f"dz{i}"].to_numpy(float)

        # pT mínimo (umbral distinto para electrones y muones).
        pt_ok &= ((es_e & (pt > C["pt_e_min"])) | (es_mu & (pt > C["pt_mu_min"])))
        # |eta| máximo: región angular donde el detector mide bien.
        eta_ok &= ((es_e & (eta < C["eta_e_max"])) | (es_mu & (eta < C["eta_mu_max"])))
        # Parámetros de impacto transversal (dxy) y longitudinal (dz).
        ip_ok &= (np.abs(dxy) < C["dxy_max"]) & (np.abs(dz) < C["dz_max"])

        # SIP3D. Si no se puede calcular (NaN) no se rechaza el leptón.
        sip = sip3d(dxy, df[f"dxyErr{i}"], dz, df[f"dzErr{i}"])
        sip_ok &= (sip < C["sip_max"]) | np.isnan(sip)

        # Aislamiento relativo: los muones usan cono 0.4; los electrones no
        # tienen esa variable y usan la de cono 0.3 (fillna la toma como respaldo).
        iso = df[f"pfRelIso04_all{i}"].fillna(df[f"pfRelIso03_all{i}"]).to_numpy()
        iso_ok &= iso < C["iso_max"]

    cerrar(ETAPAS[2], pt_ok)
    cerrar(ETAPAS[3], eta_ok)
    cerrar(ETAPAS[4], ip_ok)
    cerrar(ETAPAS[5], sip_ok)
    cerrar(ETAPAS[6], iso_ok)

    # --- Etapa 7: carga total cero -------------------------------------------
    # Dos Z neutras dan 2 leptones positivos y 2 negativos: suma de cargas = 0.
    cerrar(ETAPAS[7], (q[0] + q[1] + q[2] + q[3]) == 0)

    # --- Etapa 8: pT de los dos leptones más duros ---------------------------
    # Se ordenan los 4 pT de mayor a menor y se exige que los dos primeros
    # superen un umbral (el evento tuvo que activar el trigger).
    pts = np.sort(np.stack([df[f"pt{i}"].to_numpy() for i in range(1, 5)], axis=1), axis=1)[:, ::-1]
    cerrar(ETAPAS[8], (pts[:, 0] > C["pt1_min"]) & (pts[:, 1] > C["pt2_min"]))

    # --- mll mínima sobre TODOS los pares de carga opuesta (cualquier sabor) -
    # Sirve para eliminar resonancias de baja masa (J/psi, etc.). Se usa en la etapa 12.
    mll_min = np.full(n, np.inf)
    for a in range(4):
        for b in range(a + 1, 4):
            os_ = (q[a] + q[b]) == 0                       # par de carga opuesta (OS)
            mab = inv_mass(v[a], v[b])
            mll_min = np.where(os_, np.minimum(mll_min, mab), mll_min)

    # --- Emparejamiento en dos Z (mismo sabor y carga opuesta, SF + OS) ------
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

    # --- Etapas 9 a 13 -------------------------------------------------------
    cerrar(ETAPAS[9], np.isfinite(mZ1))                                  # existe algún emparejamiento
    cerrar(ETAPAS[10], (mZ1 > C["mz1_min"]) & (mZ1 < C["mz1_max"]))      # Z1 en ventana
    cerrar(ETAPAS[11], (mZ2 > C["mz2_min"]) & (mZ2 < C["mz2_max"]))      # Z2 en ventana
    cerrar(ETAPAS[12], mll_min > C["mll_min"])                           # sin resonancias de baja masa
    cerrar(ETAPAS[13], m4l > C["m4l_min"])                               # m4l mínima
    return res


def procesar(rutas, nombre, con_claves):
    """
    Lee uno o varios CSV por bloques y acumula los resultados de todas las etapas.

    Devuelve {etapa: (n, m4l)}: el número de eventos que pasan cada etapa y su m4l.
    Con con_claves=True (datos reales) se eliminan los eventos duplicados entre
    archivos en CADA etapa, identificándolos por (run, luminosityBlock, event).
    """
    acum = {e: {"n": 0, "m4l": [], "claves": []} for e in ETAPAS}
    leidos = 0
    for ruta in rutas:
        # usecols lee solo las columnas necesarias; chunksize lee por bloques.
        for bloque in pd.read_csv(ruta, chunksize=CHUNKSIZE,
                                  usecols=lambda c: c in NECESARIAS):
            res = evaluar(bloque, CUTS, con_claves)
            for e, (m, cl, n) in res.items():
                a = acum[e]
                a["n"] += n
                if m is not None:
                    a["m4l"].append(m)
                if cl is not None:
                    a["claves"].append(cl)
            leidos += len(bloque)
            print(f"  {nombre}: {leidos} filas leidas", end="\r")
    print()

    # Se unen los resultados de todos los bloques.
    salida = {}
    for e in ETAPAS:
        a = acum[e]
        m4l = np.concatenate(a["m4l"]) if a["m4l"] else np.array([], np.float32)
        if con_claves:
            cl = np.concatenate(a["claves"]) if a["claves"] else np.array([], np.uint64)
            _, idx = np.unique(cl, return_index=True)   # se queda con una copia de cada evento
            n = len(idx)
            if m4l.size:
                m4l = m4l[idx]
        else:
            n = a["n"]
        salida[e] = (n, m4l)
    return salida


def peso_por_evento(nombre):
    """Peso de cada evento simulado: LUMI * xsec * k / nevt (ver sección 1)."""
    m = MUESTRAS[nombre]
    return LUMI * m["xsec"] * m["k"] / m["nevt"]


# =============================================================================
# 4. TABLAS
# =============================================================================

def tablas_cutflow(res_sim, res_datos):
    """
    Construye dos tablas (filas = filtros, columnas = muestras):
      * cruda: número de eventos que pasan cada filtro, sin ponderar.
      * ponderada: eventos esperados a 11.58 fb^-1 (simulaciones multiplicadas
        por su peso). Los datos reales no se ponderan.
    """
    crudo, pond = {}, {}
    for nombre, res in res_sim.items():
        col = NOMBRES_COLUMNA[nombre]
        w = peso_por_evento(nombre)
        crudo[col] = {e: res[e][0] for e in ETAPAS}
        pond[col] = {e: res[e][0] * w for e in ETAPAS}
    crudo["Datos reales"] = {e: res_datos[e][0] for e in ETAPAS}
    pond["Datos reales"] = {e: res_datos[e][0] for e in ETAPAS}
    t_crudo = pd.DataFrame(crudo)
    t_pond = pd.DataFrame(pond).round(3)
    t_crudo.index.name = "Filtro"
    t_pond.index.name = "Filtro"
    return t_crudo, t_pond


# =============================================================================
# 5. BINNING (los intervalos del histograma)
# =============================================================================

def construir_bins(res_sim, res_datos):
    """
    Define UN solo binning para todas las gráficas, así el eje X es el mismo
    en todas. Con VENTANA se usan 36 bins entre sus extremos; sin VENTANA los
    bins miden ANCHO_BIN GeV y llegan hasta el m4l máximo (o hasta XMAX).
    """
    if VENTANA is not None:
        return np.linspace(VENTANA[0], VENTANA[1], 37)   # 37 bordes = 36 bins
    xmax = XMAX
    if xmax is None:
        # La etapa 1 contiene a todos los eventos válidos, así que ahí está el m4l máximo.
        maximos = [r[ETAPAS[1]][1].max()
                   for r in list(res_sim.values()) + [res_datos] if r[ETAPAS[1]][1].size]
        xmax = max(maximos) if maximos else 200.0
    tope = np.ceil(xmax / ANCHO_BIN) * ANCHO_BIN + ANCHO_BIN
    return np.arange(0, tope, ANCHO_BIN)


# =============================================================================
# 6. GRÁFICAS
# =============================================================================

def hists_mc(res_sim, etapa):
    """
    Histogramas de las simulaciones tras una etapa, ponderados por luminosidad
    y sección eficaz. Devuelve (ZZ, ZZ + Higgs), bin a bin.
    """
    h_zz = np.zeros(len(BINS) - 1)
    h_h = np.zeros(len(BINS) - 1)
    for nombre, res in res_sim.items():
        m4l = res[etapa][1]
        h, _ = np.histogram(m4l, bins=BINS, weights=np.full(len(m4l), peso_por_evento(nombre)))
        if MUESTRAS[nombre]["tipo"] == "senal":
            h_h += h
        else:
            h_zz += h        # los 3 canales ZZ se suman
    return h_zz, h_zz + h_h


def dibujar_datos(ax, res_datos, etapa, grande=True):
    """Histograma (barras) de los datos reales unidos. El eje Y se ajusta solo."""
    m4l = res_datos[etapa][1]
    n_visibles = ((m4l >= BINS[0]) & (m4l <= BINS[-1])).sum()
    borde = 0.8 if len(BINS) < 100 else 0.2   # con muchos bins, el borde es más fino
    ax.hist(m4l, bins=BINS, color="steelblue", edgecolor="black", linewidth=borde,
            label=f"Datos ({n_visibles} eventos)")
    ax.set_xlim(BINS[0], BINS[-1])
    if grande:
        ax.set_xlabel("Mass 4 leptons / GeV", fontsize=12)
        ax.set_ylabel("N_Events", fontsize=12)
        ax.legend()


def dibujar_mc(ax, res_sim, etapa, grande=True):
    """
    Simulaciones al estilo CMS Open Data: ZZ (3 canales juntos) en azul relleno
    y ZZ + Higgs como línea roja.
    """
    h_zz, h_tot = hists_mc(res_sim, etapa)
    ax.stairs(h_tot, BINS, color="red", linewidth=2, label="m$_H$ = 125 GeV")
    ax.stairs(h_zz, BINS, fill=True, facecolor="#9ecae1", edgecolor="black",
              linewidth=1.5 if len(BINS) < 100 else 0.5, label="ZZ")
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_ylim(0, (h_tot.max() if h_tot.max() > 0 else 1) * 1.25)   # escala propia de cada etapa
    if grande:
        ax.set_xlabel("Mass 4 leptons / GeV", fontsize=12)
        ax.set_ylabel("N_Events", fontsize=12)
        ax.text(0.02, 1.02, "CMS Open Data", transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.text(0.98, 1.02, r"$\sqrt{s}$ = 8 TeV, L$_{int}$ = 11.6 fb$^{-1}$",
                transform=ax.transAxes, fontsize=11, ha="right")
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.97, 0.95))


def figura_etapa(dibujar, res, etapa, k, titulo, archivo):
    """Guarda una figura individual con el histograma de una etapa."""
    fig, ax = plt.subplots(figsize=(8, 6))
    dibujar(ax, res, etapa, grande=True)
    fig.suptitle(f"{titulo} - {k}. {etapa}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(archivo, dpi=150)
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def panel(dibujar, res, titulo, archivo):
    """Guarda todas las etapas en una sola figura (cada una con su escala Y)."""
    etapas = ETAPAS[1:]
    ncols = 4
    nrows = int(np.ceil(len(etapas) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    for ax, (k, e) in zip(axs.ravel(), enumerate(etapas, 1)):
        dibujar(ax, res, e, grande=False)
        ax.set_title(f"{k}. {e}", fontsize=8)
    for ax in axs.ravel()[len(etapas):]:
        ax.axis("off")                      # oculta los ejes sobrantes
    fig.suptitle(f"{titulo}: m4l tras cada filtro")
    fig.tight_layout()
    fig.savefig(archivo, dpi=130)
    if MOSTRAR:
        plt.show()
    plt.close(fig)


# =============================================================================
# 7. PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)

    # 1) Procesar cada simulación por separado.
    res_sim = {}
    for nombre, ruta in RUTAS.items():
        print(f"Procesando {nombre}...")
        res_sim[nombre] = procesar([ruta], nombre, con_claves=False)

    # 2) Procesar los 4 datasets de datos reales como un solo conjunto.
    print("Procesando datos reales (4 datasets unidos)...")
    res_datos = procesar(DATOS_REALES, "Datos reales", con_claves=True)

    # 3) Tablas cutflow (se imprimen y se guardan como CSV).
    rango = "sin corte en m4l" if VENTANA is None else f"m4l en {VENTANA}"
    t_crudo, t_pond = tablas_cutflow(res_sim, res_datos)
    print(f"\n=== CUTFLOW, eventos crudos, {rango} ===")
    print(t_crudo.to_string())
    print(f"\n=== CUTFLOW, eventos esperados a 11.58 fb^-1, {rango} ===")
    print(t_pond.to_string())
    t_crudo.to_csv(os.path.join(SALIDA, "cutflow_crudo.csv"))
    t_pond.to_csv(os.path.join(SALIDA, "cutflow_ponderado.csv"))

    # 4) Binning común para todas las gráficas.
    BINS = construir_bins(res_sim, res_datos)
    print(f"\nEje X: {BINS[0]:.0f} a {BINS[-1]:.0f} GeV ({len(BINS) - 1} bins)")

    # 5) Un histograma por filtro, para datos y para simulaciones.
    for k, etapa in enumerate(ETAPAS[1:], 1):
        figura_etapa(dibujar_datos, res_datos, etapa, k, "Datos reales",
                     os.path.join(SALIDA, f"datos_etapa{k:02d}.png"))
        figura_etapa(dibujar_mc, res_sim, etapa, k, "Simulaciones",
                     os.path.join(SALIDA, f"mc_etapa{k:02d}.png"))

    # 6) Paneles con todas las etapas juntas.
    panel(dibujar_datos, res_datos, "Datos reales", os.path.join(SALIDA, "datos_panel.png"))
    panel(dibujar_mc, res_sim, "Simulaciones (ZZ + Higgs)", os.path.join(SALIDA, "mc_panel.png"))
    print("Listo. Resultados en:", SALIDA)
