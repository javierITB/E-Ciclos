import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import requests
import numpy as np
from scipy.spatial import KDTree
import time


# ===========================================
#   1) ESTRUCTURAS GLOBALES (se llenan una vez)
# ===========================================
NODOS = {}
WAYS = []
KD_IDS = []
KD_COORDS = None
KD_TREE = None
INDEX_CALLES = {}


# ===========================================
#   2) CARGA INICIAL OSM + PREPARACIÓN
# ===========================================
def preparar_osm(ruta):
    global NODOS, WAYS, KD_IDS, KD_COORDS, KD_TREE, INDEX_CALLES

    print("Cargando archivo OSM...")
    tree = ET.parse(ruta)
    root = tree.getroot()

    # ---- NODOS ----
    NODOS = {}
    for node in root.findall("node"):
        node_id = int(node.attrib["id"])
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        NODOS[node_id] = (lat, lon)

    # ---- WAYS ----
    WAYS = []
    INDEX_CALLES = {}

    for way in root.findall("way"):
        nombre = None
        node_refs = []

        for child in way:
            if child.tag == "nd":
                node_refs.append(int(child.attrib["ref"]))
            elif child.tag == "tag":
                if child.attrib.get("k") == "name":
                    nombre = child.attrib.get("v")

        if nombre:
            WAYS.append({"nombre": nombre, "nodos": node_refs})

            key = nombre.lower()
            if key not in INDEX_CALLES:
                INDEX_CALLES[key] = []
            INDEX_CALLES[key].append(node_refs)

    # ---- KD-TREE ----
    print("Construyendo KD-Tree...")
    KD_IDS = list(NODOS.keys())
    KD_COORDS = np.array([NODOS[nid] for nid in KD_IDS])
    KD_TREE = KDTree(KD_COORDS)

    print("OSM cargado y optimizado.\n")


# ===========================================
#   3) GEOCODIFICACIÓN SOLO PARA DIRECCIONES
# ===========================================
def obtener_coordenadas_osm(direccion: str):
    url = "https://photon.komoot.io/api/"
    params = {
        "q": f"{direccion}, Región Metropolitana, Chile",
        "limit": 1
    }
    headers = {"User-Agent": "MiAppGeolocalizacion/1.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        features = data.get("features", [])
        if not features:
            return None

        lon, lat = features[0]["geometry"]["coordinates"]
        return lat, lon

    except:
        return None


# ===========================================
#   4) BUSCAR INTERSECCIÓN ENTRE CALLES
# ===========================================
def buscar_interseccion(call1, call2):
    call1 = call1.lower()
    call2 = call2.lower()

    # Ways por nombre exacto
    w1 = INDEX_CALLES.get(call1, [])
    w2 = INDEX_CALLES.get(call2, [])

    if not w1 or not w2:
        # Buscar como subcadena (fallback)
        w1 = [w["nodos"] for w in WAYS if call1 in w["nombre"].lower()]
        w2 = [w["nodos"] for w in WAYS if call2 in w["nombre"].lower()]

    if not w1 or not w2:
        return -1

    nodos1 = set(n for lista in w1 for n in lista)
    nodos2 = set(n for lista in w2 for n in lista)

    # ---- Intersección exacta ----
    inter = nodos1.intersection(nodos2)
    if inter:
        return next(iter(inter))

    # ---- Aproximación usando KD-Tree ----
    coords2 = np.array([NODOS[n] for n in nodos2])
    tree2 = KDTree(coords2)
    nodos2_list = list(nodos2)

    best_dist = float("inf")
    best_node = None

    for n1 in nodos1:
        dist, idx = tree2.query(NODOS[n1])
        if dist < best_dist:
            best_dist = dist
            best_node = nodos2_list[idx]

    return best_node if best_dist <= 100 else -1


# ===========================================
#   5) BUSCAR NODO MÁS CERCANO
# ===========================================
def nodo_mas_cercano(lat, lon, max_dist=100):
    punto = np.array([lat, lon])
    dist, idx = KD_TREE.query(punto)

    if dist <= max_dist:
        return KD_IDS[idx]
    return -1


# ===========================================
#   6) CONVERSIÓN FINAL TEXTO → NODO (con tiempo)
# ===========================================
def texto_a_nodo(texto):
    inicio = time.time()  # ← medir tiempo antes

    # Caso intersección
    if "," in texto:
        c1, c2 = [s.strip() for s in texto.split(",", 1)]
        nodo = buscar_interseccion(c1, c2)
        tiempo = f"[{(time.time() - inicio):.4f}s]"
        print(tiempo + " Nodo encontrado:", nodo)
        return nodo

    # Caso dirección normal
    coord = obtener_coordenadas_osm(texto)
    if coord is None:
        tiempo = f"[{(time.time() - inicio):.4f}s]"
        print(tiempo + " No se pudo geocodificar.")
        return -1

    lat, lon = coord
    nodo = nodo_mas_cercano(lat, lon)

    tiempo = f"[{(time.time() - inicio):.4f}s]"
    print(tiempo + " Nodo encontrado:", nodo)
    return nodo


# ===========================================
#   7) EJEMPLO
# ===========================================
if __name__ == "__main__":
    #preparar_osm("map_clean.osm")
    print(texto_a_nodo("Gorbea, Vergara"))
