import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import requests


# === 1) Geocodificación solo para direcciones, nunca para intersecciones ===
def obtener_coordenadas_osm(direccion: str):
    url = "https://photon.komoot.io/api/"
    params = {
        "q": f"{direccion}, Región Metropolitana, Chile",
        "limit": 1
    }
    headers = {
        "User-Agent": "MiAppGeolocalizacion/1.0"
    }

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



# === 2) Cargar nodos y ways del dataset OSM ===
def cargar_osm(ruta):
    tree = ET.parse(ruta)
    root = tree.getroot()

    nodos = {}      # id → (lat, lon)
    ways = []       # {nombre, nodos}

    # Nodos
    for node in root.findall("node"):
        node_id = int(node.attrib["id"])
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        nodos[node_id] = (lat, lon)

    # Ways
    for way in root.findall("way"):
        nombre = None
        node_refs = []

        for child in way:
            if child.tag == "nd":
                node_refs.append(int(child.attrib["ref"]))
            elif child.tag == "tag":
                if child.attrib.get("k") == "name":
                    nombre = child.attrib.get("v")

        if nombre is not None:
            ways.append({
                "nombre": nombre,
                "nodos": node_refs
            })

    return nodos, ways



# === 3) Encontrar intersección entre dos calles (SOLO DATASET) ===
def buscar_interseccion(call1, call2, nodos, ways):
    # Filtrar ways que coincidan por nombre
    w1 = [w for w in ways if call1.lower() in w["nombre"].lower()]
    w2 = [w for w in ways if call2.lower() in w["nombre"].lower()]

    if not w1 or not w2:
        return -1

    # Nodos de ambas calles
    nodos1 = set([n for w in w1 for n in w["nodos"]])
    nodos2 = set([n for w in w2 for n in w["nodos"]])

    # Intersección exacta
    inter = nodos1.intersection(nodos2)
    if inter:
        return list(inter)[0]

    # Si no existe: aproximar
    min_dist = float("inf")
    best = None

    for n1 in nodos1:
        for n2 in nodos2:
            d = geodesic(nodos[n1], nodos[n2]).meters
            if d < min_dist:
                min_dist = d
                best = n1

    if min_dist <= 100:
        return best

    return -1



# === 4) Buscar nodo más cercano a una coordenada dentro del dataset ===
def nodo_mas_cercano(lat, lon, nodos, max_dist=100):
    punto = (lat, lon)
    best_id = None
    best_d = float("inf")

    for nid, (nlat, nlon) in nodos.items():
        d = geodesic(punto, (nlat, nlon)).meters
        if d < best_d:
            best_d = d
            best_id = nid

    return best_id if best_d <= max_dist else -1



# === 5) Función final texto → nodo usando SOLO el dataset ===
def texto_a_nodo(texto, archivo_osm="map_clean.osm"):
    nodos, ways = cargar_osm(archivo_osm)

    # --- Intersección ---
    if "," in texto:
        c1, c2 = [s.strip() for s in texto.split(",", 1)]
        return buscar_interseccion(c1, c2, nodos, ways)

    # --- Dirección normal ---
    coord = obtener_coordenadas_osm(texto)
    if coord is None:
        return -1

    lat, lon = coord
    return nodo_mas_cercano(lat, lon, nodos)



# === EJEMPLOS ===
# 1) Intersección:
print(texto_a_nodo("Gorbea, Vergara"))

# 2) Dirección normal:
# print(texto_a_nodo("cenco ñuñoa"))
