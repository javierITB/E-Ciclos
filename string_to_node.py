# string_to_node_optimized.py - Versión optimizada para tiles
from geopy.distance import geodesic
import requests
import math
import os

# ================= CONFIGURACIÓN =================
TILE_LOADER = None  # Se inyectará desde web.py

# ================= GEOCODIFICACIÓN (IGUAL) =================
def obtener_coordenadas_osm(direccion: str):
    """Geocodificación externa - MANTENER IGUAL"""
    url = "https://photon.komoot.io/api/"
    params = {"q": f"{direccion}, Región Metropolitana, Chile", "limit": 1}
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

# ================= BÚSQUEDA EN TILES (NUEVO) =================
def inicializar_con_tile_loader(loader):
    """Inyecta el tile_loader desde web.py"""
    global TILE_LOADER
    TILE_LOADER = loader
    print(f"✅ Geocodificador optimizado listo ({loader.metadata['total_nodes']:,} nodos)")

def buscar_nodo_por_coordenadas(lat, lon, max_distance_km=0.5):
    """Busca nodo más cercano usando tiles (optimizado)"""
    if TILE_LOADER is None:
        return -1
    
    # 1. Buscar en tile local primero
    tile_x, tile_y = TILE_LOADER.latlon_to_tile(lat, lon)
    tile_data = TILE_LOADER.load_tile_data(tile_x, tile_y)
    
    if not tile_data:
        return -1
    
    # 2. Buscar en nodos de este tile
    nearest_node = None
    min_distance = float('inf')
    
    for node_id in tile_data['node_ids']:
        if node_id in TILE_LOADER.all_nodes:
            node = TILE_LOADER.all_nodes[node_id]
            
            # Distancia aproximada (grados)
            dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
            
            # Convertir a km aproximados (1° ≈ 111km en Santiago)
            dist_km = dist_deg * 111.0
            
            if dist_km < min_distance and dist_km <= max_distance_km:
                min_distance = dist_km
                nearest_node = node_id
    
    # 3. Si no encontró en tile local, buscar en tiles vecinos
    if nearest_node is None:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor_tile = TILE_LOADER.load_tile_data(tile_x + dx, tile_y + dy)
                if neighbor_tile:
                    for node_id in neighbor_tile['node_ids']:
                        if node_id in TILE_LOADER.all_nodes:
                            node = TILE_LOADER.all_nodes[node_id]
                            dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                            dist_km = dist_deg * 111.0
                            
                            if dist_km < min_distance and dist_km <= max_distance_km:
                                min_distance = dist_km
                                nearest_node = node_id
    
    return nearest_node if nearest_node is not None else -1

def buscar_interseccion_optimizada(calle1, calle2):
    """Busca intersección de calles usando tiles"""
    if TILE_LOADER is None:
        return -1
    
    calle1 = calle1.lower().strip()
    calle2 = calle2.lower().strip()
    
    # Buscar ways que contengan estos nombres
    ways_calle1 = []
    ways_calle2 = []
    
    for way_id, way_data in TILE_LOADER.all_ways.items():
        way_name = way_data['tags'].get('name', '').lower()
        if calle1 in way_name:
            ways_calle1.append(way_data['nodes'])
        if calle2 in way_name:
            ways_calle2.append(way_data['nodes'])
    
    if not ways_calle1 or not ways_calle2:
        return -1
    
    # Encontrar intersección (nodo común)
    for nodes1 in ways_calle1:
        set1 = set(nodes1)
        for nodes2 in ways_calle2:
            set2 = set(nodes2)
            intersection = set1.intersection(set2)
            if intersection:
                # Devolver el primer nodo de la intersección
                return next(iter(intersection))
    
    return -1

# ================= TEXTO A NODO OPTIMIZADO =================
def texto_a_nodo_optimizado(texto):
    """Versión optimizada que usa tiles en lugar de cargar todo"""
    # Caso intersección (ej: "Alameda, Estado")
    if "," in texto:
        partes = [p.strip() for p in texto.split(",", 1)]
        if len(partes) == 2:
            nodo = buscar_interseccion_optimizada(partes[0], partes[1])
            print(f"🔍 Intersección '{texto}' → Nodo: {nodo}")
            return nodo
    
    # Caso dirección normal (ej: "Plaza de Armas, Santiago")
    coord = obtener_coordenadas_osm(texto)
    if coord is None:
        print(f"❌ No se pudo geocodificar: '{texto}'")
        return -1
    
    lat, lon = coord
    print(f"📍 Geocodificado '{texto}' → ({lat:.6f}, {lon:.6f})")
    
    # Buscar nodo más cercano usando tiles
    nodo = buscar_nodo_por_coordenadas(lat, lon)
    
    if nodo != -1:
        print(f"✅ Nodo encontrado: {nodo}")
        # Opcional: mostrar info del nodo
        if nodo in TILE_LOADER.all_nodes:
            node_data = TILE_LOADER.all_nodes[nodo]
            print(f"   Coordenadas: ({node_data['lat']:.6f}, {node_data['lon']:.6f})")
            print(f"   Altura: {node_data['ele']}m")
    else:
        print(f"⚠️  No se encontró nodo cercano para: '{texto}'")
    
    return nodo

# ================= COMPATIBILIDAD =================
def texto_a_nodo(texto):
    """Función con mismo nombre para compatibilidad con web.py"""
    return texto_a_nodo_optimizado(texto)