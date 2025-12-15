# create_tiles_fixed.py - Preserva TODOS los datos para routing
import xml.etree.ElementTree as ET
import pickle
import os
import math
import json
from collections import defaultdict

# Configuración
OSM_FILE = "map_clean.osm"
TILES_DIR = "tiles_data_complete"
TILE_SIZE = 0.01

# Bounds
BOUNDS = {
    'minlat': -33.7509885,
    'minlon': -70.9550929,
    'maxlat': -33.1051223,
    'maxlon': -70.1976561
}

def latlon_to_tile(lat, lon):
    """Convierte coordenadas a tile"""
    tile_x = math.floor((lon - BOUNDS['minlon']) / TILE_SIZE)
    tile_y = math.floor((lat - BOUNDS['minlat']) / TILE_SIZE)
    return tile_x, tile_y

def create_complete_tiles():
    print("🔄 Creando sistema de tiles COMPLETO...")
    print(f"  Preservando: nodos, alturas, vías, direccionalidad, tags")
    
    os.makedirs(TILES_DIR, exist_ok=True)
    
    # ================= ESTRUCTURAS PRINCIPALES =================
    # 1. Nodos completos
    nodes_data = {}  # id -> {lat, lon, ele, tags}
    
    # 2. Vías completas (para reconstruir aristas)
    ways_data = {}   # id -> {nodes[], tags{}, geometry}
    
    # 3. ARISTAS (esto es lo que usa routing.py)
    edges_data = []  # Lista de aristas para NetworkX/Grafo
    
    # 4. Índices espaciales
    tile_to_nodes = defaultdict(list)  # tile -> [node_ids]
    tile_to_edges = defaultdict(list)  # tile -> [edge_indices]
    
    # ================= PARSEAR OSM =================
    print("📖 Parseando OSM (preservando todo)...")
    tree = ET.parse(OSM_FILE)
    root = tree.getroot()
    
    # Contadores
    stats = {'nodes': 0, 'ways': 0, 'edges': 0, 'oneway': 0}
    
    # ---- FASE 1: NODOS ----
    for node in root.findall("node"):
        node_id = int(node.attrib["id"])
        
        node_info = {
            'id': node_id,
            'lat': float(node.attrib["lat"]),
            'lon': float(node.attrib["lon"]),
            'ele': float(node.attrib.get("ele", "0.0")),  # ALTURA REAL
            'tags': {}
        }
        
        # Guardar TODOS los tags
        for tag in node.findall("tag"):
            k = tag.attrib["k"]
            v = tag.attrib["v"]
            node_info['tags'][k] = v
        
        nodes_data[node_id] = node_info
        
        # Asignar a tile
        tile_x, tile_y = latlon_to_tile(node_info['lat'], node_info['lon'])
        tile_to_nodes[(tile_x, tile_y)].append(node_id)
        
        stats['nodes'] += 1
        if stats['nodes'] % 10000 == 0:
            print(f"    Nodos: {stats['nodes']}")
    
    print(f"  ✅ Nodos: {stats['nodes']} (con alturas)")
    
    # ---- FASE 2: VÍAS y ARISTAS ----
    for way in root.findall("way"):
        way_id = int(way.attrib["id"])
        
        # Nodos de esta vía (en orden)
        way_nodes = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
        
        if len(way_nodes) < 2:
            continue  # Vía inválida
        
        # Tags de la vía
        way_tags = {}
        for tag in way.findall("tag"):
            k = tag.attrib["k"]
            v = tag.attrib["v"]
            way_tags[k] = v
        
        # Guardar vía completa
        ways_data[way_id] = {
            'nodes': way_nodes,
            'tags': way_tags
        }
        
        # ---- CONVERTIR A ARISTAS ----
        # Determinar si es oneway
        oneway = way_tags.get("oneway", "false").lower() == "true"
        stats['oneway'] += 1 if oneway else 0
        
        # Obtener tipo de vía para importancia
        highway_type = way_tags.get("highway", "")
        
        # Para CADA segmento (nodo_i -> nodo_i+1)
        for i in range(len(way_nodes) - 1):
            from_id = way_nodes[i]
            to_id = way_nodes[i + 1]
            
            # Verificar que ambos nodos existen
            if from_id not in nodes_data or to_id not in nodes_data:
                continue
            
            # Calcular distancia (usando Haversine aproximado)
            from_node = nodes_data[from_id]
            to_node = nodes_data[to_id]
            
            # Distancia simple (en grados, luego se convierte en routing.py)
            dist_deg = math.sqrt(
                (to_node['lat'] - from_node['lat'])**2 +
                (to_node['lon'] - from_node['lon'])**2
            )
            
            # Crear arista
            edge = {
                'id': len(edges_data),  # ID interno
                'from': from_id,
                'to': to_id,
                'way_id': way_id,
                'oneway': oneway,
                'highway': highway_type,
                'length': float(way_tags.get("length", "0.0")),
                'maxspeed': way_tags.get("maxspeed", "50"),
                'lanes': way_tags.get("lanes", "1"),
                'bicycle': way_tags.get("bicycle", ""),
                'tags': way_tags  # TODOS los tags
            }
            
            edges_data.append(edge)
            stats['edges'] += 1
            
            # Si NO es oneway, crear arista inversa
            if not oneway:
                reverse_edge = {
                    'id': len(edges_data),
                    'from': to_id,
                    'to': from_id,
                    'way_id': way_id,
                    'oneway': False,  # La inversa también es bidireccional
                    'highway': highway_type,
                    'length': float(way_tags.get("length", "0.0")),
                    'maxspeed': way_tags.get("maxspeed", "50"),
                    'lanes': way_tags.get("lanes", "1"),
                    'bicycle': way_tags.get("bicycle", ""),
                    'tags': way_tags
                }
                edges_data.append(reverse_edge)
                stats['edges'] += 1
        
        stats['ways'] += 1
        if stats['ways'] % 5000 == 0:
            print(f"    Vías: {stats['ways']}, Aristas: {stats['edges']}")
    
    # ---- FASE 3: ASIGNAR ARISTAS A TILES ----
    print("🗺️ Asignando aristas a tiles...")
    
    for edge_idx, edge in enumerate(edges_data):
        # Obtener nodos de esta arista
        from_node = nodes_data.get(edge['from'])
        to_node = nodes_data.get(edge['to'])
        
        if not from_node or not to_node:
            continue
        
        # Determinar tiles que toca esta arista
        tiles = set()
        
        # Tile del nodo origen
        tile_from = latlon_to_tile(from_node['lat'], from_node['lon'])
        tiles.add(tile_from)
        
        # Tile del nodo destino
        tile_to = latlon_to_tile(to_node['lat'], to_node['lon'])
        tiles.add(tile_to)
        
        # Agregar a todos los tiles que toca
        for tile_key in tiles:
            tile_to_edges[tile_key].append(edge_idx)
    
    # ================= GUARDAR DATOS =================
    print("💾 Guardando datos completos...")
    
    # 1. Nodos completos
    with open(os.path.join(TILES_DIR, "nodes_full.pkl"), "wb") as f:
        pickle.dump(nodes_data, f)
    
    # 2. Vías completas
    with open(os.path.join(TILES_DIR, "ways_full.pkl"), "wb") as f:
        pickle.dump(ways_data, f)
    
    # 3. Aristas para routing
    with open(os.path.join(TILES_DIR, "edges_full.pkl"), "wb") as f:
        pickle.dump(edges_data, f)
    
    # 4. Índice de tiles
    tile_index = {
        'tile_size': TILE_SIZE,
        'bounds': BOUNDS,
        'tile_to_nodes': dict(tile_to_nodes),
        'tile_to_edges': dict(tile_to_edges),
        'stats': stats
    }
    
    with open(os.path.join(TILES_DIR, "tile_index.pkl"), "wb") as f:
        pickle.dump(tile_index, f)
    
    # 5. Metadata en JSON (para debug)
    metadata = {
        'total_nodes': stats['nodes'],
        'total_ways': stats['ways'],
        'total_edges': stats['edges'],
        'oneway_edges': stats['oneway'],
        'bounds': BOUNDS,
        'tile_size': TILE_SIZE,
        'tiles_with_data': len(tile_to_nodes)
    }
    
    with open(os.path.join(TILES_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # ================= GUARDAR TILES INDIVIDUALES =================
    print("📁 Creando tiles individuales...")
    
    # Para cada tile con datos
    for (tile_x, tile_y), node_ids in tile_to_nodes.items():
        # Datos específicos de este tile
        tile_data = {
            'tile_x': tile_x,
            'tile_y': tile_y,
            'bounds': {
                'minlat': BOUNDS['minlat'] + tile_y * TILE_SIZE,
                'maxlat': BOUNDS['minlat'] + (tile_y + 1) * TILE_SIZE,
                'minlon': BOUNDS['minlon'] + tile_x * TILE_SIZE,
                'maxlon': BOUNDS['minlon'] + (tile_x + 1) * TILE_SIZE
            },
            'node_ids': node_ids,
            'edge_ids': tile_to_edges.get((tile_x, tile_y), [])
        }
        
        # Guardar tile
        tile_file = os.path.join(TILES_DIR, f"tile_{tile_x}_{tile_y}.pkl")
        with open(tile_file, "wb") as f:
            pickle.dump(tile_data, f)
    
    # ================= ESTADÍSTICAS FINALES =================
    print("\n" + "="*60)
    print("✅ SISTEMA DE TILES COMPLETO CREADO")
    print("="*60)
    
    print(f"\n📊 DATOS PRESERVADOS:")
    print(f"  Nodos: {stats['nodes']:,} (con altura 'ele')")
    print(f"  Vías: {stats['ways']:,} (con todos los tags)")
    print(f"  Aristas: {stats['edges']:,} (con direccionalidad)")
    print(f"  One-way: {stats['oneway']:,}")
    
    # Distribución
    nodes_per_tile = [len(ids) for ids in tile_to_nodes.values()]
    edges_per_tile = [len(ids) for ids in tile_to_edges.values()]
    
    print(f"\n📈 DISTRIBUCIÓN POR TILE:")
    print(f"  Tiles con datos: {len(tile_to_nodes):,}")
    print(f"  Nodos/tile: min={min(nodes_per_tile)}, avg={sum(nodes_per_tile)/len(nodes_per_tile):.0f}, max={max(nodes_per_tile)}")
    print(f"  Aristas/tile: min={min(edges_per_tile)}, avg={sum(edges_per_tile)/len(edges_per_tile):.0f}, max={max(edges_per_tile)}")
    
    # Ejemplo de datos guardados
    print(f"\n🔍 EJEMPLO DE DATOS GUARDADOS:")
    if edges_data:
        sample_edge = edges_data[0]
        print(f"  Arista #{sample_edge['id']}:")
        print(f"    From: {sample_edge['from']} → To: {sample_edge['to']}")
        print(f"    Oneway: {sample_edge['oneway']}")
        print(f"    Highway: {sample_edge.get('highway', 'N/A')}")
        print(f"    Length: {sample_edge.get('length', 'N/A')}")
        print(f"    Bicycle: {sample_edge.get('bicycle', 'N/A')}")
    
    print(f"\n📂 Archivos en '{TILES_DIR}/':")
    total_size = 0
    for fname in os.listdir(TILES_DIR):
        fpath = os.path.join(TILES_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        total_size += size_mb
        print(f"  - {fname}: {size_mb:.2f} MB")
    
    print(f"  Total: {total_size:.2f} MB")
    
    return True

# ================= LOADER OPTIMIZADO =================
class CompleteTileLoader:
    """Carga tiles manteniendo TODOS los datos para routing"""
    
    def __init__(self, tiles_dir="tiles_data_complete"):
        self.tiles_dir = tiles_dir
        
        print("📂 Cargando sistema de tiles completo...")
        
        # Cargar datos globales
        with open(os.path.join(tiles_dir, "nodes_full.pkl"), "rb") as f:
            self.all_nodes = pickle.load(f)
        
        with open(os.path.join(tiles_dir, "edges_full.pkl"), "rb") as f:
            self.all_edges = pickle.load(f)
        
        with open(os.path.join(tiles_dir, "tile_index.pkl"), "rb") as f:
            self.tile_index = pickle.load(f)
        
        with open(os.path.join(tiles_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        print(f"  ✅ {self.metadata['total_nodes']:,} nodos")
        print(f"  ✅ {self.metadata['total_edges']:,} aristas")
        print(f"  ✅ {self.metadata['tiles_with_data']:,} tiles")
        
        # Cache
        self.tile_cache = {}
        self.loaded_edges = set()
    
    def latlon_to_tile(self, lat, lon):
        """Convierte coordenadas a tile"""
        tile_x = math.floor((lon - self.tile_index['bounds']['minlon']) / self.tile_index['tile_size'])
        tile_y = math.floor((lat - self.tile_index['bounds']['minlat']) / self.tile_index['tile_size'])
        return tile_x, tile_y
    
    def get_viewport_tiles(self, center_lat, center_lon, zoom=13):
        """Obtiene tiles en viewport"""
        # Radio aproximado (ajustable)
        if zoom <= 10:
            radius = 0.5
        elif zoom <= 13:
            radius = 0.2
        elif zoom <= 15:
            radius = 0.1
        else:
            radius = 0.05
        
        min_lat = center_lat - radius
        max_lat = center_lat + radius
        min_lon = center_lon - radius
        max_lon = center_lon + radius
        
        # Calcular tiles
        min_tx, min_ty = self.latlon_to_tile(min_lat, min_lon)
        max_tx, max_ty = self.latlon_to_tile(max_lat, max_lon)
        
        tiles = []
        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                tiles.append((tx, ty))
        
        return tiles
    
    def load_tile_data(self, tile_x, tile_y):
        """Carga datos de un tile específico"""
        tile_key = (tile_x, tile_y)
        
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        tile_file = os.path.join(self.tiles_dir, f"tile_{tile_x}_{tile_y}.pkl")
        
        if os.path.exists(tile_file):
            with open(tile_file, "rb") as f:
                tile_data = pickle.load(f)
                self.tile_cache[tile_key] = tile_data
                return tile_data
        
        return None
    
    def get_visible_data(self, center_lat, center_lon, zoom=13):
        """Obtiene nodos y aristas visibles para Plotly"""
        tiles = self.get_viewport_tiles(center_lat, center_lon, zoom)
        
        nodes_for_plot = []
        edges_for_plot = []
        
        for tile_x, tile_y in tiles:
            tile_data = self.load_tile_data(tile_x, tile_y)
            if not tile_data:
                continue
            
            # 1. Nodos para Plotly
            for node_id in tile_data['node_ids']:
                if node_id in self.all_nodes:
                    node = self.all_nodes[node_id]
                    nodes_for_plot.append({
                        'id': node_id,
                        'lat': node['lat'],
                        'lon': node['lon'],
                        'ele': node['ele'],
                        'tags': node.get('tags', {})
                    })
            
            # 2. Aristas para Plotly (líneas)
            for edge_idx in tile_data['edge_ids']:
                if edge_idx >= len(self.all_edges):
                    continue
                
                edge = self.all_edges[edge_idx]
                from_node = self.all_nodes.get(edge['from'])
                to_node = self.all_nodes.get(edge['to'])
                
                if from_node and to_node:
                    # Línea para Plotly
                    edges_for_plot.append({
                        'from_lat': from_node['lat'],
                        'from_lon': from_node['lon'],
                        'to_lat': to_node['lat'],
                        'to_lon': to_node['lon'],
                        'edge_data': edge  # Datos completos para routing
                    })
        
        return nodes_for_plot, edges_for_plot
    
    def get_edges_for_routing(self, node_ids):
        """Obtiene aristas conectadas a nodos específicos (para Dijkstra/A*)"""
        edges = []
        
        # Buscar aristas donde from o to estén en node_ids
        for edge in self.all_edges:
            if edge['from'] in node_ids or edge['to'] in node_ids:
                edges.append(edge)
        
        return edges
    
    def find_nearest_node(self, lat, lon, max_distance_deg=0.001):
        """Encuentra nodo más cercano para selección"""
        # Buscar en tile local primero
        tile_x, tile_y = self.latlon_to_tile(lat, lon)
        tile_data = self.load_tile_data(tile_x, tile_y)
        
        if not tile_data:
            return None
        
        nearest = None
        min_dist = float('inf')
        
        for node_id in tile_data['node_ids']:
            if node_id in self.all_nodes:
                node = self.all_nodes[node_id]
                dist = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                
                if dist < min_dist and dist <= max_distance_deg:
                    min_dist = dist
                    nearest = {
                        'id': node_id,
                        'lat': node['lat'],
                        'lon': node['lon'],
                        'ele': node['ele']
                    }
        
        return nearest

# ================= EJECUCIÓN =================
if __name__ == "__main__":
    success = create_complete_tiles()
    
    if success:
        print("\n" + "="*60)
        print("🧪 PRUEBA DE CARGA:")
        print("="*60)
        
        loader = CompleteTileLoader()
        
        # Prueba con un punto conocido
        test_point = (-33.4429009, -70.6462541)  # Nodo 1000000
        
        print(f"\n📍 Punto de prueba: {test_point}")
        
        # Encontrar nodo más cercano
        nearest = loader.find_nearest_node(*test_point)
        if nearest:
            print(f"  Nodo más cercano: ID {nearest['id']}")
            print(f"  Altura: {nearest['ele']}m")
            
            # Obtener aristas conectadas
            connected_edges = loader.get_edges_for_routing([nearest['id']])
            print(f"  Aristas conectadas: {len(connected_edges)}")
            
            if connected_edges:
                sample = connected_edges[0]
                print(f"  Ejemplo arista: {sample['from']}→{sample['to']}")
                print(f"    Oneway: {sample['oneway']}")
                print(f"    Highway: {sample.get('highway', 'N/A')}")
        
        # Obtener datos para vista
        print(f"\n👁️  Datos para vista (zoom 15):")
        nodes, edges = loader.get_visible_data(*test_point, zoom=15)
        print(f"  Nodos visibles: {len(nodes)}")
        print(f"  Aristas visibles: {len(edges)}")
        
        print("\n✅ ¡Sistema listo para integrar en web.py!")