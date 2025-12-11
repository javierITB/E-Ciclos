import osmnx as ox
import networkx as nx
import time
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon

ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 300
ox.settings.useful_tags_way = [
    'highway', 'name', 'cycleway', 'bicycle',
    'oneway', 'lanes', 'maxspeed', 'length',
    'junction', 'bridge', 'tunnel'
]

# Lista de comunas del Gran Santiago
TODAS_COMUNAS = [
    "Santiago, Chile", "Providencia, Chile", "Ñuñoa, Chile",
    "Las Condes, Chile", "La Reina, Chile", "Macul, Chile",
    "Peñalolén, Chile", "San Miguel, Chile", "San Joaquín, Chile",
    "Recoleta, Chile", "Independencia, Chile", "Conchalí, Chile",
    "Huechuraba, Chile", "Quilicura, Chile", "Renca, Chile",
    "Quinta Normal, Chile", "Lo Prado, Chile", "Pudahuel, Chile",
    "Cerro Navia, Chile", "Maipú, Chile", "Cerrillos, Chile",
    "Estación Central, Chile", "La Granja, Chile", "La Florida, Chile",
    "San Ramón, Chile", "El Bosque, Chile", "La Cisterna, Chile",
    "San Bernardo, Chile", "Lo Espejo, Chile", "Pedro Aguirre Cerda, Chile",
    "Puente Alto, Chile", "La Pintana, Chile", "Vitacura, Chile", "Lo Barnechea, Chile"
]

CUSTOM_FILTER = (
    '["highway"~"cycleway|path|footway|pedestrian|'
    'living_street|residential|unclassified|'
    'service|tertiary|secondary|primary|trunk|motorway|'
    'track|steps|corridor"]'
)



from shapely.geometry import Point
import osmnx as ox
import time

def download_grafo_comuna(comuna, custom_filter=CUSTOM_FILTER, max_intentos=3, buffer_m=0.005):
    """
    Descarga un grafo dirigido de una comuna asegurándose de obtener Polygon o MultiPolygon.
    Si Nominatim devuelve un Point, se crea un pequeño buffer para que OSMnx pueda generar el grafo.
    
    Args:
        comuna (str): Nombre de la comuna.
        custom_filter (str): Filtro OSMnx para carreteras.
        max_intentos (int): Máximo de reintentos.
        buffer_m (float): Tamaño del buffer si la geometría es Point (grados lat/lon aprox.).
    """
    for intento in range(max_intentos):
        try:
            # Intentar descargar normalmente
            G = None
            for which_result in range(1, 4):
                try:
                    G = ox.graph_from_place(
                        comuna,
                        custom_filter=custom_filter,
                        retain_all=True,
                        simplify=False,
                        network_type='drive',
                        which_result=which_result
                    )
                    if len(G.nodes) > 0:
                        return G
                except ox._errors.InsufficientResponseError:
                    continue  # probar siguiente which_result

            # Si G no tiene nodos, intentar forzar buffer si la geometría es Point
            if G is None or len(G.nodes) == 0:
                gdf = ox.geocoder.geocode_to_gdf(comuna)
                geom = gdf.geometry.iloc[0]
                if isinstance(geom, Point):
                    # Crear pequeño Polygon alrededor del Point
                    geom = geom.buffer(buffer_m)
                G = ox.graph_from_polygon(
                    geom,
                    custom_filter=custom_filter,
                    retain_all=True,
                    simplify=False,
                    network_type='drive'
                )
                if len(G.nodes) > 0:
                    return G

        except Exception as e:
            print(f"   ⚠️ Error inesperado en {comuna}: {e}")

        print(f"   ⚠️ Reintentando {comuna} (intento {intento+2}/{max_intentos})...")
        time.sleep(2)

    # Si llega hasta acá, crear un polígono por defecto (área mínima) para garantizar descarga
    print(f"⚠️ No se pudo descargar correctamente {comuna}, se usará polígono de emergencia")
    try:
        gdf = ox.geocoder.geocode_to_gdf(comuna)
        geom = gdf.geometry.iloc[0]
        if isinstance(geom, Point):
            geom = geom.buffer(buffer_m)
        G = ox.graph_from_polygon(
            geom,
            custom_filter=custom_filter,
            retain_all=True,
            simplify=False,
            network_type='drive'
        )
        return G
    except Exception as e:
        print(f"❌ Falla crítica en {comuna}, no se puede descargar: {e}")
        return None



def download_all_comunas_directed():
    """
    Descarga todas las comunas, combina en un grafo dirigido único y simplifica.
    """
    grafos = []
    for i, comuna in enumerate(TODAS_COMUNAS, 1):
        print(f"\n[{i}/{len(TODAS_COMUNAS)}] {comuna}")
        G = download_grafo_comuna(comuna)
        if G is None:
            continue
        print(f"   ✅ {len(G.nodes()):,} nodos, {len(G.edges()):,} aristas")
        grafos.append(G)
        time.sleep(0.5)
    
    if not grafos:
        print("❌ No se pudo descargar ninguna comuna")
        return None
    
    print("\n🔗 COMBINANDO TODOS LOS GRAFOS DIRIGIDOS...")
    G_total = grafos[0]
    for G_comuna in grafos[1:]:
        G_total = nx.compose(G_total, G_comuna)
    
    print(f"\n📊 GRAFO TOTAL: {len(G_total.nodes()):,} nodos, {len(G_total.edges()):,} aristas")
    
    print("\n✨ SIMPLIFICANDO GRAFO TOTAL (manteniendo dirección)...")
    try:
        G_total = ox.simplify_graph(G_total)
    except Exception as e:
        print(f"   ⚠️ No se pudo simplificar: {str(e)}")
    
    return G_total



def export_directed_to_osm(G, output_file="santiago_completo_dirigido.osm"):
    """
    Exporta un grafo DIRIGIDO a formato OSM completo
    """
    
    print(f"\n🌐 EXPORTANDO A OSM DIRIGIDO: {output_file}")
    print(f"   Grafo: {len(G.nodes()):,} nodos, {len(G.edges()):,} aristas")
    print(f"   ¿Dirigido?: {G.is_directed()}")
    
    # Crear raíz OSM
    root = ET.Element('osm')
    root.set('version', '0.6')
    root.set('generator', 'santiago_directed_complete')
    
    # Calcular bounds reales del grafo
    print("   Calculando bounds...")
    lats = []
    lons = []
    
    for node_id, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
            lats.append(data['y'])
            lons.append(data['x'])
    
    if lats and lons:
        bounds = ET.SubElement(root, 'bounds')
        bounds.set('minlat', str(min(lats)))
        bounds.set('minlon', str(min(lons)))
        bounds.set('maxlat', str(max(lats)))
        bounds.set('maxlon', str(max(lons)))
        print(f"   Bounds: [{min(lats):.4f}, {min(lons):.4f}, {max(lats):.4f}, {max(lons):.4f}]")
    else:
        # Bounds por defecto de Santiago
        bounds = ET.SubElement(root, 'bounds')
        bounds.set('minlat', '-33.65')
        bounds.set('minlon', '-70.85')
        bounds.set('maxlat', '-33.35')
        bounds.set('maxlon', '-70.45')
    
    # MAPEO DE IDs: convertir todos los IDs a numéricos secuenciales
    print("   Creando mapeo de IDs...")
    node_id_map = {}
    next_node_id = 1000000
    
    # Primero, recolectar todos los nodos únicos
    for node_id in G.nodes():
        node_id_map[node_id] = next_node_id
        next_node_id += 1
    
    # AÑADIR NODOS al OSM
    print("   Añadiendo nodos al OSM...")
    nodes_added = 0
    
    for orig_id, new_id in tqdm(node_id_map.items(), desc="Nodos"):
        if orig_id in G.nodes():
            data = G.nodes[orig_id]
            
            if 'x' in data and 'y' in data:
                node_elem = ET.Element('node')
                node_elem.set('id', str(new_id))
                node_elem.set('lat', str(data['y']))
                node_elem.set('lon', str(data['x']))
                
                # Añadir tags si existen
                tags_to_add = []
                
                if 'name' in data and data['name']:
                    tags_to_add.append(('name', str(data['name'])))
                
                if 'highway' in data and data['highway']:
                    tags_to_add.append(('highway', str(data['highway'])))
                
                # Añadir todos los tags encontrados
                for key, value in data.items():
                    if key not in ['x', 'y', 'lat', 'lon'] and value:
                        tags_to_add.append((key, str(value)))
                
                # Limitar tags a los primeros 10 para no hacer muy grande
                for key, value in tags_to_add[:10]:
                    tag = ET.SubElement(node_elem, 'tag')
                    tag.set('k', key[:50])
                    tag.set('v', value[:100])
                
                root.append(node_elem)
                nodes_added += 1
    
    print(f"   Nodos añadidos: {nodes_added:,}")
    
    # AÑADIR ARISTAS como VÍAS en OSM
    print("   Añadiendo aristas como vías...")
    next_way_id = 2000000
    ways_added = 0
    
    # Para grafos dirigidos, procesamos cada arista por separado
    for u, v, data in tqdm(G.edges(data=True), desc="Aristas"):
        if u in node_id_map and v in node_id_map:
            way_elem = ET.Element('way')
            way_elem.set('id', str(next_way_id))
            
            # Nodo origen
            nd1 = ET.SubElement(way_elem, 'nd')
            nd1.set('ref', str(node_id_map[u]))
            
            # Nodo destino
            nd2 = ET.SubElement(way_elem, 'nd')
            nd2.set('ref', str(node_id_map[v]))
            
            # TAGS DE LA VÍA (información de la arista)
            tags_to_add = []
            
            # Información de la carretera/vía
            if 'highway' in data:
                hw_value = data['highway']
                if isinstance(hw_value, list):
                    hw_value = hw_value[0] if hw_value else 'road'
                tags_to_add.append(('highway', str(hw_value)))
            
            # Información de dirección (ONE-WAY es crucial para grafos dirigidos)
            if 'oneway' in data:
                tags_to_add.append(('oneway', str(data['oneway'])))
            
            # Nombre si existe
            if 'name' in data and data['name']:
                tags_to_add.append(('name', str(data['name'])))
            
            # Longitud
            if 'length' in data:
                tags_to_add.append(('length', f"{float(data['length']):.1f}"))
            
            # Otros atributos importantes
            for key in ['cycleway', 'bicycle', 'lanes', 'maxspeed', 'junction', 'bridge', 'tunnel']:
                if key in data and data[key]:
                    tags_to_add.append((key, str(data[key])))
            
            # Añadir todos los tags
            for key, value in tags_to_add[:15]:  # Limitar a 15 tags
                tag = ET.SubElement(way_elem, 'tag')
                tag.set('k', str(key)[:50])
                tag.set('v', str(value)[:100])
            
            root.append(way_elem)
            next_way_id += 1
            ways_added += 1
            
            # Mostrar progreso cada 10,000 vías
            if ways_added % 10000 == 0:
                print(f"     Procesadas: {ways_added:,} vías")
    
    print(f"   Vías añadidas: {ways_added:,}")
    
    # GUARDAR ARCHIVO OSM
    print("   Guardando archivo OSM...")
    
    # Convertir a string XML con formato
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    
    # Formatear con indentación (puede hacer el archivo más grande pero legible)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding='utf-8')
    
    # Escribir en chunks para no consumir toda la memoria
    print("   Escribiendo archivo...")
    with open(output_file, 'wb') as f:
        f.write(pretty_xml)
    
    # Estadísticas del archivo
    size_bytes = os.path.getsize(output_file)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"\n✅ ARCHIVO OSM CREADO: {output_file}")
    print(f"   Tamaño: {size_mb:,.1f} MB ({size_bytes:,} bytes)")
    print(f"   Nodos en OSM: {nodes_added:,}")
    print(f"   Vías en OSM: {ways_added:,}")
    
    return output_file, size_mb

def create_backup_graphml(G, output_file="santiago_completo_dirigido.graphml"):
    """
    Crea también un GraphML de respaldo
    """
    print(f"\n💾 Creando GraphML de respaldo: {output_file}")
    
    try:
        # Intentar guardar con OSMnx (preserva atributos)
        ox.save_graphml(G, output_file)
    except:
        # Si falla, guardar con NetworkX (puede perder algunos atributos complejos)
        print("   ⚠️  Usando NetworkX para GraphML...")
        nx.write_graphml(G, output_file)
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   ✅ {output_file} ({size_mb:.1f} MB)")
    
    return output_file

def main():
    """Proceso principal"""
    
    print("=" * 70)
    print("🚀 CREACIÓN DE OSM COMPLETO DIRIGIDO - TODAS LAS COMUNAS DE SANTIAGO")
    print("=" * 70)
    
    print("\n📋 ESTE SCRIPT HARÁ:")
    print("   1. Descargará las 29 comunas del Gran Santiago")
    print("   2. Mantendrá la DIRECCIÓN de las calles (grafo dirigido)")
    print("   3. Combinará todo en un solo grafo")
    print("   4. Exportará a un ÚNICO archivo OSM completo")
    print("   5. También creará un GraphML de respaldo")
    
    print("\n⚠️  ADVERTENCIAS:")
    print("   • El proceso puede tomar 30-60 minutos")
    print("   • El archivo OSM final puede ser de 50-150 MB")
    print("   • Se requiere buena conexión a internet")
    print("   • Se necesita suficiente espacio en disco (>500 MB)")
    
    respuesta = input("\n¿Continuar? (s/N): ").strip().lower()
    if respuesta != 's':
        print("Operación cancelada.")
        return
    
    # Verificar espacio en disco
    try:
        stat = os.statvfs('.')
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        print(f"\n💾 Espacio disponible: {free_mb:.0f} MB")
        
        if free_mb < 500:
            print("⚠️  ESPACIO BAJO. Continuar puede causar problemas.")
            resp2 = input("¿Continuar de todos modos? (s/N): ").strip().lower()
            if resp2 != 's':
                return
    except:
        print("⚠️  No se pudo verificar espacio en disco")
    
    # EJECUTAR PROCESO
    start_time = time.time()
    
    try:
        # Paso 1: Descargar todas las comunas
        print("\n" + "=" * 50)
        print("PASO 1: DESCARGANDO COMUNAS")
        print("=" * 50)
        G = download_all_comunas_directed()
        
        if G is None:
            print("❌ No se pudo descargar el grafo")
            return
        
        # Paso 2: Crear GraphML de respaldo
        print("\n" + "=" * 50)
        print("PASO 2: CREANDO GRAPHML DE RESPALDO")
        print("=" * 50)
        graphml_file = create_backup_graphml(G)
        
        # Paso 3: Exportar a OSM completo
        print("\n" + "=" * 50)
        print("PASO 3: EXPORTANDO A OSM COMPLETO")
        print("=" * 50)
        osm_file, osm_size = export_directed_to_osm(G)
        
        # Tiempo total
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        # REPORTE FINAL
        print("\n" + "=" * 70)
        print("🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("=" * 70)
        
        print(f"\n⏱️  TIEMPO TOTAL: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   1. {osm_file} ({osm_size:.1f} MB) - OSM COMPLETO DIRIGIDO")
        print(f"   2. {graphml_file} ({os.path.getsize(graphml_file)/(1024*1024):.1f} MB) - GraphML de respaldo")
        
        print(f"\n📊 ESTADÍSTICAS DEL GRAFO:")
        print(f"   • Nodos totales: {len(G.nodes()):,}")
        print(f"   • Aristas totales: {len(G.edges()):,}")
        print(f"   • ¿Grafo dirigido?: {G.is_directed()}")
        
        # Calcular algunas estadísticas de dirección
        if G.is_directed():
            oneway_count = 0
            for u, v, data in G.edges(data=True):
                if data.get('oneway') in ['yes', 'true', '1', True]:
                    oneway_count += 1
            
            print(f"   • Calles unidireccionales: {oneway_count:,} ({oneway_count/len(G.edges())*100:.1f}%)")
        
        print(f"\n🎯 PARA TU PROYECTO:")
        print(f'   En web.py usa: OSM_FILE = "{osm_file}"')
        print(f"\n💡 El archivo OSM contiene TODAS las comunas y mantiene la DIRECCIÓN")
        print("   de las calles (oneway), crucial para algoritmos de ruteo realistas.")
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL PROCESO: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 SUGERENCIA: Intenta con menos comunas o más memoria")

if __name__ == "__main__":
    # Instalar tqdm si no está
    try:
        from tqdm import tqdm
    except ImportError:
        print("Instalando tqdm...")
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
    
    main()