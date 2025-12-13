import pandas as pd
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
import numpy as np

# Cargar datos de elevación
df = pd.read_csv("alturas_santiago.csv")  # columnas: lat, lon, ele

# Crear un árbol KD para búsqueda rápida del vecino más cercano
coords = df[['lat', 'lon']].to_numpy()
elevs = df['ele'].to_numpy()
tree = cKDTree(coords)

# Cargar archivo OSM
tree_osm = ET.parse("map_clean.osm")
root = tree_osm.getroot()

# Iterar sobre todos los nodos y agregar elevación
for node in root.findall('node'):
    lat = float(node.attrib['lat'])
    lon = float(node.attrib['lon'])
    
    # Buscar índice del vecino más cercano
    dist, idx = tree.query([lat, lon])
    ele_approx = elevs[idx]
    
    # Agregar atributo 'ele' al nodo
    node.set('ele', f"{ele_approx:.3f}")

# Guardar nuevo archivo OSM con elevación
tree_osm.write("map_with_elevation.osm", encoding="utf-8", xml_declaration=True)

print("Archivo generado: map_with_elevation.osm")
