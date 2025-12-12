# osmlib.py
# ============================================
# Carga única del grafo OSM para uso global
# ============================================

import os
import osmnx as ox
from grafo import Grafo
from string_to_node import preparar_osm


# --------------------------------------------
# VARIABLES GLOBALES
# --------------------------------------------
G = None        # Aquí quedará el grafo cargado
NODOS = None    # Si quieres almacenar lista de nodos útiles
OSM_PATH = "map_clean.osm"


# --------------------------------------------
# FUNCIÓN PRINCIPAL: preparar y cargar OSM
# --------------------------------------------
def preparar_osm_archivo(osm_path: str = OSM_PATH):
    """
    Prepara el archivo OSM con tu proceso interno y carga el grafo global.
    Este método debe ejecutarse solo 1 vez al iniciar la app.
    """

    global G, NODOS

    print("🔄 Preparando archivo OSM...")
    preparar_osm(osm_path)

    print("📥 Cargando grafo desde OSM...")
    graph = ox.graph_from_xml(osm_path, simplify=False)

    print("🔧 Convirtiendo grafo a clase Grafo...")
    G = Grafo.desde_osmnx(graph)   # Ajusta este método a tu implementación real

    print("📌 Extrayendo nodos principales...")
    NODOS = list(G.grafo.nodes)

    print(f"✅ OSM cargado: {len(NODOS)} nodos totales")


# --------------------------------------------
# Inicialización automática al importar módulo
# --------------------------------------------
try:
    preparar_osm_archivo()
except Exception as e:
    print(f"⚠️ Error cargando OSM: {e}")
