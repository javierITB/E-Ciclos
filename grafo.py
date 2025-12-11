# grafo.py (MODIFICADO)
from typing import Dict, List, Optional  # Añadir List y Optional
from nodo import Nodo
from camino import Camino
import networkx as nx  # Añadir NetworkX


class Grafo:
    def __init__(self):
        self.nodos: Dict[int, Nodo] = {}
        self.caminos: Dict[int, Camino] = {}

    def agregar_nodo(self, id_nodo: int, lat: float, lon: float, alt: float = 0.0, prob_acc: float = 0.0) -> Nodo:
        if id_nodo not in self.nodos:
            # Crear y almacenar la instancia de Nodo
            self.nodos[id_nodo] = Nodo(id_nodo, lat, lon, alt, prob_acc)
        return self.nodos[id_nodo]

    def agregar_camino(self, id_camino: int, id_nodo_a: int, id_nodo_b: int, ciclovia: bool = False,
                       importancia: int = 1) -> Camino:
        if id_camino not in self.caminos:
            nodo_a = self.nodos[id_nodo_a]
            nodo_b = self.nodos[id_nodo_b]
            # La clase Camino se encarga de llamar a nodo.agregar_camino()
            self.caminos[id_camino] = Camino(id_camino, nodo_a, nodo_b, ciclovia, importancia)
        return self.caminos[id_camino]

    def cargar_desde_networkx(self, G_nx: nx.MultiDiGraph):
        """Inicializa el grafo a partir de un grafo NetworkX de OSMnx.
        
        NOTA: Crea caminos bidireccionales. Para cada arista (u, v) se crean
        dos caminos: u→v y v→u.
        """
        self.nodos.clear()
        self.caminos.clear()

        # 1. Crear Nodos
        for nid, data in G_nx.nodes(data=True):
            # Usamos los datos simulados de web.py o 0.0/0.1 por defecto
            altura = data.get('altitud_m', 400.0)
            prob_acc = data.get('peligrosidad', 0.1)

            self.agregar_nodo(
                id_nodo=nid,
                lat=data['y'],
                lon=data['x'],
                alt=altura,
                prob_acc=prob_acc
            )

        # 2. Crear Caminos (Aristas) BIDIRECCIONALES y establecer conexiones
        edge_id = 0
        for u, v, k, data in G_nx.edges(keys=True, data=True):
            # La importancia se puede simular o extraer de datos OSM si existen
            # Aquí usamos el largo de la arista (length) como una proxy simple para 'importancia'
            # (aunque el módulo routing.py espera un int, lo forzamos a 1 para simplicidad si no existe)
            importancia = int(data.get('length', 1) / 100) + 1

            # Crear camino en dirección u → v
            edge_id += 1
            self.agregar_camino(
                id_camino=edge_id,
                id_nodo_a=u,
                id_nodo_b=v,
                ciclovia=False,  # Este dato no está en el OSMnx base, se asume False
                importancia=importancia
            )

            # Crear camino en dirección v → u (BIDIRECCIONAL)
            edge_id += 1
            self.agregar_camino(
                id_camino=edge_id,
                id_nodo_a=v,
                id_nodo_b=u,
                ciclovia=False,
                importancia=importancia
            )

    def __repr__(self):
        return f"Grafo(nodos={len(self.nodos)}, caminos={len(self.caminos)})"