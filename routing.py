"""
Módulo de ruteo (en español).

Implementa Dijkstra y A* (A-estrella) sobre el grafo definido en `classes`.

Algoritmos incluidos:
- `dijkstra`: búsqueda de caminos con costes no negativos. Devuelve distancias y predecesores.
- `a_estrella`: A* usando sólo la distancia geográfica (Haversine) como heurística admisible.

Parámetros de coste utilizados por las funciones:
- `w_dist`: peso de la componente de distancia (metros).
- `w_elev`: peso de la componente de ganancia de altura positiva (metros).
- `w_seg`: peso de la componente de seguridad (se usa `nodo.prob_accidente` y `camino.importancia`).

Funciones pendientes / mejoras posibles:
- Incluir en la heurística de A* componentes relacionadas con elevación o seguridad (ahora sólo se usa distancia, para mantener heurística admisible).
- Añadir versiones vectorizadas o estructuras de datos más optimizadas para grafos muy grandes.
"""

from typing import Dict, Tuple, List, Optional
import math
import heapq

from nodo import Nodo
from camino import Camino


def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos lat/lon."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def coste_arista(camino: Camino, nodo_origen: Nodo, nodo_destino: Nodo,
                 w_dist: float = 1.0, w_elev: float = 0.0, w_seg: float = 0.0) -> float:
    """Calcula el coste de recorrer `camino` desde `nodo_origen` hacia `nodo_destino`.

    Componentes:
    - distancia (metros)
    - ganancia de altura positiva (si destino está más alto que origen)
    - seguridad: `nodo_destino.prob_accidente` penalizado por `camino.importancia`
    """
    distancia = distancia_haversine(nodo_origen.latitud, nodo_origen.longitud,
                                     nodo_destino.latitud, nodo_destino.longitud)
    ganancia_altura = max(0.0, nodo_destino.altura - nodo_origen.altura)
    seguridad = nodo_destino.prob_accidente * (1.0 / max(1.0, camino.importancia))

    return w_dist * distancia + w_elev * ganancia_altura + w_seg * seguridad


def dijkstra(grafo, inicio_id: int, objetivo_id: Optional[int] = None, *,
             w_dist: float = 1.0, w_elev: float = 0.0, w_seg: float = 0.0, **kwargs) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    # Compatibilidad con keyword `goal_id` usado en llamadas anteriores
    if 'goal_id' in kwargs and objetivo_id is None:
        objetivo_id = kwargs.get('goal_id')
    """Dijkstra clásico. Devuelve (distancias, predecesores).

    Si se pasa `objetivo_id`, la búsqueda termina cuando se asienta ese nodo.
    """
    dist: Dict[int, float] = {nid: math.inf for nid in grafo.nodos}
    prev: Dict[int, Optional[int]] = {nid: None for nid in grafo.nodos}
    dist[inicio_id] = 0.0

    cola: List[Tuple[float, int]] = [(0.0, inicio_id)]

    while cola:
        d_act, nid = heapq.heappop(cola)
        if d_act > dist[nid]:
            continue
        if objetivo_id is not None and nid == objetivo_id:
            break

        nodo = grafo.nodos[nid]
        for camino in nodo.caminos:
            otro = camino.obtener_otro_nodo(nodo)
            if otro is None:
                continue
            c = coste_arista(camino, nodo, otro, w_dist=w_dist, w_elev=w_elev, w_seg=w_seg)
            nd = d_act + c
            if nd < dist[otro.id]:
                dist[otro.id] = nd
                prev[otro.id] = nid
                heapq.heappush(cola, (nd, otro.id))

    return dist, prev


def reconstruir_camino(prev: Dict[int, Optional[int]], inicio_id: int, objetivo_id: int) -> List[int]:
    camino: List[int] = []
    actual = objetivo_id
    while actual is not None:
        camino.append(actual)
        if actual == inicio_id:
            break
        actual = prev.get(actual)
    return list(reversed(camino))


def a_estrella(grafo, inicio_id: int, objetivo_id: int,
               w_dist: float = 1.0, w_elev: float = 0.0, w_seg: float = 0.0) -> Optional[List[int]]:
    """A* (A-estrella) usando la distancia geográfica como heurística admisible.

    La heurística usa sólo la componente de distancia ponderada por `w_dist`.
    """
    inicio = grafo.nodos[inicio_id]
    objetivo = grafo.nodos[objetivo_id]

    abierto: List[Tuple[float, int]] = []
    gscore: Dict[int, float] = {nid: math.inf for nid in grafo.nodos}
    fscore: Dict[int, float] = {nid: math.inf for nid in grafo.nodos}
    viene_de: Dict[int, Optional[int]] = {nid: None for nid in grafo.nodos}

    gscore[inicio_id] = 0.0
    h0 = w_dist * distancia_haversine(inicio.latitud, inicio.longitud, objetivo.latitud, objetivo.longitud)
    fscore[inicio_id] = h0
    heapq.heappush(abierto, (fscore[inicio_id], inicio_id))

    while abierto:
        _, actual_id = heapq.heappop(abierto)
        if actual_id == objetivo_id:
            return reconstruir_camino(viene_de, inicio_id, objetivo_id)

        actual = grafo.nodos[actual_id]
        for camino in actual.caminos:
            vecino = camino.obtener_otro_nodo(actual)
            if vecino is None:
                continue
            tent_g = gscore[actual_id] + coste_arista(camino, actual, vecino, w_dist=w_dist, w_elev=w_elev, w_seg=w_seg)
            if tent_g < gscore[vecino.id]:
                viene_de[vecino.id] = actual_id
                gscore[vecino.id] = tent_g
                h = w_dist * distancia_haversine(vecino.latitud, vecino.longitud, objetivo.latitud, objetivo.longitud)
                fscore[vecino.id] = tent_g + h
                heapq.heappush(abierto, (fscore[vecino.id], vecino.id))

    return None


# Alias de retrocompatibilidad (inglés)
astar = a_estrella
haversine = distancia_haversine
edge_cost = coste_arista


def asignar_indicador_seguridad(grafo, scores: Dict[str, float], nodo_attr: str = 'comuna') -> None:
    """Asigna valores de seguridad a los nodos del grafo.

    - `scores`: dict donde las claves son nombres de grupo (p. ej. 'RENCA') y los valores el score.
    - `nodo_attr`: nombre del atributo en cada `Nodo` que contiene el grupo (por defecto 'comuna').

    La búsqueda del atributo es flexible: se toma el valor del atributo si existe
    (puede añadirse dinámicamente a los nodos) y se hace match en mayúsculas.
    Si no se encuentra un valor para un nodo, no se modifica su `prob_accidente`.
    """
    # Normalizar claves de scores
    scores_norm = {str(k).strip().upper(): float(v) for k, v in scores.items()}

    for nodo in grafo.nodos.values():
        grupo = None
        if hasattr(nodo, nodo_attr):
            grupo = getattr(nodo, nodo_attr)
        else:
            # intentar minúscula por si el atributo fue seteado en otra forma
            if hasattr(nodo, nodo_attr.lower()):
                grupo = getattr(nodo, nodo_attr.lower())

        if grupo is None:
            continue

        key = str(grupo).strip().upper()
        if key in scores_norm:
            nodo.prob_accidente = scores_norm[key]