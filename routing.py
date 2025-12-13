"""
Módulo de ruteo (en español).

Implementa Dijkstra Lexicográfico (Distancia, Esfuerzo) y A* (A-estrella).

Costos para Dijkstra: (Distancia Plana Acumulada, Esfuerzo Efectivo Acumulado).
Costos para A*: Escalares (ponderados por w_dist, w_elev, w_seg).
"""

from typing import Dict, Tuple, List, Optional
import math
import heapq

# Asumo que nodo y camino están disponibles en el entorno de ejecución
from nodo import Nodo
from camino import Camino

# --- 1. Parámetros de Esfuerzo ---
# PENDIENTE_MAX = tan(15 grados)
PENDIENTE_MAX = math.tan(math.radians(15.0)) # Aprox. 0.2679

# Tipo de Costo Lexicográfico: (Distancia Plana Acumulada, Esfuerzo Efectivo Acumulado)
CostoLexicografico = Tuple[float, float]
INF_LEX = (math.inf, math.inf)


def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos lat/lon (distancia horizontal)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# --- Función de Costo de Tramo para Dijkstra Lexicográfico ---
def _costo_esfuerzo_tramo(nodo_origen: Nodo, nodo_destino: Nodo) -> Optional[CostoLexicografico]:
    """
    Calcula el costo lexicográfico de un tramo: (d_i, F_i).
    Retorna None si el tramo excede la pendiente máxima (15°).
    """
    
    d = distancia_haversine(nodo_origen.latitud, nodo_origen.longitud,
                            nodo_destino.latitud, nodo_destino.longitud)
    
    delta_h = nodo_destino.altura - nodo_origen.altura

    # 1. Filtro de pendiente máxima (solo subidas)
    # Se añade un pequeño epsilon (1e-6) para evitar división por cero en tramos muy cortos
    if d > 1e-6 and (delta_h / d) > PENDIENTE_MAX:
        return None # Tramo no transitable (pendiente > 15 grados)

    # 2. Esfuerzo por tramo (F_i)
    delta_h_pos = max(0.0, delta_h)
    
    if d < 1e-6:
        # Caso especial para nodos coincidentes o muy cercanos
        F_i = delta_h_pos
    else:
        # F_i = sqrt(d_i^2 + max(0, Delta h_i)^2)
        F_i = math.sqrt(d**2 + delta_h_pos**2)

    return (d, F_i)


def coste_arista(camino: Camino, nodo_origen: Nodo, nodo_destino: Nodo,
                 w_dist: float = 1.0, w_elev: float = 0.0, w_seg: float = 0.0) -> float:
    """Calcula el coste escalar, usado principalmente por el A* (antiguo)."""
    # Se mantiene la lógica original para que A* siga funcionando con pesos
    distancia = distancia_haversine(nodo_origen.latitud, nodo_origen.longitud,
                                     nodo_destino.latitud, nodo_destino.longitud)
    ganancia_altura = max(0.0, nodo_destino.altura - nodo_origen.altura)
    seguridad = nodo_destino.prob_accidente * (1.0 / max(1.0, camino.importancia))

    return w_dist * distancia + w_elev * ganancia_altura + w_seg * seguridad


def dijkstra(grafo, inicio_id: int, objetivo_id: Optional[int] = None, *,
             w_dist: float = 1.0, w_elev: float = 0.0, w_seg: float = 0.0, **kwargs) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Dijkstra SOBRESCRITO: Implementa la minimización lexicográfica (Distancia Plana, Esfuerzo).
    
    Devuelve SÓLO la Distancia Plana Total (float) para compatibilidad externa.
    """
    
    # Compatibilidad con keyword `goal_id` usado en llamadas anteriores
    if 'goal_id' in kwargs and objetivo_id is None:
        objetivo_id = kwargs.get('goal_id')
    
    # Costo Interno: (Distancia Plana Acumulada, Esfuerzo Efectivo Acumulado)
    dist_lex: Dict[int, CostoLexicografico] = {nid: INF_LEX for nid in grafo.nodos}
    prev: Dict[int, Optional[int]] = {nid: None for nid in grafo.nodos}
    dist_lex[inicio_id] = (0.0, 0.0)

    # Cola de prioridad que usa la comparación lexicográfica de tuplas (costo, id)
    cola: List[Tuple[CostoLexicografico, int]] = [((0.0, 0.0), inicio_id)]

    while cola:
        d_act_lex, nid = heapq.heappop(cola)
        
        if d_act_lex > dist_lex[nid]: # Comparación lexicográfica
            continue
        if objetivo_id is not None and nid == objetivo_id:
            break

        nodo = grafo.nodos[nid]
        for camino in nodo.caminos:
            otro = camino.obtener_otro_nodo(nodo)
            if otro is None:
                continue
            
            # Obtener el costo del tramo (d_i, F_i) con filtro de pendiente
            costo_tramo = _costo_esfuerzo_tramo(nodo, otro)
            
            if costo_tramo is None:
                continue 

            d_i, F_i = costo_tramo
            D_act, F_act = d_act_lex

            # Nuevo costo acumulado: (D_act + d_i, F_act + F_i)
            nd_i_lex = (D_act + d_i, F_act + F_i)
            
            if nd_i_lex < dist_lex[otro.id]: # Comparación lexicográfica
                dist_lex[otro.id] = nd_i_lex
                prev[otro.id] = nid
                heapq.heappush(cola, (nd_i_lex, otro.id))

    # COMPATIBILIDAD EXTERNA: Devolver solo la Distancia Plana total como float
    dist_escalar: Dict[int, float] = {}
    for nid, (D, F) in dist_lex.items():
        dist_escalar[nid] = D
        
    return dist_escalar, prev


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
    """A* (A-estrella) - Se mantiene la versión escalar original para compatibilidad."""
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
            
            # --- NOTA ---
            # Para A*, se recomienda añadir aquí el filtro de pendiente:
            # if _costo_esfuerzo_tramo(actual, vecino) is None: continue 
            # Se omite para mantener la lógica original del A* de tu código.
            
            coste_escalar = coste_arista(camino, actual, vecino, w_dist=w_dist, w_elev=w_elev, w_seg=w_seg)
            tent_g = gscore[actual_id] + coste_escalar
            
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
    # ... (Se mantiene igual)
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


# --- Función Adicional de Cálculo de Esfuerzo de Ruta ---
def porcentaje_esfuerzoruta(ruta: List[int], grafo) -> Optional[float]:
    """
    Calcula el porcentaje de subidas de una ruta.
    esfuerzo% = (sum(F_i) - sum(d_i)) / sum(d_i)
    """

    distancia_plana = 0.0
    distancia_efectiva = 0.0

    for i in range(len(ruta) - 1):
        u_id = ruta[i]
        v_id = ruta[i + 1]
        
        # Debe manejar el caso de nodos no encontrados (aunque en una ruta válida no debería pasar)
        if u_id not in grafo.nodos or v_id not in grafo.nodos:
            return None
            
        u = grafo.nodos[u_id]
        v = grafo.nodos[v_id]

        costo_tramo = _costo_esfuerzo_tramo(u, v)

        if costo_tramo is None:
            # Si un tramo es intransitable (pendiente > 15), la ruta es inválida
            return None

        d_i, F_i = costo_tramo
        distancia_plana += d_i
        distancia_efectiva += F_i

    if distancia_plana == 0:
        return None

    return (distancia_efectiva - distancia_plana) / distancia_plana