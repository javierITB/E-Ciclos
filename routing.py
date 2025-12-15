"""
Módulo de ruteo (en español).

Implementa Dijkstra y A* (A-estrella) sobre el grafo definido en `classes`.

Algoritmos incluidos:
- `dijkstra`: búsqueda de caminos con costes no negativos. Devuelve distancias y predecesores.
- `a_estrella`: A* usando la distancia geográfica (Haversine) como heurística admisible.
- `a_estrella_multiruta`: Genera múltiples rutas optimizando diferentes criterios (distancia, seguridad, esfuerzo).

Parámetros de coste utilizados por las funciones:
- `w_dist`: peso de la componente de distancia (metros).
- `w_elev`: peso de la componente de ganancia de altura positiva (metros).
- `w_seg`: peso de la componente de seguridad (se usa `nodo.prob_accidente` y `camino.importancia`).

Filtros disponibles:
- `filtro_seguridad`: Filtra rutas por índice de seguridad mínimo.
- `filtro_esfuerzo`: Filtra rutas por esfuerzo máximo (basado en elevación).

Funciones de utilidad:
- `calcular_indicador_seguridad_desde_excel`: Lee datos de accidentes y calcula índice de seguridad.
- `normalizar_scores`: Normaliza scores de seguridad al rango [0,1].
- `asignar_indicador_seguridad`: Asigna valores de seguridad a los nodos del grafo.

Última actualización: Diciembre 2025 - Integración de índice de seguridad desde GrafosYDosRuedas.
"""

from typing import Dict, Tuple, List, Optional, Callable
from enum import Enum
import math
import heapq
import os

from nodo import Nodo
from camino import Camino


# =============================================================================
# ENUMERACIONES Y CONSTANTES PARA CONFIGURACIÓN
# =============================================================================

class TipoFiltro(Enum):
    """
    Tipos de filtro disponibles para la búsqueda de rutas.
    
    - NINGUNO: Sin filtro adicional.
    - SEGURIDAD: Filtra por índice de seguridad (evita zonas peligrosas).
    - ESFUERZO: Filtra por cálculo de esfuerzo físico (evita pendientes).
    - SEGURIDAD_Y_ESFUERZO: Combina ambos filtros.
    """
    NINGUNO = "ninguno"
    SEGURIDAD = "seguridad"
    ESFUERZO = "esfuerzo"
    SEGURIDAD_Y_ESFUERZO = "seguridad_y_esfuerzo"


class TipoRuta(Enum):
    """
    Tipos de optimización de ruta disponibles.
    
    - DISTANCIA: Optimiza por distancia ponderada (comportamiento original).
    - PELIGROSIDAD: Optimiza minimizando el índice de peligrosidad.
    - ESFUERZO: Optimiza minimizando el esfuerzo físico (elevación).
    - BALANCEADA: Balancea distancia, seguridad y esfuerzo.
    """
    DISTANCIA = "distancia"
    PELIGROSIDAD = "peligrosidad"
    ESFUERZO = "esfuerzo"
    BALANCEADA = "balanceada"


# Ruta por defecto al archivo Excel de índice de seguridad
RUTA_EXCEL_SEGURIDAD_DEFAULT = os.path.join(
    os.path.dirname(__file__), 'data', 'os2_sin_2025_08.xlsx'
)


# =============================================================================
# FUNCIONES DE CÁLCULO DE ÍNDICE DE SEGURIDAD (desde GrafosYDosRuedas)
# =============================================================================

def calcular_indicador_seguridad_desde_excel(
    ruta_excel: str = None, 
    agrupar_por: str = 'COMUNA'
) -> Dict[str, float]:
    """
    Lee un archivo Excel de accidentes y devuelve un diccionario
    {grupo: indicador_seguridad} donde `grupo` es el valor de la columna `agrupar_por`.

    Indicador propuesto (simple y explicable):
    - Pondera más los eventos con fallecidos y lesiones graves.
    - indicador = (5*FALLECIDO + 3*GRAVE + 2*M/GRAVE + 1*LEVE) / sqrt(numero_eventos)

    La raíz cuadrada en el denominador reduce la penalización de zonas con muchos eventos,
    ayudando a estabilizar el índice.
    
    Args:
        ruta_excel: Ruta al archivo Excel con datos de accidentes.
                    Si es None, usa RUTA_EXCEL_SEGURIDAD_DEFAULT.
        agrupar_por: Columna por la cual agrupar los datos (default: 'COMUNA').
    
    Returns:
        Dict[str, float]: Diccionario con el indicador de seguridad por grupo.
    
    Raises:
        RuntimeError: Si pandas no está instalado.
        FileNotFoundError: Si el archivo Excel no existe.
    """
    if ruta_excel is None:
        ruta_excel = RUTA_EXCEL_SEGURIDAD_DEFAULT
    
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError('pandas es requerido para procesar el Excel. Instalar con: pip install pandas openpyxl') from e

    if not os.path.exists(ruta_excel):
        raise FileNotFoundError(f"No se encontró el archivo Excel de seguridad: {ruta_excel}")

    df = pd.read_excel(ruta_excel, dtype={agrupar_por: str})
    # Normalizar nombres para agrupamiento
    df[agrupar_por] = df[agrupar_por].astype(str).str.strip().str.upper()

    # Asegurar columnas numéricas
    for col in ['FALLECIDO', 'GRAVE', 'M/GRAVE', 'LEVE']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    grupos = df.groupby(agrupar_por)
    indicador: Dict[str, float] = {}

    for nombre, grupo in grupos:
        eventos = len(grupo)
        peso = (5 * grupo['FALLECIDO'].sum() +
                3 * grupo['GRAVE'].sum() +
                2 * grupo['M/GRAVE'].sum() +
                1 * grupo['LEVE'].sum())
        if eventos == 0:
            indicador[nombre] = 0.0
        else:
            indicador[nombre] = float(peso) / (eventos ** 0.5)

    return indicador


def normalizar_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza los scores a rango [0,1], donde 1 = más inseguro.
    Si todos los scores son 0, retorna todos 0.
    
    Args:
        scores: Diccionario con scores sin normalizar.
    
    Returns:
        Dict[str, float]: Diccionario con scores normalizados en [0,1].
    """
    if not scores:
        return {}
    vals = list(scores.values())
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


# =============================================================================
# FUNCIONES DE DISTANCIA Y COSTE
# =============================================================================

def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos lat/lon."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def calcular_esfuerzo(nodo_origen: Nodo, nodo_destino: Nodo, distancia: float) -> float:
    """
    Calcula el esfuerzo físico estimado para recorrer un tramo.
    
    El esfuerzo considera:
    - Ganancia de altura positiva (subidas).
    - Pendiente del tramo (porcentaje).
    
    Args:
        nodo_origen: Nodo de inicio del tramo.
        nodo_destino: Nodo de destino del tramo.
        distancia: Distancia horizontal en metros.
    
    Returns:
        float: Valor de esfuerzo (mayor = más difícil).
    """
    ganancia_altura = max(0.0, nodo_destino.altura - nodo_origen.altura)
    if distancia > 0:
        # Pendiente como porcentaje (ganancia / distancia horizontal * 100)
        pendiente = (ganancia_altura / distancia) * 100
        # Penalización extra para pendientes pronunciadas (>5%)
        factor_pendiente = 1.0 + max(0, pendiente - 5) * 0.1
    else:
        factor_pendiente = 1.0
    
    # Esfuerzo = ganancia de altura + penalización por pendiente
    return ganancia_altura * factor_pendiente


def coste_arista(
    camino: Camino, 
    nodo_origen: Nodo, 
    nodo_destino: Nodo,
    w_dist: float = 1.0, 
    w_elev: float = 0.0, 
    w_seg: float = 0.0,
    tipo_ruta: TipoRuta = TipoRuta.DISTANCIA
) -> float:
    """
    Calcula el coste de recorrer `camino` desde `nodo_origen` hacia `nodo_destino`.

    Componentes:
    - distancia (metros)
    - ganancia de altura positiva (si destino está más alto que origen)
    - seguridad: `nodo_destino.prob_accidente` penalizado por `camino.importancia`
    
    El tipo de ruta modifica cómo se ponderan las componentes:
    - DISTANCIA: Prioriza distancia mínima.
    - PELIGROSIDAD: Prioriza evitar zonas con alto índice de peligrosidad.
    - ESFUERZO: Prioriza minimizar el esfuerzo físico.
    - BALANCEADA: Balancea todas las componentes.
    
    Args:
        camino: Camino/arista a evaluar.
        nodo_origen: Nodo de inicio.
        nodo_destino: Nodo de destino.
        w_dist: Peso de la componente de distancia.
        w_elev: Peso de la componente de elevación.
        w_seg: Peso de la componente de seguridad.
        tipo_ruta: Tipo de optimización de ruta.
    
    Returns:
        float: Coste calculado de la arista.
    """
    distancia = distancia_haversine(
        nodo_origen.latitud, nodo_origen.longitud,
        nodo_destino.latitud, nodo_destino.longitud
    )
    ganancia_altura = max(0.0, nodo_destino.altura - nodo_origen.altura)
    
    # Componente de seguridad basada en prob_accidente del nodo destino
    # y la importancia del camino (caminos más importantes = más tráfico = más riesgo)
    importancia_camino = getattr(camino, 'importancia', 1.0)
    seguridad = nodo_destino.prob_accidente * (1.0 / max(1.0, importancia_camino))
    
    # Ajustar pesos según el tipo de ruta
    if tipo_ruta == TipoRuta.PELIGROSIDAD:
        # Prioriza seguridad: aumenta peso de seguridad
        w_dist_eff = w_dist * 0.3
        w_elev_eff = w_elev * 0.3
        w_seg_eff = max(w_seg, 5.0)  # Mínimo peso de seguridad = 5
    elif tipo_ruta == TipoRuta.ESFUERZO:
        # Prioriza minimizar esfuerzo: aumenta peso de elevación
        w_dist_eff = w_dist * 0.5
        w_elev_eff = max(w_elev, 3.0)  # Mínimo peso de elevación = 3
        w_seg_eff = w_seg * 0.5
    elif tipo_ruta == TipoRuta.BALANCEADA:
        # Balance entre todas las componentes
        w_dist_eff = max(w_dist, 1.0)
        w_elev_eff = max(w_elev, 1.0)
        w_seg_eff = max(w_seg, 1.0)
    else:
        # DISTANCIA: comportamiento original
        w_dist_eff = w_dist
        w_elev_eff = w_elev
        w_seg_eff = w_seg

    return w_dist_eff * distancia + w_elev_eff * ganancia_altura + w_seg_eff * seguridad


# =============================================================================
# FUNCIONES DE FILTRADO
# =============================================================================

def filtro_seguridad(
    nodo: Nodo, 
    umbral_seguridad: float = 0.5
) -> bool:
    """
    Filtro por índice de seguridad.
    
    Retorna True si el nodo es seguro (bajo índice de peligrosidad).
    
    Args:
        nodo: Nodo a evaluar.
        umbral_seguridad: Umbral máximo de prob_accidente permitido (default: 0.5).
                          Valores más bajos = más restrictivo.
    
    Returns:
        bool: True si el nodo pasa el filtro de seguridad.
    """
    return nodo.prob_accidente <= umbral_seguridad


def filtro_esfuerzo(
    nodo_origen: Nodo, 
    nodo_destino: Nodo, 
    umbral_pendiente: float = 8.0
) -> bool:
    """
    Filtro por cálculo de esfuerzo (pendiente máxima).
    
    Retorna True si el tramo tiene una pendiente aceptable.
    
    Args:
        nodo_origen: Nodo de inicio.
        nodo_destino: Nodo de destino.
        umbral_pendiente: Pendiente máxima permitida en porcentaje (default: 8%).
    
    Returns:
        bool: True si el tramo pasa el filtro de esfuerzo.
    """
    distancia = distancia_haversine(
        nodo_origen.latitud, nodo_origen.longitud,
        nodo_destino.latitud, nodo_destino.longitud
    )
    if distancia < 1.0:  # Evitar división por cero
        return True
    
    ganancia_altura = nodo_destino.altura - nodo_origen.altura
    pendiente = abs(ganancia_altura / distancia) * 100
    
    return pendiente <= umbral_pendiente


def aplicar_filtros(
    nodo_origen: Nodo,
    nodo_destino: Nodo,
    tipo_filtro: TipoFiltro = TipoFiltro.NINGUNO,
    umbral_seguridad: float = 0.5,
    umbral_pendiente: float = 8.0
) -> bool:
    """
    Aplica los filtros seleccionados por el usuario.
    
    Args:
        nodo_origen: Nodo de inicio del tramo.
        nodo_destino: Nodo de destino del tramo.
        tipo_filtro: Tipo de filtro a aplicar.
        umbral_seguridad: Umbral para filtro de seguridad.
        umbral_pendiente: Umbral para filtro de esfuerzo.
    
    Returns:
        bool: True si el tramo pasa todos los filtros seleccionados.
    """
    if tipo_filtro == TipoFiltro.NINGUNO:
        return True
    
    if tipo_filtro == TipoFiltro.SEGURIDAD:
        return filtro_seguridad(nodo_destino, umbral_seguridad)
    
    if tipo_filtro == TipoFiltro.ESFUERZO:
        return filtro_esfuerzo(nodo_origen, nodo_destino, umbral_pendiente)
    
    if tipo_filtro == TipoFiltro.SEGURIDAD_Y_ESFUERZO:
        return (filtro_seguridad(nodo_destino, umbral_seguridad) and 
                filtro_esfuerzo(nodo_origen, nodo_destino, umbral_pendiente))
    
    return True


# =============================================================================
# ALGORITMOS DE BÚSQUEDA
# =============================================================================


def dijkstra(
    grafo, 
    inicio_id: int, 
    objetivo_id: Optional[int] = None, 
    *,
    w_dist: float = 1.0, 
    w_elev: float = 0.0, 
    w_seg: float = 0.0,
    tipo_filtro: TipoFiltro = TipoFiltro.NINGUNO,
    tipo_ruta: TipoRuta = TipoRuta.DISTANCIA,
    umbral_seguridad: float = 0.5,
    umbral_pendiente: float = 8.0,
    **kwargs
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Dijkstra clásico mejorado con filtros configurables.
    
    Devuelve (distancias, predecesores).
    Si se pasa `objetivo_id`, la búsqueda termina cuando se asienta ese nodo.
    
    Args:
        grafo: Grafo sobre el cual buscar.
        inicio_id: ID del nodo inicial.
        objetivo_id: ID del nodo objetivo (opcional).
        w_dist: Peso de la componente de distancia.
        w_elev: Peso de la componente de elevación.
        w_seg: Peso de la componente de seguridad.
        tipo_filtro: Tipo de filtro a aplicar (NINGUNO, SEGURIDAD, ESFUERZO, SEGURIDAD_Y_ESFUERZO).
        tipo_ruta: Tipo de optimización de ruta (DISTANCIA, PELIGROSIDAD, ESFUERZO, BALANCEADA).
        umbral_seguridad: Umbral para filtro de seguridad (default: 0.5).
        umbral_pendiente: Umbral para filtro de esfuerzo (default: 8.0%).
    
    Returns:
        Tuple[Dict[int, float], Dict[int, Optional[int]]]: Diccionarios de distancias y predecesores.
    """
    # Compatibilidad con keyword `goal_id` usado en llamadas anteriores
    if 'goal_id' in kwargs and objetivo_id is None:
        objetivo_id = kwargs.get('goal_id')
    
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
            
            # Aplicar filtros configurables
            if not aplicar_filtros(nodo, otro, tipo_filtro, umbral_seguridad, umbral_pendiente):
                continue
            
            c = coste_arista(
                camino, nodo, otro, 
                w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
                tipo_ruta=tipo_ruta
            )
            nd = d_act + c
            if nd < dist[otro.id]:
                dist[otro.id] = nd
                prev[otro.id] = nid
                heapq.heappush(cola, (nd, otro.id))

    return dist, prev


def reconstruir_camino(prev: Dict[int, Optional[int]], inicio_id: int, objetivo_id: int) -> List[int]:
    """
    Reconstruye el camino desde inicio hasta objetivo usando el diccionario de predecesores.
    
    Args:
        prev: Diccionario de predecesores.
        inicio_id: ID del nodo inicial.
        objetivo_id: ID del nodo objetivo.
    
    Returns:
        List[int]: Lista de IDs de nodos desde inicio hasta objetivo.
    """
    camino: List[int] = []
    actual = objetivo_id
    while actual is not None:
        camino.append(actual)
        if actual == inicio_id:
            break
        actual = prev.get(actual)
    return list(reversed(camino))


def calcular_heuristica(
    nodo: Nodo, 
    objetivo: Nodo, 
    w_dist: float = 1.0,
    tipo_ruta: TipoRuta = TipoRuta.DISTANCIA
) -> float:
    """
    Calcula la heurística para A* considerando el tipo de ruta.
    
    La heurística debe ser admisible (no sobreestimar el coste real).
    Para DISTANCIA usa solo la distancia Haversine.
    Para otros tipos, ajusta la heurística manteniendo admisibilidad.
    
    Args:
        nodo: Nodo actual.
        objetivo: Nodo objetivo.
        w_dist: Peso de la distancia.
        tipo_ruta: Tipo de optimización de ruta.
    
    Returns:
        float: Valor heurístico estimado.
    """
    dist_base = distancia_haversine(
        nodo.latitud, nodo.longitud, 
        objetivo.latitud, objetivo.longitud
    )
    
    if tipo_ruta == TipoRuta.PELIGROSIDAD:
        # Para peligrosidad, reducimos el peso de distancia en la heurística
        # para mantener admisibilidad (nunca sobreestimar)
        return w_dist * 0.3 * dist_base
    elif tipo_ruta == TipoRuta.ESFUERZO:
        # Para esfuerzo, usamos distancia reducida
        return w_dist * 0.5 * dist_base
    elif tipo_ruta == TipoRuta.BALANCEADA:
        # Para balanceada, usamos la distancia completa
        return w_dist * dist_base
    else:
        # DISTANCIA: heurística original
        return w_dist * dist_base


def a_estrella(
    grafo, 
    inicio_id: int, 
    objetivo_id: int,
    w_dist: float = 1.0, 
    w_elev: float = 0.0, 
    w_seg: float = 0.0,
    tipo_filtro: TipoFiltro = TipoFiltro.NINGUNO,
    tipo_ruta: TipoRuta = TipoRuta.DISTANCIA,
    umbral_seguridad: float = 0.5,
    umbral_pendiente: float = 8.0
) -> Optional[List[int]]:
    """
    A* (A-estrella) mejorado con índice de seguridad y filtros configurables.
    
    Usa la distancia geográfica (Haversine) como heurística admisible base,
    ajustada según el tipo de ruta seleccionado.
    
    Args:
        grafo: Grafo sobre el cual buscar.
        inicio_id: ID del nodo inicial.
        objetivo_id: ID del nodo objetivo.
        w_dist: Peso de la componente de distancia (default: 1.0).
        w_elev: Peso de la componente de elevación/esfuerzo (default: 0.0).
        w_seg: Peso de la componente de seguridad (default: 0.0).
        tipo_filtro: Tipo de filtro a aplicar:
            - NINGUNO: Sin filtro adicional.
            - SEGURIDAD: Filtra nodos con alto índice de peligrosidad.
            - ESFUERZO: Filtra tramos con pendiente excesiva.
            - SEGURIDAD_Y_ESFUERZO: Combina ambos filtros.
        tipo_ruta: Tipo de optimización de ruta:
            - DISTANCIA: Prioriza distancia mínima (comportamiento original).
            - PELIGROSIDAD: Prioriza evitar zonas peligrosas.
            - ESFUERZO: Prioriza minimizar esfuerzo físico.
            - BALANCEADA: Balancea todas las componentes.
        umbral_seguridad: Umbral máximo de prob_accidente permitido (default: 0.5).
        umbral_pendiente: Pendiente máxima permitida en % (default: 8.0).
    
    Returns:
        Optional[List[int]]: Lista de IDs de nodos del camino, o None si no existe.
    
    Ejemplo de uso:
        # Ruta optimizando seguridad con filtro de pendiente
        ruta = a_estrella(
            grafo, inicio, objetivo,
            tipo_filtro=TipoFiltro.ESFUERZO,
            tipo_ruta=TipoRuta.PELIGROSIDAD,
            umbral_pendiente=5.0
        )
    """
    if inicio_id not in grafo.nodos or objetivo_id not in grafo.nodos:
        return None
    
    inicio = grafo.nodos[inicio_id]
    objetivo = grafo.nodos[objetivo_id]

    abierto: List[Tuple[float, int]] = []
    gscore: Dict[int, float] = {nid: math.inf for nid in grafo.nodos}
    fscore: Dict[int, float] = {nid: math.inf for nid in grafo.nodos}
    viene_de: Dict[int, Optional[int]] = {nid: None for nid in grafo.nodos}

    gscore[inicio_id] = 0.0
    h0 = calcular_heuristica(inicio, objetivo, w_dist, tipo_ruta)
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
            
            # Aplicar filtros configurables por el usuario
            if not aplicar_filtros(actual, vecino, tipo_filtro, umbral_seguridad, umbral_pendiente):
                continue
            
            # Calcular coste de arista con tipo de ruta
            tent_g = gscore[actual_id] + coste_arista(
                camino, actual, vecino, 
                w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
                tipo_ruta=tipo_ruta
            )
            
            if tent_g < gscore[vecino.id]:
                viene_de[vecino.id] = actual_id
                gscore[vecino.id] = tent_g
                h = calcular_heuristica(vecino, objetivo, w_dist, tipo_ruta)
                fscore[vecino.id] = tent_g + h
                heapq.heappush(abierto, (fscore[vecino.id], vecino.id))

    return None


# =============================================================================
# FUNCIONES PARA GENERAR MÚLTIPLES RUTAS
# =============================================================================

def a_estrella_multiruta(
    grafo, 
    inicio_id: int, 
    objetivo_id: int,
    w_dist: float = 1.0, 
    w_elev: float = 0.0, 
    w_seg: float = 0.0,
    tipo_filtro: TipoFiltro = TipoFiltro.NINGUNO,
    umbral_seguridad: float = 0.5,
    umbral_pendiente: float = 8.0
) -> Dict[str, Optional[List[int]]]:
    """
    Genera múltiples rutas optimizando diferentes criterios.
    
    Calcula dos rutas principales:
    1. Ruta por distancia ponderada (comportamiento original/actual).
    2. Ruta ponderando el índice de peligrosidad (prioriza seguridad).
    
    Opcionalmente genera rutas adicionales según la configuración.
    
    Args:
        grafo: Grafo sobre el cual buscar.
        inicio_id: ID del nodo inicial.
        objetivo_id: ID del nodo objetivo.
        w_dist: Peso base de la componente de distancia.
        w_elev: Peso base de la componente de elevación.
        w_seg: Peso base de la componente de seguridad.
        tipo_filtro: Tipo de filtro a aplicar a todas las rutas.
        umbral_seguridad: Umbral para filtro de seguridad.
        umbral_pendiente: Umbral para filtro de esfuerzo.
    
    Returns:
        Dict[str, Optional[List[int]]]: Diccionario con las rutas generadas:
            - 'ruta_distancia': Ruta optimizada por distancia.
            - 'ruta_segura': Ruta optimizada por seguridad (menor peligrosidad).
            - 'ruta_esfuerzo': Ruta optimizada por menor esfuerzo físico.
            - 'ruta_balanceada': Ruta balanceada entre todos los criterios.
    
    Ejemplo de uso:
        rutas = a_estrella_multiruta(grafo, inicio, objetivo)
        
        # Obtener ruta más corta
        ruta_corta = rutas['ruta_distancia']
        
        # Obtener ruta más segura
        ruta_segura = rutas['ruta_segura']
    """
    rutas: Dict[str, Optional[List[int]]] = {}
    
    # 1. Ruta por distancia ponderada (comportamiento actual/original)
    rutas['ruta_distancia'] = a_estrella(
        grafo, inicio_id, objetivo_id,
        w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
        tipo_filtro=tipo_filtro,
        tipo_ruta=TipoRuta.DISTANCIA,
        umbral_seguridad=umbral_seguridad,
        umbral_pendiente=umbral_pendiente
    )
    
    # 2. Ruta ponderando el índice de peligrosidad (prioriza seguridad)
    rutas['ruta_segura'] = a_estrella(
        grafo, inicio_id, objetivo_id,
        w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
        tipo_filtro=tipo_filtro,
        tipo_ruta=TipoRuta.PELIGROSIDAD,
        umbral_seguridad=umbral_seguridad,
        umbral_pendiente=umbral_pendiente
    )
    
    # 3. Ruta optimizada por esfuerzo físico
    rutas['ruta_esfuerzo'] = a_estrella(
        grafo, inicio_id, objetivo_id,
        w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
        tipo_filtro=tipo_filtro,
        tipo_ruta=TipoRuta.ESFUERZO,
        umbral_seguridad=umbral_seguridad,
        umbral_pendiente=umbral_pendiente
    )
    
    # 4. Ruta balanceada
    rutas['ruta_balanceada'] = a_estrella(
        grafo, inicio_id, objetivo_id,
        w_dist=w_dist, w_elev=w_elev, w_seg=w_seg,
        tipo_filtro=tipo_filtro,
        tipo_ruta=TipoRuta.BALANCEADA,
        umbral_seguridad=umbral_seguridad,
        umbral_pendiente=umbral_pendiente
    )
    
    return rutas


def calcular_metricas_ruta(
    grafo, 
    ruta: List[int]
) -> Dict[str, float]:
    """
    Calcula métricas de una ruta para comparación.
    
    Args:
        grafo: Grafo con los nodos.
        ruta: Lista de IDs de nodos de la ruta.
    
    Returns:
        Dict[str, float]: Diccionario con métricas:
            - 'distancia_total': Distancia total en metros.
            - 'ganancia_altura': Ganancia de altura acumulada en metros.
            - 'peligrosidad_promedio': Índice de peligrosidad promedio.
            - 'pendiente_maxima': Pendiente máxima del recorrido en %.
    """
    if not ruta or len(ruta) < 2:
        return {
            'distancia_total': 0.0,
            'ganancia_altura': 0.0,
            'peligrosidad_promedio': 0.0,
            'pendiente_maxima': 0.0
        }
    
    distancia_total = 0.0
    ganancia_altura = 0.0
    peligrosidades = []
    pendiente_maxima = 0.0
    
    for i in range(len(ruta) - 1):
        nodo_actual = grafo.nodos[ruta[i]]
        nodo_siguiente = grafo.nodos[ruta[i + 1]]
        
        # Distancia
        dist = distancia_haversine(
            nodo_actual.latitud, nodo_actual.longitud,
            nodo_siguiente.latitud, nodo_siguiente.longitud
        )
        distancia_total += dist
        
        # Ganancia de altura
        dif_altura = nodo_siguiente.altura - nodo_actual.altura
        if dif_altura > 0:
            ganancia_altura += dif_altura
        
        # Peligrosidad
        peligrosidades.append(nodo_siguiente.prob_accidente)
        
        # Pendiente
        if dist > 0:
            pendiente = abs(dif_altura / dist) * 100
            pendiente_maxima = max(pendiente_maxima, pendiente)
    
    return {
        'distancia_total': distancia_total,
        'ganancia_altura': ganancia_altura,
        'peligrosidad_promedio': sum(peligrosidades) / len(peligrosidades) if peligrosidades else 0.0,
        'pendiente_maxima': pendiente_maxima
    }


# =============================================================================
# FUNCIONES DE ASIGNACIÓN DE SEGURIDAD AL GRAFO
# =============================================================================

def asignar_indicador_seguridad(
    grafo, 
    scores: Dict[str, float] = None, 
    nodo_attr: str = 'comuna',
    ruta_excel: str = None,
    normalizar: bool = True
) -> None:
    """
    Asigna valores de seguridad a los nodos del grafo.
    
    Puede recibir scores precalculados o calcularlos desde el archivo Excel.

    Args:
        grafo: Grafo con nodos a los que asignar seguridad.
        scores: Dict donde las claves son nombres de grupo (p. ej. 'RENCA') y los valores el score.
                Si es None, se calculan desde el archivo Excel.
        nodo_attr: Nombre del atributo en cada `Nodo` que contiene el grupo (default: 'comuna').
        ruta_excel: Ruta al archivo Excel de seguridad. Si es None, usa el default.
        normalizar: Si True, normaliza los scores al rango [0,1] antes de asignar.

    La búsqueda del atributo es flexible: se toma el valor del atributo si existe
    (puede añadirse dinámicamente a los nodos) y se hace match en mayúsculas.
    Si no se encuentra un valor para un nodo, no se modifica su `prob_accidente`.
    """
    # Si no se proporcionan scores, calcularlos desde Excel
    if scores is None:
        try:
            scores = calcular_indicador_seguridad_desde_excel(
                ruta_excel=ruta_excel, 
                agrupar_por=nodo_attr.upper()
            )
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el índice de seguridad desde Excel: {e}")
            return
    
    # Normalizar si se solicita
    if normalizar:
        scores = normalizar_scores(scores)
    
    # Normalizar claves de scores
    scores_norm = {str(k).strip().upper(): float(v) for k, v in scores.items()}

    nodos_actualizados = 0
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
            nodos_actualizados += 1
    
    print(f"Índice de seguridad asignado a {nodos_actualizados} nodos.")


def cargar_seguridad_desde_excel(
    grafo,
    ruta_excel: str = None,
    agrupar_por: str = 'COMUNA',
    nodo_attr: str = 'comuna',
    normalizar: bool = True
) -> Dict[str, float]:
    """
    Función de conveniencia para cargar seguridad desde Excel y asignarla al grafo.
    
    Args:
        grafo: Grafo al cual asignar los valores de seguridad.
        ruta_excel: Ruta al archivo Excel. Si None, usa el default.
        agrupar_por: Columna del Excel por la cual agrupar.
        nodo_attr: Atributo del nodo que contiene el grupo.
        normalizar: Si normalizar los scores a [0,1].
    
    Returns:
        Dict[str, float]: Diccionario de scores calculados.
    """
    scores = calcular_indicador_seguridad_desde_excel(ruta_excel, agrupar_por)
    
    if normalizar:
        scores = normalizar_scores(scores)
    
    asignar_indicador_seguridad(grafo, scores, nodo_attr, normalizar=False)
    
    return scores


# =============================================================================
# ALIAS DE RETROCOMPATIBILIDAD (INGLÉS)
# =============================================================================

# Funciones principales
astar = a_estrella
astar_multi = a_estrella_multiruta
haversine = distancia_haversine
edge_cost = coste_arista

# Enumeraciones (para uso desde código en inglés)
FilterType = TipoFiltro
RouteType = TipoRuta