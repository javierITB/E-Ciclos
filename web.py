# web.py (Versión Final Integrada - Con Tiles para Optimización)
import osmnx as ox
import plotly.express as px
import pandas as pd
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import random
import plotly.graph_objects as go
from typing import List, Optional
import routing
from grafo import Grafo
from string_to_node import texto_a_nodo, inicializar_con_tile_loader
from create_tiles import CompleteTileLoader  # <-- NUEVO IMPORT
import math  # <-- NUEVO IMPORT

# ================= INICIALIZAR TILE LOADER =================
print("🔄 Inicializando sistema de tiles...")
try:
    tile_loader = CompleteTileLoader("tiles_data_complete")
    print(f"✅ Sistema de tiles cargado: {tile_loader.metadata['total_nodes']:,} nodos")
    
    inicializar_con_tile_loader(tile_loader)
    
    # Variables globales (mantener compatibilidad)
    G = None  # No usamos NetworkX completo
    G_CUSTOM = None  # Lo construiremos bajo demanda
    PUNTO_ORIGEN = None
    PUNTO_DESTINO = None
    RUTA_ASTAR = None
    
    # Variables de vista actual
    CURRENT_VIEWPORT = {
        'center_lat': -33.45,
        'center_lon': -70.65,
        'zoom': 13
    }
    
except Exception as e:
    print(f"❌ Error cargando tiles: {e}")
    tile_loader = None
    G = None
    G_CUSTOM = None
    PUNTO_ORIGEN = None
    PUNTO_DESTINO = None
    RUTA_ASTAR = None
    CURRENT_VIEWPORT = None

# ================= FUNCIONES CON TILES =================
def generar_datos_por_viewport(center_lat=None, center_lon=None, zoom=None):
    """Reemplaza a generar_datos_para_plotear - Solo carga lo visible"""
    if tile_loader is None:
        return pd.DataFrame(), pd.DataFrame()
    
    # Usar valores por defecto si no se especifican
    if center_lat is None or center_lon is None or zoom is None:
        if CURRENT_VIEWPORT:
            center_lat = CURRENT_VIEWPORT['center_lat']
            center_lon = CURRENT_VIEWPORT['center_lon']
            zoom = CURRENT_VIEWPORT['zoom']
        else:
            center_lat = -33.45
            center_lon = -70.65
            zoom = 13
    
    # Actualizar viewport actual
    CURRENT_VIEWPORT.update({
        'center_lat': center_lat,
        'center_lon': center_lon,
        'zoom': zoom
    })
    
    # Obtener datos visibles desde tile loader
    nodes_list, edges_list = tile_loader.get_visible_data(center_lat, center_lon, zoom)
    
    # Convertir a DataFrame (manteniendo estructura original)
    if nodes_list:
        nodes_df = pd.DataFrame(nodes_list)
        # Añadir columnas que espera Plotly (mantener nombres originales)
        nodes_df['altitud_m'] = nodes_df['ele']  # Usar altura REAL del OSM
        nodes_df['peligrosidad'] = 0.1  # Valor por defecto (igual que antes)
        
        # Renombrar columnas para compatibilidad
        nodes_df = nodes_df.rename(columns={
            'lat': 'lat',
            'lon': 'lon'
        })
    else:
        nodes_df = pd.DataFrame(columns=['lat', 'lon', 'altitud_m', 'peligrosidad'])
    
    # Preparar aristas para Plotly (mantener formato original)
    lon_list = []
    lat_list = []
    
    for edge in edges_list:
        lon_list.extend([edge['from_lon'], edge['to_lon'], None])
        lat_list.extend([edge['from_lat'], edge['to_lat'], None])
    
    if lon_list:
        edges_df = pd.DataFrame({'lon': lon_list, 'lat': lat_list})
    else:
        edges_df = pd.DataFrame({'lon': [], 'lat': []})
    
    return nodes_df, edges_df

def get_nearest_node(lat, lon, max_distance_km=0.2):
    """Versión mejorada - Encuentra nodo más cercano usando tiles"""
    if tile_loader is None:
        return None
    
    # 1. Buscar en el tile local PRIMERO (más rápido)
    tile_x, tile_y = tile_loader.latlon_to_tile(lat, lon)
    tile_data = tile_loader.load_tile_data(tile_x, tile_y)
    
    nearest_node = None
    min_distance = float('inf')
    
    if tile_data:
        for node_id in tile_data['node_ids']:
            if node_id in tile_loader.all_nodes:
                node = tile_loader.all_nodes[node_id]
                # Distancia en grados (aproximación rápida)
                dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                dist_km = dist_deg * 111.0  # 1 grado ≈ 111 km
                
                if dist_km < min_distance:
                    min_distance = dist_km
                    nearest_node = node_id
    
    # 2. Si no encontró en el tile local, buscar en tiles vecinos (hasta 1km)
    if nearest_node is None or min_distance > max_distance_km:
        # Buscar en un radio de 1km (aproximadamente 0.009 grados)
        search_radius_deg = 0.009
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor_tile = tile_loader.load_tile_data(tile_x + dx, tile_y + dy)
                if neighbor_tile:
                    for node_id in neighbor_tile['node_ids']:
                        if node_id in tile_loader.all_nodes:
                            node = tile_loader.all_nodes[node_id]
                            dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                            dist_km = dist_deg * 111.0
                            
                            if dist_km < min_distance and dist_km <= max_distance_km:
                                min_distance = dist_km
                                nearest_node = node_id
    
    if nearest_node and min_distance <= max_distance_km:
        print(f"📍 Clic en ({lat:.6f}, {lon:.6f}) → Nodo {nearest_node} a {min_distance*1000:.0f}m")
        return nearest_node
    
    print(f"⚠️  No se encontró nodo cerca de ({lat:.6f}, {lon:.6f})")
    return None

def build_custom_graph_for_routing(origin_id, destination_id):
    """Construye grafo personalizado SOLO para el área de routing"""
    if tile_loader is None:
        return None
    
    # Obtener nodos origen/destino
    origin_node = tile_loader.all_nodes.get(origin_id)
    dest_node = tile_loader.all_nodes.get(destination_id)
    
    if not origin_node or not dest_node:
        print(f"❌ Nodos no encontrados: {origin_id} o {destination_id}")
        return None
    
    # Calcular área de búsqueda (buffer alrededor de los puntos)
    buffer_deg = 0.03  # ~3.3km
    min_lat = min(origin_node['lat'], dest_node['lat']) - buffer_deg
    max_lat = max(origin_node['lat'], dest_node['lat']) + buffer_deg
    min_lon = min(origin_node['lon'], dest_node['lon']) - buffer_deg
    max_lon = max(origin_node['lon'], dest_node['lon']) + buffer_deg
    
    # Encontrar tiles en esta área
    min_tx, min_ty = tile_loader.latlon_to_tile(min_lat, min_lon)
    max_tx, max_ty = tile_loader.latlon_to_tile(max_lat, max_lon)
    
    # Recopilar TODAS las aristas en esta área
    all_edges_in_area = []
    edges_seen = set()  # Para evitar duplicados
    
    print(f"🔍 Buscando en área: {min_lat:.4f},{min_lon:.4f} a {max_lat:.4f},{max_lon:.4f}")
    print(f"   Tiles: ({min_tx},{min_ty}) a ({max_tx},{max_ty})")
    
    for tx in range(min_tx, max_tx + 1):
        for ty in range(min_ty, max_ty + 1):
            tile_data = tile_loader.load_tile_data(tx, ty)
            if tile_data:
                for edge_idx in tile_data['edge_ids']:
                    if edge_idx < len(tile_loader.all_edges):
                        edge = tile_loader.all_edges[edge_idx]
                        edge_key = f"{edge['from']}-{edge['to']}"
                        if edge_key not in edges_seen:
                            edges_seen.add(edge_key)
                            all_edges_in_area.append(edge)
    
    print(f"  📊 Área de routing: {len(all_edges_in_area):,} aristas únicas")
    
    # Construir grafo personalizado (usando tus clases originales)
    grafo = Grafo()
    
    # Primero, agregar nodos relevantes
    node_ids_in_area = set()
    for edge in all_edges_in_area:
        node_ids_in_area.add(edge['from'])
        node_ids_in_area.add(edge['to'])
    
    print(f"  📍 Nodos en área: {len(node_ids_in_area):,}")
    
    for node_id in node_ids_in_area:
        if node_id in tile_loader.all_nodes:
            node_data = tile_loader.all_nodes[node_id]
            
            # Usar datos REALES del OSM
            altura = node_data['ele']  # Altura real del OSM
            prob_accidente = 0.1  # Valor por defecto (igual que antes)
            
            grafo.agregar_nodo(
                node_id,
                node_data['lat'],
                node_data['lon'],
                altura,
                prob_accidente
            )
    
    # Luego, agregar caminos (aristas)
    camino_id = 0
    for edge in all_edges_in_area:
        # Calcular importancia basada en datos OSM (simple)
        importancia = 1
        
        # Ajustar por tipo de vía
        highway = edge.get('highway', '')
        if highway in ['cycleway', 'path']:
            importancia = 1  # Mejor para ciclistas
        elif highway in ['primary', 'secondary', 'trunk']:
            importancia = 3  # Más tráfico
        elif highway in ['residential', 'tertiary', 'unclassified']:
            importancia = 2
        
        # Ajustar por carriles
        try:
            lanes = int(edge.get('lanes', 1))
            if lanes > 2:
                importancia = max(1, importancia - 1)
        except:
            pass
        
        # Crear camino (respetando direccionalidad)
        if edge.get('oneway', False):
            # Solo dirección forward
            camino_id += 1
            try:
                grafo.agregar_camino(
                    camino_id,
                    edge['from'],
                    edge['to'],
                    ciclovia=(edge.get('bicycle') in ['yes', 'designated'] or highway == 'cycleway'),
                    importancia=importancia
                )
            except Exception as e:
                # Si hay error, continuar con la siguiente arista
                continue
        else:
            # Bidireccional: crear dos caminos
            camino_id += 1
            try:
                grafo.agregar_camino(
                    camino_id,
                    edge['from'],
                    edge['to'],
                    ciclovia=(edge.get('bicycle') in ['yes', 'designated'] or highway == 'cycleway'),
                    importancia=importancia
                )
            except:
                pass
            
            camino_id += 1
            try:
                grafo.agregar_camino(
                    camino_id,
                    edge['to'],
                    edge['from'],
                    ciclovia=(edge.get('bicycle') in ['yes', 'designated'] or highway == 'cycleway'),
                    importancia=importancia
                )
            except:
                pass
    
    print(f"  ✅ Grafo construido: {len(grafo.nodos):,} nodos, {len(grafo.caminos):,} caminos")
    
    # Verificar que origen y destino están en el grafo
    if origin_id not in grafo.nodos:
        print(f"⚠️  Origen {origin_id} no encontrado en grafo de routing")
        return None
    if destination_id not in grafo.nodos:
        print(f"⚠️  Destino {destination_id} no encontrado en grafo de routing")
        return None
    
    return grafo

# ================= INICIALIZAR DATOS PARA PLOTLY =================
# Cargar vista inicial (mantener compatibilidad)
if tile_loader is not None:
    nodes_df, edges_lines = generar_datos_por_viewport()
else:
    nodes_df, edges_lines = pd.DataFrame(), pd.DataFrame()

# ================= INICIALIZACIÓN DE LA APLICACIÓN DASH =================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

if tile_loader is None:
    app.layout = dbc.Container([
        html.H1("Ruteo Interactivo", className="my-4 text-center text-danger"),
        dbc.Alert("⛔ Error: No se pudo cargar el sistema de tiles. Verifique 'tiles_data_complete/'.", color="danger"),
    ], fluid=True)
else:
    app.layout = dbc.Container([
        html.Hr(className="mb-4"),

        dbc.Row([
            # 1. SIDEBAR (MANTENER EXACTAMENTE IGUAL)
            dbc.Col([
                html.H4("🧭 Controles de Ruta", className="mt-2 mb-3 text-secondary"),
                html.P(f"Total de nodos disponibles: {tile_loader.metadata['total_nodes']:,}", className="text-muted small"),

                # --- ETIQUETA DE MODO CLIC ---
                html.Div(id='click-mode-label', className="text-center fw-bold text-info border p-2 mb-3 rounded-lg",
                         children="Modo Click: Seleccionar Origen"),
                # -----------------------------------

                # --- ALMACENES DE VALOR (OCULTOS) ---
                dcc.Store(id='store-origen-id', data=None),
                dcc.Store(id='store-destino-id', data=None),
                dcc.Store(id='store-show-graph', data=False),
                # ------------------------------------

                dbc.Label("📍 Origen (Texto o ID de Nodo):", className="mt-3 form-label"),
                dbc.Input(id='input-origen', type='text', placeholder='Ej: Plaza de Armas o 139158914',
                          className="mb-2"),
                dbc.Button('Fijar Origen', id='btn-fijar-origen', n_clicks=0, color="info", size="sm",
                           className="w-100"),

                dbc.Label("🏁 Destino (Texto o ID de Nodo):", className="mt-4 form-label"),
                dbc.Input(id='input-destino', type='text', placeholder='Ej: Cerro San Cristóbal o 139162483',
                          className="mb-2"),
                dbc.Button('Fijar Destino', id='btn-fijar-destino', n_clicks=0, color="danger", size="sm",
                           className="w-100"),

                # Mensaje de estado
                dbc.Alert(id="selection-status", color="light", className="text-center mt-3 py-2",
                          children="Fija Origen y Destino usando texto o IDs."),
                html.Hr(className="my-4"),
                dbc.Button('🛣️ Calcular Ruta Óptima', id='btn-ruta', n_clicks=0, color="success",
                           className="w-100 mt-2 btn-lg"),

                # Contenedores para Output del Callback 5
                html.Div(id='ruta-output', className="my-3 text-center"),
                html.Div(id='ruta-nodos-list', className="mt-4 p-2 bg-white border rounded shadow-sm",
                         style={'maxHeight': '300px', 'overflowY': 'auto', 'fontSize': '0.8rem'}),
                # Fin Contenedores

                dbc.Button('🔄 Resetear Selección', id='btn-reset', n_clicks=0, color="warning", className="w-100 mt-4"),
                dbc.Button('Ver rutas disponibles', id='btn-show-graph', n_clicks=0, color="primary", className="w-100 mt-3"),
                html.Hr(className="my-4"),

            ], width=3, className="p-4 bg-light border-end shadow-sm",
                style={'minHeight': '90vh', 'overflowY': 'auto'}),

            # 2. MAPA PRINCIPAL (MANTENER EXACTAMENTE IGUAL)
            dbc.Col([
                html.Div([
                    dcc.Graph(
                        id='mapa-2d-interactivo',
                        style={'height': '100%', 'width': '100%'},
                        config={
                            'displayModeBar': False,
                            'scrollZoom': True,
                            'doubleClick': 'reset'
                        }
                    ),
                    html.Div([
                        dbc.Button('➕', id='btn-zoom-in', n_clicks=0, color="secondary", size="sm", className="mb-1"),
                        dbc.Button('➖', id='btn-zoom-out', n_clicks=0, color="secondary", size="sm"),
                    ], style={'position': 'absolute', 'bottom': '10px', 'right': '10px', 'z-index': '1000'}),
                ], style={'position': 'relative', 'height': '100%', 'width': '100%'}),
                dbc.Collapse(
                    html.Div([
                        html.H5("Vista 3D (Altitud y Peligrosidad)", className="mt-3 text-center text-info"),
                        dcc.Graph(id='mapa-3d-altitud', style={'height': '60vh'})
                    ], className="p-3 bg-light border rounded shadow-sm mt-3"),
                    id="collapse-3d",
                    is_open=False,
                ),
            ], width=9, className="p-0", style={'height': '90vh'}),
        ], className="g-0"),

        # --- COMPONENTE DE ESTADO INVISIBLE PARA EL MODO DE CLIC ---
        dcc.Store(id='store-click-mode', data={'next_selection': 'ORIGEN'}),
        # ----------------------------------------------------------
        html.Hr(className="mb-4"),
    ], fluid=True, className="p-0")

# ================= CALLBACKS (MANTENER LÓGICA ORIGINAL) =================

# Callback de Modo de Clic: Actualiza la etiqueta de modo
@app.callback(
    Output('click-mode-label', 'children'),
    [Input('store-click-mode', 'data'),
     Input('store-origen-id', 'data'),
     Input('store-destino-id', 'data')]
)
def update_click_mode_label_mejorado(mode_data, origen_id, destino_id):
    """Etiqueta más informativa que muestra estado actual"""
    mode = mode_data.get('next_selection', 'ORIGEN')
    
    if mode == 'ORIGEN':
        status = html.Div([
            html.H6("🟢 Modo: SELECCIONAR ORIGEN", className="mb-1"),
            html.Small("Haz clic en el mapa para elegir el punto de partida", className="text-muted")
        ])
    else:
        status = html.Div([
            html.H6("🔴 Modo: SELECCIONAR DESTINO", className="mb-1"),
            html.Small("Haz clic en el mapa para elegir el punto de llegada", className="text-muted")
        ])
    
    # Añadir información de lo que ya está seleccionado
    info_extra = html.Div([
        html.Hr(className="my-2"),
        html.Small("Seleccionados:", className="text-muted d-block"),
        html.Small(f"📍 Origen: {origen_id if origen_id else 'No seleccionado'}", 
                  className="text-success" if origen_id else "text-muted"),
        html.Br(),
        html.Small(f"🏁 Destino: {destino_id if destino_id else 'No seleccionado'}", 
                  className="text-danger" if destino_id else "text-muted")
    ], className="mt-2")
    
    return html.Div([status, info_extra])

# Callback de Click en el Mapa: Versión optimizada con tiles

# ================= CALLBACK PARA CLICS SIMPLES (SELECCIÓN) =================
@app.callback(
    [
        Output('store-origen-id', 'data', allow_duplicate=True),
        Output('store-destino-id', 'data', allow_duplicate=True),
        Output('store-click-mode', 'data', allow_duplicate=True),
        Output('selection-status', 'children', allow_duplicate=True)
    ],
    Input('mapa-2d-interactivo', 'clickData'),  # <-- CAMBIADO a clickData
    [
        State('store-click-mode', 'data'),
        State('store-origen-id', 'data'),
        State('store-destino-id', 'data')
    ],
    prevent_initial_call=True
)
def handle_map_click(clickData, mode_data, current_origen, current_destino):
    """Callback para clics en el mapa"""
    
    if not clickData:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    print(f"🖱️ CLIC DETECTADO EN MAPA")
    
    # 1. Obtener coordenadas
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']
    
    print(f"   Coordenadas: ({lat:.6f}, {lon:.6f})")
    
    # 2. Encontrar nodo más cercano
    node_id = get_nearest_node(lat, lon)
    
    if not node_id:
        error_msg = html.Div([
            html.Span("❌ ", className="text-danger"),
            html.Span("No hay nodos cerca de este punto."),
            html.Br(),
            html.Small("Haz clic cerca de una calle.", className="text-muted")
        ])
        return dash.no_update, dash.no_update, dash.no_update, error_msg
    
    print(f"   Nodo seleccionado: {node_id}")
    
    # 3. Lógica de selección
    current_mode = mode_data.get('next_selection', 'ORIGEN')
    
    if current_mode == 'ORIGEN':
        # Establecer origen
        new_mode = {'next_selection': 'DESTINO'}
        msg = html.Div([
            html.Span("✅ ", className="text-success"),
            html.B(f"Origen establecido: "),
            html.Span(f"Nodo {node_id}"),
            html.Br(),
            html.Small("Ahora selecciona el DESTINO.", className="text-muted")
        ])
        return node_id, dash.no_update, new_mode, msg
    
    else:  # DESTINO
        # Establecer destino
        new_mode = {'next_selection': 'ORIGEN'}
        msg = html.Div([
            html.Span("✅ ", className="text-success"),
            html.B(f"Destino establecido: "),
            html.Span(f"Nodo {node_id}"),
            html.Br(),
            html.Small("¡Ahora puedes calcular la ruta!", className="text-muted")
        ])
        return dash.no_update, node_id, new_mode, msg


# ================= CALLBACK PARA DOBLE CLIC (ZOOM/RESET) =================
@app.callback(
    Output('mapa-2d-interactivo', 'figure', allow_duplicate=True),
    Input('mapa-2d-interactivo', 'dblclickData'),
    State('mapa-2d-interactivo', 'figure'),
    prevent_initial_call=True
)
def handle_double_click(dblclickData, current_figure):
    """Manejar doble clic para resetear zoom"""
    if not dblclickData or not current_figure:
        return dash.no_update
    
    # Resetear a vista inicial
    new_figure = current_figure.copy()
    if 'layout' in new_figure:
        if 'mapbox' in new_figure['layout']:
            new_figure['layout']['mapbox']['zoom'] = 13
            new_figure['layout']['mapbox']['center'] = {"lat": -33.45, "lon": -70.65}
    
    return new_figure

def get_nearest_node(lat, lon, max_distance_km=0.2):
    """Versión mejorada - Encuentra nodo más cercano usando tiles"""
    if tile_loader is None:
        print("❌ Tile loader no disponible")
        return None
    
    print(f"🔍 Buscando nodo cercano a ({lat:.6f}, {lon:.6f})")
    
    # 1. Buscar en el tile local PRIMERO (más rápido)
    tile_x, tile_y = tile_loader.latlon_to_tile(lat, lon)
    tile_data = tile_loader.load_tile_data(tile_x, tile_y)
    
    nearest_node = None
    min_distance = float('inf')
    node_coords = None
    
    if tile_data:
        print(f"   Tile ({tile_x}, {tile_y}) tiene {len(tile_data['node_ids'])} nodos")
        
        # Buscar en TODOS los nodos del tile
        for node_id in tile_data['node_ids']:
            if node_id in tile_loader.all_nodes:
                node = tile_loader.all_nodes[node_id]
                # Distancia en grados (aproximación rápida)
                dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                dist_km = dist_deg * 111.0  # 1 grado ≈ 111 km
                
                if dist_km < min_distance and dist_km <= max_distance_km:
                    min_distance = dist_km
                    nearest_node = node_id
                    node_coords = (node['lat'], node['lon'])
    
    # 2. Si no encontró en el tile local, buscar en tiles vecinos
    if nearest_node is None:
        print(f"   No se encontró en tile principal, buscando en vecinos...")
        search_radius_deg = 0.009  # ~1km
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor_tile = tile_loader.load_tile_data(tile_x + dx, tile_y + dy)
                if neighbor_tile:
                    for node_id in neighbor_tile['node_ids']:
                        if node_id in tile_loader.all_nodes:
                            node = tile_loader.all_nodes[node_id]
                            dist_deg = math.sqrt((node['lat'] - lat)**2 + (node['lon'] - lon)**2)
                            dist_km = dist_deg * 111.0
                            
                            if dist_km < min_distance and dist_km <= max_distance_km:
                                min_distance = dist_km
                                nearest_node = node_id
                                node_coords = (node['lat'], node['lon'])
    
    if nearest_node:
        print(f"✅ Nodo encontrado: {nearest_node} a {min_distance*1000:.0f}m")
        print(f"   Coordenadas nodo: ({node_coords[0]:.6f}, {node_coords[1]:.6f})")
        return nearest_node
    
    print(f"❌ No se encontró nodo cerca de ({lat:.6f}, {lon:.6f})")
    return None

# Callback de Manejo de Texto: MANTENER EXACTAMENTE IGUAL
@app.callback(
    [Output('store-origen-id', 'data', allow_duplicate=True),
     Output('store-destino-id', 'data', allow_duplicate=True),
     Output('selection-status', 'children', allow_duplicate=True)],
    [Input('btn-fijar-origen', 'n_clicks'),
     Input('btn-fijar-destino', 'n_clicks'),
     Input('btn-reset', 'n_clicks')],
    [State('input-origen', 'value'),
     State('input-destino', 'value'),
     State('store-origen-id', 'data'),
     State('store-destino-id', 'data')],
    prevent_initial_call=True
)
def handle_text_input(n_origen, n_destino, n_reset,
                      origen_text, destino_text,
                      current_origen_id, current_destino_id):
    global PUNTO_ORIGEN, PUNTO_DESTINO, RUTA_ASTAR

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    # 1. Lógica de Reseteo
    if trigger_id == 'btn-reset':
        RUTA_ASTAR = None
        return None, None, html.Div(["👆 Fija Origen y Destino usando texto o IDs."], className="fw-bold")

    # Inicializar las salidas
    new_origen_id = current_origen_id
    new_destino_id = current_destino_id
    status_msg = dash.no_update

    # 2. Manejo de Origen
    if trigger_id == 'btn-fijar-origen':
        if not origen_text:
            new_origen_id = None
            status_msg = dash.no_update
        else:
            try:
                # Usar tu función original texto_a_nodo
                node_id = texto_a_nodo(origen_text)

                if node_id == -1:
                    try:
                        potential_id = int(origen_text)
                        # Verificar que existe en los datos
                        if potential_id in tile_loader.all_nodes:
                            node_id = potential_id
                    except ValueError:
                        pass

                if node_id != -1 and node_id in tile_loader.all_nodes:
                    new_origen_id = int(node_id)
                    RUTA_ASTAR = None
                    status_msg = dash.no_update
                else:
                    new_origen_id = None
                    status_msg = html.Div([f"⛔ Error: Origen '{origen_text}' no encontrado o inválido."],
                                          className="text-danger")

            except Exception as e:
                new_origen_id = None
                status_msg = html.Div([f"⛔ Error en la búsqueda de Origen: {e}"], className="text-danger")

    # 3. Manejo de Destino
    elif trigger_id == 'btn-fijar-destino':
        if not destino_text:
            new_destino_id = None
            status_msg = dash.no_update
        else:
            try:
                node_id = texto_a_nodo(destino_text)

                if node_id == -1:
                    try:
                        potential_id = int(destino_text)
                        if potential_id in tile_loader.all_nodes:
                            node_id = potential_id
                    except ValueError:
                        pass

                # Validación adicional
                is_valid = node_id != -1 and node_id in tile_loader.all_nodes and int(node_id) != current_origen_id

                if is_valid:
                    new_destino_id = int(node_id)
                    RUTA_ASTAR = None
                    status_msg = dash.no_update
                else:
                    new_destino_id = None
                    if node_id != -1 and int(node_id) == current_origen_id:
                        status_msg = html.Div(["⛔ Error: Destino no puede ser igual al Origen."],
                                              className="text-danger")
                    else:
                        status_msg = html.Div([f"⛔ Error: Destino '{destino_text}' no encontrado o inválido."],
                                              className="text-danger")

            except Exception as e:
                new_destino_id = None
                status_msg = html.Div([f"⛔ Error en la búsqueda de Destino: {e}"], className="text-danger")

    return new_origen_id, new_destino_id, status_msg

# Callback principal del mapa: VERSIÓN OPTIMIZADA CON TILES
@app.callback(
    [Output('mapa-2d-interactivo', 'figure'),
     Output('store-show-graph', 'data')],
    [Input('btn-reset', 'n_clicks'),
     Input('ruta-output', 'children'),
     Input('store-origen-id', 'data'),
     Input('store-destino-id', 'data'),
     Input('btn-zoom-in', 'n_clicks'),
     Input('btn-zoom-out', 'n_clicks'),
     Input('btn-show-graph', 'n_clicks')],
    [State('mapa-2d-interactivo', 'figure'),
     State('store-show-graph', 'data')]
)
def update_map_and_selection(reset_clicks,
                             ruta_output_children,
                             origen_id, destino_id,
                             zoom_in_clicks, zoom_out_clicks,
                             show_graph_clicks,
                             current_figure,
                             current_show_graph):
    global PUNTO_ORIGEN, PUNTO_DESTINO, RUTA_ASTAR, CURRENT_VIEWPORT

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    # Obtener zoom y centro actuales del viewport
    if current_figure:
        layout = current_figure.get('layout', {})
        mapbox = layout.get('mapbox', {})
        current_zoom = mapbox.get('zoom', 13)
        current_center = mapbox.get('center', {"lat": -33.45, "lon": -70.65})
    else:
        current_zoom = 13
        current_center = {"lat": -33.45, "lon": -70.65}

    # Manejar zoom con botones
    if trigger_id == 'btn-zoom-in':
        current_zoom = min(current_zoom + 1, 20)
    elif trigger_id == 'btn-zoom-out':
        current_zoom = max(current_zoom - 1, 1)

    # Manejar mostrar grafo
    if trigger_id == 'btn-show-graph':
        new_show_graph = not current_show_graph
    else:
        new_show_graph = current_show_graph

    # 1. Lógica de Reseteo
    if trigger_id == 'btn-reset':
        PUNTO_ORIGEN = None
        PUNTO_DESTINO = None
        RUTA_ASTAR = None

    # 2. Actualizar puntos globales desde stores
    PUNTO_ORIGEN = origen_id if origen_id is not None and tile_loader and origen_id in tile_loader.all_nodes else None
    PUNTO_DESTINO = destino_id if destino_id is not None and tile_loader and destino_id in tile_loader.all_nodes else None

    # 3. Generar datos SOLO para el viewport actual
    nodes_df, edges_lines = generar_datos_por_viewport(
        current_center['lat'],
        current_center['lon'],
        current_zoom
    )
    
    # 4. Generar la figura base
    fig = go.Figure()

    if not nodes_df.empty:
        # Tomar algunos nodos como puntos clickeables
        sample_nodes = nodes_df.head(200)  # 200 nodos para clics
        
        fig.add_trace(go.Scattermapbox(
            lat=sample_nodes['lat'],
            lon=sample_nodes['lon'],
            mode='markers',
            marker=dict(
                size=3,  # Pequeño
                color='rgba(255, 0, 0, 0.3)',  # ROJO SEMITRANSPARENTE (para debug)
                opacity=0.3  # Visible pero tenue
                
            ),
            name='_clickable_points',
            hoverinfo='none',
            customdata=sample_nodes.index,
            hovertemplate='',
            showlegend=False
        ))
        print(f"✅ Añadidos {len(sample_nodes)} puntos clickeables invisibles")

    # Añadir una traza invisible para forzar el renderizado
    fig.add_trace(go.Scattermapbox(
        lat=[nodes_df['lat'].mean() if not nodes_df.empty else -33.45],
        lon=[nodes_df['lon'].mean() if not nodes_df.empty else -70.65],
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 5. Lógica para centrar en punto recién fijado
    target_node_id = None
    if trigger_id in ['store-origen-id', 'store-destino-id'] and (PUNTO_ORIGEN or PUNTO_DESTINO):
        if trigger_id == 'store-origen-id' and PUNTO_ORIGEN is not None:
            target_node_id = PUNTO_ORIGEN
        elif trigger_id == 'store-destino-id' and PUNTO_DESTINO is not None:
            target_node_id = PUNTO_DESTINO

    # Configuración de Layout
    layout_updates = dict(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        hovermode="closest",
        title="Mapa de Calles de Santiago (Optimizado con Tiles)",
        mapbox_zoom=current_zoom,
        mapbox_center=current_center,
        uirevision=str(PUNTO_ORIGEN) + str(PUNTO_DESTINO) + str(bool(RUTA_ASTAR))
    )

    # Centrar en punto si se acaba de fijar
    if target_node_id and target_node_id in tile_loader.all_nodes:
        node_data = tile_loader.all_nodes[target_node_id]
        layout_updates['mapbox_zoom'] = 16
        layout_updates['mapbox_center'] = {"lat": node_data['lat'], "lon": node_data['lon']}
        layout_updates['uirevision'] = 'center_on_' + str(target_node_id)

    fig.update_layout(**layout_updates)

    # Leyenda
    fig.update_layout(
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Gray",
            borderwidth=1,
            title=None,
        ),
    )

    # 6. Dibujar calles si está activado (USANDO TILES)
    if new_show_graph and not edges_lines.empty:
        fig.add_trace(go.Scattermapbox(
            lat=edges_lines['lat'],
            lon=edges_lines['lon'],
            mode='lines',
            line=dict(width=1, color='blue'),
            name='Calles',
            showlegend=True
        ))

    # 7. Dibujar marcadores de selección (MANTENER LÓGICA ORIGINAL)
    if PUNTO_ORIGEN is not None and PUNTO_ORIGEN in tile_loader.all_nodes:
        data = tile_loader.all_nodes[PUNTO_ORIGEN]
        lat_o = [data['lat']]
        lon_o = [data['lon']]

        # Contorno
        fig.add_trace(go.Scattermapbox(
            lat=lat_o,
            lon=lon_o,
            mode='markers',
            marker=dict(size=20, color='black', symbol='circle', opacity=1.0),
            name='Origen (Contorno)',
            hoverinfo='skip',
            showlegend=False
        ))

        # Marcador principal
        fig.add_trace(go.Scattermapbox(
            lat=lat_o,
            lon=lon_o,
            mode='markers',
            marker=dict(size=14, color='#28a745', symbol='circle', opacity=1.0),
            name='Origen',
            hovertext=[f'Origen: {PUNTO_ORIGEN}'],
            hovertemplate='Nodo ID: %{hovertext}<extra></extra>',
            showlegend=True
        ))

    if PUNTO_DESTINO is not None and PUNTO_DESTINO in tile_loader.all_nodes:
        data = tile_loader.all_nodes[PUNTO_DESTINO]
        lat_d = [data['lat']]
        lon_d = [data['lon']]

        # Contorno
        fig.add_trace(go.Scattermapbox(
            lat=lat_d,
            lon=lon_d,
            mode='markers',
            marker=dict(size=20, color='black', symbol='circle', opacity=1.0),
            name='Destino (Contorno)',
            hoverinfo='skip',
            showlegend=False
        ))

        # Marcador principal
        fig.add_trace(go.Scattermapbox(
            lat=lat_d,
            lon=lon_d,
            mode='markers',
            marker=dict(size=14, color='#dc3545', symbol='circle', opacity=1.0),
            name='Destino',
            hovertext=[f'Destino: {PUNTO_DESTINO}'],
            hovertemplate='Nodo ID: %{hovertext}<extra></extra>',
            showlegend=True
        ))

    # 8. Dibujar rutas (MANTENER LÓGICA ORIGINAL)
    def dibujar_ruta(ruta_nodos: Optional[List[int]], nombre: str, color: str, opacidad: float = 1.0):
        if ruta_nodos and tile_loader is not None:
            route_coords = []
            for nid in ruta_nodos:
                if nid in tile_loader.all_nodes:
                    node = tile_loader.all_nodes[nid]
                    route_coords.append({'lat': node['lat'], 'lon': node['lon']})
            
            if route_coords:
                route_lats = [coord['lat'] for coord in route_coords]
                route_lons = [coord['lon'] for coord in route_coords]
                
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lat=route_lats,
                    lon=route_lons,
                    line=dict(width=5, color=color),
                    opacity=opacidad,
                    name=nombre,
                    showlegend=True
                ))

    dibujar_ruta(RUTA_ASTAR, "Ruta A* (Heurística)", "orange", 1.0)

    # 9. Dibujar nodos intermedios de rutas
    def dibujar_nodos_ruta(ruta_nodos: Optional[List[int]], nombre: str, color: str):
        if ruta_nodos and tile_loader is not None:
            node_ids_to_plot = []
            for nid in ruta_nodos:
                if (nid != PUNTO_ORIGEN and nid != PUNTO_DESTINO and 
                    nid in tile_loader.all_nodes):
                    node_ids_to_plot.append(nid)

            if not node_ids_to_plot:
                return

            route_coords = []
            for nid in node_ids_to_plot:
                node = tile_loader.all_nodes[nid]
                route_coords.append({'lat': node['lat'], 'lon': node['lon']})
            
            route_lats = [coord['lat'] for coord in route_coords]
            route_lons = [coord['lon'] for coord in route_coords]

            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lat=route_lats,
                lon=route_lons,
                marker=dict(size=6, color=color, opacity=0.8),
                name=f"Nodos {nombre}",
                customdata=node_ids_to_plot,
                hovertemplate='Nodo ID: %{customdata}<extra></extra>',
                showlegend=True
            ))

    dibujar_nodos_ruta(RUTA_ASTAR, "A*", "#E63946")

    return fig, new_show_graph

# Callback para actualizar estado de selección (MANTENER IGUAL)
@app.callback(
    Output("selection-status", "children", allow_duplicate=True),
    [Input('store-origen-id', 'data'),
     Input('store-destino-id', 'data')],
    prevent_initial_call=True
)
def update_status_text_from_store(origen_id, destino_id):
    if origen_id is None and destino_id is None:
        return html.Div(["👆 Fija Origen y Destino usando texto o IDs."], className="fw-bold")
    elif origen_id is not None and destino_id is None:
        return html.Div([
            html.B("✅ Origen Fijado. "),
            html.Span(f"ID: {origen_id}. Ahora selecciona el Destino.")
        ])
    elif origen_id is not None and destino_id is not None:
        return html.Div([
            html.B("🎉 Listo: "),
            html.Span(f"Origen ({origen_id}) y Destino ({destino_id}). ¡Calcula la ruta!")
        ], className="text-success fw-bold")
    else:
        return html.Div(["Fija Origen y Destino usando texto o IDs."], className="fw-bold")

# Callback para calcular rutas: VERSIÓN OPTIMIZADA
@app.callback(
    [Output('ruta-output', 'children'),
     Output('ruta-nodos-list', 'children')],
    [Input('btn-ruta', 'n_clicks')],
    [State('store-origen-id', 'data'),
     State('store-destino-id', 'data')],
    prevent_initial_call=True
)
def calcular_rutas(n_clicks, origen_id, destino_id):
    global RUTA_ASTAR, PUNTO_ORIGEN, PUNTO_DESTINO, G_CUSTOM

    # Actualizar globales
    PUNTO_ORIGEN = origen_id
    PUNTO_DESTINO = destino_id

    if PUNTO_ORIGEN is None or PUNTO_DESTINO is None:
        return dbc.Alert("Seleccione Origen y Destino válidos antes de calcular.", color="danger"), None

    print(f"📍 Calculando ruta: {PUNTO_ORIGEN} → {PUNTO_DESTINO}")
    
    # Parámetros de peso (mantener igual)
    W_DIST = 1.0
    W_ELEV = 0.0
    W_SEG = 1000.0

    # 1. Construir grafo SOLO para el área necesaria
    print("🔄 Construyendo grafo para routing...")
    G_CUSTOM = build_custom_graph_for_routing(PUNTO_ORIGEN, PUNTO_DESTINO)
    
    if G_CUSTOM is None:
        return dbc.Alert("No se pudo construir el grafo para la ruta seleccionada.", color="danger"), None

    # 2. Ejecutar A*
    RUTA_ASTAR = None
    try:
        print("🔍 Ejecutando A*...")
        RUTA_ASTAR = routing.a_estrella(G_CUSTOM, PUNTO_ORIGEN, PUNTO_DESTINO,
                                        w_dist=W_DIST, w_elev=W_ELEV, w_seg=W_SEG)
        print(f"✅ Ruta encontrada: {len(RUTA_ASTAR) if RUTA_ASTAR else 0} nodos")
    except Exception as e:
        print(f"❌ Error en A*: {e}")
        import traceback
        traceback.print_exc()

    # 3. Generación del Listado de Nodos
    ruta_nodos_html = []
    astar_encontrada = bool(RUTA_ASTAR)

    if astar_encontrada:
        ruta_nodos_html.append(html.B("Ruta A* (IDs):", className="d-block text-warning mt-3"))
        astar_ids_str = ' → '.join(map(str, RUTA_ASTAR))
        ruta_nodos_html.append(html.P(astar_ids_str, style={'wordBreak': 'break-all'}))
    else:
        ruta_nodos_html.append(html.P("A*: No se encontró camino.", className="text-muted"))

    # 4. Respuesta Final
    if astar_encontrada:
        a_len = len(RUTA_ASTAR)
        mensaje = f"Ruta calculada con A*. {a_len} nodos. El mapa se ha actualizado."
        msg = dbc.Alert(mensaje, color="success")
        return msg, html.Div(ruta_nodos_html)
    else:
        return dbc.Alert("No se pudo encontrar ninguna ruta entre los puntos seleccionados.", color="danger"), None

# ================= NUEVO CALLBACK PARA ACTUALIZAR AL MOVER ZOOM =================
@app.callback(
    Output('mapa-2d-interactivo', 'figure', allow_duplicate=True),
    Input('mapa-2d-interactivo', 'relayoutData'),
    State('mapa-2d-interactivo', 'figure'),
    prevent_initial_call=True
)
def update_on_map_move(relayout_data, current_figure):
    """Actualiza los tiles cuando el usuario mueve/zoomea el mapa"""
    if not relayout_data or not current_figure:
        return dash.no_update
    
    # Extraer nueva posición del evento
    new_center = None
    new_zoom = None
    
    if 'mapbox.center' in relayout_data:
        new_center = relayout_data['mapbox.center']
    if 'mapbox.zoom' in relayout_data:
        new_zoom = relayout_data['mapbox.zoom']
    
    # También verificar drag events
    if 'mapbox._derived' in relayout_data:
        derived = relayout_data['mapbox._derived']
        if 'coordinates' in derived:
            coords = derived['coordinates']
            if isinstance(coords, dict) and 'lat' in coords and 'lon' in coords:
                new_center = {'lat': coords['lat'], 'lon': coords['lon']}
    
    # Si no hay cambios en posición, no actualizar
    if not new_center and not new_zoom:
        return dash.no_update
    
    # Usar nuevos valores o mantener actuales
    if new_center:
        center = new_center
    else:
        center = current_figure.get('layout', {}).get('mapbox', {}).get('center', {"lat": -33.45, "lon": -70.65})
    
    if new_zoom:
        zoom = new_zoom
    else:
        zoom = current_figure.get('layout', {}).get('mapbox', {}).get('zoom', 13)
    
    # Actualizar viewport global
    if CURRENT_VIEWPORT:
        CURRENT_VIEWPORT.update({
            'center_lat': center['lat'],
            'center_lon': center['lon'],
            'zoom': zoom
        })
    
    # Solo actualizar si hay cambios significativos
    return dash.no_update  # Dejamos que el callback principal se encargue

# Añade este nuevo callback después de los existentes

# ================= INICIO DE LA APLICACIÓN =================
if __name__ == '__main__':
    if tile_loader is not None:
        print("\n" + "="*60)
        print("🚀 Iniciando aplicación Dash con sistema de tiles...")
        print(f"   Nodos totales: {tile_loader.metadata['total_nodes']:,}")
        print(f"   Aristas totales: {tile_loader.metadata['total_edges']:,}")
        print(f"   Tiles disponibles: {tile_loader.metadata['tiles_with_data']:,}")
        print("="*60)
        
        print("\n🔍 CALLBACKS REGISTRADOS:")
        for callback in app.callback_map.values():
            inputs = [str(inp) for inp in callback['inputs']]
            if 'clickData' in str(inputs):
                print(f"  📌 Callback con clickData: {inputs}")
        
        # SOLO UNA VEZ app.run()
        app.run(debug=False, host="0.0.0.0", port=8050)
    else:
        print("❌ No se pudo iniciar la aplicación: tile_loader no disponible")