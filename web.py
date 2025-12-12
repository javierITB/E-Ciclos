# web.py (Versión Final Integrada - Con Nodos de Ruta y Correcciones)
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
from string_to_node import texto_a_nodo  # Asegúrate de que string_to_node.py esté en el mismo directorio
import osmlib

# Cargar grafo completo una sola vez
osmlib.preparar_osm("map_clean.osm")


# -------------------------

# --- CONFIGURACIÓN Y CARGA DE DATOS ---
OSM_FILE = "./map_clean.osm"

# --- Variables globales ---
G = None  # Grafo NetworkX
G_CUSTOM = None  # Grafo personalizado (Grafo)
PUNTO_ORIGEN = None
PUNTO_DESTINO = None
RUTA_DIJKSTRA = None
RUTA_ASTAR = None


def cargar_grafo_osm(osm_file_path):
    """Carga el grafo desde el archivo OSM y el grafo personalizado."""
    global G, G_CUSTOM

    if G is not None:
        return G

    try:
        print("Cargando grafo OSM...")

        if not os.path.exists(osm_file_path):
            print(f"Error: El archivo '{osm_file_path}' no se encuentra.")
            return None

        G = ox.graph_from_xml(osm_file_path, simplify=False)
        print(f"Grafo cargado: {len(G.nodes)} nodos, {len(G.edges)} aristas.")

        for node, data in G.nodes(data=True):
            data['lon'] = data['x']
            data['lat'] = data['y']

            if 'altitud_m' not in data:
                data['altitud_m'] = 400 + (data['y'] * -100) + random.uniform(0, 50)
                data['peligrosidad'] = random.uniform(0.1, 0.9)

        G_CUSTOM = Grafo()
        G_CUSTOM.cargar_desde_networkx(G)
        print(f"Grafo personalizado cargado con {len(G_CUSTOM.nodos)} nodos para ruteo.")

        return G

    except Exception as e:
        print(f"Error al cargar el grafo: {e}")
        G = None
        G_CUSTOM = None
        return None


def generar_datos_para_plotear(G):
    """Extrae nodos y caminos del grafo para Plotly."""
    if G is None:
        return pd.DataFrame(), pd.DataFrame()

    nodes_df = ox.graph_to_gdfs(G, edges=False).reset_index()

    # 1. Enriquecer datos (Simulación de Altitud y Peligrosidad)
    nodes_df['altitud_m'] = nodes_df['lat'].apply(lambda x: 400 + (x * -100) + random.uniform(0, 50))
    nodes_df['peligrosidad'] = nodes_df.index.to_series().apply(lambda x: random.uniform(0.1, 0.9))

    # 2. Extraer aristas (caminos) con sus geometrías
    edges_df = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    lon_list = []
    lat_list = []

    for geom in edges_df.geometry:
        if geom.geom_type == 'LineString':
            lon, lat = geom.xy
            lon_list.extend(lon)
            lat_list.extend(lat)
            lon_list.append(None)
            lat_list.append(None)

    if lon_list:
        edges_lines = pd.DataFrame({'lon': lon_list, 'lat': lat_list})
    else:
        edges_lines = pd.DataFrame({'lon': [], 'lat': []})

    return nodes_df, edges_lines


# Cargar el grafo globalmente al inicio
G = cargar_grafo_osm(OSM_FILE)
nodes_df, edges_lines = (generar_datos_para_plotear(G) if G is not None else (pd.DataFrame(), pd.DataFrame()))

# --- INICIALIZACIÓN DE LA APLICACIÓN DASH ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

if G is None:
    app.layout = dbc.Container([
        html.H1("Ruteo Interactivo", className="my-4 text-center text-danger"),
        dbc.Alert("⛔ Error: No se pudo cargar el archivo OSM. Verifique 'map_clean.osm'.", color="danger"),
    ], fluid=True)
else:
    app.layout = dbc.Container([
        html.Hr(className="mb-4"),

        dbc.Row([
            # 1. SIDEBAR (VUELVE A SER UNA COLUMNA FIJA)
            dbc.Col([
                html.H4("🧭 Controles de Ruta", className="mt-2 mb-3 text-secondary"),
                html.P(f"Total de nodos disponibles: {len(G.nodes)}", className="text-muted small"),

                # --- NUEVA ETIQUETA DE MODO CLIC ---
                html.Div(id='click-mode-label', className="text-center fw-bold text-info border p-2 mb-3 rounded-lg",
                         children="Modo Click: Seleccionar Origen"),
                # -----------------------------------

                # --- ALMACENES DE VALOR (OCULTOS) ---
                # Usaremos estos dcc.Store para guardar los IDs de nodo numéricos
                dcc.Store(id='store-origen-id', data=None),
                dcc.Store(id='store-destino-id', data=None),
                # ------------------------------------

                dbc.Label("📍 Origen (Texto o ID de Nodo):", className="mt-3 form-label"),
                # CAMBIO: type='text' para permitir la entrada de nombres de lugares
                dbc.Input(id='input-origen', type='text', placeholder='Ej: Plaza de Armas o 139158914',
                          className="mb-2"),
                dbc.Button('Fijar Origen', id='btn-fijar-origen', n_clicks=0, color="info", size="sm",
                           className="w-100"),

                dbc.Label("🏁 Destino (Texto o ID de Nodo):", className="mt-4 form-label"),
                # CAMBIO: type='text' para permitir la entrada de nombres de lugares
                dbc.Input(id='input-destino', type='text', placeholder='Ej: Cerro San Cristóbal o 139162483',
                          className="mb-2"),
                dbc.Button('Fijar Destino', id='btn-fijar-destino', n_clicks=0, color="danger", size="sm",
                           className="w-100"),

                # CAMBIO: El mensaje de estado ahora es más genérico
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
                html.Hr(className="my-4"),

            ], width=3, className="p-4 bg-light border-end shadow-sm",
                style={'minHeight': '90vh', 'overflowY': 'auto'}),  # Mantener minHeight para alinear

            # 2. MAPA PRINCIPAL (Ocupa el espacio restante)
            dbc.Col([
                dcc.Graph(
                    id='mapa-2d-interactivo',
                    # Aseguramos que la altura sea 100% del contenedor Col
                    style={'height': '100%', 'width': '100%'},
                    config={
                        'displayModeBar': False,
                        'scrollZoom': True,
                        'doubleClick': 'reset'
                    }
                ),
                dbc.Collapse(
                    html.Div([
                        html.H5("Vista 3D (Altitud y Peligrosidad)", className="mt-3 text-center text-info"),
                        dcc.Graph(id='mapa-3d-altitud', style={'height': '60vh'})
                    ], className="p-3 bg-light border rounded shadow-sm mt-3"),
                    id="collapse-3d",
                    is_open=False,
                ),
            ], width=9, className="p-0", style={'height': '90vh'}),  # Fijamos la altura del contenedor del mapa
        ], className="g-0"),

        # --- COMPONENTE DE ESTADO INVISIBLE PARA EL MODO DE CLIC ---
        dcc.Store(id='store-click-mode', data={'next_selection': 'ORIGEN'}),
        # ----------------------------------------------------------
        html.Hr(className="mb-4"),
    ], fluid=True, className="p-0")

# --- Función de Ayuda: Encontrar el Nodo Más Cercano ---
import numpy as np
import dash.exceptions
from dash import no_update


def get_nearest_node(graph, lat, lon):
    """Encuentra el ID del nodo en el grafo más cercano a las coordenadas (lat, lon) dadas."""
    if graph is None or not graph.nodes:
        return None

    node_data = {node: (data['x'], data['y'])
                 for node, data in graph.nodes(data=True)
                 if 'x' in data and 'y' in data}

    if not node_data:
        return None

    node_ids = list(node_data.keys())
    coords = np.array(list(node_data.values()))

    # Punto objetivo (lon, lat)
    target = np.array([lon, lat])

    # Calcular la distancia cuadrada
    distances = np.sum((coords - target) ** 2, axis=1)

    # Encontrar el índice del nodo más cercano
    nearest_node_index = np.argmin(distances)
    nearest_node_id = node_ids[nearest_node_index]

    return nearest_node_id


# --- CALLBACKS (LÓGICA INTERACTIVA DE PYTHON) ---

# Callback de Modo de Clic: Actualiza la etiqueta de modo
@app.callback(
    Output('click-mode-label', 'children'),
    Input('store-click-mode', 'data')
)
def update_click_mode_label(mode_data):
    mode = mode_data['next_selection']
    if mode == 'ORIGEN':
        return html.Span(["Modo Click: ", html.B("Seleccionar Origen")], className="text-info")
    else:
        return html.Span(["Modo Click: ", html.B("Seleccionar Destino")], className="text-danger")


# Callback de Click en el Mapa: Lee el clic, encuentra el nodo y actualiza los inputs
# CAMBIO: Ahora actualiza los STORES de ID, no los inputs de texto.
@app.callback(
    [
        Output('store-origen-id', 'data', allow_duplicate=True),
        Output('store-destino-id', 'data', allow_duplicate=True),
        Output('store-click-mode', 'data', allow_duplicate=True)
    ],
    Input('mapa-2d-interactivo', 'clickData'),
    State('store-click-mode', 'data'),
    prevent_initial_call=True
)
def handle_map_click(clickData, mode_data):
    """
    Maneja el evento de click en el mapa para seleccionar el nodo más cercano
    y alternar entre Origen y Destino.
    """
    global G

    if clickData is None or G is None:
        return no_update, no_update, no_update

    # 1. Obtener coordenadas
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']

    # 2. Encontrar el nodo más cercano
    node_id = get_nearest_node(G, lat, lon)

    if node_id is None:
        return no_update, no_update, no_update

    node_id_int = int(node_id)  # Se almacena como INT para consistencia

    # 3. Determinar qué campo actualizar y alternar el modo
    current_mode = mode_data['next_selection']

    if current_mode == 'ORIGEN':
        # Actualizar Origen y cambiar el modo a DESTINO
        new_mode_data = {'next_selection': 'DESTINO'}
        return node_id_int, no_update, new_mode_data

    elif current_mode == 'DESTINO':
        # Actualizar Destino y cambiar el modo a ORIGEN
        new_mode_data = {'next_selection': 'ORIGEN'}
        return no_update, node_id_int, new_mode_data

    return no_update, no_update, no_update



# --- Callback de Manejo de Texto: Convierte texto a ID y actualiza Stores ---
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
    global PUNTO_ORIGEN, PUNTO_DESTINO, RUTA_DIJKSTRA, RUTA_ASTAR

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    # 1. Lógica de Reseteo
    if trigger_id == 'btn-reset':
        RUTA_DIJKSTRA = None
        RUTA_ASTAR = None
        return None, None, html.Div(["👆 Fija Origen y Destino usando texto o IDs."], className="fw-bold")

    # Inicializar las salidas
    new_origen_id = current_origen_id
    new_destino_id = current_destino_id
    status_msg = no_update

    # 2. Manejo de Origen
    if trigger_id == 'btn-fijar-origen':
        # Si el campo de texto está vacío, limpiamos el origen
        if not origen_text:
            new_origen_id = None
            status_msg = no_update  # Se actualizará con el callback de estado general
        else:
            try:
                # Intentar buscar el nodo (Nota: texto_a_nodo en tu string_to_node.py solo necesita el texto)
                node_id = texto_a_nodo(
                    origen_text)  # Asumimos que G no es necesario si usa 'map_clean.osm' internamente.

                # Validar si el texto introducido es el ID de un nodo válido (alternativa)
                if node_id == -1:
                    try:
                        potential_id = int(origen_text)
                        if potential_id in G:
                            node_id = potential_id
                    except ValueError:
                        pass  # No es un ID numérico, seguimos con -1

                if node_id != -1 and node_id in G:
                    new_origen_id = int(node_id)
                    # Limpiamos las rutas
                    RUTA_DIJKSTRA = None
                    RUTA_ASTAR = None
                    status_msg = no_update  # Se actualizará con el callback de estado general
                else:
                    new_origen_id = None  # Limpia el store si hay error
                    status_msg = html.Div([f"⛔ Error: Origen '{origen_text}' no encontrado o inválido."],
                                          className="text-danger")

            except Exception as e:
                new_origen_id = None
                # ***ESTE MENSAJE PERSISTIRÁ porque este callback es el que lo genera***
                status_msg = html.Div([f"⛔ Error en la búsqueda de Origen: {e}"], className="text-danger")


    # 3. Manejo de Destino
    elif trigger_id == 'btn-fijar-destino':
        if not destino_text:
            new_destino_id = None
            status_msg = no_update  # Se actualizará con el callback de estado general
        else:
            try:
                node_id = texto_a_nodo(
                    destino_text)  # Asumimos que G no es necesario si usa 'map_clean.osm' internamente.

                # Validar si el texto introducido es el ID de un nodo válido (alternativa)
                if node_id == -1:
                    try:
                        potential_id = int(destino_text)
                        if potential_id in G:
                            node_id = potential_id
                    except ValueError:
                        pass  # No es un ID numérico, seguimos con -1

                # Validación adicional: Destino no puede ser igual a Origen
                is_valid = node_id != -1 and node_id in G and int(node_id) != current_origen_id

                if is_valid:
                    new_destino_id = int(node_id)

                    # Limpiamos las rutas
                    RUTA_DIJKSTRA = None
                    RUTA_ASTAR = None
                    status_msg = no_update  # Se actualizará con el callback de estado general
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
                # ***ESTE MENSAJE PERSISTIRÁ porque este callback es el que lo genera***
                status_msg = html.Div([f"⛔ Error en la búsqueda de Destino: {e}"], className="text-danger")

    return new_origen_id, new_destino_id, status_msg



# Callback 2: Dibujar el Mapa 2D y la Ruta (Versión Estabilizada y unificada)
@app.callback(
    Output('mapa-2d-interactivo', 'figure'),
    [Input('btn-reset', 'n_clicks'),
     Input('ruta-output', 'children'),
     # CAMBIO: Usamos los valores de los STORES para disparar el redibujo
     Input('store-origen-id', 'data'),
     Input('store-destino-id', 'data')],
    [State('mapa-2d-interactivo', 'figure')]
)
def update_map_and_selection(reset_clicks,
                             ruta_output_children,
                             origen_id, destino_id,  # Los valores de los STORES
                             current_figure):
    global PUNTO_ORIGEN, PUNTO_DESTINO, RUTA_DIJKSTRA, RUTA_ASTAR, G, nodes_df, edges_lines

    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    # 1. Lógica de Reseteo (Se mantiene)
    if trigger_id == 'btn-reset':
        PUNTO_ORIGEN = None
        PUNTO_DESTINO = None
        RUTA_DIJKSTRA = None
        RUTA_ASTAR = None
        # Ya no hace falta anular aquí, pues lo hace el callback de texto.

    # --- LÓGICA DE ASIGNACIÓN DE PUNTO (Activado por Store) ---
    # Lo más importante es que las variables globales se actualicen
    # con los valores que vienen de los STORES.
    # El store de Origen y Destino solo se actualiza si el valor es un ID VÁLIDO o None.

    PUNTO_ORIGEN = origen_id if origen_id is not None and origen_id in G else None
    PUNTO_DESTINO = destino_id if destino_id is not None and destino_id in G else None

    # --- FIN LÓGICA DE ASIGNACIÓN ---

    # --- INICIO DE DIBUJO ---
    # 4. Generar la figura base (sin traza de calles de fondo)
    fig = go.Figure()

    # Añadir una traza invisible para forzar el renderizado del mapa base
    fig.add_trace(go.Scattermapbox(
        lat=[nodes_df['lat'].mean()],
        lon=[nodes_df['lon'].mean()],
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Las líneas de fondo del grafo han sido removidas para simplificar la vista

    # --- Lógica de Control de Zoom y Centro ---

    # 5. Configuración de Layout base
    layout_updates = dict(
        mapbox_style="open-street-map",  # Cambiado a OpenStreetMap para mostrar mapa base sin token
        mapbox_bounds={"west": nodes_df['lon'].min(), "east": nodes_df['lon'].max(),
                       "south": nodes_df['lat'].min(), "north": nodes_df['lat'].max()},

        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        hovermode="closest",

        # uirevision por defecto (mantiene la vista)
        uirevision=str(PUNTO_ORIGEN) + str(PUNTO_DESTINO) + str(bool(RUTA_DIJKSTRA)) + str(bool(RUTA_ASTAR)),

        title="Mapa de Calles de Santiago (Selección por ID/Texto)",
    )

    # 5.1. Lógica para centrar en el punto recién fijado (SOLO SI SE FIJÓ UN PUNTO)
    target_node_id = None

    # Si el trigger vino del callback que actualiza los stores (por click o por botón)
    if trigger_id in ['store-origen-id', 'store-destino-id'] and (PUNTO_ORIGEN or PUNTO_DESTINO):

        # Si el origen se acaba de fijar (o cambió) y no es nulo
        if trigger_id == 'store-origen-id' and PUNTO_ORIGEN is not None:
            target_node_id = PUNTO_ORIGEN
        # Si el destino se acaba de fijar (o cambió) y no es nulo
        elif trigger_id == 'store-destino-id' and PUNTO_DESTINO is not None:
            target_node_id = PUNTO_DESTINO

    if target_node_id and target_node_id in G:
        # Centrar en el punto recién fijado y hacer zoom (nivel 16 es a nivel de calle)
        layout_updates['mapbox_zoom'] = 16
        layout_updates['mapbox_center'] = {"lat": G.nodes[target_node_id]['y'], "lon": G.nodes[target_node_id]['x']}
        # IMPORTANTE: Rompemos uirevision para forzar el cambio de vista/zoom
        layout_updates['uirevision'] = 'center_on_' + str(target_node_id)

    elif (PUNTO_ORIGEN is None and PUNTO_DESTINO is None) or (trigger_id == 'btn-reset'):
        # Centrar en la vista general si es el estado inicial
        layout_updates['mapbox_zoom'] = 13
        layout_updates['mapbox_center'] = {"lat": nodes_df['lat'].mean(), "lon": nodes_df['lon'].mean()}
        # Si es reset, forzamos el reinicio de la vista
        if trigger_id == 'btn-reset':
            layout_updates['uirevision'] = 'reset_view'

    # Aplicar el layout (incluyendo la corrección de altura y la nueva lógica de centrado)
    fig.update_layout(**layout_updates)

    # La leyenda debe flotar sobre el mapa (corrección anterior)
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

    # 6. Dibujar los marcadores de selección (Origen/Destino) - CORREGIDO PARA VISIBILIDAD
    if PUNTO_ORIGEN is not None and PUNTO_ORIGEN in G:
        data = G.nodes[PUNTO_ORIGEN]
        lat_o = [data['y']]
        lon_o = [data['x']]

        # 🟢 Traza 1: Contorno del marcador (Negro)
        fig.add_trace(go.Scattermapbox(
            lat=lat_o,
            lon=lon_o,
            mode='markers',
            marker=dict(size=20, color='black', symbol='circle', opacity=1.0),
            name='Origen (Contorno)',
            hoverinfo='skip',
            showlegend=False
        ))

        # 🟢 Traza 2: Marcador principal (Color brillante)
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

    if PUNTO_DESTINO is not None and PUNTO_DESTINO in G:
        data = G.nodes[PUNTO_DESTINO]
        lat_d = [data['y']]
        lon_d = [data['x']]

        # 🔴 Traza 1: Contorno del marcador (Negro)
        fig.add_trace(go.Scattermapbox(
            lat=lat_d,
            lon=lon_d,
            mode='markers',
            marker=dict(size=20, color='black', symbol='circle', opacity=1.0),
            name='Destino (Contorno)',
            hoverinfo='skip',
            showlegend=False
        ))

        # 🔴 Traza 2: Marcador principal (Color brillante)
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

    # 7. Dibujar las Rutas (Líneas)
    def dibujar_ruta(ruta_nodos: Optional[List[int]], nombre: str, color: str, opacidad: float = 1.0):
        """Función auxiliar para añadir una ruta (línea) al Plotly figure."""
        if ruta_nodos and G is not None:
            route_coords = [G.nodes[nid] for nid in ruta_nodos if nid in G.nodes]
            route_lats = [coord['y'] for coord in route_coords]
            route_lons = [coord['x'] for coord in route_coords]
            
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lat=route_lats,
                lon=route_lons,
                line=dict(width=5, color=color),
                opacity=opacidad,
                name=nombre,
                showlegend=True
            ))

    dibujar_ruta(RUTA_DIJKSTRA, "Ruta Dijkstra (Costo Mínimo)", "blue", 1.0)
    dibujar_ruta(RUTA_ASTAR, "Ruta A* (Heurística)", "orange", 1.0)

    # 8. Dibujar los Nodos Intermedios de las Rutas (Puntos)
    def dibujar_nodos_ruta(ruta_nodos: Optional[List[int]], nombre: str, color: str):
        """Función auxiliar para añadir los nodos de una ruta al Plotly figure."""
        if ruta_nodos and G is not None:
            node_ids_to_plot = [nid for nid in ruta_nodos if
                                nid != PUNTO_ORIGEN and nid != PUNTO_DESTINO and nid in G.nodes]

            if not node_ids_to_plot:
                return

            route_coords = [G.nodes[nid] for nid in node_ids_to_plot]
            route_lats = [coord['y'] for coord in route_coords]
            route_lons = [coord['x'] for coord in route_coords]

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

    dibujar_nodos_ruta(RUTA_DIJKSTRA, "Dijkstra", "#0077B6")
    dibujar_nodos_ruta(RUTA_ASTAR, "A*", "#E63946")

    return fig


# Callback 3: Actualizar el estado de la selección (ACTUALIZADO para solo manejar estados OK)
@app.callback(
    Output("selection-status", "children", allow_duplicate=True),
    [Input('store-origen-id', 'data'),
     Input('store-destino-id', 'data')],
    # Se eliminó el 'btn-reset' de Input, ya que el callback de texto lo maneja.
    prevent_initial_call=True
)
def update_status_text_from_store(origen_id, destino_id):
    ctx = dash.callback_context
    # Si el trigger fue de un store que se acaba de limpiar (poner en None),
    # significa que hubo un error que el callback de texto ya reportó.
    # Evitamos sobrescribir ese error.

    # Si origen_id y destino_id son None (después de un reset o un error de texto en ambos)
    if origen_id is None and destino_id is None:
        return html.Div(["👆 Fija Origen y Destino usando texto o IDs."], className="fw-bold")

    # Si solo origen está fijo
    elif origen_id is not None and destino_id is None:
        return html.Div([
            html.B("✅ Origen Fijado. "),
            html.Span(f"ID: {origen_id}. Ahora selecciona el Destino.")
        ])

    # Si ambos están fijos
    elif origen_id is not None and destino_id is not None:
        return html.Div([
            html.B("🎉 Listo: "),
            html.Span(f"Origen ({origen_id}) y Destino ({destino_id}). ¡Calcula la ruta!")
        ], className="text-success fw-bold")

    # Si solo destino está fijo (caso poco probable pero cubierto)
    else:
        return html.Div(["Fija Origen y Destino usando texto o IDs."], className="fw-bold")



# Callback 5: Calcular y almacenar las rutas (Corregido el error de Dijkstra)
@app.callback(
    [Output('ruta-output', 'children'),
     Output('ruta-nodos-list', 'children')],
    [Input('btn-ruta', 'n_clicks')],
    # CAMBIO: Usamos los STORES para obtener el Origen/Destino final
    [State('store-origen-id', 'data'),
     State('store-destino-id', 'data')],
    prevent_initial_call=True
)
def calcular_rutas(n_clicks, origen_id, destino_id):
    global RUTA_DIJKSTRA, RUTA_ASTAR, PUNTO_ORIGEN, PUNTO_DESTINO

    # Importante: Actualizamos las globales de ruteo con los datos de los stores
    PUNTO_ORIGEN = origen_id
    PUNTO_DESTINO = destino_id

    if G_CUSTOM is None or PUNTO_ORIGEN is None or PUNTO_DESTINO is None:
        return dbc.Alert("Seleccione Origen y Destino válidos antes de calcular.", color="danger"), None

    W_DIST = 1.0
    W_ELEV = 0.0
    W_SEG = 1000.0

    # 1. Ejecutar Dijkstra
    RUTA_DIJKSTRA = None
    try:
        # Se asume que routing.dijkstra devuelve (costos, predecesores)
        cost_dijkstra, prev_dijkstra = routing.dijkstra(G_CUSTOM, PUNTO_ORIGEN, PUNTO_DESTINO,
                                                        w_dist=W_DIST, w_elev=W_ELEV, w_seg=W_SEG)

        # Corrección: Asegurarse de que el camino fue encontrado antes de reconstruir
        if PUNTO_DESTINO in prev_dijkstra and prev_dijkstra[PUNTO_DESTINO] is not None:
            RUTA_DIJKSTRA = routing.reconstruir_camino(prev_dijkstra, PUNTO_ORIGEN, PUNTO_DESTINO)
        else:
            print("Dijkstra no encontró camino.")
    except Exception as e:
        print(f"Error en Dijkstra: {e}")

    # 2. Ejecutar A*
    RUTA_ASTAR = None
    try:
        RUTA_ASTAR = routing.a_estrella(G_CUSTOM, PUNTO_ORIGEN, PUNTO_DESTINO,
                                        w_dist=W_DIST, w_elev=W_ELEV, w_seg=W_SEG)
    except Exception as e:
        print(f"Error en A*: {e}")

    # 3. Generación del Listado de Nodos
    ruta_nodos_html = []
    dijkstra_encontrada = bool(RUTA_DIJKSTRA)
    astar_encontrada = bool(RUTA_ASTAR)

    if dijkstra_encontrada:
        ruta_nodos_html.append(html.B("Ruta Dijkstra (IDs):", className="d-block text-primary mt-2"))
        dijkstra_ids_str = ' → '.join(map(str, RUTA_DIJKSTRA))
        ruta_nodos_html.append(html.P(dijkstra_ids_str, style={'wordBreak': 'break-all'}))
    else:
        ruta_nodos_html.append(html.P("Dijkstra: No se encontró camino.", className="text-muted mt-2"))

    if astar_encontrada:
        ruta_nodos_html.append(html.B("Ruta A* (IDs):", className="d-block text-warning mt-3"))
        astar_ids_str = ' → '.join(map(str, RUTA_ASTAR))
        ruta_nodos_html.append(html.P(astar_ids_str, style={'wordBreak': 'break-all'}))
    else:
        ruta_nodos_html.append(html.P("A*: No se encontró camino.", className="text-muted"))

    # 4. Respuesta Final
    if dijkstra_encontrada or astar_encontrada:
        d_len = len(RUTA_DIJKSTRA) if dijkstra_encontrada else 0
        a_len = len(RUTA_ASTAR) if astar_encontrada else 0

        # Muestra si las rutas son iguales, abordando tu pregunta.
        rutas_iguales = dijkstra_encontrada and astar_encontrada and RUTA_DIJKSTRA == RUTA_ASTAR

        mensaje = f"Rutas calculadas. Dijkstra: {d_len} nodos. A*: {a_len} nodos."

        if rutas_iguales:
            mensaje += " **¡Ambos algoritmos encontraron la misma ruta óptima!**"

        mensaje += " El mapa se ha actualizado."

        msg = dbc.Alert(mensaje, color="success")
        return msg, html.Div(ruta_nodos_html)
    else:
        return dbc.Alert("No se pudo encontrar ninguna ruta entre los puntos seleccionados.", color="danger"), None


# --- INICIO DE LA APLICACIÓN --- (SE MANTIENE IGUAL)
if __name__ == '__main__':
    if G is not None:
        print("Iniciando aplicación Dash...")
        app.run(debug=False, host="0.0.0.0", port=8050)
    else:
        pass