"""
Script para generar gráficas de análisis del proyecto E-Ciclos.
Genera visualizaciones basadas en los datos disponibles del grafo y algoritmos.
"""

import os
import sys
import time
import random
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para generar imágenes sin GUI
import numpy as np
from typing import List, Tuple, Dict

# Importar módulos del proyecto
try:
    from grafo import Grafo
    from nodo import Nodo
    from camino import Camino
    import routing
    import osmnx as ox
except ImportError as e:
    print(f"Error importando módulos del proyecto: {e}")
    sys.exit(1)

# Crear directorio para imágenes
OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cargar_grafo_real(osm_file="map_clean.osm"):
    """Carga el grafo real desde el archivo OSM de Santiago."""
    print(f"Cargando grafo desde {osm_file}...")
    
    if not os.path.exists(osm_file):
        print(f"ERROR: No se encontró el archivo {osm_file}")
        print("Asegúrate de que map_clean.osm está en el directorio actual.")
        sys.exit(1)
    
    try:
        # Cargar grafo desde OSM usando osmnx
        G_nx = ox.graph_from_xml(osm_file, simplify=False)
        print(f"Grafo NetworkX cargado: {len(G_nx.nodes)} nodos, {len(G_nx.edges)} aristas")
        
        # Enriquecer datos: altitud y peligrosidad
        print("Asignando datos de altura y peligrosidad...")
        alturas_reales = 0
        alturas_simuladas = 0
        
        for node, data in G_nx.nodes(data=True):
            data['lon'] = data['x']
            data['lat'] = data['y']
            
            # Intentar usar altitud real desde OSM (atributo 'ele')
            if 'altitud_m' not in data:
                try:
                    # Primero intentar leer 'ele' directamente del nodo OSM
                    if 'ele' in data:
                        ele_value = data['ele']
                        # Convertir a float de forma segura
                        if isinstance(ele_value, (int, float)):
                            data['altitud_m'] = float(ele_value)
                            alturas_reales += 1
                        elif isinstance(ele_value, str):
                            data['altitud_m'] = float(ele_value)
                            alturas_reales += 1
                        else:
                            raise ValueError("Formato de 'ele' no válido")
                    else:
                        # Fallback: altitud simulada si no existe 'ele'
                        data['altitud_m'] = 400 + (data['y'] * -100) + random.uniform(0, 50)
                        alturas_simuladas += 1
                except (ValueError, TypeError, KeyError):
                    # Si hay error al parsear 'ele', usar simulación
                    data['altitud_m'] = 400 + (data['y'] * -100) + random.uniform(0, 50)
                    alturas_simuladas += 1
            
            # Peligrosidad simulada (mejorable con datos CONASET)
            if 'peligrosidad' not in data:
                data['peligrosidad'] = random.uniform(0.1, 0.9)
        
        print(f"  - Alturas reales (desde OSM 'ele'): {alturas_reales}")
        print(f"  - Alturas simuladas (fallback): {alturas_simuladas}")
        
        # Convertir a grafo personalizado
        print("Convirtiendo a estructura de Grafo personalizada...")
        grafo = Grafo()
        grafo.cargar_desde_networkx(G_nx)
        
        print(f"✓ Grafo personalizado cargado: {len(grafo.nodos)} nodos, {len(grafo.caminos)} caminos")
        return grafo
        
    except Exception as e:
        print(f"ERROR cargando grafo: {e}")
        sys.exit(1)

def calcular_pendientes(grafo: Grafo) -> List[float]:
    """Calcula las pendientes de todas las aristas del grafo."""
    pendientes = []
    
    for camino in grafo.caminos.values():
        nodo_a, nodo_b = camino.nodos
        
        # Distancia horizontal (Haversine simplificado)
        d_h = routing.distancia_haversine(
            nodo_a.latitud, nodo_a.longitud,
            nodo_b.latitud, nodo_b.longitud
        )
        
        if d_h > 0:
            delta_h = nodo_b.altura - nodo_a.altura
            theta_rad = math.atan(delta_h / d_h)
            theta_deg = math.degrees(theta_rad)
            pendientes.append(theta_deg)
    
    return pendientes

def grafica_1_distribucion_pendientes(grafo: Grafo):
    """Gráfica 1: Distribución de pendientes en la red vial."""
    print("\n[1/6] Generando: Distribución de pendientes...")
    
    pendientes = calcular_pendientes(grafo)
    
    if not pendientes:
        return False
    
    plt.figure(figsize=(10, 6))
    plt.hist(pendientes, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Límite 15° (máx. transitable)')
    plt.axvline(x=-15, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Terreno plano')
    
    plt.xlabel('Pendiente (grados)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Pendientes en la Red Vial de Santiago', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    filepath = os.path.join(OUTPUT_DIR, 'pendientes.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    print(f"  - Pendientes analizadas: {len(pendientes)}")
    print(f"  - Rango: {min(pendientes):.1f}° a {max(pendientes):.1f}°")
    print(f"  - Media: {np.mean(pendientes):.1f}°")
    
    return True

def grafica_2_comparacion_esfuerzo(grafo: Grafo, origen_id: int, destino_id: int):
    """Gráfica 2: Comparación de esfuerzo entre rutas con diferentes ponderaciones."""
    print("\n[2/6] Generando: Comparación de esfuerzo...")
    
    # Verificar que los nodos existen
    if origen_id not in grafo.nodos or destino_id not in grafo.nodos:
        print("✗ Nodos de origen/destino no válidos")
        return False
    
    # Configuraciones de pesos a comparar
    configs = [
        ("Mínima distancia", {"w_dist": 1.0, "w_elev": 0.0, "w_seg": 0.0}),
        ("Mínimo esfuerzo", {"w_dist": 0.3, "w_elev": 0.6, "w_seg": 0.1}),
        ("Máxima seguridad", {"w_dist": 0.2, "w_elev": 0.1, "w_seg": 0.7}),
    ]
    
    resultados = []
    
    for nombre, pesos in configs:
        try:
            ruta = routing.a_estrella(grafo, origen_id, destino_id, **pesos)
            
            if ruta:
                # Calcular métricas de la ruta
                distancia_total = 0
                esfuerzo_total = 0
                riesgo_promedio = 0
                
                for i in range(len(ruta) - 1):
                    nodo_a = grafo.nodos[ruta[i]]
                    nodo_b = grafo.nodos[ruta[i + 1]]
                    
                    # Encontrar el camino entre estos nodos
                    camino = None
                    for c in nodo_a.caminos:
                        if c.obtener_otro_nodo(nodo_a) == nodo_b:
                            camino = c
                            break
                    
                    if camino:
                        d = routing.distancia_haversine(nodo_a.latitud, nodo_a.longitud,
                                                        nodo_b.latitud, nodo_b.longitud)
                        delta_h = nodo_b.altura - nodo_a.altura
                        F_i = math.sqrt(d**2 + max(0, delta_h)**2)
                        
                        distancia_total += d
                        esfuerzo_total += (F_i - d)
                        riesgo_promedio += nodo_b.prob_accidente
                
                riesgo_promedio /= len(ruta)
                
                resultados.append({
                    'nombre': nombre,
                    'distancia_km': distancia_total / 1000,
                    'esfuerzo_adicional_m': esfuerzo_total,
                    'riesgo_promedio': riesgo_promedio
                })
        except Exception as e:
            print(f"  Error calculando ruta para {nombre}: {e}")
    
    if len(resultados) < 2:
        print("✗ No se pudieron calcular suficientes rutas")
        return False
    
    # Crear gráfica de barras múltiples
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    nombres = [r['nombre'] for r in resultados]
    colores = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Distancia
    axes[0].bar(nombres, [r['distancia_km'] for r in resultados], color=colores, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Distancia (km)', fontsize=11)
    axes[0].set_title('Distancia Total', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Esfuerzo
    axes[1].bar(nombres, [r['esfuerzo_adicional_m'] for r in resultados], color=colores, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Esfuerzo adicional (m)', fontsize=11)
    axes[1].set_title('Esfuerzo por Elevación', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    # Riesgo
    axes[2].bar(nombres, [r['riesgo_promedio'] for r in resultados], color=colores, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Riesgo promedio [0-1]', fontsize=11)
    axes[2].set_title('Seguridad Vial', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.suptitle('Comparación de Rutas según Criterio de Optimización', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'comparacion_esfuerzo.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    for r in resultados:
        print(f"  - {r['nombre']}: {r['distancia_km']:.2f} km, esfuerzo +{r['esfuerzo_adicional_m']:.1f} m")
    
    return True

def grafica_3_mapa_calor_peligrosidad(grafo: Grafo):
    """Gráfica 3: Mapa de calor de peligrosidad."""
    print("\n[3/6] Generando: Mapa de calor de peligrosidad...")
    
    # Extraer coordenadas y peligrosidad
    lats = [nodo.latitud for nodo in grafo.nodos.values()]
    lons = [nodo.longitud for nodo in grafo.nodos.values()]
    peligrosidad = [nodo.prob_accidente for nodo in grafo.nodos.values()]
    
    plt.figure(figsize=(12, 8))
    
    # Usar hexbin para agrupar y suavizar datos (reduce ruido visual)
    # gridsize controla el nivel de suavizado (más bajo = más suavizado)
    hexbin = plt.hexbin(lons, lats, C=peligrosidad, gridsize=50, 
                        cmap='YlOrRd', reduce_C_function=np.mean,
                        mincnt=1, alpha=0.8, edgecolors='none')
    
    plt.colorbar(hexbin, label='Índice de Peligrosidad [0-1]')
    plt.xlabel('Longitud', fontsize=12)
    plt.ylabel('Latitud', fontsize=12)
    plt.title('Mapa de Calor: Peligrosidad Vial en Santiago', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    filepath = os.path.join(OUTPUT_DIR, 'mapa_calor_seguridad.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    print(f"  - Nodos analizados: {len(grafo.nodos)}")
    print(f"  - Peligrosidad media: {np.mean(peligrosidad):.3f}")
    print(f"  - Técnica: Agrupación hexagonal (gridsize=50)")
    
    return True

def grafica_4_tiempos_algoritmos(grafo: Grafo, origen_id: int, destino_id: int, repeticiones=10):
    """Gráfica 4: Tiempo de cálculo por algoritmo (Dijkstra vs A*)."""
    print("\n[4/6] Generando: Tiempos de algoritmos...")
    
    if origen_id not in grafo.nodos or destino_id not in grafo.nodos:
        print("✗ Nodos de origen/destino no válidos")
        return False
    
    tiempos_dijkstra = []
    tiempos_astar = []
    
    print(f"  Ejecutando {repeticiones} repeticiones de cada algoritmo...")
    
    for i in range(repeticiones):
        # Dijkstra
        start = time.perf_counter()
        _, _ = routing.dijkstra(grafo, origen_id, objetivo_id=destino_id)
        tiempo_dij = (time.perf_counter() - start) * 1000  # ms
        tiempos_dijkstra.append(tiempo_dij)
        
        # A*
        start = time.perf_counter()
        _ = routing.a_estrella(grafo, origen_id, destino_id)
        tiempo_a = (time.perf_counter() - start) * 1000  # ms
        tiempos_astar.append(tiempo_a)
    
    # Crear gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1.boxplot([tiempos_dijkstra, tiempos_astar], labels=['Dijkstra', 'A*'])
    ax1.set_ylabel('Tiempo (ms)', fontsize=11)
    ax1.set_title('Distribución de Tiempos', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Barras con promedio
    promedios = [np.mean(tiempos_dijkstra), np.mean(tiempos_astar)]
    desviaciones = [np.std(tiempos_dijkstra), np.std(tiempos_astar)]
    
    ax2.bar(['Dijkstra', 'A*'], promedios, yerr=desviaciones, 
            color=['#3498db', '#e74c3c'], alpha=0.7, capsize=5, edgecolor='black')
    ax2.set_ylabel('Tiempo promedio (ms)', fontsize=11)
    ax2.set_title('Comparación de Rendimiento', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Anotar speedup
    speedup = np.mean(tiempos_dijkstra) / np.mean(tiempos_astar)
    ax2.text(0.5, max(promedios) * 0.9, f'Speedup: {speedup:.2f}x', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Comparación de Tiempos de Cálculo (n={repeticiones} ejecuciones)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'tiempos_algoritmos.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    print(f"  - Dijkstra: {np.mean(tiempos_dijkstra):.2f} ± {np.std(tiempos_dijkstra):.2f} ms")
    print(f"  - A*: {np.mean(tiempos_astar):.2f} ± {np.std(tiempos_astar):.2f} ms")
    print(f"  - Speedup: {speedup:.2f}x")
    
    return True

def grafica_5_pendientes_descartadas(grafo: Grafo):
    """Gráfica 5: Proporción de rutas descartadas por pendiente (>15°)."""
    print("\n[5/6] Generando: Proporción de pendientes descartadas...")
    
    pendientes = calcular_pendientes(grafo)
    
    if not pendientes:
        return False
    
    total = len(pendientes)
    descartadas = sum(1 for p in pendientes if abs(p) > 15)
    validas = total - descartadas
    
    # Gráfica de pastel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    sizes = [validas, descartadas]
    labels = [f'Transitables\n({validas} aristas)', f'No transitables\n({descartadas} aristas)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Clasificación de Aristas por Pendiente', fontsize=12, fontweight='bold')
    
    # Histograma con zonas
    ax2.hist(pendientes, bins=50, color='gray', alpha=0.5, edgecolor='black')
    ax2.axvspan(-90, -15, alpha=0.3, color='red', label='No transitable (< -15°)')
    ax2.axvspan(15, 90, alpha=0.3, color='red', label='No transitable (> 15°)')
    ax2.axvspan(-15, 15, alpha=0.2, color='green', label='Transitable (-15° a 15°)')
    ax2.axvline(x=15, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=-15, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Pendiente (grados)', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.set_title('Distribución con Límites de Transitabilidad', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Análisis de Transitabilidad: Filtro de Pendiente Máxima (15°)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'pendientes_descartadas.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    print(f"  - Total aristas: {total}")
    print(f"  - Transitables: {validas} ({100*validas/total:.1f}%)")
    print(f"  - Descartadas: {descartadas} ({100*descartadas/total:.1f}%)")
    
    return True

def grafica_6_impacto_pesos(grafo: Grafo, origen_id: int, destino_id: int):
    """Gráfica 6: Impacto de los pesos (w_dist, w_elev, w_seg) en la ruta."""
    print("\n[6/6] Generando: Impacto de pesos...")
    
    if origen_id not in grafo.nodos or destino_id not in grafo.nodos:
        print("✗ Nodos de origen/destino no válidos")
        return False
    
    # Variar cada peso mientras se mantienen los otros constantes
    variaciones = np.linspace(0, 1, 11)
    
    resultados_dist = []
    resultados_elev = []
    resultados_seg = []
    
    print("  Calculando rutas con diferentes ponderaciones...")
    
    for w in variaciones:
        # Variar w_dist (mantener w_elev y w_seg bajos y balanceados)
        try:
            ruta = routing.a_estrella(grafo, origen_id, destino_id, 
                                      w_dist=w, w_elev=(1-w)*0.5, w_seg=(1-w)*0.5)
            if ruta:
                costo = calcular_costo_total(grafo, ruta, w, (1-w)*0.5, (1-w)*0.5)
                resultados_dist.append(costo)
            else:
                resultados_dist.append(None)
        except:
            resultados_dist.append(None)
        
        # Variar w_elev
        try:
            ruta = routing.a_estrella(grafo, origen_id, destino_id,
                                      w_dist=(1-w)*0.5, w_elev=w, w_seg=(1-w)*0.5)
            if ruta:
                costo = calcular_costo_total(grafo, ruta, (1-w)*0.5, w, (1-w)*0.5)
                resultados_elev.append(costo)
            else:
                resultados_elev.append(None)
        except:
            resultados_elev.append(None)
        
        # Variar w_seg
        try:
            ruta = routing.a_estrella(grafo, origen_id, destino_id,
                                      w_dist=(1-w)*0.5, w_elev=(1-w)*0.5, w_seg=w)
            if ruta:
                costo = calcular_costo_total(grafo, ruta, (1-w)*0.5, (1-w)*0.5, w)
                resultados_seg.append(costo)
            else:
                resultados_seg.append(None)
        except:
            resultados_seg.append(None)
    
    # Filtrar None
    resultados_dist = [r for r in resultados_dist if r is not None]
    resultados_elev = [r for r in resultados_elev if r is not None]
    resultados_seg = [r for r in resultados_seg if r is not None]
    
    if not resultados_dist and not resultados_elev and not resultados_seg:
        print("✗ No se pudieron calcular rutas con variaciones de pesos")
        return False
    
    plt.figure(figsize=(12, 7))
    
    if resultados_dist:
        plt.plot(variaciones[:len(resultados_dist)], resultados_dist, 
                'o-', label='Prioridad: Distancia (w_dist)', linewidth=2, markersize=6, color='#3498db')
    if resultados_elev:
        plt.plot(variaciones[:len(resultados_elev)], resultados_elev,
                's-', label='Prioridad: Esfuerzo (w_elev)', linewidth=2, markersize=6, color='#e74c3c')
    if resultados_seg:
        plt.plot(variaciones[:len(resultados_seg)], resultados_seg,
                '^-', label='Prioridad: Seguridad (w_seg)', linewidth=2, markersize=6, color='#2ecc71')
    
    plt.xlabel('Peso del criterio prioritario', fontsize=12)
    plt.ylabel('Costo total de la ruta', fontsize=12)
    plt.title('Sensibilidad de la Ruta Óptima a las Ponderaciones', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    filepath = os.path.join(OUTPUT_DIR, 'impacto_pesos.jpg')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Guardada: {filepath}")
    print(f"  - Variaciones evaluadas: {len(variaciones)}")
    
    return True

def calcular_costo_total(grafo: Grafo, ruta: List[int], w_dist: float, w_elev: float, w_seg: float) -> float:
    """Calcula el costo total de una ruta."""
    costo = 0.0
    for i in range(len(ruta) - 1):
        nodo_a = grafo.nodos[ruta[i]]
        nodo_b = grafo.nodos[ruta[i + 1]]
        
        # Encontrar el camino
        camino = None
        for c in nodo_a.caminos:
            if c.obtener_otro_nodo(nodo_a) == nodo_b:
                camino = c
                break
        
        if camino:
            costo += routing.coste_arista(camino, nodo_a, nodo_b, w_dist, w_elev, w_seg)
    
    return costo

def main():
    """Función principal que ejecuta la generación de todas las gráficas."""
    print("="*60)
    print("GENERACIÓN DE GRÁFICAS - PROYECTO E-CICLOS")
    print("="*60)
    
    # Cargar grafo REAL desde map_clean.osm
    grafo = cargar_grafo_real("map_clean.osm")
    
    # Seleccionar nodos de origen y destino aleatorios del grafo real
    nodos_ids = list(grafo.nodos.keys())
    origen_id = random.choice(nodos_ids)
    destino_id = random.choice(nodos_ids)
    
    while destino_id == origen_id:
        destino_id = random.choice(nodos_ids)
    
    print(f"\nNodos seleccionados para pruebas de rutas:")
    print(f"  - Origen: {origen_id}")
    print(f"  - Destino: {destino_id}")
    
    # Ejecutar generación de gráficas
    resultados = {
        "Distribución de pendientes": False,
        "Comparación de esfuerzo": False,
        "Mapa de calor de peligrosidad": False,
        "Tiempos de algoritmos": False,
        "Pendientes descartadas": False,
        "Impacto de pesos": False
    }
    
    try:
        resultados["Distribución de pendientes"] = grafica_1_distribucion_pendientes(grafo)
    except Exception as e:
        print(f"✗ Error en gráfica 1: {e}")
    
    try:
        resultados["Comparación de esfuerzo"] = grafica_2_comparacion_esfuerzo(grafo, origen_id, destino_id)
    except Exception as e:
        print(f"✗ Error en gráfica 2: {e}")
    
    try:
        resultados["Mapa de calor de peligrosidad"] = grafica_3_mapa_calor_peligrosidad(grafo)
    except Exception as e:
        print(f"✗ Error en gráfica 3: {e}")
    
    try:
        resultados["Tiempos de algoritmos"] = grafica_4_tiempos_algoritmos(grafo, origen_id, destino_id, repeticiones=10)
    except Exception as e:
        print(f"✗ Error en gráfica 4: {e}")
    
    try:
        resultados["Pendientes descartadas"] = grafica_5_pendientes_descartadas(grafo)
    except Exception as e:
        print(f"✗ Error en gráfica 5: {e}")
    
    try:
        resultados["Impacto de pesos"] = grafica_6_impacto_pesos(grafo, origen_id, destino_id)
    except Exception as e:
        print(f"✗ Error en gráfica 6: {e}")
    
    # Generar reporte
    print("\n" + "="*60)
    print("REPORTE FINAL")
    print("="*60)
    
    exitosas = sum(1 for v in resultados.values() if v)
    print(f"\nGráficas generadas exitosamente: {exitosas}/6")
    print(f"Directorio de salida: {OUTPUT_DIR}/")
    
    print("\n--- GRÁFICAS VIABLES ---")
    for nombre, exito in resultados.items():
        if exito:
            print(f"  ✓ {nombre}")
    
    print("\n--- GRÁFICAS NO VIABLES ---")
    for nombre, exito in resultados.items():
        if not exito:
            print(f"  ✗ {nombre}")
            print(f"    Razón: Error durante la generación o datos insuficientes")
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    
    # Guardar reporte en archivo
    with open(os.path.join(OUTPUT_DIR, 'REPORTE.txt'), 'w', encoding='utf-8') as f:
        f.write("REPORTE DE GENERACIÓN DE GRÁFICAS - PROYECTO E-CICLOS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total de gráficas generadas: {exitosas}/6\n\n")
        
        f.write("GRÁFICAS VIABLES:\n")
        for nombre, exito in resultados.items():
            if exito:
                f.write(f"  ✓ {nombre}\n")
        
        f.write("\nGRÁFICAS NO VIABLES:\n")
        for nombre, exito in resultados.items():
            if not exito:
                f.write(f"  ✗ {nombre}\n")
                f.write(f"    Justificación: Error durante procesamiento o datos simulados insuficientes\n")
        
        f.write("\nDATOS UTILIZADOS:\n")
        f.write("  - Grafo: map_clean.osm (red vial real de Santiago)\n")
        f.write("  - Alturas: Simuladas (mejorable con alturas_santiago.csv)\n")
        f.write("  - Peligrosidad: Simulada (mejorable con datos CONASET)\n")
    
    print(f"\nReporte guardado en: {os.path.join(OUTPUT_DIR, 'REPORTE.txt')}")

if __name__ == "__main__":
    main()
