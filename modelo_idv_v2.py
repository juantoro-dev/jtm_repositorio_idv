import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

# Configuración
warnings.filterwarnings('ignore')
CRS_9377 = "EPSG:9377"

def optimized_map_matching_with_progress(points_gdf, lines_gdf, max_distance, sample_interval=20, k_neighbors=5):
    """
    Map matching optimizado para grandes datasets con barra de progreso
    """
    print("INICIANDO MAP MATCHING CON CRS 9377")
    print(f"Puntos a procesar: {len(points_gdf):,}")
    print(f"Líneas disponibles: {len(lines_gdf):,}")
    
    # Hacer copias para no modificar los originales
    points = points_gdf.copy()
    lines = lines_gdf.copy()
    
    # Definir columnas específicas a preservar
    columns_to_preserve = [
        'time_stamp_event', 'plate', 'imei', 'company_name',
        'manufacturer_name', 'reference_name', 'event', 'longitude', 'latitude', 'ignition_status'
    ]
    
    # Asegurar que las columnas existen en points
    for col in columns_to_preserve:
        if col not in points.columns:
            # Si no existe, intentar buscar algo similar o llenar con NaN/placeholder
            if col == 'manufacturer_name': points[col] = 'Unknown'
            elif col == 'reference_name': points[col] = 'Unknown'
            elif col == 'company_name': points[col] = 'Unknown'
            elif col == 'event': points[col] = 'Unknown'
        
    print("\n CONVIRTIENDO A CRS 9377...")
    if points.crs != CRS_9377:
        points = points.to_crs(CRS_9377)
    if lines.crs != CRS_9377:
        lines = lines.to_crs(CRS_9377)
    print("Conversión CRS completada")
    
    print("\nAPLICANDO FILTRO ESPACIAL...")
    points_buffer = points.copy()
    points_buffer.geometry = points_buffer.geometry.buffer(max_distance * 2)
    
    candidate_pairs = gpd.sjoin(points_buffer, lines, how='inner', predicate='intersects')
    print(f"Filtro espacial completado: {len(candidate_pairs):,} pares candidatos encontrados")
    
    if len(candidate_pairs) == 0:
        print("No se encontraron candidatos. Revisa los datos o aumenta max_distance")
        return gpd.GeoDataFrame(columns=['point_id', 'line_id', 'distance', 'geometry'] + columns_to_preserve, crs=CRS_9377)
    
    print("\nMUESTREANDO PUNTOS EN LÍNEAS...")
    line_samples = []
    line_indices = []
    line_geometries = []
    
    unique_line_ids = candidate_pairs.index_right.unique()
    
    for line_idx in tqdm(unique_line_ids, desc="Muestreando líneas"):
        line = lines.loc[line_idx]
        geom = line.geometry
        
        if geom.length > 0:
            num_samples = max(2, int(geom.length / sample_interval))
            distances = np.linspace(0, geom.length, num_samples)
            
            for dist in distances:
                sample_point = geom.interpolate(dist)
                line_samples.append([sample_point.x, sample_point.y])
                line_indices.append(line_idx)
                line_geometries.append(geom)
    
    print(f"Muestreo completado: {len(line_samples):,} puntos de referencia")
    
    if not line_samples:
        raise ValueError("No se pudieron extraer puntos de las líneas")
    
    # Construir índice espacial con BallTree
    print("\nCONSTRUYENDO ÍNDICE ESPACIAL...")
    line_points_array = np.array(line_samples)
    tree = BallTree(line_points_array, metric='euclidean')
    print("Índice espacial construido")
    
    print("\nREALIZANDO MAP MATCHING...")
    results = []
    points_with_candidates = candidate_pairs.index.unique()
    
    for point_idx in tqdm(points_with_candidates, desc="Procesando puntos"):
        point_row = points.loc[point_idx]
        point_geom = point_row.geometry
        point_coord = [[point_geom.x, point_geom.y]]
        
        # Buscar k vecinos más cercanos
        distances, indices = tree.query(point_coord, k=min(k_neighbors, len(line_samples)), return_distance=True)
        
        best_match = None
        best_distance = float('inf')
        
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= max_distance:
                line_idx = line_indices[idx]
                line_geom = line_geometries[idx]
                
                # Calcular punto exacto más cercano en la línea
                closest_point = line_geom.interpolate(line_geom.project(point_geom))
                actual_distance = point_geom.distance(closest_point)
                
                if actual_distance <= max_distance and actual_distance < best_distance:
                    best_distance = actual_distance
                    
                    best_match = {
                        'point_id': point_idx,
                        'line_id': line_idx,
                        'distance': actual_distance,
                        'geometry': closest_point,
                        'original_point_geometry': point_geom
                    }
                    
                    # Añadir todas las columnas específicas a preservar
                    for col in columns_to_preserve:
                        if col in point_row:
                            best_match[col] = point_row[col]
        
        if best_match:
            results.append(best_match)
    
    # Crear GeoDataFrame con resultados
    if not results:
        print("No se encontraron matches dentro de la distancia especificada")
        return gpd.GeoDataFrame(columns=['point_id', 'line_id', 'distance', 'geometry', 'original_point_geometry', 'ignition_status'] + columns_to_preserve, crs=CRS_9377)
    
    results_df = pd.DataFrame(results)
    
    if 'geometry' in results_df.columns and len(results_df) > 0:
        result_gdf = gpd.GeoDataFrame(
            results_df, 
            geometry='geometry',
            crs=CRS_9377
        )
    else:
        print("No hay geometrías válidas en los resultados")
        return gpd.GeoDataFrame(columns=['point_id', 'line_id', 'distance', 'geometry', 'original_point_geometry', 'ignition_status'] + columns_to_preserve, crs=CRS_9377)
    
    # Mostrar estadísticas finales
    print("\nESTADÍSTICAS FINALES:")
    print(f"   • Puntos procesados: {len(points):,}")
    print(f"   • Matches encontrados: {len(result_gdf):,}")
    print(f"   • Tasa de éxito: {len(result_gdf)/len(points)*100:.1f}%")
    
    if len(result_gdf) > 0:
        print(f"   • Distancia promedio: {result_gdf['distance'].mean():.2f} m")
        print(f"   • Distancia mínima: {result_gdf['distance'].min():.2f} m")
        print(f"   • Distancia máxima: {result_gdf['distance'].max():.2f} m")
        print(f"   • Puntos sin match: {len(points) - len(result_gdf):,}")
    
    print("MAP MATCHING COMPLETADO")
    return result_gdf

def preparar_datos(points_gdf, lines_gdf):
    """
    Prepara los datos para el map matching
    """
    print("VERIFICANDO DATOS...")
    # Verificar que hay geometrías válidas
    points_valid = points_gdf[points_gdf.geometry.is_valid & points_gdf.geometry.notna()]
    lines_valid = lines_gdf[lines_gdf.geometry.is_valid & lines_gdf.geometry.notna()]   
    return points_valid, lines_valid

def crear_rutas_por_viaje(gdf_puntos, gdf_vias, buffer_distancia=50, crs_proyectado='EPSG:9377', verbose=True):
    """
    Crea rutas segmentadas por tipo de vía a partir de puntos de mapmatching
    usando ignition_status para identificar viajes
    """
    
    def log(msg):
        if verbose:
            print(msg)
    
    log("="*70)
    log("INICIANDO CREACIÓN DE RUTAS POR VIAJE")
    log("="*70)
        
    # 1. VERIFICAR Y REPARAR GEOMETRÍAS
    puntos_invalidos = (~gdf_puntos.geometry.is_valid).sum()
    vias_invalidas = (~gdf_vias.geometry.is_valid).sum()
    
    if puntos_invalidos > 0:
        gdf_puntos = gdf_puntos[gdf_puntos.geometry.is_valid].copy()
    
    if vias_invalidas > 0:
        gdf_vias = gdf_vias[gdf_vias.geometry.is_valid].copy()
    
    # 2. CONVERTIR A CRS PROYECTADO
    crs_original = gdf_puntos.crs
    
    def convertir_a_proyectado(gdf, nombre, crs_objetivo):
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
            gdf = gdf.to_crs(crs_objetivo)
        elif str(gdf.crs) != crs_objetivo:
            log(f"   {nombre}: de {gdf.crs} a {crs_objetivo}")
            try:
                gdf = gdf.to_crs(crs_objetivo)
            except Exception as e:
                gdf = gdf.to_crs('EPSG:4326')
                gdf = gdf.to_crs(crs_objetivo)
        return gdf
    
    gdf_puntos_proj = convertir_a_proyectado(gdf_puntos, "Puntos", crs_proyectado)
    gdf_vias_proj = convertir_a_proyectado(gdf_vias, "Vías", crs_proyectado)
    
    # 3. VERIFICAR SUPERPOSICIÓN
    bbox_puntos = gdf_puntos_proj.total_bounds
    bbox_vias = gdf_vias_proj.total_bounds
    
    superposicion_x = (bbox_puntos[0] <= bbox_vias[2]) and (bbox_puntos[2] >= bbox_vias[0])
    superposicion_y = (bbox_puntos[1] <= bbox_vias[3]) and (bbox_puntos[3] >= bbox_vias[1])
    
    if not (superposicion_x and superposicion_y):
        centro_puntos = gdf_puntos_proj.unary_union.centroid
        centro_vias = gdf_vias_proj.unary_union.centroid
        distancia_centros = centro_puntos.distance(centro_vias)
                
        if distancia_centros > 10000:
            nuevo_buffer = min(5000, int(distancia_centros * 0.1))
            buffer_distancia = nuevo_buffer

    # 4. VERIFICAR IGNITION_STATUS
    if 'ignition_status' not in gdf_puntos_proj.columns:
        log(" ERROR: No se encontró la columna 'ignition_status'")
        return gpd.GeoDataFrame()
    
    # 5. PROCESAR TIMESTAMPS
    if not pd.api.types.is_datetime64_any_dtype(gdf_puntos_proj['time_stamp_event']):
        try:
            gdf_puntos_proj['time_stamp_event'] = pd.to_datetime(gdf_puntos_proj['time_stamp_event'])
        except Exception as e:
            log(f" Error convirtiendo timestamps: {e}")
            return gpd.GeoDataFrame()

    # 6. CONVERTIR IGNITION_STATUS A BOOLEANO
    if gdf_puntos_proj['ignition_status'].dtype == 'object':
        gdf_puntos_proj['ignition_bool'] = gdf_puntos_proj['ignition_status'].apply(
            lambda x: str(x).lower().strip() in ['true', '1', 'yes', 'on', 'encendido', 'si', 't', 'sí', 'encendida', 'verdadero']
        )
    else:
        gdf_puntos_proj['ignition_bool'] = gdf_puntos_proj['ignition_status'].astype(bool)

    # 7. IDENTIFICAR VIAJES
    log("\n IDENTIFICANDO VIAJES:")
    gdf_puntos_proj = gdf_puntos_proj.sort_values(['imei', 'plate', 'time_stamp_event'])
    gdf_puntos_proj['viaje_id'] = None
    
    viaje_actual = 0
    
    # Agrupar por vehículo
    for (imei, plate), puntos_vehiculo in gdf_puntos_proj.groupby(['imei', 'plate']):
        # Copia para trabajar
        puntos_vehiculo = puntos_vehiculo.sort_values('time_stamp_event')
        
        # Lógica de detección de viajes vectorizada (más o menos) o iterativa rápida
        # Mantenemos la lógica iterativa original que funciona bien para cambios de estado
        
        indices = puntos_vehiculo.index
        ignitions = puntos_vehiculo['ignition_bool'].values
        
        # Detectar cambios de estado y continuidad
        # 0: apagado, 1: encendido
        # Viaje: secuencia de encendidos.
        
        # Simplificación: un viaje es una secuencia continua donde ignition es True
        # Y tal vez tolerar pequeños gaps. Pero la lógica original usaba cambios de estado.
        
        # Vamos a usar una lógica vectorizada para asignar IDs de viaje
        # Un nuevo viaje comienza cuando ignition pasa de False a True
        # O es el primer registro y es True.
        
        # Shift para comparar con anterior
        prev_ignition = np.roll(ignitions, 1)
        prev_ignition[0] = False # Asumir anterior apagado para el primero
        
        # Inicio de viaje: Actual True y (Anterior False o es el primero)
        starts = (ignitions == True) & (prev_ignition == False)
        
        # Asignar ID de viaje acumulativo
        # Cumsum de los starts nos da un ID único incrementing
        # Solo válido donde ignition es True
        trip_ids = np.cumsum(starts)
        
        # Donde es False, poner NaN o 0? Mejor poner NaN y luego filtrar
        final_trip_ids = pd.Series(trip_ids, index=indices).where(ignitions == True, np.nan)
        
        # Asignar al dataframe original
        # Convertir a string para que no sea float
        if not final_trip_ids.isna().all():
             # Crear ID único global: imei_viaje_X
             # Usamos el índice original para asignar
             gdf_puntos_proj.loc[indices, 'viaje_id'] = final_trip_ids.apply(
                 lambda x: f"{imei}_viaje_{int(x)}" if not np.isnan(x) else None
             )

    gdf_viajes = gdf_puntos_proj[gdf_puntos_proj['viaje_id'].notna()].copy()
    log(f"   Viajes identificados: {gdf_viajes['viaje_id'].nunique()}")
    log(f"   Puntos en viajes: {len(gdf_viajes):,}")
    
    if len(gdf_viajes) == 0:
        log("No se identificaron viajes completos")
        return gpd.GeoDataFrame()
    
    # 8. JOIN ESPACIAL MEJORADO
    log(f"\n REALIZANDO CRUCE ESPACIAL (buffer={buffer_distancia}m):")
    log(" Creando spatial index para vías...")
    vias_sindex = gdf_vias_proj.sindex
    
    fclass_results = []
    distancia_results = []
    indices = []
    
    log(f" Procesando {len(gdf_viajes):,} puntos...")
    
    # Optimización: procesar en chunks o usar sjoin_nearest si geopandas es reciente
    # Pero mantendremos el bucle si se requiere lógica muy específica, 
    # aunque sjoin_nearest es mucho más rápido.
    # Dado que ya tenemos map matching previo, esto es para clasificar sobre la red vial completa (tipos de vía)
    
    # Vamos a intentar usar sjoin_nearest para acelerar
    try:
        if hasattr(gpd, 'sjoin_nearest'):
            # Usar sjoin_nearest es mucho más rápido
            log(" Usando sjoin_nearest vectorizado...")
            
            # Solo columnas necesarias de vias
            vias_min = gdf_vias_proj[['geometry', 'fclass_reclass']].copy()
            
            joined = gpd.sjoin_nearest(
                gdf_viajes[['geometry']], 
                vias_min, 
                how='left', 
                max_distance=buffer_distancia, 
                distance_col='dist_nearest'
            )
            
            # Eliminar duplicados si un punto tiene varios nearest (toma el primero)
            joined = joined[~joined.index.duplicated(keep='first')]
            
            # Asignar resultados
            gdf_viajes['fclass_reclass'] = joined['fclass_reclass']
            gdf_viajes['distancia_a_via'] = joined['dist_nearest']
            
        else:
            raise AttributeError("No sjoin_nearest")
            
    except Exception as e:
        log(" Fallback a método iterativo (lento)...")
        for idx, punto in tqdm(gdf_viajes.iterrows(), total=len(gdf_viajes), disable=not verbose):
            buffer_punto = punto.geometry.buffer(buffer_distancia)
            posibles_vias_idx = list(vias_sindex.query(buffer_punto))
            
            if posibles_vias_idx:
                vias_candidatas = gdf_vias_proj.iloc[posibles_vias_idx]
                mask_intersect = vias_candidatas.intersects(buffer_punto)
                vias_intersectan = vias_candidatas[mask_intersect]
                
                if len(vias_intersectan) > 0:
                    distancias = vias_intersectan.geometry.distance(punto.geometry)
                    idx_min = distancias.idxmin()
                    via_mas_cercana = vias_intersectan.loc[idx_min]
                    fclass_results.append(via_mas_cercana['fclass_reclass'])
                    distancia_results.append(distancias.min())
                else:
                    fclass_results.append('Desconocido')
                    distancia_results.append(np.inf)
            else:
                fclass_results.append('Desconocido')
                distancia_results.append(np.inf)
            
            indices.append(idx)
            
        if len(indices) > 0:
             # Agregar resultados
            resultados_df = pd.DataFrame({
                'fclass_reclass': fclass_results,
                'distancia_a_via': distancia_results
            }, index=indices)
            
            gdf_viajes = gdf_viajes.join(resultados_df)
    
    gdf_viajes['fclass_reclass'] = gdf_viajes['fclass_reclass'].fillna('Desconocido')
    
    # 9. CREAR SEGMENTOS DE RUTA
    log("\n CREANDO SEGMENTOS DE RUTA...")
    segmentos_rutas = []
    
    # Agrupar y procesar segmentos
    # Vectorizar esto es difícil, mantenemos lógica iterativa por grupos
    viajes_grupos = gdf_viajes.groupby(['imei', 'viaje_id'])
    
    for (vehiculo, viaje_id), grupo in viajes_grupos:
        grupo = grupo.sort_values('time_stamp_event')
        
        if len(grupo) < 2:
            continue
            
        # Detectar cambios de fclass_reclass
        # Crear grupos de segmentos consecutivos con misma fclass
        grupo['fclass_shift'] = grupo['fclass_reclass'].shift()
        grupo['segment_change'] = grupo['fclass_reclass'] != grupo['fclass_shift']
        grupo['segment_id'] = grupo['segment_change'].cumsum()
        
        # Agrupar por sub-segmentos
        for seg_id, subgrupo in grupo.groupby('segment_id'):
            if len(subgrupo) < 2: 
                continue
                
            try:
                geom_puntos = subgrupo.geometry.tolist()
                linea = LineString(geom_puntos)
                
                inicio = subgrupo['time_stamp_event'].iloc[0]
                fin = subgrupo['time_stamp_event'].iloc[-1]
                duracion = (fin - inicio).total_seconds() / 60
                
                dist_prom = subgrupo['distancia_a_via'].mean() if 'distancia_a_via' in subgrupo.columns else 0
                
                segmentos_rutas.append({
                    'imei': vehiculo,
                    'viaje_id': viaje_id,
                    'fclass_reclass': subgrupo['fclass_reclass'].iloc[0],
                    'inicio_viaje': grupo['time_stamp_event'].iloc[0],
                    'fin_viaje': grupo['time_stamp_event'].iloc[-1],
                    'duracion_viaje_minutos': (grupo['time_stamp_event'].iloc[-1] - grupo['time_stamp_event'].iloc[0]).total_seconds()/60,
                    'inicio_segmento': inicio,
                    'fin_segmento': fin,
                    'duracion_segmento_minutos': duracion,
                    'num_puntos_segmento': len(subgrupo),
                    'distancia_promedio_via': dist_prom,
                    'geometry': linea
                })
            except Exception:
                continue
    
    # 10. CREAR GEO DATAFRAME FINAL    
    if segmentos_rutas:
        gdf_rutas = gpd.GeoDataFrame(segmentos_rutas, crs=crs_proyectado)
        gdf_rutas['longitud_metros'] = gdf_rutas['geometry'].length
        
        if crs_original and str(crs_original) != str(crs_proyectado):
            log(f"   🔄 Convirtiendo resultados a CRS original: {crs_original}")
            try:
                gdf_rutas = gdf_rutas.to_crs(crs_original)
            except Exception as e:
                log(f" No se pudo convertir a CRS original: {e}")
        
        log(f"\n RUTAS CREADAS EXITOSAMENTE")
        log("="*50)
        log(f"   - Total segmentos: {len(gdf_rutas):,}")
        log(f"   - Viajes únicos: {gdf_rutas['viaje_id'].nunique()}")
        
        return gdf_rutas
    else:
        log("No se pudieron crear segmentos de ruta")
        return gpd.GeoDataFrame(columns=['geometry'], crs=crs_original or crs_proyectado)

def procesar_archivo_csv(archivo_csv, vias_gdf):
    """
    Procesa un archivo CSV individual y retorna el DataFrame con los resultados del modelo IDV
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO ARCHIVO: {os.path.basename(archivo_csv)}")
    print(f"{'='*70}")
    
    # Cargar datos del vehículo
    vehicles_df = pd.read_csv(archivo_csv)
    
    # Limpieza y preparación de datos
    vehicles_df = vehicles_df[vehicles_df['valid'] == True]
    
    # RENOMBRAMIENTO CLAVE: devicetime -> time_stamp_event
    if 'devicetime' in vehicles_df.columns:
        vehicles_df = vehicles_df.rename(columns={'devicetime': 'time_stamp_event'})
    elif 'fixtime' in vehicles_df.columns:
        print("Aviso: No se encontró 'devicetime', usando 'fixtime' en su lugar.")
        vehicles_df = vehicles_df.rename(columns={'fixtime': 'time_stamp_event'})
    else:
        raise ValueError("El archivo no contiene columna 'devicetime' ni 'fixtime'.")
    
    # Asegurar datetime
    vehicles_df['time_stamp_event'] = pd.to_datetime(vehicles_df['time_stamp_event'], format='mixed')
    
    # Extraer ignition
    vehicles_df['ignition_status'] = vehicles_df['attributes'].str.extract(r'"ignition":\s*(true|false)').replace({'true': True, 'false': False})
    
    # Extraer nombre del archivo sin extensión para usar como identificador
    nombre_archivo = os.path.splitext(os.path.basename(archivo_csv))[0]
    
    # Datos completos (ficticios/default donde falten)
    vehicles_df['plate'] = nombre_archivo
    vehicles_df['imei'] = '123123123' # Placeholder si no está en datos
    vehicles_df['company_name'] = 'EQUIRRENT'
    vehicles_df['manufacturer_name'] = 'Teltonika'
    vehicles_df['reference_name'] = 'FMC920'
    
    gdf_plate_druid = gpd.GeoDataFrame(vehicles_df, geometry=gpd.points_from_xy(vehicles_df.longitude, vehicles_df.latitude), crs='EPSG:4326')
    
    # Map Matching
    puntos_preparados, lineas_preparadas = preparar_datos(gdf_plate_druid, vias_gdf)
    resultado = optimized_map_matching_with_progress(puntos_preparados, lineas_preparadas, max_distance=100)
    
    if resultado is None or resultado.empty:
        print(f"No se encontraron resultados para {nombre_archivo}")
        return None
    
    # Crear rutas segmentadas (metodo preferido)
    print(f"Creando rutas segmentadas para {nombre_archivo}...")
    gdf_rutas_creadas = crear_rutas_por_viaje(
        resultado,
        vias_gdf,
        buffer_distancia=100,
        crs_proyectado='EPSG:9377',
        verbose=False  # Reducir verbosidad para múltiples archivos
    )
    
    df_final = None # Inicializar
    
    if gdf_rutas_creadas.empty:
        print(f"No se pudieron crear rutas para {nombre_archivo}. Usando metodo alternativo por puntos.")
        df_alt_resumen = resumen_idv_desde_mapmatching(
            resultado_mm=resultado,
            vias_gdf=vias_gdf,
            nombre_archivo=nombre_archivo,
            buffer_distancia=100,
            crs_proyectado='EPSG:9377'
        )
        if df_alt_resumen is None or df_alt_resumen.empty:
            return None
        print(f"Procesamiento completado para {nombre_archivo}")
        return df_alt_resumen
    else:
        # Generar reporte resumen desde rutas
        df_final = gdf_rutas_creadas.merge(
            resultado[['imei', 'plate']].drop_duplicates(),
            on='imei',
            how='left'
        )
    
    # ========================================================================================
    # LÓGICA DE AGREGACIÓN IDV V2: Segmentación por Año, Mes y Quincena
    # ========================================================================================
    
    # Asegurar columna de fecha de referencia
    if 'inicio_segmento' in df_final.columns:
        df_final['fecha_ref'] = pd.to_datetime(df_final['inicio_segmento'])
    else:
        # Fallback a inicio_viaje o timestamp generico
        df_final['fecha_ref'] = pd.to_datetime(df_final['inicio_viaje'])
        
    df_final['año'] = df_final['fecha_ref'].dt.year
    df_final['mes'] = df_final['fecha_ref'].dt.month
    df_final['dia'] = df_final['fecha_ref'].dt.day
    
    # Calcular Quincena Vectorizado
    # 1: días 1-15, 2: días 16-fin
    df_final['quincena'] = np.where(df_final['dia'] <= 15, 1, 2)
    
    # Agrupar por imei, plate, año, mes, quincena
    # Calculamos sumas condicionales
    
    df_resumen = df_final.groupby(['imei', 'plate', 'año', 'mes', 'quincena']).agg(
        total_km_recorridos=('longitud_metros', lambda x: x.sum() / 1000),
        # Vectorizado condicional dentro de agg es complejo, mejor sumar columnas pre-calculadas
    ).reset_index()
    
    # Pre-calcular columnas para pavimentada y sin pavimentar para facilitar suma
    df_final['km_pav'] = np.where(df_final['fclass_reclass'] == 'Pavimentada', df_final['longitud_metros'], 0)
    df_final['km_sin_pav'] = np.where(df_final['fclass_reclass'] == 'Sin pavimentar', df_final['longitud_metros'], 0)
    
    # Re-agrupar con todas las métricas
    df_resumen = df_final.groupby(['imei', 'plate', 'año', 'mes', 'quincena']).agg(
        total_km_recorridos=('longitud_metros', lambda x: x.sum() / 1000),
        total_recorridos_pavimentada=('km_pav', lambda x: x.sum() / 1000),
        total_recorridos_sin_pavimentar=('km_sin_pav', lambda x: x.sum() / 1000)
    ).reset_index()

    # Calcular IDV Vectorizado
    w_np = 3
    
    # Evitar división por cero
    df_resumen['IDV'] = np.where(
        df_resumen['total_km_recorridos'] > 0,
        (w_np * df_resumen['total_recorridos_sin_pavimentar'] + df_resumen['total_recorridos_pavimentada']) / df_resumen['total_km_recorridos'],
        np.nan
    )
    
    # Redondear
    cols_float = ['total_km_recorridos', 'total_recorridos_pavimentada', 'total_recorridos_sin_pavimentar', 'IDV']
    df_resumen[cols_float] = df_resumen[cols_float].round(2)
    
    # Agregar nombre del archivo origen
    df_resumen['archivo_origen'] = nombre_archivo
    df_resumen['metodo_calculo'] = 'rutas_viaje_v2'
    
    print(f"Procesamiento completado para {nombre_archivo}")
    return df_resumen

def _clasificar_puntos_por_via(gdf_puntos_proj, gdf_vias_proj, buffer_distancia=100):
    """
    Asigna a cada punto la fclass_reclass de la via mas cercana.
    """
    if 'fclass_reclass' not in gdf_vias_proj.columns:
        raise ValueError("La capa de vias no tiene la columna 'fclass_reclass'.")

    # Usar sjoin_nearest si es posible
    if hasattr(gpd, 'sjoin_nearest'):
        vias_min = gdf_vias_proj[['geometry', 'fclass_reclass']]
        joined = gpd.sjoin_nearest(
            gdf_puntos_proj[['geometry']], 
            vias_min, 
            how='left', 
            max_distance=buffer_distancia, 
            distance_col='dist_nearest'
        )
        joined = joined[~joined.index.duplicated(keep='first')]
        out = gdf_puntos_proj.copy()
        out['fclass_reclass'] = joined['fclass_reclass']
        out['distancia_a_via'] = joined['dist_nearest']
        out['fclass_reclass'] = out['fclass_reclass'].fillna('Desconocido')
        out['distancia_a_via'] = out['distancia_a_via'].fillna(float('inf'))
        return out
    else:
        # Fallback iterativo (simplificado del original)
        vias_sindex = gdf_vias_proj.sindex
        fclass_results = []
        distancia_results = []
        
        for _, punto in gdf_puntos_proj.iterrows():
            buffer_punto = punto.geometry.buffer(buffer_distancia)
            posibles_vias_idx = list(vias_sindex.query(buffer_punto))
            if posibles_vias_idx:
                vias_candidatas = gdf_vias_proj.iloc[posibles_vias_idx]
                distancias = vias_candidatas.geometry.distance(punto.geometry)
                idx_min = distancias.idxmin()
                fclass_results.append(vias_candidatas.loc[idx_min, 'fclass_reclass'])
                distancia_results.append(distancias.min())
            else:
                fclass_results.append('Desconocido')
                distancia_results.append(float('inf'))
                
        out = gdf_puntos_proj.copy()
        out['fclass_reclass'] = fclass_results
        out['distancia_a_via'] = distancia_results
        return out

def resumen_idv_desde_mapmatching(resultado_mm, vias_gdf, nombre_archivo, buffer_distancia=100, crs_proyectado='EPSG:9377'):
    """
    Fallback IDV V2 para puntos.
    """
    if resultado_mm is None or resultado_mm.empty:
        return None

    puntos = resultado_mm.copy()
    if 'time_stamp_event' in puntos.columns:
        puntos['time_stamp_event'] = pd.to_datetime(puntos['time_stamp_event'], errors='coerce')
    else:
        return None

    # CRS
    if puntos.crs is None: puntos = puntos.set_crs('EPSG:4326')
    puntos_proj = puntos.to_crs(crs_proyectado)
    vias_proj = vias_gdf.to_crs(crs_proyectado)

    # Clasificar
    puntos_clas = _clasificar_puntos_por_via(puntos_proj, vias_proj, buffer_distancia=buffer_distancia)

    # Calcular distancias
    puntos_clas = puntos_clas.sort_values(['imei', 'plate', 'time_stamp_event'])
    
    # Vectorized distance calculation
    # Shift geometry
    puntos_clas['prev_geom'] = puntos_clas.groupby(['imei', 'plate'])['geometry'].shift()
    
    # Distance function handling first element (running distance from previous)
    # Using apply because distance is geometric method
    # Optimization: points distance vectorization is not native in shapely/geopandas without applying
    # But we can iterate over coordinates which is faster
    # For now, let's stick to apply or a simplified approach
    
    # However, for accuracy we must use geometry.distance
    # To speed up, we can align indices and prevent invalid comparisons
    
    mask_valid = puntos_clas['prev_geom'].notna()
    puntos_clas['distancia_tramo_m'] = 0.0
    
    if mask_valid.any():
        # Iterate only valid - still slow but correct
        # Alternatively, extract X/Y and use numpy euclidean
        coords = np.array([(g.x, g.y) for g in puntos_clas.geometry])
        
        # Shift coords array
        # Create groups to ensure we don't calculate distance across diff cars
        # But we already did shift inside group, so indices are aligned with 'prev_geom'
        # Can't easily use numpy on prev_geom column directly if it contains shapely objects mixed with NaT
        
        # Let's fallback to the iterative approach but slightly cleaner:
        puntos_clas.loc[mask_valid, 'distancia_tramo_m'] = puntos_clas[mask_valid].apply(
            lambda row: row['geometry'].distance(row['prev_geom']), axis=1
        )

    # --- AGREGACION IDV V2 ---
    puntos_clas['año'] = puntos_clas['time_stamp_event'].dt.year
    puntos_clas['mes'] = puntos_clas['time_stamp_event'].dt.month
    puntos_clas['dia'] = puntos_clas['time_stamp_event'].dt.day
    puntos_clas['quincena'] = np.where(puntos_clas['dia'] <= 15, 1, 2)
    
    puntos_clas['km_pav'] = np.where(puntos_clas['fclass_reclass'] == 'Pavimentada', puntos_clas['distancia_tramo_m'], 0)
    puntos_clas['km_sin_pav'] = np.where(puntos_clas['fclass_reclass'] == 'Sin pavimentar', puntos_clas['distancia_tramo_m'], 0)
    
    df_resumen = puntos_clas.groupby(['imei', 'plate', 'año', 'mes', 'quincena']).agg(
        total_km_recorridos=('distancia_tramo_m', lambda x: x.sum() / 1000),
        total_recorridos_pavimentada=('km_pav', lambda x: x.sum() / 1000),
        total_recorridos_sin_pavimentar=('km_sin_pav', lambda x: x.sum() / 1000)
    ).reset_index()
    
    w_np = 3
    df_resumen['IDV'] = np.where(
        df_resumen['total_km_recorridos'] > 0,
        (w_np * df_resumen['total_recorridos_sin_pavimentar'] + df_resumen['total_recorridos_pavimentada']) / df_resumen['total_km_recorridos'],
        np.nan
    )
    
    cols_float = ['total_km_recorridos', 'total_recorridos_pavimentada', 'total_recorridos_sin_pavimentar', 'IDV']
    df_resumen[cols_float] = df_resumen[cols_float].round(2)
    
    df_resumen['archivo_origen'] = nombre_archivo
    df_resumen['metodo_calculo'] = 'fallback_puntos_v2'
    
    return df_resumen

def main():
    print("="*70)
    print("PROCESO DE MODELO IDV V2 - SEGMENTACIÓN QUINCENAL")
    print("="*70)
    
    # Cargar Vías (una sola vez)
    print("\nCargando Vías...")
    vias_gdf = gpd.read_file('Data/vias_colombia.gpkg')
    print(f"Vías cargadas: {len(vias_gdf):,} líneas")
    
    # Encontrar todos los archivos CSV en Data
    print("\nBuscando archivos CSV en carpeta Data...")
    data_dir = 'Data'
    archivos_csv = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                   if f.endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))
                   and 'resultado' not in f] # Evitar procesar archivos de resultados
    
    if not archivos_csv:
        print("No se encontraron archivos CSV en la carpeta Data")
        return
    
    print(f"Encontrados {len(archivos_csv)} archivo(s) CSV para procesar.")
    
    # Procesar cada archivo y acumular resultados
    resultados_consolidados = []
    
    for archivo_csv in archivos_csv:
        try:
            resultado_archivo = procesar_archivo_csv(archivo_csv, vias_gdf)
            if resultado_archivo is not None and not resultado_archivo.empty:
                resultados_consolidados.append(resultado_archivo)
        except Exception as e:
            print(f"Error procesando {os.path.basename(archivo_csv)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Consolidar todos los resultados
    if resultados_consolidados:
        print(f"\n{'='*70}")
        print("CONSOLIDANDO RESULTADOS V2")
        print(f"{'='*70}")
        
        df_final_consolidado = pd.concat(resultados_consolidados, ignore_index=True)
        
        # Reordenar columnas
        columnas_base = ['archivo_origen', 'imei', 'plate', 'año', 'mes', 'quincena', 
                         'total_km_recorridos', 'total_recorridos_pavimentada', 
                         'total_recorridos_sin_pavimentar', 'IDV']
        
        # Asegurar todas las columnas en el orden deseado, agregando las extras al final
        cols_finales = columnas_base + [c for c in df_final_consolidado.columns if c not in columnas_base]
        df_final_consolidado = df_final_consolidado[cols_finales]
        
        # Exportar CSV consolidado
        archivo_salida = 'Data/resultados_idv_v2_consolidado.csv'
        df_final_consolidado.to_csv(archivo_salida, index=False)
        
        print(f"\nPROCESO COMPLETADO")
        print(f"   • Archivos procesados: {len(resultados_consolidados)}")
        print(f"   • Total registros: {len(df_final_consolidado):,}")
        print(f"   • Archivo de salida: {archivo_salida}")
        print("\nResumen consolidado (primeras filas):")
        print(df_final_consolidado.head().to_string(index=False))
    else:
        print("\nNo se generaron resultados válidos de ningún archivo")

if __name__ == "__main__":
    main()
