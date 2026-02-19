import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from shapely.geometry import Point, LineString
from sklearn.neighbors import BallTree
import warnings
import traceback

warnings.filterwarnings('ignore')

# Configuración de CRS
CRS_9377 = "EPSG:9377"

def preparar_datos(points_gdf, lines_gdf):
    """Prepara los datos para el map matching"""
    points_valid = points_gdf[points_gdf.geometry.is_valid & points_gdf.geometry.notna()]
    lines_valid = lines_gdf[lines_gdf.geometry.is_valid & lines_gdf.geometry.notna()]   
    return points_valid, lines_valid

def optimized_map_matching_with_progress(points_gdf, lines_gdf, max_distance, sample_interval=20, k_neighbors=5):
    """Map matching optimizado para grandes datasets"""
    if points_gdf.empty:
        return None
        
    points = points_gdf.copy()
    lines = lines_gdf.copy()
    
    columns_to_preserve = [
        'time_stamp_event', 'plate', 'imei', 'ignition_status'
    ]
    
    if points.crs != CRS_9377:
        points = points.to_crs(CRS_9377)
    if lines.crs != CRS_9377:
        lines = lines.to_crs(CRS_9377)
    
    points_buffer = points.copy()
    points_buffer.geometry = points_buffer.geometry.buffer(max_distance * 2)
    
    candidate_pairs = gpd.sjoin(points_buffer, lines, how='inner', predicate='intersects')
    
    if len(candidate_pairs) == 0:
        return None
    
    line_samples = []
    line_indices = []
    line_geometries = []
    
    unique_line_ids = candidate_pairs.index_right.unique()
    
    for line_idx in unique_line_ids:
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
    
    if not line_samples:
        return None
    
    tree = BallTree(np.array(line_samples), metric='euclidean')
    results = []
    points_with_candidates = candidate_pairs.index.unique()
    
    for point_idx in points_with_candidates:
        point_row = points.loc[point_idx]
        point_geom = point_row.geometry
        point_coord = [[point_geom.x, point_geom.y]]
        
        distances, indices = tree.query(point_coord, k=min(k_neighbors, len(line_samples)), return_distance=True)
        
        best_match = None
        best_distance = float('inf')
        
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= max_distance:
                line_idx = line_indices[idx]
                line_geom = line_geometries[idx]
                closest_point = line_geom.interpolate(line_geom.project(point_geom))
                actual_distance = point_geom.distance(closest_point)
                
                if actual_distance <= max_distance and actual_distance < best_distance:
                    best_distance = actual_distance
                    best_match = {
                        'point_id': point_idx,
                        'line_id': line_idx,
                        'distance': actual_distance,
                        'geometry': closest_point,
                        'fclass_reclass': lines.loc[line_idx, 'fclass_reclass'] if 'fclass_reclass' in lines.columns else 'Pavimentada'
                    }
                    for col in columns_to_preserve:
                        if col in point_row:
                            best_match[col] = point_row[col]
        
        if best_match:
            results.append(best_match)
    
    if not results:
        return None
        
    return gpd.GeoDataFrame(results, crs=CRS_9377)

def crear_rutas_por_viaje(resultado_mm, vias_gdf, buffer_distancia=100, crs_proyectado='EPSG:9377', verbose=False):
    """Crea rutas a partir de puntos de map matching"""
    if resultado_mm is None or resultado_mm.empty:
        return gpd.GeoDataFrame()
    
    res = resultado_mm.sort_values(by=['imei', 'time_stamp_event'])
    rutas = []
    
    for imei, group in res.groupby('imei'):
        if len(group) < 2:
            continue
            
        group = group.copy()
        group['next_geom'] = group.geometry.shift(-1)
        group['next_time'] = group.time_stamp_event.shift(-1)
        
        for idx, row in group.iterrows():
            if pd.isna(row['next_geom']):
                continue
                
            line = LineString([row['geometry'], row['next_geom']])
            if line.length > 0:
                rutas.append({
                    'imei': imei,
                    'plate': row['plate'],
                    'inicio_viaje': row['time_stamp_event'],
                    'longitud_metros': line.length,
                    'fclass_reclass': row['fclass_reclass'],
                    'geometry': line
                })
                
    if not rutas:
        return gpd.GeoDataFrame()
        
    return gpd.GeoDataFrame(rutas, crs=crs_proyectado)

def agrupar_quincenal(df):
    """Aplica la lógica de IDV quincenal"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    # Asegurar fecha
    if 'inicio_viaje' in df.columns:
        df['fecha_ref'] = pd.to_datetime(df['inicio_viaje'])
    elif 'time_stamp_event' in df.columns:
        df['fecha_ref'] = pd.to_datetime(df['time_stamp_event'])
    else:
        return pd.DataFrame()
        
    df['ano'] = df['fecha_ref'].dt.year
    df['mes'] = df['fecha_ref'].dt.month
    df['quincena'] = np.where(df['fecha_ref'].dt.day <= 15, 1, 2)
    
    # Lógica de pavimentación: Si no es 'Sin pavimentar', asumimos 'Pavimentada'
    df['km_sin_pav'] = np.where(df['fclass_reclass'] == 'Sin pavimentar', df['longitud_metros'] / 1000, 0)
    df['km_pav'] = np.where(df['fclass_reclass'] != 'Sin pavimentar', df['longitud_metros'] / 1000, 0)
    
    resumen = df.groupby(['imei', 'plate', 'ano', 'mes', 'quincena']).agg(
        total_km_recorridos=('longitud_metros', lambda x: x.sum() / 1000),
        total_recorridos_pavimentada=('km_pav', 'sum'),
        total_recorridos_sin_pavimentar=('km_sin_pav', 'sum')
    ).reset_index()
    
    # Cálculo IDV (w_np = 3)
    w_np = 3
    resumen['IDV'] = np.where(
        resumen['total_km_recorridos'] > 0,
        (w_np * resumen['total_recorridos_sin_pavimentar'] + resumen['total_recorridos_pavimentada']) / resumen['total_km_recorridos'],
        1.0 # Default para vehículos estacionados
    )
    
    cols_float = ['total_km_recorridos', 'total_recorridos_pavimentada', 'total_recorridos_sin_pavimentar', 'IDV']
    resumen[cols_float] = resumen[cols_float].round(2)
    
    return resumen

def procesar_archivo_csv(archivo_csv, vias_gdf):
    print(f"\n{'-'*50}\nPROCESANDO: {os.path.basename(archivo_csv)}\n{'-'*50}")
    
    try:
        df = pd.read_csv(archivo_csv)
    except Exception as e:
        print(f"Error al leer CSV: {e}")
        return None
        
    if 'valid' in df.columns:
        df = df[df['valid'].astype(str).str.lower().isin(['true', '1', '1.0'])]
        
    if df.empty:
        print("Sin registros válidos.")
        return None
        
    # Priorizar fixtime para evitar el error de 1970 en devicetime
    if 'fixtime' in df.columns:
        df['time_stamp_event'] = pd.to_datetime(df['fixtime'], format='mixed', errors='coerce')
    elif 'devicetime' in df.columns:
        df['time_stamp_event'] = pd.to_datetime(df['devicetime'], format='mixed', errors='coerce')
    
    df = df.dropna(subset=['time_stamp_event'])
    
    # Identificar columnas de placa e imei
    plate_col = 'placa' if 'placa' in df.columns else ('plate' if 'plate' in df.columns else None)
    imei_col = 'deviceid' if 'deviceid' in df.columns else ('imei' if 'imei' in df.columns else None)
    
    if not plate_col:
        df['plate'] = os.path.splitext(os.path.basename(archivo_csv))[0]
        plate_col = 'plate'
    else:
        df['plate'] = df[plate_col].astype(str)
        
    if not imei_col:
        df['imei'] = '123123123'
    else:
        df['imei'] = df[imei_col].astype(str)
        
    # Extraer ignition
    df['ignition_status'] = df['attributes'].str.extract(r'"ignition":\s*(true|false)').replace({'true': True, 'false': False})
    df['ignition_status'] = df['ignition_status'].fillna(True)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    puntos, lineas = preparar_datos(gdf, vias_gdf)
    
    placas_validas = df['plate'].unique()
    print(f"Placas encontradas: {len(placas_validas)}")
    
    resultados_archivo = []
    
    # Procesar por grupos de placa para reporte detallado
    for placa in tqdm(placas_validas, desc="Placas"):
        gdf_p = puntos[puntos['plate'] == placa]
        res_mm = optimized_map_matching_with_progress(gdf_p, lineas, max_distance=100)
        
        if res_mm is None or res_mm.empty:
            continue
            
        rutas = crear_rutas_por_viaje(res_mm, vias_gdf)
        
        if not rutas.empty:
            resumen = agrupar_quincenal(rutas)
        else:
            # Fallback por puntos si no hay rutas
            res_mm['longitud_metros'] = 0 # En puntos no hay dist recorrido
            resumen = agrupar_quincenal(res_mm)
            
        if not resumen.empty:
            resultados_archivo.append(resumen)
            
    if not resultados_archivo:
        return pd.DataFrame()
        
    return pd.concat(resultados_archivo, ignore_index=True)

def main():
    print("="*70)
    print("PROCESO IDV - HANDLING MULTI-PLATE & YEAR 1970 FIX")
    print("="*70)
    
    vias_path = 'Data/vias_colombia.gpkg'
    if not os.path.exists(vias_path):
        print(f"Error: {vias_path} no encontrado.")
        return
        
    vias_gdf = gpd.read_file(vias_path)
    
    data_dir = 'Data'
    archivos = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
               if f.endswith('.csv') and 'resultado' not in f.lower()]
    
    if not archivos:
        print("No se encontraron archivos CSV.")
        return
        
    consolidados = []
    for f in archivos:
        res = procesar_archivo_csv(f, vias_gdf)
        if res is not None and not res.empty:
            consolidados.append(res)
            
    if consolidados:
        df_final = pd.concat(consolidados, ignore_index=True)
        salida = 'Data/resultados_idv_placas_consolidado.csv'
        df_final.to_csv(salida, index=False, encoding='utf-8-sig')
        
        print(f"\nPROCESO COMPLETADO")
        print(f"Archivo generado: {salida}")
        print(f"Total placas procesadas exitosamente: {df_final['plate'].nunique()}")
        print("\nPrimeras filas:")
        print(df_final.head())
    else:
        print("\nNo se generaron resultados.")

if __name__ == "__main__":
    main()
