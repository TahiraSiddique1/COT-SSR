# import s3fs
# import os
# import shutil
# import pandas as pd
# import numpy as np
# from satpy.scene import Scene
# from glob import glob
# import warnings
# import concurrent.futures

# # Suppress Satpy/Dask warnings
# warnings.filterwarnings("ignore")

# # --- CONFIGURATION ---
# START_DATE = "2023-01-01"
# END_DATE   = "2024-12-31" 
# HOURS = ["0200", "0300", "0400", "0500", "0600", "0700", "0800", "0900", "1000"]
# FREQ = "1D" 

# # PERFORMANCE SETTINGS
# MAX_WORKERS = 4  # How many hours to process at the SAME time (Keep low to save RAM/Disk)
# DOWNLOAD_THREADS = 8 # How many files to download at once

# # WHERE TO SAVE
# base_dir = os.getcwd() 
# output_dir = os.path.join(base_dir, "training_data_final")
# os.makedirs(output_dir, exist_ok=True)

# # TEMP FOLDER
# temp_dir_root = os.path.join(base_dir, "temp_fast_processing")
# if os.path.exists(temp_dir_root):
#     shutil.rmtree(temp_dir_root) # Clean start
# os.makedirs(temp_dir_root, exist_ok=True)

# # Bangladesh Bounding Box
# BD_BBOX = (87, 20, 93, 27)
# TARGET_SEGMENTS = ["S0310", "S0410"]

# def create_physics_label(ir_data):
#     """Generates COT label from IR Temperature."""
#     ir_clean = np.nan_to_num(ir_data, nan=300.0)
#     ir_clean = np.clip(ir_clean, 180, 310)
#     fake_cot = (310 - ir_clean) 
#     fake_cot = fake_cot / (310-180) * 60.0
#     fake_cot[ir_clean > 290] = 0
#     return fake_cot

# def download_file(fs, remote_path, local_path):
#     """Helper function for threaded downloading."""
#     if not os.path.exists(local_path):
#         try:
#             fs.get(remote_path, local_path)
#         except Exception:
#             pass # Skip errors to keep moving

# def process_one_slot(task_info):
#     """
#     Handles the lifecycle of ONE hour:
#     Download (Targeted) -> Process -> Save -> Delete
#     """
#     date, time_str, fs = task_info
    
#     yyyy = date.strftime("%Y")
#     mm = date.strftime("%m")
#     dd = date.strftime("%d")
#     date_str = f"{yyyy}{mm}{dd}"
    
#     # 1. Check if done
#     save_name = f"train_{date_str}_{time_str}.npz"
#     save_path = os.path.join(output_dir, save_name)
#     if os.path.exists(save_path):
#         return f"⏩ {date_str} {time_str} Skipped"

#     # Create unique temp folder for this thread (avoids file collisions)
#     thread_temp = os.path.join(temp_dir_root, f"{date_str}_{time_str}")
#     os.makedirs(thread_temp, exist_ok=True)

#     try:
#         # 2. Targeted Download (Parallelized)
#         bucket_path = f"noaa-himawari9/AHI-L1b-FLDK/{yyyy}/{mm}/{dd}/{time_str}/"
#         download_tasks = []
        
#         for band in ["B03", "B13"]:
#             all_files = fs.glob(f"{bucket_path}*{band}*")
            
#             for remote in all_files:
#                 if any(seg in remote for seg in TARGET_SEGMENTS):
#                     fname = remote.split('/')[-1]
#                     local = os.path.join(thread_temp, fname)
#                     download_tasks.append((remote, local))

#         if not download_tasks:
#             return f"⚠️ No data {date_str} {time_str}"

#         # Run downloads in parallel threads
#         with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as dl_executor:
#             futures = [dl_executor.submit(download_file, fs, r, l) for r, l in download_tasks]
#             concurrent.futures.wait(futures)

#         # 3. Process
#         local_files = glob(os.path.join(thread_temp, "*"))
#         if len(local_files) < 2:
#             return f"⚠️ Incomplete Download {date_str} {time_str}"

#         scn = Scene(filenames=local_files, reader='ahi_hsd')
#         scn.load(['B03', 'B13'])
#         cropped = scn.crop(ll_bbox=BD_BBOX)
#         resampled = cropped.resample(cropped['B13'].attrs['area'], resampler='native')
        
#         vis = resampled['B03'].values
#         ir = resampled['B13'].values
        
#         # Dark check
#         if np.mean(vis) < 1.0:
#             return f"🌑 Night Image {date_str} {time_str}"

#         label = create_physics_label(ir)
        
#         # Save
#         vis = np.nan_to_num(vis, nan=0)
#         ir = np.nan_to_num(ir, nan=0)
#         X_data = np.stack([vis, ir], axis=-1)
#         Y_data = np.expand_dims(label, axis=-1)
#         np.savez(save_path, X=X_data, y=Y_data)
        
#         return f"✅ Saved {date_str} {time_str}"

#     except Exception as e:
#         return f"❌ Error {date_str} {time_str}: {str(e)}"
    
#     finally:
#         # 4. Cleanup (Delete the specific temp folder)
#         try:
#             shutil.rmtree(thread_temp)
#         except:
#             pass

# def main():
#     print("🚀 Starting FAST Parallel Pipeline")
#     print(f"   Workers: {MAX_WORKERS} (Concurrent Times)")
#     print(f"   Threads: {DOWNLOAD_THREADS} (Downloads per Time)")
#     print(f"   Optimization: Downloading only Segments {TARGET_SEGMENTS}")
    
#     fs = s3fs.S3FileSystem(anon=True)
#     dates = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)
    
#     # Prepare list of all tasks
#     tasks = []
#     for date in dates:
#         for time_str in HOURS:
#             tasks.append((date, time_str, fs))
            
#     print(f"   Queue: {len(tasks)} items to process.")
    
#     # Process in Parallel
#     # We use ThreadPoolExecutor because the bottleneck is Network I/O, not CPU.
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         results = executor.map(process_one_slot, tasks)
        
#         # Print results as they finish
#         for res in results:
#             print(res)

#     print("\n🎉 DONE! Check training_data_final/")

# if __name__ == "__main__":
#     main()







import s3fs
import os
import shutil
import pandas as pd
import numpy as np
from satpy.scene import Scene
from glob import glob
import warnings
import concurrent.futures
import cv2
import xarray as xr
import earthaccess
import dask # Added to control Satpy's CPU usage
warnings.filterwarnings("ignore")

START_DATE = "2023-01-01"
END_DATE   = "2023-12-31" 
HOURS = ["0200", "0300", "0400", "0500", "0600", "0700", "0800", "0900", "1000"]
FREQ = "1D" 

# Directories
AUX_DATA_ROOT = "G:\\Other computers\\Latitude\\INFRYNE\\resarrch\\Model\\aux_data1" # Where you downloaded ERA5 and CERES data

# PERFORMANCE TUNING
MAX_WORKERS = 4 # Number of parallel time-slots (match this to your physical CPU cores)
DOWNLOAD_THREADS = 4 # S3 downloads per slot
dask.config.set(num_workers=2) # Stop Satpy from stealing all CPU cores per worker

# PATHS
base_dir = os.getcwd() 
output_dir = os.path.join(base_dir, "training_data_master")
os.makedirs(output_dir, exist_ok=True)
temp_dir_root = os.path.join(base_dir, "temp_integrated")
os.makedirs(temp_dir_root, exist_ok=True)

BD_BBOX = (87, 20, 93, 27)
TARGET_SEGMENTS = ["S0310", "S0410"]

# --- AUXILIARY DATA HANDLER ---
def get_aux_data(date, hour_str, target_shape, ceres_file_path):
    yyyy, mm, dd = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    month_dir = os.path.join(AUX_DATA_ROOT, f"era5_bd_{yyyy}_{mm}")
    era5_cube = np.zeros((*target_shape, 3), dtype=np.float32) 
    
    if os.path.exists(month_dir):
        try:
            inst_path = os.path.join(month_dir, "data_stream-oper_stepType-instant.nc")
            accum_path = os.path.join(month_dir, "data_stream-oper_stepType-accum.nc")
            time_query = f"{yyyy}-{mm}-{dd}T{hour_str[:2]}:00:00"

            # 1. Process Instantaneous Stream (Temperature and Cloud Cover)
            if os.path.exists(inst_path):
                with xr.open_dataset(inst_path, engine="h5netcdf") as ds_inst:
                    # Using valid_time as found in your specific CDS download
                    slice_inst = ds_inst.sel(valid_time=time_query, method="nearest")
                    
                    tcc = slice_inst['tcc'].values # Cloud Cover
                    t2m = slice_inst['t2m'].values # 2m Temperature
                    
                    era5_cube[:,:,0] = cv2.resize(tcc, (target_shape[1], target_shape[0]))
                    era5_cube[:,:,2] = cv2.resize(t2m, (target_shape[1], target_shape[0]))

            # 2. Process Accumulated Stream (Solar Radiation)
            if os.path.exists(accum_path):
                with xr.open_dataset(accum_path, engine="h5netcdf") as ds_accum:
                    # Using valid_time as found in your specific CDS download
                    slice_accum = ds_accum.sel(valid_time=time_query, method="nearest")
                    
                    ssrd = slice_accum['ssrd'].values # Solar Radiation
                    
                    era5_cube[:,:,1] = cv2.resize(ssrd, (target_shape[1], target_shape[0]))
                    
        except Exception as e:
            print(f"⚠️ ERA5 Error for {yyyy}-{mm}-{dd}: {e}")

    # --- CERES NASA LABEL PROCESSING ---
    ceres_cot = np.full(target_shape, np.nan, dtype=np.float32)
    
    if ceres_file_path and os.path.exists(ceres_file_path):
        try:
            with xr.open_dataset(ceres_file_path, engine="h5netcdf") as ds_ceres:
                lat = np.squeeze(ds_ceres['latitude'].values)
                lon = np.squeeze(ds_ceres['longitude'].values)
                raw_cot = np.squeeze(ds_ceres['cloud_visible_optical_depth'].values)
                
                min_lon, min_lat, max_lon, max_lat = BD_BBOX
                valid_mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
                valid_coords = np.where(valid_mask)
                
                if len(valid_coords[0]) > 0:
                    min_row, max_row = np.min(valid_coords[0]), np.max(valid_coords[0])
                    min_col, max_col = np.min(valid_coords[1]), np.max(valid_coords[1])
                    clipped_cot = raw_cot[min_row:max_row+1, min_col:max_col+1]
                    ceres_cot = cv2.resize(clipped_cot, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"⚠️ CERES Error for {yyyy}-{mm}-{dd}: {e}")

    # Return: [TCC, SSRD, T2M] cube, Official COT array, SSRD Target
    return era5_cube, ceres_cot, era5_cube[:,:,1] 

# --- MAIN LOGIC ---
def download_file(fs, remote_path, local_path):
    if not os.path.exists(local_path):
        try: 
            fs.get(remote_path, local_path)
        except Exception as e:
            # Delete corrupted partial files if the network drops
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"⚠️ Network drop during download of {remote_path}: {e}")

def process_one_slot(task_info):
    # Note: earthaccess.login() is intentionally NOT here anymore.
    # It is handled by the initialize_worker function instead.
    
    date, time_str, fs, ceres_granule_dict = task_info
    yyyy, mm, dd = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    date_str = f"{yyyy}{mm}{dd}"
    
    save_name = f"master_{date_str}_{time_str}.npz"
    save_path = os.path.join(output_dir, save_name)
    if os.path.exists(save_path): return f"⏩ Skipped {date_str} {time_str}"

    thread_temp = os.path.join(temp_dir_root, f"{date_str}_{time_str}")
    os.makedirs(thread_temp, exist_ok=True)

    try:
        # --- 1. SATELLITE DOWNLOAD ---
        bucket_path = f"noaa-himawari9/AHI-L1b-FLDK/{yyyy}/{mm}/{dd}/{time_str}/"
        download_tasks = []
        for band in ["B03", "B13"]:
            all_files = fs.glob(f"{bucket_path}*{band}*")
            for remote in all_files:
                if any(seg in remote for seg in TARGET_SEGMENTS):
                    local = os.path.join(thread_temp, remote.split('/')[-1])
                    download_tasks.append((remote, local))

        if not download_tasks: return f"⚠️ No Sat Data {date_str} {time_str}"

        # ThreadPool inside the process is perfectly fine for network I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as dl_executor:
            futures = [dl_executor.submit(download_file, fs, r, l) for r, l in download_tasks]
            concurrent.futures.wait(futures)
            
        # --- 2. FAST CERES DOWNLOAD ---
        ceres_local_path = None
        if ceres_granule_dict:
            # We already have the granule mapping, just download it!
            downloaded = earthaccess.download([ceres_granule_dict], thread_temp)
            if downloaded: ceres_local_path = downloaded[0]

        # --- 3. SATELLITE PROCESSING ---
        local_files = glob(os.path.join(thread_temp, "*_B*"))
        if len(local_files) < 2: return f"⚠️ Missing Bands {date_str}"

        scn = Scene(filenames=local_files, reader='ahi_hsd')
        scn.load(['B03', 'B13'])
        cropped = scn.crop(ll_bbox=BD_BBOX)
        resampled = cropped.resample(cropped['B13'].attrs['area'], resampler='native')
        
        vis = np.nan_to_num(resampled['B03'].values, nan=0)
        ir = np.nan_to_num(resampled['B13'].values, nan=0)
        
        if np.mean(vis) < 1.0: return f"🌑 Night {date_str}"
        era5_data, official_cot, ceres_truth = get_aux_data(date, time_str, vis.shape, ceres_local_path)

        vis_norm = vis / 100.0
        ir_norm = (ir - 200) / 100.0
        era5_tcc = era5_data[:,:,0] 
        era5_rad = era5_data[:,:,1] / 1000.0 
        era5_temp = (era5_data[:,:,2] - 273.15) / 50.0 
        
        X_final = np.stack([vis_norm, ir_norm, era5_tcc, era5_rad, era5_temp], axis=-1)
        np.savez(save_path, X=X_final, y_cot=official_cot, y_solar=ceres_truth)
        
        return f"✅ Saved Master {date_str} {time_str}"

    except Exception as e:
        return f"❌ Error {date_str}: {e}"
    finally:
        try: shutil.rmtree(thread_temp)
        except: pass

def get_ceres_dictionary():
    """Fetches metadata for all CERES granules for the whole year ONCE."""
    print("☀️ Authenticating with NASA Earthdata for main process...")
    earthaccess.login()
    print("🔍 Pre-fetching CERES metadata for the entire year (This takes ~15 seconds)...")
    results = earthaccess.search_data(
        short_name="CER_GEO_Ed4_HIM09_NH",
        temporal=(START_DATE, END_DATE),
        bounding_box=BD_BBOX
    )
    
    # Map them by Date and Hour: e.g., "20250101_0200" -> granule object
    ceres_dict = {}
    for r in results:
        # Example timestamp: '2025-01-01T02:00:00.000Z'
        time_str = r['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        key = f"{time_str[:4]}{time_str[5:7]}{time_str[8:10]}_{time_str[11:13]}00"
        ceres_dict[key] = r
    
    print(f"📊 Cached {len(ceres_dict)} CERES granules in memory.")
    return ceres_dict

def initialize_worker():
    """This runs exactly once per CPU core when the process starts."""
    import earthaccess
    try:
        earthaccess.login()
    except Exception:
        pass

def main():
    print("🚀 Starting ULTRA-FAST INTEGRATED PIPELINE")
    ceres_lookup = get_ceres_dictionary()
    fs = s3fs.S3FileSystem(anon=True)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)
    
    tasks = []
    for date in dates:
        for time_str in HOURS:
            yyyy, mm, dd = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
            key = f"{yyyy}{mm}{dd}_{time_str}"
            granule = ceres_lookup.get(key, None)
            tasks.append((date, time_str, fs, granule))
            
    # Switch to ProcessPoolExecutor for CPU-bound performance with the new initializer
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=initialize_worker) as executor:
        results = executor.map(process_one_slot, tasks)
        for res in results: print(res)

if __name__ == "__main__":
    main()