import os
import argparse
import calendar
import zipfile
import shutil
import s3fs
import pandas as pd
import numpy as np
from satpy.scene import Scene
from glob import glob
import warnings
import concurrent.futures
import cv2
import xarray as xr
import earthaccess
import tempfile
import cdsapi
import pvlib

warnings.filterwarnings("ignore")

# --- 1. VALIDATION REGION & SATELLITE CONFIGURATION ---
REGIONS = {
    "us": {"bbox": (-105.0, 30.0, -90.0, 40.0), "ceres_name": "CER_GEO_Ed4_GOES16_NH"},
    "pacific": {"bbox": (160.0, 0.0, 170.0, 10.0), "ceres_name": "CER_GEO_Ed4_HIM8"}, # Himawari-8/9 CERES
    "sahara": {"bbox": (10.0, 20.0, 20.0, 30.0), "ceres_name": "CER_GEO_Ed4_MET8"},  # Meteosat CERES
    "congo": {"bbox": (15.0, -5.0, 25.0, 5.0), "ceres_name": "CER_GEO_Ed4_MET8"},
    "se_asia": {"bbox": (110.0, -5.0, 120.0, 5.0), "ceres_name": "CER_GEO_Ed4_HIM8"}, # GK2A/Himawari CERES proxy
    "bangladesh":{"bbox":(87, 20, 93, 27), "ceres_name": "CER_GEO_Ed4_HIM09_NH"}
}

# --- 2. COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Universal Data Downloader for Azure ML")
parser.add_argument("--satellite", type=str, required=True, choices=["GOES16", "HIMAWARI9", "SEVIRI", "GK2A"])
parser.add_argument("--region", type=str, required=True, choices=["us", "pacific", "sahara", "congo", "se_asia"])
parser.add_argument("--start_date", type=str, default="2023-06-15")
parser.add_argument("--end_date", type=str, default="2023-06-20")
parser.add_argument("--output_dir", type=str, default="./outputs/test_data")

# Credentials
parser.add_argument("--cds_key", type=str, required=True)
parser.add_argument("--earthdata_user", type=str, required=True)
parser.add_argument("--earthdata_pass", type=str, required=True)
args = parser.parse_args()

# --- INJECT CREDENTIALS ---
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = args.cds_key
os.environ["EARTHDATA_USERNAME"] = args.earthdata_user
os.environ["EARTHDATA_PASSWORD"] = args.earthdata_pass

BBOX = REGIONS[args.region]["bbox"]
CERES_DATASET = REGIONS[args.region]["ceres_name"]
OUTPUT_DIR = os.path.join(args.output_dir, args.region)
AUX_DATA_ROOT = os.path.join(OUTPUT_DIR, "aux_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUX_DATA_ROOT, exist_ok=True)

# --- 3. ERA5 DOWNLOADER (Dynamic BBox) ---
def download_era5():
    """Downloads monthly ERA5 data for auxiliary channels."""
    c = cdsapi.Client()
    dataset = 'reanalysis-era5-single-levels'
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq="1D")
    
    for ym in dates.to_period('M').unique():
        year, month_str = str(ym.year), f"{ym.month:02d}"
        final_folder = os.path.join(AUX_DATA_ROOT, f"era5_{args.region}_{year}_{month_str}")
        if os.path.exists(final_folder): continue

        temp_outfile = os.path.join(AUX_DATA_ROOT, f"raw_era5_{year}_{month_str}.tmp")
        last_day = calendar.monthrange(int(year), ym.month)[1]
        
        print(f"⬇️ Requesting ERA5: {year}-{month_str} for {args.region}")
        try:
            era5_area = [BBOX[3], BBOX[0], BBOX[1], BBOX[2]]
            
            result = c.retrieve(dataset, {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['total_cloud_cover', 'surface_solar_radiation_downwards', '2m_temperature'],
                'year': year, 'month': month_str, 'day': [f"{i:02d}" for i in range(1, last_day + 1)],
                'time': [f"{i:02d}:00" for i in range(24)],
                'area': era5_area, 
            })
            result.download(temp_outfile)
            
            os.makedirs(final_folder, exist_ok=True)
            if zipfile.is_zipfile(temp_outfile):
                with zipfile.ZipFile(temp_outfile, 'r') as zip_ref:
                    zip_ref.extractall(final_folder)
            else:
                shutil.move(temp_outfile, os.path.join(final_folder, "era5_data.nc"))
            if os.path.exists(temp_outfile): os.remove(temp_outfile)
        except Exception as e:
            print(f"❌ ERA5 Error: {e}")

# --- FIXED: SOLAR ZENITH ANGLE CALCULATOR ---
def get_solar_zenith_grid(date_time_obj, target_shape, resampled_area):
    """
    Computes a 2D grid of solar zenith angles matching the satellite image shape 
    using pvlib and Satpy grid coordinates (flattening 2D arrays to 1D for pvlib compatibility).
    """
    try:
        lons, lats = resampled_area.get_lonlats()
        
        # pvlib requires 1D arrays, so we flatten the grids here
        flat_lats = lats.ravel()
        flat_lons = lons.ravel()
        
        times = pd.DatetimeIndex([date_time_obj] * len(flat_lats))
        
        solpos = pvlib.solarposition.get_solarposition(
            time=times,
            latitude=flat_lats,
            longitude=flat_lons
        )
        
        zenith = solpos['apparent_zenith'].values
        zenith = np.nan_to_num(zenith, nan=90.0)
        
        # Reshape back to the original target 2D shape (H, W)
        return zenith.reshape(target_shape).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Solar Zenith Calculation Error: {e}")
        return np.zeros(target_shape, dtype=np.float32)

# --- 4. SATELLITE & LABEL PROCESSING ---
def get_aux_data(date, hour_str, target_shape, ceres_file_path):
    yyyy, mm, dd = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    month_dir = os.path.join(AUX_DATA_ROOT, f"era5_{args.region}_{yyyy}_{mm}")
    era5_cube = np.zeros((*target_shape, 3), dtype=np.float32)
    
    if os.path.exists(month_dir):
        nc_files = glob(os.path.join(month_dir, "*.nc"))
        if nc_files:
            time_query = f"{yyyy}-{mm}-{dd}T{hour_str[:2]}:00:00"
            for nc_file in nc_files:
                with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
                    if 'expver' in ds.dims: ds = ds.isel(expver=0)
                    slice_ds = ds.sel(valid_time=time_query, method="nearest")
            
                    if 'tcc' in ds.variables: era5_cube[:,:,0] = cv2.resize(slice_ds['tcc'].values, (target_shape[1], target_shape[0]))
                    if 'ssrd' in ds.variables: era5_cube[:,:,1] = cv2.resize(slice_ds['ssrd'].values, (target_shape[1], target_shape[0]))
                    if 't2m' in ds.variables: era5_cube[:,:,2] = cv2.resize(slice_ds['t2m'].values, (target_shape[1], target_shape[0]))

    ceres_cot = np.full(target_shape, np.nan, dtype=np.float32)
    if ceres_file_path and os.path.exists(ceres_file_path):
        with xr.open_dataset(ceres_file_path, engine="h5netcdf") as ds_ceres:
            lat = np.squeeze(ds_ceres['latitude'].values)
            lon = np.squeeze(ds_ceres['longitude'].values)
            raw_cot = np.squeeze(ds_ceres['cloud_visible_optical_depth'].values)
            
            min_lon, min_lat, max_lon, max_lat = BBOX
            valid_mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            valid_coords = np.where(valid_mask)
            
            if len(valid_coords[0]) > 0:
                min_row, max_row = np.min(valid_coords[0]), np.max(valid_coords[0])
                min_col, max_col = np.min(valid_coords[1]), np.max(valid_coords[1])
                clipped_cot = raw_cot[min_row:max_row+1, min_col:max_col+1]
                
                smoothed_cot = cv2.GaussianBlur(clipped_cot, (3, 3), 0)
                ceres_cot = cv2.resize(smoothed_cot, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    return era5_cube, ceres_cot, era5_cube[:,:,1]

# --- 5. DYNAMIC SATELLITE DOWNLOADER ---
def download_satellite_data(fs, date, hh, thread_temp):
    yyyy, mm, dd = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    julian_day = date.strftime("%j") 
    download_tasks = []
    reader_type = ""
    bands_to_load = []
    
    if args.satellite == "GOES16":
        bucket_path = f"noaa-goes19/ABI-L1b-RadF/{yyyy}/{julian_day}/{hh}/"
        reader_type, bands_to_load = 'abi_l1b', ['C02', 'C13']
        for band in bands_to_load:
            all_files = fs.glob(f"{bucket_path}*{band}*")
            if all_files:
                local = os.path.join(thread_temp, all_files[0].split('/')[-1])
                fs.get(all_files[0], local)
                download_tasks.append(local)
                
    elif args.satellite == "HIMAWARI9":
        bucket_path = f"noaa-himawari9/AHI-L1b-FLDK/{yyyy}/{mm}/{dd}/{hh}00/"
        reader_type, bands_to_load = 'ahi_hsd', ['B03', 'B13']
        for band in bands_to_load:
            all_files = fs.glob(f"{bucket_path}*{band}*")
            if all_files:
                local = os.path.join(thread_temp, all_files[0].split('/')[-1])
                fs.get(all_files[0], local)
                download_tasks.append(local)
                
    elif args.satellite == "GK2A":
        bucket_path = f"noaa-gk2a-pds/AMI-L1b-FD/{yyyy}/{mm}/{dd}/{hh}/"
        reader_type, bands_to_load = 'ami_l1b', ['VI064', 'IR105']
        for band in bands_to_load:
            all_files = fs.glob(f"{bucket_path}*{band}*")
            if all_files:
                local = os.path.join(thread_temp, all_files[0].split('/')[-1])
                fs.get(all_files[0], local)
                download_tasks.append(local)
        
    elif args.satellite == "SEVIRI":
        pass

    return download_tasks, reader_type, bands_to_load

def find_closest_granule(target_dt, granules_list):
    closest_granule = None
    min_diff = pd.Timedelta(days=999)
    for g in granules_list:
        t_str = g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        g_dt = pd.to_datetime(t_str)
        diff = abs(g_dt.tz_localize(None) - target_dt.tz_localize(None))
        if diff < min_diff:
            min_diff = diff
            closest_granule = g
    if min_diff <= pd.Timedelta(minutes=45):
        return closest_granule
    return None

def process_one_slot(task_info):
    date, time_str, fs, results_list = task_info
    date_str = f"{date.strftime('%Y%m%d')}"
    
    save_name = f"master_{args.satellite}_{args.region}_{date_str}_{time_str}.npz"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    if os.path.exists(save_path): return f"⏩ Skipped {save_name}"

    thread_temp = tempfile.mkdtemp()
    try:
        # 1. Download Satellite Data
        download_tasks, reader, bands = download_satellite_data(fs, date, time_str[:2], thread_temp)
        if len(download_tasks) < 2 and args.satellite == "GOES16": return f"⚠️ Missing Bands {date_str}"
        if len(download_tasks) < 2 and args.satellite == "HIMAWARI9": return f"⚠️ Missing Bands {date_str}"

        # 2. Match and Download CERES Data dynamically by time proximity
        hour_int = int(time_str[:2])
        minute_int = int(time_str[2:])
        slot_dt = pd.to_datetime(date).replace(hour=hour_int, minute=minute_int)

        ceres_granule_dict = find_closest_granule(slot_dt, results_list)
        ceres_local_path = None
        if ceres_granule_dict:
            downloaded = earthaccess.download([ceres_granule_dict], thread_temp)
            if downloaded: ceres_local_path = downloaded[0]

        # 3. Satpy Processing (Sensor Agnostic)
        scn = Scene(filenames=download_tasks, reader=reader)
        scn.load(bands)
        cropped = scn.crop(ll_bbox=BBOX)
        resampled = cropped.resample(cropped[bands[1]].attrs['area'], resampler='native')
        
        vis = np.nan_to_num(resampled[bands[0]].values, nan=0)
        ir = np.nan_to_num(resampled[bands[1]].values, nan=0)
        
        # Compute Solar Zenith Angle Grid matching image dimensions
        solar_zenith = get_solar_zenith_grid(slot_dt, vis.shape, resampled[bands[1]].attrs['area'])

        # 4. Aux Data & Stack
        # NOTE: get_aux_data still returns the full era5_cube (tcc, ssrd, t2m) because
        # channel index 1 (ssrd) is required below as the y_solar TARGET. It must NOT
        # also be placed in X_final, or the model is handed its own label as an input
        # feature (target leakage) — this was silently happening in previous versions
        # and is almost certainly why prior SSRD R^2 looked inflated.
        era5_data, official_cot, ceres_truth = get_aux_data(date, time_str, vis.shape, ceres_local_path)

        vis_norm = vis / 100.0 if np.max(vis) > 1.0 else vis
        ir_norm = (ir - 200) / 100.0
        solar_zenith_norm = solar_zenith / 90.0  # Normalize zenith angle to roughly [0, 1] range

        # Final 5-channel input stack: VIS, IR, ERA5 TCC, ERA5 T2M, Solar Zenith Angle.
        # ERA5 SSRD (era5_data[:,:,1]) is deliberately EXCLUDED here — it is the same
        # physical quantity as the y_solar label below, so including it would leak the
        # target into the input. It is only ever used downstream as the label.
        X_final = np.stack([
            vis_norm,
            ir_norm,
            era5_data[:,:,0],                     # ERA5 total cloud cover
            (era5_data[:,:,2] - 273.15) / 50.0,    # ERA5 2m temperature (K -> normalized C)
            solar_zenith_norm
        ], axis=-1)

        np.savez_compressed(save_path, X=X_final, y_cot=official_cot, y_solar=ceres_truth / 1000000.0)
        return f"✅ Saved {save_name}"
    except Exception as e:
        return f"❌ Error {date_str}: {e}"
    finally:
        shutil.rmtree(thread_temp)

def main():
    print(f"🚀 Starting Pipeline for {args.satellite} over {args.region.upper()}")
    download_era5()
    
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=CERES_DATASET,
        temporal=(args.start_date, args.end_date),
        bounding_box=BBOX
    )
    
    fs = s3fs.S3FileSystem(anon=True)
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq="1D")
    
    # Process 1600 to 2000 UTC, passing the full results list for dynamic temporal lookups
    tasks = [(date, f"{h:02d}00", fs, results) for date in dates for h in range(16, 21)]
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for res in executor.map(process_one_slot, tasks): print(res)

if __name__ == "__main__":
    main()
