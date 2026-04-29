import cdsapi
import earthaccess
import os
import calendar
import zipfile
import shutil
from datetime import datetime

YEARS = ['2023']
DATA_ROOT = "aux_data1"
os.makedirs(DATA_ROOT, exist_ok=True)

def download_era5():
    """Downloads monthly ERA5 data and extracts if zipped."""
    c = cdsapi.Client()
    dataset = 'reanalysis-era5-single-levels'
    
    for year in YEARS:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            temp_outfile = os.path.join(DATA_ROOT, f"raw_era5_{year}_{month_str}.tmp")
            final_folder = os.path.join(DATA_ROOT, f"era5_bd_{year}_{month_str}")
            if os.path.exists(final_folder):
                print(f"✅ Data for {year}-{month_str} already exists in {final_folder}")
                continue

            last_day = calendar.monthrange(int(year), month)[1]
            days = [f"{i:02d}" for i in range(1, last_day + 1)]

            print(f"⬇️ Requesting ERA5: {year}-{month_str}")
            try:
                result = c.retrieve(dataset, {
                    'product_type': 'reanalysis',
                    'format': 'netcdf', 
                    'variable': [
                        'total_cloud_cover', 
                        'surface_solar_radiation_downwards',
                        '2m_temperature'
                    ],
                    'year': year,   
                    'month': month_str,
                    'day': days,
                    'time': [f"{i:02d}:00" for i in range(24)],
                    'area': [27, 87, 20, 93], 
                })
                result.download(temp_outfile)
                if zipfile.is_zipfile(temp_outfile):
                    print(f"📦 ZIP detected for {month_str}. Extracting...")
                    os.makedirs(final_folder, exist_ok=True)
                    with zipfile.ZipFile(temp_outfile, 'r') as zip_ref:
                        zip_ref.extractall(final_folder)
                    os.remove(temp_outfile)
                else:
                    os.makedirs(final_folder, exist_ok=True)
                    shutil.move(temp_outfile, os.path.join(final_folder, f"era5_data.nc"))

                print(f"✨ {year}-{month_str} ready in {final_folder}")

            except Exception as e:
                print(f"❌ Error for {year}-{month_str}: {e}")
                if os.path.exists(temp_outfile):
                    os.remove(temp_outfile)

if __name__ == "__main__":
    download_era5()
    print("🎉 All Auxiliary Data Ready!")