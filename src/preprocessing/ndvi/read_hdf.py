import rasterio
import numpy as np

def read_ndvi(hdf_path):
    ndvi_layer = (
        f'HDF4_EOS:EOS_GRID:"{hdf_path}":'
        'MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"'
    )

    with rasterio.open(ndvi_layer) as src:
        ndvi = src.read(1).astype("float32")
        ndvi[ndvi <= -2000] = np.nan
        ndvi = ndvi * 0.0001

        return ndvi, src.transform, src.crs
