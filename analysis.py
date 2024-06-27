import logging, beautifullogger
import sys
from Pipeline.pipeline import Pipeline, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import preprocessing

logger = logging.getLogger(__name__)
beautifullogger.setup()


pipeline = Pipeline()
folder = Path("/home/julienb/Documents/Data/Lea/")






if __name__ == "__main__":
    preprocessing.add_preprocessing_pipeline(pipeline, folder)
    p = pipeline.initialize()
    print(p)
    # print(p.get_coords(["session", "block"]))
    # print(p.get_locations("raw_edf_data").to_string())
    p.compute("bipolar_sig")
    # print(p.get_coords(["session", "electrode", "block", "depth_pair"]).reset_index())