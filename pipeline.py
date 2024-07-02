import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
import edf_preprocessing, signal_preprocessing, event_preprocessing, poly_preprocessing

logger = logging.getLogger(__name__)

 
def mk_pipeline(folder):
    pipeline = Database("Human Jerome database")
    pipeline.declare(CoordComputer(coords={"folder"}, dependencies=set(), compute=lambda db, df: df.assign(folder=str(folder))))
    pipeline = Database.join(pipeline, edf_preprocessing.pipeline)
    pipeline = Database.join(pipeline, signal_preprocessing.pipeline)
    # signal_preprocessing.add_signal_preprocessing_pipeline(pipeline, folder)
    # event_preprocessing.add_event_preprocessing_pipeline(pipeline, folder)
    # poly_preprocessing.add_poly_preprocessing_pipeline(pipeline, folder)
    return pipeline






if __name__ == "__main__":
    beautifullogger.setup(displayLevel=logging.INFO)
    folder = Path("/home/julienb/Documents/Data/Lea/")
    p = mk_pipeline(folder).initialize()
    print(p)
    p.run_action("plot", "stft_bipolar_spectrogram")
    # print(p.get_coords(["session", "block"]))
    # print(p.get_locations("raw_edf_data").to_string())
    # selection = dict(subject="P01-CP", block=1)
    # p.compute("trial_indexed_events", selection)
    # p.compute("stft_bipolar_spectrogram", selection)
    # p.compute("scipy_bipolar_spectrogram", selection)
    # p.compute("event_dataset")
    # print(p.get_locations("poly_data").to_string())
    # p.compute("poly_events", selection)
    # p.compute("windowed_data", selection)
    # print(p.get_coords(["session", "electrode", "block", "depth_pair"]).reset_index())