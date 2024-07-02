import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

def add_event_preprocessing_pipeline(p: DatabaseInstance, folder):

    @p.register
    @Data.from_class
    class TrialsComputer:
        name = "trial_indexed_events"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "trial_indexed_events.tsv"
    
        @staticmethod
        @cache(lambda f, a: a.to_csv(f, sep="\t"))
        def compute(out_location, selection): 
            events = pd.read_csv(p.compute_unique("raw_events", selection), sep="\t")
            events["Event"] = np.where(events["Marker"] == "MARQUEUR 1", "PadReady",
                              np.where(events["Marker"] == "MARQUEUR 2", "Cue",
                              np.where(events["Marker"] == "MARQUEUR 4", "PadLift",
                              np.where(events["Marker"] == "MARQUEUR 128", "LeverPress",
            "Unknown"))))
            events = events.sort_values("t")
            if ((events["Event"] == "PadReady").shift(1, fill_value=0) != (events["Event"] == "Cue")).any():
                raise Exception("Problem")
            events["Trial"] = (events["Event"] == "PadReady").cumsum() -1
            return events.loc[events["Event"]!="Unknown", ["t", "Trial", "Event"]]


    @p.register
    @Data.from_class
    class EventDataset:
        name = "event_dataset"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "event_dataset.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f), force_recompute=True)
        def compute(out_location, selection): 
            events = pd.read_csv(p.compute_unique("trial_indexed_events", selection), sep="\t")
            def compute_trial_info(d):
                res = dict(trial_pattern = ",".join(d["Event"].to_list()), trial_start=d["t"].iat[0])
                return pd.Series(res)
            trials = events.groupby("Trial").apply(compute_trial_info)
            event_times = events.drop_duplicates(["Trial", "Event"], keep="first").set_index(["Trial", "Event"])
            d = xr.Dataset.from_dataframe(trials)
            d["event_times"] = xr.DataArray.from_series(event_times["t"])
            d["trial_type"] = xr.where(d["trial_pattern"] == "PadReady,Cue", "NoGoSucess",
                              xr.where(d["trial_pattern"] == "PadReady,Cue,PadLift,LeverPress", "GoSucess",
                                       "Other"
            ))
            if (d["trial_type"]=="Other").sum() > 0:
                logger.warning(f'Found Trials with unexpected pattern for {selection}. Patterns in file are \n{d[["trial_pattern", "trial_type"]].to_dataframe().value_counts()}')
            return d


    @p.register
    @CoordComputer.from_function()
    def event_name():
        return ["PadReady", "Cue", "PadLift", "LeverPress"]
    
    @p.register
    @CoordComputer.from_function()
    def signal_property():
        return ["stft_bipolar_spectrogram", "scipy_bipolar_spectrogram"]
    

    @p.register
    @Data.from_class
    class EventWindowing:
        name = "windowed_data"

        @staticmethod
        def location(session, block, electrode, depth_pair, signal_property):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / f"event_{signal_property}.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            events = pd.read_csv(p.compute_unique("trial_indexed_events", selection), sep="\t")
            events=events.set_index(["Trial", "Event"])["t"]
            events = xr.DataArray.from_series(events)
            # events= events.loc[events["Event"] == selection["event_name"]]
            match selection["signal_property"]:
                case "stft_bipolar_spectrogram":
                    data = xr.open_dataset(p.compute_unique(selection["signal_property"], selection))["spectrogram"]
                case "scipy_bipolar_spectrogram":
                    data = xr.open_dataset(p.compute_unique(selection["signal_property"], selection))["spectrogram"]
            
            print(events)
            exit()
            return 