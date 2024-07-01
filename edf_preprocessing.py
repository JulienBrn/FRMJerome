import logging, beautifullogger
import sys
from Pipeline.pipeline import Pipeline, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

def add_edf_preprocessing_pipeline(p, folder):
    @p.register
    @Data.from_class
    class AllEDFFiles:
        name = "all_edf_files"

        @staticmethod
        def location():
            return  folder/"all_edf_files.txt"
        
        @staticmethod
        @cache(lambda *args: np.savetxt(*args, fmt="%s"))
        def compute(out_location: Path, selection):
            return np.array([str(f.relative_to(folder)) for f in folder.glob("**/*.edf")])

    @p.register
    @CoordComputer.from_function(vectorized=False, adapt_return=False, coords=["session","subject", "block"])
    def session():
        p.compute("all_edf_files")
        l=np.loadtxt(p.get_single_location("all_edf_files"), dtype=str)
        l = [f for f in l if "BAGOSMOV" in f]
        df = pd.DataFrame()
        df["session"] = [str(Path(f).parent.parent) for f in l]
        df["subject"] = [str(Path(f).parents[-3].stem) for f in l]
        df["block"] = [int(Path(f).stem[-2:]) for f in l]
        return df

    @p.register
    @Data.from_class
    class RecordingData:
        name = "raw_edf_data"

        @staticmethod
        def location(session, block):
            return singleglob(folder / session, f"*_BAGOSMOV/*{block}.edf")

    @p.register
    @Data.from_class
    class RecordingData:
        name = "edf_extracted_data"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data"
        
        @staticmethod
        def compute(out_location: Path, selection):
            signal_loc = p.get_single_location("raw_signals", selection)
            event_loc = p.get_single_location("raw_events", selection)
            metadata_loc = p.get_single_location("raw_metadata", selection)
            exists = [loc.exists() for loc in (signal_loc, event_loc, metadata_loc)]
            if np.all(exists):
                return
            out_location.mkdir(exist_ok=True, parents=True)
            edf_loc = p.get_single_location("raw_edf_data", selection)
            from pyedflib import highlevel
            signals, signal_headers, header = highlevel.read_edf(str(edf_loc))
            annotations = pd.DataFrame(header.pop("annotations"), columns=["t", "?", "Marker"])
            signal_headers=pd.DataFrame(signal_headers).reset_index(names=["sig_index"])
            np.save(signal_loc, signals)
            annotations.to_csv(event_loc, sep="\t")
            signal_headers.to_csv(metadata_loc, sep="\t")


    @p.register
    @Data.from_class
    class RawSignals:
        name = "raw_signals"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "signals.npy"
        
        @staticmethod
        def compute(out_location, selection): p.compute("edf_extracted_data", selection)

    @p.register
    @Data.from_class
    class RawEvents:
        name = "raw_events"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "events.tsv"
        
        @staticmethod
        def compute(out_location, selection): p.compute("edf_extracted_data", selection)

    @p.register
    @Data.from_class
    class RawMetadata:
        name = "raw_metadata"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "metadata.tsv"
        
        @staticmethod
        def compute(out_location, selection): p.compute("edf_extracted_data", selection)
