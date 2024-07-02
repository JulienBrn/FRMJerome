import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)


p = Database("edf_preprocessing")

@p.register
@Data.from_class()
class AllEDFFiles:
    name = "all_edf_files"

    @staticmethod
    def location(folder):
        return  Path(folder)/"all_edf_files.txt"
    
    @staticmethod
    @cache(lambda *args: np.savetxt(*args, fmt="%s"))
    def compute(db: DatabaseInstance, out_location: Path, selection):
        folder=Path(selection["folder"])
        return np.array([str(f.relative_to(folder)) for f in folder.glob("**/*.edf")])

    @staticmethod
    def load(db: DatabaseInstance, out_location, selection):
        loc = db.compute_single("all_edf_files", selection)
        return np.loadtxt(loc, dtype=str)
    
    @staticmethod
    def show(db, out_location, selection):
        v = db.run_action("load", "all_edf_files", selection, single=True)
        print(v)

@p.register
@CoordComputer.from_function(vectorized=False, adapt_return=False, coords=["session","subject", "block"], database_arg="db")
def session(db):
    l = db.run_action("load", "all_edf_files", single=True)
    l = [f for f in l if "BAGOSMOV" in f]
    df = pd.DataFrame()
    df["session"] = [str(Path(f).parent.parent) for f in l]
    df["subject"] = [str(Path(f).parents[-3].stem) for f in l]
    df["block"] = [int(Path(f).stem[-2:]) for f in l]
    return df

@p.register
@Data.from_class()
class RecordingData:
    name = "raw_edf_data"

    @staticmethod
    def location(folder, session, block):
        return singleglob(Path(folder) / session, f"*_BAGOSMOV/*{block}.edf")

@p.register
@Data.from_class()
class RecordingData:
    name = "edf_extracted_data"

    @staticmethod
    def location(folder, session, block):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data"
    
    @staticmethod
    def compute(db, out_location: Path, selection):
        signal_loc = db.get_single_location("raw_signals", selection)
        event_loc = db.get_single_location("raw_events", selection)
        metadata_loc = db.get_single_location("raw_metadata", selection)
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
        return out_location


@p.register
@Data.from_class()
class RawSignals:
    name = "raw_signals"

    @staticmethod
    def location(folder, session, block):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "signals.npy"
    
    @staticmethod
    def compute(db, out_location, selection): db.compute("edf_extracted_data", selection)

    @staticmethod
    def load(db, out_location, selection): return np.load(out_location)

    @staticmethod
    def show(db, out_location, selection): print(db.run_action("load", "raw_signals", selection, single=True))


@p.register
@Data.from_class()
class RawEvents:
    name = "raw_events"

    @staticmethod
    def location(folder, session, block):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "events.tsv"
    
    @staticmethod
    def compute(db, out_location, selection): db.compute("edf_extracted_data", selection)

    @staticmethod
    def load(db, out_location, selection): return pd.read_csv(out_location, sep="\t")

    @staticmethod
    def show(db, out_location, selection): return print(db.run_action("load", "raw_events", selection, single=True))

@p.register
@Data.from_class()
class RawMetadata:
    name = "raw_metadata"

    @staticmethod
    def location(folder, session, block):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "edf_extracted_data" / "metadata.tsv"
    
    @staticmethod
    def compute(db, out_location, selection): db.compute("edf_extracted_data", selection)

    @staticmethod
    def load(db, out_location, selection): return pd.read_csv(out_location, sep="\t")

    @staticmethod
    def show(db, out_location, selection): return print(db.run_action("load", "raw_metadata", selection, single=True))

pipeline = p