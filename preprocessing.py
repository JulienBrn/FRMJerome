import logging, beautifullogger
import sys
from Pipeline.pipeline import Pipeline, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

def add_preprocessing_pipeline(p, folder):
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
    @CoordComputer.from_function(vectorized=False, adapt_return=False, coords=["session", "block"])
    def session():
        p.compute("all_edf_files")
        l=np.loadtxt(p.get_single_location("all_edf_files"), dtype=str)
        l = [f for f in l if "BAGOSMOV" in f]
        df = pd.DataFrame()
        df["session"] = [str(Path(f).parent.parent) for f in l]
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

    @p.register
    @CoordComputer.from_function(coords=["signal", "electrode", "contact_depth"])
    def signal(session, block):
        p.compute("raw_metadata", session=session, block=block)
        metadata = pd.read_csv(p.get_single_location("raw_metadata", session=session, block=block), sep="\t")
        ret = pd.DataFrame()
        ret["signal"] = metadata["label"]
        ret["contact_depth"] = metadata["label"].str.extract("(\d+)", expand=False).astype(pd.Int64Dtype())
        ret["electrode"] = metadata["label"].str.extract("(\D+)", expand=False)
        return ret

    @p.register
    @Data.from_class
    class Signal:
        name = "raw_recording"

        @staticmethod
        def location(session, block, electrode, contact_depth):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "Signals" / f"{contact_depth:02d}" / "raw_recording.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            all_signals = np.load(p.compute_unique("raw_signals", selection))
            metadata = pd.read_csv(p.compute_unique("raw_metadata", selection), sep="\t")
            reduced = metadata.loc[metadata["label"] == selection["signal"], :]
            if len(reduced.index) > 1:
                raise Exception(f"Many corresponding sig_index...\n {reduced}")
            elif len(reduced.index) ==0:
                raise Exception("No sig_index...")
            else:
                res = xr.Dataset()
                res["signal"] = xr.DataArray(all_signals[reduced["sig_index"].iat[0], :], dims="t")
                res["t"] = xr.DataArray(np.arange(res["signal"].size)/reduced["sample_rate"].iat[0])
                return res


    @p.register
    @CoordComputer.from_function(coords=["depth_1", "depth_2", "depth_pair"])
    def contact_pair_num(session, block, electrode):
        contacts = p.get_coords(["contact_depth"], session=session, block=block, electrode=electrode)["contact_depth"].to_list()
        contacts.sort()
        return [(contacts[i], contacts[i+1], f'{contacts[i+1]:02d}-{contacts[i]:02d}') for i in range(len(contacts)-1)]

    @p.register
    @Data.from_class
    class BipolarSignal:
        name = "bipolar_sig"

        @staticmethod
        def location(session, block, electrode, depth_pair):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_recording.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            sig1 = xr.open_dataset(p.compute_unique("raw_recording", selection, contact_depth=selection["depth_1"]))
            sig2 = xr.open_dataset(p.compute_unique("raw_recording", selection, contact_depth=selection["depth_2"]))
            return sig2-sig1