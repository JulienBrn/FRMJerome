import logging, beautifullogger
import sys
from Pipeline.pipeline import Pipeline,PipelineInstance, cache, Data, CoordComputer, singleglob, get_fs
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

def add_signal_preprocessing_pipeline(p: PipelineInstance, folder):
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
                res["t"] = xr.DataArray(np.arange(res["signal"].size)/reduced["sample_rate"].iat[0], dims="t")
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
            d = sig2-sig1
            return d
        
    @p.register
    @Data.from_class
    class BipolarSignalZscored:
        name = "bipolar_sig_zscored"

        @staticmethod
        def location(session, block, electrode, depth_pair):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_zscored_recording.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            d = xr.open_dataset(p.compute_unique("bipolar_sig", selection))
            zscore_part = d["signal"].sel(t=slice(0, 5))
            d["z-scored"] = (d["signal"] - zscore_part.mean())/zscore_part.std()
            return d["z-scored"]
        

    @p.register
    @CoordComputer.from_function()
    def f():
        return np.arange(120)
    

    @p.register
    @Data.from_class
    class BipolarSignalStft:
        name = "bipolar_stft"

        @staticmethod
        def location(session, block, electrode, depth_pair):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_stft.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f, engine="h5netcdf", invalid_netcdf=True))
        def compute(out_location, selection): 
            d = xr.open_dataset(p.compute_unique("bipolar_sig_zscored", selection))
            fs = get_fs(d["t"].to_numpy())
            f = p.get_coords(["f"])["f"].to_numpy()
            freq_fs = get_fs(f)
            import scipy.signal
            stft_obj = scipy.signal.ShortTimeFFT(scipy.signal.windows.hamming(int(fs/freq_fs)), 
                          hop=int(fs*0.01), 
                          fs=fs, 
                          scale_to="psd"
            )
            arr_size, p0, p1 = d["t"].size, stft_obj.lower_border_end[1], stft_obj.upper_border_begin(d["t"].size)[1]
            stft: xr.DataArray = xr.apply_ufunc(lambda a: stft_obj.stft(a, p0=p0, p1=p1, axis=-1), d["z-scored"],  input_core_dims=[["t"]], output_core_dims=[["f", "t_bin"]])
            stft = stft.rename(t_bin="t")
            stft["t"] = stft_obj.t(arr_size, p0, p1) + d["t"].isel(t=0).item()
            stft["f"] = stft_obj.f
            stft=stft.interp(f=f)
            stft = stft.to_dataset(name="stft")
            return stft
        
    @p.register
    @Data.from_class
    class BipolarSignalSpectrogram:
        name = "stft_bipolar_spectrogram"

        @staticmethod
        def location(session, block, electrode, depth_pair):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "stft_bipolar_spectrogram.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            d = xr.open_dataset(p.compute_unique("bipolar_stft", selection), engine="h5netcdf")
            return (np.abs(d["stft"])**2).rename("spectrogram")
        
    @p.register
    @Data.from_class
    class BipolarSignalSpectrogram2:
        name = "scipy_bipolar_spectrogram"

        @staticmethod
        def location(session, block, electrode, depth_pair):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "scipy_bipolar_spectrogram.nc"
        
        @staticmethod
        @cache(lambda f, a: a.to_netcdf(f))
        def compute(out_location, selection): 
            d = xr.open_dataset(p.compute_unique("bipolar_sig_zscored", selection))
            fs = get_fs(d["t"].to_numpy())
            f = p.get_coords(["f"])["f"].to_numpy()
            freq_fs = get_fs(f)
            import scipy.signal
            stft_obj = scipy.signal.ShortTimeFFT(scipy.signal.windows.hamming(int(fs/freq_fs)), 
                          hop=int(fs*0.01), 
                          fs=fs, 
                          scale_to="psd"
            )
            arr_size, p0, p1 = d["t"].size, stft_obj.lower_border_end[1], stft_obj.upper_border_begin(d["t"].size)[1]
            stft: xr.DataArray = xr.apply_ufunc(lambda a: stft_obj.spectrogram(a, p0=p0, p1=p1, axis=-1), d["z-scored"],  input_core_dims=[["t"]], output_core_dims=[["f", "t_bin"]])
            stft = stft.rename(t_bin="t")
            stft["t"] = stft_obj.t(arr_size, p0, p1) + d["t"].isel(t=0).item()
            stft["f"] = stft_obj.f
            stft=stft.interp(f=f)
            stft = stft.to_dataset(name="spectrogram")
            return stft

    