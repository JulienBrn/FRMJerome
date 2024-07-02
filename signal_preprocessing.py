import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

p = Database("signal_preprocessing")

@p.register
@CoordComputer.from_function(coords=["signal", "electrode", "contact_depth"], database_arg="db")
def signal(db, session, block):
    metadata = db.run_action("load", "raw_metadata", session=session, block=block, single=True)
    ret = pd.DataFrame()
    ret["signal"] = metadata["label"]
    ret["contact_depth"] = metadata["label"].str.extract("(\d+)", expand=False).astype(pd.Int64Dtype())
    ret["electrode"] = metadata["label"].str.extract("(\D+)", expand=False)
    return ret

@p.register
@Data.from_class()
class Signal:
    name = "raw_recording"

    @staticmethod
    def location(folder, session, block, electrode, contact_depth):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "Signals" / f"{contact_depth:02d}" / "raw_recording.nc"
    
    @staticmethod
    @cache(lambda f, a: a.to_netcdf(f))
    def compute(db, out_location, selection): 
        all_signals = db.run_action("load", "raw_signals", selection, single=True)
        metadata = db.run_action("load", "raw_metadata", selection, single=True)
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
        
    @staticmethod
    def load(db, out_location, selection): return xr.open_dataset(out_location)

    @staticmethod
    def plot(db, out_location, selection): 
        import matplotlib.pyplot as plt
        db.run_action("load", "raw_recording", selection, single=True)["signal"].plot()
        plt.suptitle(str(selection))
        plt.show()

@p.register
@CoordComputer.from_function(coords=["depth_1", "depth_2", "depth_pair"], database_arg="db")
def contact_pair_num(db, session, block, electrode):
    contacts = db.get_coords(["contact_depth"], session=session, block=block, electrode=electrode)["contact_depth"].to_list()
    contacts.sort()
    return [(contacts[i], contacts[i+1], f'{contacts[i+1]:02d}-{contacts[i]:02d}') for i in range(len(contacts)-1)]

@p.register
@Data.from_class()
class BipolarSignal:
    name = "bipolar_sig"

    @staticmethod
    def location(folder, session, block, electrode, depth_pair):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_recording.nc"
    
    @staticmethod
    @cache(lambda f, a: a.to_netcdf(f))
    def compute(db: DatabaseInstance, out_location, selection): 
        sig1 = db.run_action("load", "raw_recording", selection, contact_depth=selection["depth_1"], single=True)
        sig2 = db.run_action("load", "raw_recording", selection, contact_depth=selection["depth_2"], single=True)
        d = sig2-sig1
        return d
    
    @staticmethod
    def load(db, out_location, selection): return xr.open_dataset(out_location)

    @staticmethod
    def plot(db, out_location, selection): 
        import matplotlib.pyplot as plt
        db.run_action("load", "bipolar_sig", selection, single=True)["signal"].plot()
        plt.suptitle(str(selection))
        plt.show()
    
@p.register
@Data.from_class()
class BipolarSignalZscored:
    name = "bipolar_sig_zscored"

    @staticmethod
    def location(folder, session, block, electrode, depth_pair):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_zscored_recording.nc"
    
    @staticmethod
    @cache(lambda f, a: a.to_netcdf(f))
    def compute(db, out_location, selection): 
        d = db.run_action("load", "bipolar_sig", selection, single=True)
        zscore_part = d["signal"].sel(t=slice(0, 5))
        d["z-scored"] = (d["signal"] - zscore_part.mean())/zscore_part.std()
        return d["z-scored"]
    
    @staticmethod
    def load(db, out_location, selection): return xr.open_dataset(out_location)

    @staticmethod
    def plot(db, out_location, selection): 
        import matplotlib.pyplot as plt
        db.run_action("load", "bipolar_sig_zscored", selection, single=True).plot()
        plt.suptitle(str(selection))
        plt.show()

@p.register
@CoordComputer.from_function()
def f():
    return np.arange(120)


@p.register
@Data.from_class()
class BipolarSignalStft:
    name = "bipolar_stft"

    @staticmethod
    def location(folder, session, block, electrode, depth_pair):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "bipolar_stft.nc"
    
    @staticmethod
    @cache(lambda f, a: a.to_netcdf(f, engine="h5netcdf", invalid_netcdf=True))
    def compute(db, out_location, selection): 
        d = db.run_action("load", "bipolar_sig_zscored", selection, single=True)
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
    
    @staticmethod
    def load(db, out_location, selection): return xr.open_dataset(out_location, engine="h5netcdf")
    
@p.register
@Data.from_class()
class BipolarSignalSpectrogram:
    name = "stft_bipolar_spectrogram"

    @staticmethod
    def location(folder, session, block, electrode, depth_pair):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "stft_bipolar_spectrogram.nc"
    
    @staticmethod
    @cache(lambda f, a: a.to_netcdf(f))
    def compute(db, out_location, selection): 
        d = xr.open_dataset(p.compute_unique("bipolar_stft", selection), engine="h5netcdf")
        return (np.abs(d["stft"])**2).rename("spectrogram")
    
    @staticmethod
    def load(db, out_location, selection): return xr.open_dataset(out_location)

    @staticmethod
    def plot(db, out_location, selection): 
        import matplotlib.pyplot as plt
        np.log(db.run_action("load", "stft_bipolar_spectrogram", selection, single=True)["spectrogram"]).plot.pcolormesh(x="t", y="f")
        plt.suptitle(str(selection))
        plt.show()


    
# @p.register
# @Data.from_class()
# class BipolarSignalSpectrogram2:
#     name = "scipy_bipolar_spectrogram"

#     @staticmethod
#     def location(folder, session, block, electrode, depth_pair):
#         return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "Electrodes" / f"{electrode}" / "BipolarSignals" / f"{depth_pair}" / "scipy_bipolar_spectrogram.nc"
    
#     @staticmethod
#     @cache(lambda f, a: a.to_netcdf(f))
#     def compute(out_location, selection): 
#         d = xr.open_dataset(p.compute_unique("bipolar_sig_zscored", selection))
#         fs = get_fs(d["t"].to_numpy())
#         f = p.get_coords(["f"])["f"].to_numpy()
#         freq_fs = get_fs(f)
#         import scipy.signal
#         stft_obj = scipy.signal.ShortTimeFFT(scipy.signal.windows.hamming(int(fs/freq_fs)), 
#                         hop=int(fs*0.01), 
#                         fs=fs, 
#                         scale_to="psd"
#         )
#         arr_size, p0, p1 = d["t"].size, stft_obj.lower_border_end[1], stft_obj.upper_border_begin(d["t"].size)[1]
#         stft: xr.DataArray = xr.apply_ufunc(lambda a: stft_obj.spectrogram(a, p0=p0, p1=p1, axis=-1), d["z-scored"],  input_core_dims=[["t"]], output_core_dims=[["f", "t_bin"]])
#         stft = stft.rename(t_bin="t")
#         stft["t"] = stft_obj.t(arr_size, p0, p1) + d["t"].isel(t=0).item()
#         stft["f"] = stft_obj.f
#         stft=stft.interp(f=f)
#         stft = stft.to_dataset(name="spectrogram")
#         return stft

pipeline = p