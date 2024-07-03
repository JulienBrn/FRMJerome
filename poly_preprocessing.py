import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

p = Database("polypreprocessing")

@p.register
@Data.from_class()
class PolyData:
    name = "poly_data"

    @staticmethod
    def location(folder, session, subject, block):
        if subject == "P01-CP":
            return singleglob(Path(folder) / session / "PolyData", f"**/{subject}*B{block:02d}*.dat", f"**/P01-PC*B{block:02d}*.dat", f"**/P1-PC*B{block:02d}*.dat")
        elif subject == "P02-LC":
            return singleglob(Path(folder) / session / "PolyData",f"**/P02-LC*BLOCK{block:01d}*.dat", f"**/P01_LC*BLOCK{block:01d}*.dat", f"**/P02_LC*BLOCK{block:01d}*.dat")
        else:
            raise Exception("Not a known subject")
        
    @staticmethod
    def load(db, out_location, selection): return pd.read_csv(out_location, sep="\t", 
                                                    names=['time (ms)', 'family', 'nbre', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'], skiprows=13)

@p.register
@Data.from_class()
class PolyData:
    name = "poly_events"

    @staticmethod
    def location(folder, session, block):
        return Path(folder) / session / "Blocks" / f"Block_{block:02d}" / "poly_events.tsv"
    
    @staticmethod
    @cache(lambda f, a: a.to_csv(f, sep="\t"), force_recompute=True)
    def compute(db: DatabaseInstance, out_location, selection): 
        data = db.run_action("load", "poly_data", selection, single=True)
        left_cues = []
        right_cues = []
        left_pads = []
        right_pads = []
        left_levers=[]
        right_levers = []
        for _, row in data.iterrows():
            t = row["time (ms)"]
            match row["family"], row["nbre"]:
                case 1,2:
                    if row["_L"] > 0:
                        left_cues.append([t, row["_P"], row["_L"]])
                case 1, 4:
                    if row["_L"] > 0:
                        right_cues.append([t, row["_P"], row["_L"]])
                case 6, 22:
                    if row['_V'] == 0:
                        right_pads.append([t, row["_P"]])
                    elif row['_V'] == 1:
                        left_pads.append([t, row["_P"]])
                    else:
                        raise Exception(f'Error unknwon pad {row["_P"]}')
                case 2,1:
                    left_levers.append([t, row["_V"]])
                case 2,2:
                    right_levers.append([t, row["_V"]])
                case _: pass

        dfs = {}
        errors = []
        for which, l in dict(Cue_left=left_cues, Cue_right=right_cues, Pad_left=left_pads, Pad_right=right_pads, Lever_left=left_levers, Lever_right=right_levers).items():
            try:
                [ev_type, direction] = which.split("_")
                r = xr.Dataset()
                on = [t[0]/1000 for t in l if t[1]==1]
                off = [t[0]/1000 for t in l if t[1]==0]
                on.sort()
                off.sort()
                if len(off) > 0:
                    if len(on) ==0 or on[0] > off[0]:
                        on =[-1] + on
                if len(on) > 0:
                    if len(off) ==0 or off[-1] < on[-1]:
                        off = off + [np.nan]
                if len(on) != len(off):
                    seq = pd.concat([pd.Series(on).to_frame(name="t").assign(which="on"), pd.Series(off).to_frame(name="t").assign(which="off")]).sort_values("t").reset_index(drop=True)
                    s = seq["which"].to_numpy()
                    indices = np.flatnonzero(s[: -1] == s[1:])
                    reduced_seq = seq.iloc[np.sort(np.unique(np.concatenate([indices+i for i in range(-3, 4) if indices+i >=0 and indices+i < len(seq.index)]))), :]
                    raise Exception(f"Problem with on/off. n_on = {len(on)}, n_off={len(off)}\n{reduced_seq.to_string()}")
                
                
                r["t"] = xr.DataArray(on, dims="event")
                r["t_end"] = xr.DataArray(off, dims="event")
                r["duration"] = r["t_end"] - r["t"] 
                if ev_type=="Cue":
                    if ((r["duration"] < 0) | (r["duration"] > 0.300)).any():
                        raise Exception(f'Problem figuring out cue events... {r["duration"]}')
                    r["number"] = xr.DataArray([t[2] for t in l if t[1] ==0], dims="event")
                    dfs[f"{which}_go"] = r.where(r["number"] ==1, drop=True)["t"].to_dataframe()
                    dfs[f"{which}_nogo"] = r.where(r["number"] ==2, drop=True)["t"].to_dataframe()
                elif ev_type == "Pad":
                    dfs[f"{which}_Press"] = r["t"].to_dataframe()
                    dfs[f"{which}_Lift"] = r[["t_end"]].rename(t_end="t")["t"].to_dataframe()
                elif ev_type=="Lever":
                    dfs[f"{which}"] = r["t"].to_dataframe()
            except Exception as e:
                e.add_note(f'While computing {which} events')
                if not db.continue_on_error:
                    raise e
                else:
                    errors.append(e)

        if len(errors) > 0:
            raise ExceptionGroup("Error while computing events", errors)
        res = pd.concat(dfs).reset_index(level=0, names=["Event"]).sort_values("t").reset_index().drop(columns="event")
        res["Trial"] = res["Event"].str.contains("Cue").cumsum() -1
        return res


pipeline = p