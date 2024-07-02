import logging, beautifullogger
import sys
from database import Database, DatabaseInstance, cache, Data, CoordComputer, singleglob
import pandas as pd, numpy as np, xarray as xr
from pathlib import Path
logger = logging.getLogger(__name__)

def add_poly_preprocessing_pipeline(p: DatabaseInstance, folder: Path):
    
    @p.register
    @Data.from_class
    class PolyData:
        name = "poly_data"

        @staticmethod
        def location(session, subject, block):
            if subject == "P01-CP":
                return singleglob(folder / session / "PolyData", f"**/{subject}*B{block:02d}*.dat", f"**/P01-PC*B{block:02d}*.dat", f"**/P1-PC*B{block:02d}*.dat")
            elif subject == "P02-LC":
                return singleglob(folder / session / "PolyData",f"**/P02-LC*BLOCK{block:01d}*.dat", f"**/P01_LC*BLOCK{block:01d}*.dat", f"**/P02_LC*BLOCK{block:01d}*.dat")
            else:
                raise Exception("Not a known subject")
    
    @p.register
    @Data.from_class
    class PolyData:
        name = "poly_events"

        @staticmethod
        def location(session, block):
            return folder / session / "Blocks" / f"Block_{block:02d}" / "poly_events.tsv"
        
        @staticmethod
        @cache(lambda f, a: a.to_csv(f, sep="\t"), force_recompute=True)
        def compute(out_location, selection): 
            data = pd.read_csv(p.get_single_location("poly_data", selection), sep="\t", names=['time (ms)', 'family', 'nbre', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'], skiprows=13)
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
            for which, l in dict(Cue_left=left_cues, Cue_right=right_cues, Pad_left=left_pads, Pad_right=right_pads, Lever_left=left_levers, Lever_right=right_levers).items():
                [ev_type, direction] = which.split("_")
                r = xr.Dataset()
                on = [t[0]/1000 for t in l if t[1]==1]
                off = [t[0]/1000 for t in l if t[1]==0]
                if len(on) > len(off):
                    off = [0] + off
                if len(off) > len(on):
                    on = [0]+on
                r["t"] = xr.DataArray(on, dims="event")
                r["t_end"] = xr.DataArray(off, dims="event")
                r["duration"] = r["t_end"] - r["t"] 
                if ev_type=="Cue":
                    if ((r["duration"] < 0) | (r["duration"] > 0.300)).any():
                        raise Exception('Problem figuring out cue events...')
                    r["number"] = xr.DataArray([t[2] for t in l if t[1] ==0], dims="event")
                    dfs[f"{which}_go"] = r.where(r["number"] ==1, drop=True)["t"].to_dataframe()
                    dfs[f"{which}_nogo"] = r.where(r["number"] ==2, drop=True)["t"].to_dataframe()
                elif ev_type == "Pad":
                    dfs[f"{which}_Press"] = r["t"].to_dataframe()
                    dfs[f"{which}_Lift"] = r[["t_end"]].rename(t_end="t")["t"].to_dataframe()
                elif ev_type=="Lever":
                    dfs[f"{which}"] = r["t"].to_dataframe()


            res = pd.concat(dfs).reset_index(level=0, names=["Event"]).sort_values("t").reset_index().drop(columns="event")
            res["Trial"] = res["Event"].str.contains("Cue").cumsum() -1
            return res


        