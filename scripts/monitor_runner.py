
import glob, numpy as np, os, time

log  = "C:/Users/yupei/Desktop/SHP/analyze_run.log"
html = "C:/Users/yupei/Desktop/SHP/analysis_all.html"
mon  = "C:/Users/yupei/Desktop/SHP/monitor.log"
orig_mtime = os.path.getmtime(html)

while True:
    caches = glob.glob("C:/Users/yupei/Desktop/SHP/SCAN/*/*/analysis_cache.npz")
    v2 = sum(1 for c in caches if "cache_version" in (d:=np.load(c,allow_pickle=True)) and str(d["cache_version"])=="v2")
    cur_mtime = os.path.getmtime(html)
    with open(log) as f:
        last = f.readlines()[-1].strip()
    msg = "[" + time.strftime("%H:%M:%S") + "] " + str(v2) + "/84 | " + last
    with open(mon, "a") as m:
        m.write(msg + "
")
    if cur_mtime > orig_mtime:
        with open(mon, "a") as m:
            m.write("DONE\! HTML saved.
")
        break
    time.sleep(120)
