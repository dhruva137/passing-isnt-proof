"""Render SVG illustrations to PNG with the real web fonts, through headless Edge."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, os.environ.get("ILLUS_DIR", "illus"))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
FONTS = os.path.join(HERE, "fonts").replace("\\", "/")
CSS = (f"@font-face{{font-family:Manrope;src:url('file:///{FONTS}/ManropeExtraLight-Regular-VF.ttf');font-weight:200 800}}"
       f"@font-face{{font-family:'JetBrains Mono';src:url('file:///{FONTS}/JetBrainsMono-Regular-VF.ttf');font-weight:100 800}}"
       f"@font-face{{font-family:Newsreader;src:url('file:///{FONTS}/Newsreader16pt16pt-Regular-VF.ttf');font-weight:200 800}}"
       "html,body{margin:0;background:transparent}")
for f in sorted(os.listdir(D)):
    if not f.endswith(".svg") or (len(sys.argv) > 1 and sys.argv[1] not in f): continue
    s = open(os.path.join(D, f), encoding="utf-8").read()
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
    html = os.path.join(D, "_r.html")
    open(html, "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{s}")
    out = os.path.join(D, f[:-4] + ".png")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={int(w)},{int(h)}", f"--screenshot={out}", "--virtual-time-budget=3000",
                    "file:///" + html.replace("\\", "/")], capture_output=True, timeout=60)
    print(f, os.path.exists(out))
os.remove(os.path.join(D, "_r.html"))
