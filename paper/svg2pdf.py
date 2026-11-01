"""Vector PDFs of the storyboard SVGs (with the real fonts), via headless Edge, for the LaTeX report."""
import os, re, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "illus")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
F = os.path.join(HERE, "fonts").replace("\\", "/")
for f in sorted(os.listdir(D)):
    if not f.endswith(".svg"): continue
    s = open(os.path.join(D, f), encoding="utf-8").read()
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
    css = (f"@font-face{{font-family:Manrope;src:url('file:///{F}/ManropeExtraLight-Regular-VF.ttf');font-weight:200 800}}"
           f"@font-face{{font-family:'JetBrains Mono';src:url('file:///{F}/JetBrainsMono-Regular-VF.ttf');font-weight:100 800}}"
           f"@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0}}svg{{display:block;width:{w}px;height:{h}px}}")
    html = os.path.join(D, "_p.html"); open(html, "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><style>{css}</style>{s}")
    out = os.path.join(D, f[:-4] + ".pdf")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "--virtual-time-budget=3000",
                    "file:///" + html.replace("\\", "/")], capture_output=True, timeout=90)
    print(f, os.path.exists(out))
os.remove(os.path.join(D, "_p.html"))
