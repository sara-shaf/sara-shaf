"""Weekly research-map updater for the GitHub profile (sara-shaf/sara-shaf).

1. Fetches works from ORCID and adds papers not yet on the map (nothing is ever removed).
2. Redraws assets/research-map.svg and assets/research-evolution.svg from research-map-data.json.
"""
import json, math, os, re, sys, urllib.request
from collections import Counter
from html import escape

ORCID_ID = "0000-0001-9433-5060"
DATA_FILE = "research-map-data.json"
EXCLUDE = ["retraction note", "improving financial literacy and supporting financial decisions"]
THEME_RULES = [
    ("ai",    r"agentic|generative ai|synthetic data|deep learning|anomaly|llm|large language|judge|neural|simulation-based|manufacturing system reconfiguration|product and manufacturing design"),
    ("rec",   r"recommender|recommendation|bert|online reviews|consumer|co-demanded|community|user implicit|co-creation|customisation model|innovation strategy|expert systems"),
    ("aec",   r"\baec\b|construction|modulari|building|off-site|facade|fa\u00e7ade"),
    ("agile", r"scrum|agile|design thinking|model-driven|behavio[u]?r-driven|rational unified|computer-aided software|user-cent|use-oriented|knowledge domains|ui/ux|traditional versus"),
    ("org",   r"digital transformation|organi[sz]ation|leadership|team learning|reintegration|patient|resistance to innovation|supply chain|well-being"),
    ("other", r"antibacterial|nanoparticle|nano-|earnings|teachers|language teaching|textbook"),
    ("pcs",   r"."),
]
COL = {"pcs": "#7dd3fc", "agile": "#a78bfa", "aec": "#34d399", "rec": "#f472b6", "org": "#94a3b8", "ai": "#fbbf24", "other": "#fb923c"}
HUB = {"pcs": (290, 290), "agile": (560, 160), "rec": (860, 180), "ai": (690, 420), "aec": (130, 500), "other": (440, 520), "org": (960, 490)}
ORDER = ["pcs", "agile", "aec", "org", "rec", "ai", "other"]


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())[:70]


def theme_for(title):
    for k, rx in THEME_RULES:
        if re.search(rx, title.lower()):
            return k
    return "pcs"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "research-map-updater"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def update_from_orcid(data):
    try:
        groups = get(f"https://pub.orcid.org/v3.0/{ORCID_ID}/works").get("group", [])
    except Exception as e:
        print("ORCID not reachable, keeping current data:", e)
        return
    known = {norm(p["t"]) for p in data["pubs"]}
    for g in groups:
        s = (g.get("work-summary") or [{}])[0]
        title = (((s.get("title") or {}).get("title") or {}).get("value") or "").strip()
        year = ((s.get("publication-date") or {}).get("year") or {}).get("value")
        if not title or not year or norm(title) in known or any(title.lower().startswith(x) for x in EXCLUDE):
            continue
        data["pubs"].append({"t": title, "y": int(year), "v": ((s.get("journal-title") or {}).get("value") or ""),
                             "a": ["S. Shafiee"], "j": s.get("type") == "journal-article", "th": theme_for(title)})
        known.add(norm(title))
        print("Added:", year, title)


def wrap(text, n=22):
    lines, cur = [], ""
    for w in text.split():
        if len(cur + " " + w) > n and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def draw_map(data, path):
    W, H = 1100, 700
    names = {t["k"]: t["n"] for t in data["themes"]}
    latest = max(p["y"] for p in data["pubs"])
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Research map: {len(data["pubs"])} publications grouped by theme">',
         '<defs><radialGradient id="bg" cx=".5" cy=".45" r=".75"><stop offset="0" stop-color="#1b2150"/><stop offset="1" stop-color="#0f1226"/></radialGradient>'
         '<radialGradient id="glow"><stop offset="0" stop-color="#fbbf24" stop-opacity=".55"/><stop offset="1" stop-color="#fbbf24" stop-opacity="0"/></radialGradient></defs>',
         f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>']
    golden = math.pi * (3 - math.sqrt(5))
    for k in ORDER:
        ps = sorted([p for p in data["pubs"] if p["th"] == k], key=lambda p: p["y"])
        if not ps:
            continue
        hx, hy, c = *HUB[k], COL[k]
        pts = []
        for i, p in enumerate(ps):
            rr = 34 + 9.2 * math.sqrt(i + 1)
            a = i * golden
            pts.append((hx + rr * math.cos(a), hy + rr * math.sin(a), p))
        for x, y, p in pts:
            o.append(f'<line x1="{hx}" y1="{hy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{c}" stroke-opacity=".15"/>')
        o.append(f'<circle cx="{hx}" cy="{hy}" r="27" fill="{c}" fill-opacity=".13" stroke="{c}" stroke-opacity=".6">'
                 f'<animate attributeName="r" values="25;30;25" dur="4s" repeatCount="indefinite"/></circle>')
        o.append(f'<circle cx="{hx}" cy="{hy}" r="8" fill="{c}"/>')
        for i, (x, y, p) in enumerate(pts):
            r = 6.5 if p.get("j") else 4.2
            if p["y"] >= latest - 1:
                o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*2.6:.1f}" fill="url(#glow)">'
                         f'<animate attributeName="opacity" values=".2;1;.2" dur="3s" begin="{(i%5)*.6}s" repeatCount="indefinite"/></circle>')
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" stroke="#0f1226" stroke-width="1.2"><title>{p["y"]}: {escape(p["t"])}</title></circle>')
        bottom = max(y for _, y, _ in pts) + 26
        label = wrap(names[k]) + [f"{len(ps)} publications"]
        for layer in ("halo", "ink"):
            attrs = 'stroke="#0f1226" stroke-width="5" stroke-linejoin="round" fill="#0f1226"' if layer == "halo" else 'fill="#e6e8f2"'
            t = [f'<text x="{hx}" y="{bottom:.0f}" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="14" font-weight="600" {attrs}>']
            for i, l in enumerate(label):
                last = i == len(label) - 1
                extra = ' font-weight="400" font-size="12.5"' + ('' if layer == "halo" else ' fill="#9aa3bf"') if last else ''
                t.append(f'<tspan x="{hx}" dy="{0 if i == 0 else 17}"{extra}>{escape(l)}</tspan>')
            o.append("".join(t) + "</text>")
    o.append(f'<text x="24" y="{H-20}" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12.5" fill="#9aa3bf">Each dot is a publication · larger dots are journal articles · glowing dots are from {latest-1}–{latest}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def draw_evolution(data, path):
    W, H, L, R, T, B = 1100, 400, 44, 16, 20, 130
    years = list(range(min(p["y"] for p in data["pubs"]), max(p["y"] for p in data["pubs"]) + 1))
    cnt = Counter((p["y"], p["th"]) for p in data["pubs"])
    mx = max(sum(cnt[(y, k)] for k in ORDER) for y in years)
    mx = int(math.ceil(mx / 4.0) * 4)
    pw, ph = W - L - R, H - T - B
    bw = pw / len(years)
    sy = lambda v: T + ph - v / mx * ph
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Publications per year by theme">',
         f'<rect width="{W}" height="{H}" rx="18" fill="#0f1226"/>']
    F = 'font-family="Segoe UI, Helvetica, Arial, sans-serif"'
    for v in range(0, mx + 1, 4):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#2a2f52"/><text x="{L-8}" y="{sy(v)+4:.1f}" text-anchor="end" {F} font-size="12" fill="#9aa3bf">{v}</text>')
    for i, y in enumerate(years):
        x = L + i * bw + bw * 0.18
        acc = 0
        for k in ORDER:
            n = cnt[(y, k)]
            if n:
                o.append(f'<rect x="{x:.1f}" y="{sy(acc+n):.1f}" width="{bw*0.64:.1f}" height="{sy(acc)-sy(acc+n):.1f}" rx="2" fill="{COL[k]}"><title>{y}: {n}</title></rect>')
                acc += n
        o.append(f'<text x="{x + bw*0.32:.1f}" y="{T+ph+18}" text-anchor="middle" {F} font-size="12" fill="#9aa3bf">{y}</text>')
    names = {t["k"]: t["n"] for t in data["themes"]}
    colw = (W - L - R) / 2
    for i, k in enumerate(ORDER):
        lx = L + (i % 2) * colw
        ly = H - 78 + (i // 2) * 20
        o.append(f'<circle cx="{lx+5:.1f}" cy="{ly-4}" r="5" fill="{COL[k]}"/><text x="{lx+15:.1f}" y="{ly}" {F} font-size="12.5" fill="#c9cde0">{escape(names[k])}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def main():
    data = json.load(open(DATA_FILE, encoding="utf-8"))
    if "--no-fetch" not in sys.argv:
        update_from_orcid(data)
    data["pubs"].sort(key=lambda p: (-p["y"], p["t"]))
    json.dump(data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs("assets", exist_ok=True)
    draw_map(data, "assets/research-map.svg")
    draw_evolution(data, "assets/research-evolution.svg")
    print(f"Drew research map with {len(data['pubs'])} publications.")


if __name__ == "__main__":
    main()
