"""Weekly research-map updater for the GitHub profile (sara-shaf/sara-shaf).

1. Fetches works from ORCID and adds papers not yet on the map (nothing is ever removed).
2. Embeds all paper titles (TF-IDF + t-SNE) and draws assets/research-map.svg,
   plus a per-year chart in assets/research-evolution.svg. Both are animated SVGs.
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
SHORT = {"pcs": "Product configuration", "agile": "Agile & model-driven", "aec": "Construction & modular",
         "rec": "Recommender systems", "org": "Digital transformation", "ai": "AI agents & manufacturing", "other": "Beyond engineering"}
ORDER = ["pcs", "agile", "aec", "org", "rec", "ai", "other"]
MONO = 'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"'
SANS = 'font-family="Segoe UI, Helvetica, Arial, sans-serif"'


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


def embed(pubs):
    """2-D map of paper titles: TF-IDF features (+ theme hint) -> SVD -> t-SNE."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform([p["t"] for p in pubs])
    X = TruncatedSVD(n_components=min(40, X.shape[1] - 1), random_state=7).fit_transform(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    onehot = np.array([[1.0 if p["th"] == k else 0.0 for k in ORDER] for p in pubs]) * 1.1
    Z = TSNE(n_components=2, perplexity=min(15, len(pubs) - 1), random_state=7, init="pca").fit_transform(np.hstack([X, onehot]))
    return Z, X


def halo_text(x, y, text, size, fill, weight=700, anchor="middle", extra=""):
    base = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" {SANS} font-size="{size}" font-weight="{weight}" {extra}'
    return (f'<text {base} stroke="#0b0e22" stroke-width="7" stroke-linejoin="round" fill="#0b0e22">{escape(text)}</text>'
            f'<text {base} fill="{fill}">{escape(text)}</text>')


def draw_map(data, path):
    import numpy as np
    pubs = data["pubs"]
    Z, X = embed(pubs)
    W, H = 1100, 820
    x0, x1, y0, y1 = 70, 1030, 165, 720
    mn, mx = Z.min(0), Z.max(0)
    P = (Z - mn) / (mx - mn + 1e-9)
    P[:, 0] = x0 + P[:, 0] * (x1 - x0)
    P[:, 1] = y0 + P[:, 1] * (y1 - y0)
    years = [p["y"] for p in pubs]
    ymin, ymax = min(years), max(years)
    span = ymax - ymin + 1
    LOOP = 18.0                                   # seconds per animation loop
    grow = 0.72                                   # share of the loop used to "train"
    t_of = lambda y: (y - ymin) / span * grow
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="Research embedding map of {len(pubs)} publications in {len(ORDER)} clusters, {ymin} to {ymax}">',
         '<defs><radialGradient id="bg" cx=".5" cy=".5" r=".8"><stop offset="0" stop-color="#171d48"/><stop offset="1" stop-color="#0b0e22"/></radialGradient>'
         '<radialGradient id="gold"><stop offset="0" stop-color="#fbbf24" stop-opacity=".6"/><stop offset="1" stop-color="#fbbf24" stop-opacity="0"/></radialGradient>'
         '<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#6366f1" stroke-opacity=".08"/></pattern></defs>',
         f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/><rect x="0" y="120" width="{W}" height="{H-185}" fill="url(#grid)"/>']
    # terminal header
    o.append(f'<rect x="18" y="16" width="{W-36}" height="88" rx="12" fill="#0b0e22" stroke="#2a2f52"/>')
    for i, c in enumerate(["#f87171", "#fbbf24", "#34d399"]):
        o.append(f'<circle cx="{40+i*20}" cy="36" r="6" fill="{c}"/>')
    o.append(f'<text x="40" y="68" {MONO} font-size="21" fill="#7dd3fc">$ embed --papers {len(pubs)} --method tfidf+tsne</text>')
    o.append(f'<text x="40" y="94" {MONO} font-size="19" fill="#9aa3bf">&#8594; {len(ORDER)} research clusters &#183; {ymin}&#8211;{ymax}</text>')
    o.append(f'<text x="{W-120}" y="68" text-anchor="end" {MONO} font-size="19" fill="#9aa3bf">epoch</text>')
    for y in range(ymin, ymax + 1):
        a, b = t_of(y), (t_of(y + 1) if y < ymax else 1.0)
        vis = "visible" if y == ymax else "hidden"
        o.append(f'<text x="{W-60}" y="94" text-anchor="end" {MONO} font-size="30" font-weight="700" fill="#fbbf24" visibility="{vis}">{y}'
                 f'<animate attributeName="visibility" values="hidden;visible;hidden" keyTimes="0;{a:.4f};{min(b,0.9999):.4f}" calcMode="discrete" dur="{LOOP}s" repeatCount="indefinite"/></text>')
    o.append(f'<rect x="{W-54}" y="70" width="12" height="26" fill="#7dd3fc"><animate attributeName="opacity" values="1;0;1" dur="1.1s" repeatCount="indefinite"/></rect>')
    # nearest-neighbour graph (in embedding space) for a neural-network look
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, 1e9)
    edges = set()
    for i in range(len(pubs)):
        for j in np.argsort(D[i])[:3]:
            if D[i, j] > 130 ** 2:
                continue
            edges.add(tuple(sorted((i, int(j)))))
    for i, j in sorted(edges):
        a = t_of(max(years[i], years[j]))
        o.append(f'<line x1="{P[i,0]:.1f}" y1="{P[i,1]:.1f}" x2="{P[j,0]:.1f}" y2="{P[j,1]:.1f}" stroke="#818cf8" stroke-opacity=".22" stroke-width="1.2">'
                 f'<animate attributeName="stroke-opacity" values="0;0;.22;.22" keyTimes="0;{a:.4f};{a+.03:.4f};1" dur="{LOOP}s" repeatCount="indefinite"/></line>')
    # signals travelling between cluster centres
    cent = {k: P[[i for i, p in enumerate(pubs) if p["th"] == k]].mean(0) for k in ORDER if any(p["th"] == k for p in pubs)}
    route = [k for k in ["pcs", "agile", "rec", "ai", "aec", "org", "other"] if k in cent]
    for n, (a, b) in enumerate(zip(route, route[1:] + route[:1])):
        (ax, ay), (bx, by) = cent[a], cent[b]
        o.append(f'<path id="sig{n}" d="M{ax:.1f},{ay:.1f} L{bx:.1f},{by:.1f}" stroke="#fbbf24" stroke-opacity=".10" stroke-dasharray="3 7"/>')
        o.append(f'<circle r="3.5" fill="#fde68a"><animateMotion dur="{3.2+n*.4:.1f}s" begin="{n*.5:.1f}s" repeatCount="indefinite"><mpath href="#sig{n}"/></animateMotion></circle>')
    # dots
    for i, p in enumerate(pubs):
        x, y = P[i]
        a = t_of(p["y"])
        r = 8 if p.get("j") else 5.5
        c = COL[p["th"]]
        anim = f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{a:.4f};{a+.025:.4f};1" dur="{LOOP}s" repeatCount="indefinite"/>'
        if p["y"] >= ymax - 1:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*2.8:.1f}" fill="url(#gold)">{anim}'
                     f'<animate attributeName="r" values="{r*2:.1f};{r*3.2:.1f};{r*2:.1f}" dur="2.4s" repeatCount="indefinite"/></circle>')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" stroke="#0b0e22" stroke-width="1.5"><title>{p["y"]}: {escape(p["t"])}</title>{anim}</circle>')
    # cluster labels (large, readable on phones)
    placed = []
    for k in ORDER:
        if k not in cent:
            continue
        idx = [i for i, p in enumerate(pubs) if p["th"] == k]
        cx = float(np.median(P[idx, 0])); top = float(P[idx, 1].min())
        half = len(f"{SHORT[k]} · {len(idx)}") * 24 * 0.30
        ly = max(150, top - 16)
        lx = min(max(cx, half + 24), W - half - 24)
        moved = True
        while moved:
            moved = False
            for (px, py, ph) in placed:
                if abs(px - lx) < ph + half + 12 and abs(py - ly) < 32:
                    ly = py - 34 if ly <= py else py + 34
                    moved = True
        placed.append((lx, ly, half))
        o.append(halo_text(lx, ly, f"{SHORT[k]} · {len(idx)}", 24, COL[k]))
    o.append(halo_text(W / 2, H - 52, "each dot is a paper · closeness = similar topics (TF-IDF + t-SNE of titles)", 19, "#c9cde0", 400))
    o.append(halo_text(W / 2, H - 24, f"bigger = journal article · glowing = {ymax-1}–{ymax} · updates weekly from ORCID", 19, "#9aa3bf", 400))
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def draw_evolution(data, path):
    pubs = data["pubs"]
    W, H, L, R, T = 1100, 560, 60, 24, 130
    years = list(range(min(p["y"] for p in pubs), max(p["y"] for p in pubs) + 1))
    cnt = Counter((p["y"], p["th"]) for p in pubs)
    top = max(sum(cnt[(y, k)] for k in ORDER) for y in years)
    top = int(math.ceil(top / 4.0) * 4)
    ph = 260
    pw = W - L - R
    bw = pw / len(years)
    sy = lambda v: T + ph - v / top * ph
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Publications per year by research cluster">',
         f'<rect width="{W}" height="{H}" rx="18" fill="#0b0e22"/>',
         f'<rect x="18" y="16" width="{W-36}" height="66" rx="12" fill="#0f1226" stroke="#2a2f52"/>']
    for i, c in enumerate(["#f87171", "#fbbf24", "#34d399"]):
        o.append(f'<circle cx="{40+i*20}" cy="36" r="6" fill="{c}"/>')
    o.append(f'<text x="40" y="68" {MONO} font-size="21" fill="#7dd3fc">$ plot papers_per_year --hue cluster</text>')
    for v in range(0, top + 1, 4):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#2a2f52"/>'
                 f'<text x="{L-10}" y="{sy(v)+7:.1f}" text-anchor="end" {SANS} font-size="20" fill="#9aa3bf">{v}</text>')
    for i, y in enumerate(years):
        x = L + i * bw + bw * 0.14
        acc, delay = 0, i * 0.18
        for k in ORDER:
            n = cnt[(y, k)]
            if not n:
                continue
            ya, yb = sy(acc + n), sy(acc)
            o.append(f'<rect x="{x:.1f}" y="{ya:.1f}" width="{bw*0.72:.1f}" height="{yb-ya:.1f}" rx="3" fill="{COL[k]}"><title>{y}: {n} · {escape(SHORT[k])}</title>'
                     f'<animate attributeName="height" from="0" to="{yb-ya:.1f}" begin="{delay:.2f}s" dur=".6s" fill="freeze"/>'
                     f'<animate attributeName="y" from="{sy(0):.1f}" to="{ya:.1f}" begin="{delay:.2f}s" dur=".6s" fill="freeze"/></rect>')
            acc += n
        if i % 2 == 0 or len(years) <= 8:
            o.append(f'<text x="{x + bw*0.36:.1f}" y="{T+ph+30}" text-anchor="middle" {SANS} font-size="20" fill="#9aa3bf">{y}</text>')
    colw = (W - L - R) / 2
    for i, k in enumerate(ORDER):
        lx = L + (i % 2) * colw
        ly = T + ph + 74 + (i // 2) * 30
        o.append(f'<circle cx="{lx+8:.1f}" cy="{ly-7}" r="8" fill="{COL[k]}"/><text x="{lx+24:.1f}" y="{ly}" {SANS} font-size="21" fill="#e6e8f2">{escape(SHORT[k])}</text>')
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def main():
    data = json.load(open(DATA_FILE, encoding="utf-8"))
    if "--no-fetch" not in sys.argv:
        update_from_orcid(data)
    data["pubs"].sort(key=lambda p: (-p["y"], p["t"]))
    json.dump(data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs("assets", exist_ok=True)
    draw_map(data, "assets/research-map-v2.svg")
    draw_evolution(data, "assets/research-evolution-v2.svg")
    print(f"Drew research map with {len(data['pubs'])} publications.")


if __name__ == "__main__":
    main()
