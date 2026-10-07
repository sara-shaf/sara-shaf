"""Weekly research-map updater for the GitHub profile (sara-shaf/sara-shaf).

1. Fetches works from ORCID and adds papers not yet on the map (nothing is ever removed).
2. Embeds all paper titles (TF-IDF + t-SNE) and draws assets/research-map.svg,
   plus a per-year chart, and refreshes the "Explore the clusters" list in README.md.
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
         "rec": "Recommender systems", "org": "Digital transformation", "ai": "AI for manufacturing", "other": "Beyond engineering"}
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


GENERIC = set("""study case review approach analysis framework systems system using based case-study literature
literature review company companies research towards paper new evaluation process processes industry industrial high intermediate performance development""".split())


def cluster_keywords(pubs, k, n=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    titles = [p["t"] for p in pubs]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = v.fit_transform(titles).toarray()
    terms = v.get_feature_names_out()
    inside = np.array([p["th"] == k for p in pubs])
    if inside.sum() == 0:
        return []
    score = X[inside].mean(0) - 0.5 * X[~inside].mean(0)
    out = []
    for i in np.argsort(-score):
        t = terms[i]
        own = set(re.findall(r"[a-z]+", SHORT[k].lower()))
        if any(w in GENERIC or w in own or w.rstrip("s") in own for w in t.split()) or score[i] <= 0:
            continue
        if any(t in o or o in t for o in out):
            continue
        out.append(t)
        if len(out) == n:
            break
    return out


def halo_text(x, y, text, size, fill, weight=700, anchor="middle", extra=""):
    base = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" {SANS} font-size="{size}" font-weight="{weight}" {extra}'
    return (f'<text {base} stroke="#0b0e22" stroke-width="7" stroke-linejoin="round" fill="#0b0e22">{escape(text)}</text>'
            f'<text {base} fill="{fill}">{escape(text)}</text>')


def draw_map(data, path):
    import numpy as np
    pubs = data["pubs"]
    Z, X = embed(pubs)
    W, H = 1100, 820
    x0, x1, y0, y1 = 50, 640, 150, 715
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
    newest = max(pubs, key=lambda p: p["y"])
    nt = newest["t"].split(":")[0] if len(newest["t"].split(":")[0]) <= 58 else newest["t"][:56].rsplit(" ", 1)[0] + "…"
    o.append(f'<text x="{W-40}" y="68" text-anchor="end" {MONO} font-size="17" fill="#9aa3bf">newest ({newest["y"]})</text>')
    o.append(f'<text x="{W-40}" y="94" text-anchor="end" {MONO} font-size="17" fill="#fbbf24">{escape(nt)}</text>')
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
        o.append(f'<line x1="{P[i,0]:.1f}" y1="{P[i,1]:.1f}" x2="{P[j,0]:.1f}" y2="{P[j,1]:.1f}" stroke="#818cf8" stroke-opacity=".25" stroke-width="1.2"/>')
    cent = {k: P[[i for i, p in enumerate(pubs) if p["th"] == k]].mean(0) for k in ORDER if any(p["th"] == k for p in pubs)}
    # dots
    for i, p in enumerate(pubs):
        x, y = P[i]
        r = 8 if p.get("j") else 5.5
        c = COL[p["th"]]
        if p["y"] >= ymax - 1:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r*2.6:.1f}" fill="url(#gold)"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" stroke="#0b0e22" stroke-width="1.5"><title>{p["y"]}: {escape(p["t"])}</title></circle>')
    # legend panel (right): one row per cluster, newest activity first
    o.append(f'<rect x="680" y="130" width="{W-700}" height="{H-230}" rx="14" fill="#0b0e22" fill-opacity=".72" stroke="#2a2f52"/>')
    o.append(f'<text x="704" y="166" {MONO} font-size="17" fill="#9aa3bf">clusters · papers · top topics</text>')
    keys = sorted([k for k in ORDER if k in cent], key=lambda k: (-sum(p["y"] >= ymax - 2 for p in pubs if p["th"] == k), -max(p["y"] for p in pubs if p["th"] == k)))
    rowh = (H - 230 - 60) / len(keys)
    for n, k in enumerate(keys):
        idx = [i for i, p in enumerate(pubs) if p["th"] == k]
        y = 200 + n * rowh
        kw = " · ".join(cluster_keywords(pubs, k))
        o.append(f'<circle cx="714" cy="{y+12:.1f}" r="9" fill="{COL[k]}"/>')
        o.append(f'<text x="732" y="{y+19:.1f}" {SANS} font-size="22" font-weight="700" fill="{COL[k]}">{escape(SHORT[k])}</text>')
        o.append(f'<text x="{W-40}" y="{y+19:.1f}" text-anchor="end" {SANS} font-size="22" font-weight="700" fill="#e6e8f2">{len(idx)}</text>')
        if kw:
            o.append(f'<text x="732" y="{y+44:.1f}" {SANS} font-size="17" fill="#9aa3bf">{escape(kw)}</text>')
    o.append(halo_text(W / 2, H - 52, "each dot is a paper · close dots = similar topics (TF-IDF + t-SNE of titles)", 19, "#c9cde0", 400))
    o.append(halo_text(W / 2, H - 24, f"bigger = journal article · glowing = {ymax-1}–{ymax} · click the map to explore every cluster", 19, "#9aa3bf", 400))
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
            o.append(f'<rect x="{x:.1f}" y="{ya:.1f}" width="{bw*0.72:.1f}" height="{yb-ya:.1f}" rx="3" fill="{COL[k]}"><title>{y}: {n} · {escape(SHORT[k])}</title></rect>')
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


def readme_block(data):
    pubs = data["pubs"]
    names = {t["k"]: t["n"] for t in data["themes"]}
    dots = {"pcs": "🔵", "agile": "🟣", "aec": "🟢", "rec": "🩷", "org": "⚪", "ai": "🟡", "other": "🟠"}
    lines = ["### Explore the clusters", "",
             "Click a cluster to see its papers. Newest first.", ""]
    ymax = max(p["y"] for p in pubs)
    for k in sorted(ORDER, key=lambda k: (-sum(p["y"] >= ymax - 2 for p in pubs if p["th"] == k), -max([p["y"] for p in pubs if p["th"] == k] or [0]))):
        ps = sorted([p for p in pubs if p["th"] == k], key=lambda p: (-p["y"], p["t"]))
        if not ps:
            continue
        kw = ", ".join(cluster_keywords(pubs, k))
        lines.append(f"<details><summary>{dots[k]} <b>{escape(names[k])}</b> · {len(ps)} papers · {ps[-1]['y']}–{ps[0]['y']}{' · <i>' + escape(kw) + '</i>' if kw else ''}</summary>")
        lines.append("")
        for p in ps:
            url = f"https://doi.org/{p['doi']}" if p.get("doi") else "https://scholar.google.com/scholar?q=" + urllib.request.quote(p["t"])
            code = f" · [code]({p['code']})" if p.get("code") else ""
            venue = f" · *{p['v']}*" if p.get("v") else ""
            lines.append(f"- **{p['y']}** · [{p['t']}]({url}){venue}{code}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def update_readme(data, path="README.md"):
    if not os.path.exists(path):
        return
    s = open(path, encoding="utf-8").read()
    a, b = "<!--CLUSTERS:START-->", "<!--CLUSTERS:END-->"
    if a not in s or b not in s:
        return
    new = s[:s.index(a) + len(a)] + "\n" + readme_block(data) + "\n" + s[s.index(b):]
    if new != s:
        open(path, "w", encoding="utf-8").write(new)


def main():
    data = json.load(open(DATA_FILE, encoding="utf-8"))
    if "--no-fetch" not in sys.argv:
        update_from_orcid(data)
    data["pubs"].sort(key=lambda p: (-p["y"], p["t"]))
    json.dump(data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs("assets", exist_ok=True)
    draw_map(data, "assets/research-map-v3.svg")
    draw_evolution(data, "assets/research-evolution-v3.svg")
    update_readme(data)
    print(f"Drew research map with {len(data['pubs'])} publications.")


if __name__ == "__main__":
    main()
