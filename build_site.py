"""Render index.html from the joint paper, with a results-first landing block."""
import base64, pathlib, re, subprocess

HERE = pathlib.Path(__file__).resolve().parent
LAND = """
# Coverage and max-min diversity in synthetic data generation

Both halves of this paper are built on one method, **Recursive Axis Conditioning**
(RAC): the generator is asked to name the axes along which its own outputs can
differ, those axes are ranked and their most-different values chosen, and an axis
that runs out of new directions to offer is split into finer sub-axes that apply
only inside the region that exhausted it. Section II.4 gives the scoring rule and
Section I.5 the stack.

**Coverage.** Scale-free coverage of a held-out human-written reference, twelve
corpora at matched evaluated *n* = 450, radii taken from the reference alone:

| rank | corpus | coverage AUC |
|---|---|---|
| 1 | **RAC-coverage, retrieval-aimed (2.4k)** | **0.4441** |
| 2 | Alpaca (Self-Instruct, 52k) | 0.3722 |
| 3 | RAC-coverage, selective 1-of-8 (304) | 0.2591 |
| 4 | PersonaHub (50k) | 0.2532 |
| 5 | WizardLM Evol-Instruct (143k) | 0.2329 |
| 6–9 | budget-matched ablation arms (1.2k–2.4k) | 0.2102–0.2188 |
| 10–12 | unseeded arms (450 each) | 0.095–0.101 |

First place on one-twentieth of Alpaca's generation budget, with the highest
precision in the field (0.973).

**Max-min.** At matched *n* across seven arms and two domains, RAC wins every
literal and latent measure. On psychometric items it reaches 124.8 mean-centered
Vendi where persona conditioning reaches 99.7 and naive prompting 7.1, with a
0.000 exact-duplicate rate against naive's 0.668.

---
---
"""

def main():
    css = re.search(r"<style>(.*?)</style>", (HERE / "index.html").read_text(), re.S).group(1)
    tmp = HERE / ".site.md"
    tmp.write_text(LAND + (HERE / "paper" / "paper.md").read_text())
    body = subprocess.run(
        ["pandoc", "-f", "gfm+tex_math_dollars", "-t", "html5", "--mathml", str(tmp)],
        capture_output=True, text=True, check=True).stdout
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

    def embed(m):
        src = m.group(1)
        path = HERE / "paper" / src
        if not path.is_file():
            path = HERE / "figures" / pathlib.Path(src).name
        if not path.is_file():
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, "data:image/png;base64," + data)

    body = re.sub(r'<img[^>]*src="([^"]+)"', embed, body)
    banner = ('<div class="banner">Paper: <a href="paper/paper.pdf">PDF</a> &middot; '
              '<a href="paper/paper.tex">LaTeX</a> &middot; '
              '<a href="https://github.com/thehalleyyoung/coverage-max-min">code &amp; data</a></div>')
    (HERE / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Coverage and Max-Min Diversity in Synthetic Data Generation</title>\n'
        f'<style>{css}</style>\n</head>\n<body>\n<main>\n{banner}\n{body}\n</main>\n</body>\n</html>\n')
    tmp.unlink()
    n = (HERE / "index.html").read_text().count("data:image/png;base64,")
    print(f"index.html written ({n} figures embedded)")

if __name__ == "__main__":
    main()
