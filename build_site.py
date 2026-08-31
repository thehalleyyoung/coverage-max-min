"""Render index.html from the joint paper, with a results-first landing block."""
import base64, pathlib, re, subprocess

HERE = pathlib.Path(__file__).resolve().parent
LAND = """
# Coverage and Max-Min Diversity in Synthetic Data Generation

**One method for two objectives, and the conditional dimension that limits both**

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
