"""QR code for the poster, linking to the interactive calculator.

Usage (from the repo root, with the venv active):
    python -m src.make_qr

Writes poster/qr_calculator.svg (vector, for the poster file) and poster/qr_calculator.png (print resolution).
Error correction level H keeps it scannable if the poster gets scuffed or a logo is placed over the center.
"""

from pathlib import Path

import segno

URL = "https://trentrice.github.io/pwhl-breakeven-breakaway/"
OUT = Path(__file__).resolve().parents[1] / "poster"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    qr = segno.make(URL, error="h", micro=False)
    qr.save(OUT / "qr_calculator.svg", scale=10, border=4, dark="#000000", light="#ffffff")
    # 2.5 inches wide at 600 dpi: plenty for a poster QR scanned from a few feet away
    side_modules = qr.symbol_size(border=4)[0]
    scale = max(1, round(2.5 * 600 / side_modules))
    qr.save(OUT / "qr_calculator.png", scale=scale, border=4, dark="#000000", light="#ffffff", dpi=600)
    print(f"version {qr.version}, error {qr.error}, {side_modules} modules; wrote {OUT}/qr_calculator.svg and .png")


if __name__ == "__main__":
    main()
