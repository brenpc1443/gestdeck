"""
slide_reader.py — Lectura básica de archivos .pptx.

En modo virtual solo nos interesa extraer cada slide como imagen de fondo
para mostrarla en el canvas del frontend. No renderizamos animaciones ni
elementos interactivos del PowerPoint original: el .pptx vive como "el
fondo visual" y GestDeck dibuja los objetos interactivos encima.

Estrategias de conversión (en orden de preferencia):
    1. LibreOffice (soffice) — más fiel, disponible en Linux/macOS/Windows
    2. PyMuPDF (si ya existe un PDF del .pptx)
    3. python-pptx + PIL — genera un render estilizado con títulos, viñetas
       y paleta oscura (no es pixel-perfect pero se ve bien en la demo)

Uso desde CLI (compatible con IPC de Electron):
    python -m core.slide_reader <archivo.pptx> <carpeta_salida>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 16:9 aproximado a 1920×1080 (fidelidad alta sin ser pesado)
SLIDE_W, SLIDE_H = 1920, 1080


def convert_pptx_to_images(
    pptx_path: str | Path,
    output_dir: str | Path,
    dpi: int = 150,
    fmt: str = "png",
) -> List[Path]:
    """Convierte un .pptx a una lista de imágenes PNG (una por slide)."""
    pptx_path = Path(pptx_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar salidas previas para evitar mezclar slides de conversiones
    for old in output_dir.glob(f"*.{fmt}"):
        try:
            old.unlink()
        except OSError:
            pass

    # --- Estrategia 1: LibreOffice ---------------------------------------
    soffice = _find_soffice()
    if soffice is not None:
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf",
                 "--outdir", str(output_dir), str(pptx_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            pdf = output_dir / (pptx_path.stem + ".pdf")
            if pdf.exists():
                imgs = _pdf_to_pngs(pdf, output_dir, dpi=dpi, fmt=fmt)
                if imgs:
                    # borrar el PDF intermedio
                    try:
                        pdf.unlink()
                    except OSError:
                        pass
                    return imgs
        except subprocess.CalledProcessError:
            pass
        except subprocess.TimeoutExpired:
            pass

    # --- Estrategia 2: fallback python-pptx + PIL ------------------------
    if not HAS_PPTX or not HAS_PIL:
        missing = []
        if not HAS_PPTX:
            missing.append("python-pptx")
        if not HAS_PIL:
            missing.append("Pillow")
        raise RuntimeError(
            "No se pudo convertir el .pptx.\n"
            "Opción A (recomendado): instala LibreOffice — https://es.libreoffice.org/\n"
            f"Opción B: pip install {' '.join(missing)}  "
            "(render simplificado, sin layouts originales)."
        )
    return _fallback_pptx_to_images(pptx_path, output_dir, fmt=fmt)


def list_slides(pptx_path: str | Path) -> List[dict]:
    """Devuelve metadatos básicos por slide: título, número de shapes."""
    if not HAS_PPTX:
        return []
    prs = Presentation(str(pptx_path))
    out = []
    for i, slide in enumerate(prs.slides, start=1):
        titulo = ""
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text:
                titulo = shape.text_frame.text.split("\n")[0]
                break
        out.append({
            "slide_id": i,
            "titulo": titulo,
            "num_shapes": len(slide.shapes),
        })
    return out


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------
def _find_soffice():
    """Busca el ejecutable de LibreOffice/soffice."""
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    # Rutas típicas en Windows
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _pdf_to_pngs(pdf: Path, output_dir: Path, dpi: int, fmt: str) -> List[Path]:
    """Convierte PDF a PNG usando pdftoppm (poppler) o PyMuPDF."""
    pdftoppm = shutil.which("pdftoppm")
    prefix = output_dir / "slide"
    if pdftoppm is not None:
        subprocess.run(
            [pdftoppm, f"-{fmt}", "-r", str(dpi), str(pdf), str(prefix)],
            check=True,
        )
        return sorted(output_dir.glob(f"slide-*.{fmt}"))

    # Fallback: PyMuPDF
    try:
        import fitz   # PyMuPDF
    except ImportError:
        return []
    doc = fitz.open(str(pdf))
    outs = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out = output_dir / f"slide-{i:03d}.{fmt}"
        pix.save(str(out))
        outs.append(out)
    return outs


# ----------------------------------------------------------------------
# Fallback visual — sin LibreOffice, sin PDF.
# Generamos imágenes estilizadas por slide basándonos en python-pptx.
# ----------------------------------------------------------------------
def _fallback_pptx_to_images(pptx_path: Path, output_dir: Path, fmt: str = "png") -> List[Path]:
    prs = Presentation(str(pptx_path))
    outs: List[Path] = []
    for i, slide in enumerate(prs.slides, start=1):
        title, bullets = _extract_slide_text(slide)
        img = _render_slide_png(i, title, bullets)
        out = output_dir / f"slide-{i:03d}.{fmt}"
        img.save(out, fmt.upper() if fmt.lower() == "png" else fmt.upper())
        outs.append(out)
    return outs


def _extract_slide_text(slide) -> tuple:
    title = ""
    bullets: List[str] = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if not txt:
            continue
        if not title:
            # Primer shape con texto asumido como título
            parts = txt.split("\n", 1)
            title = parts[0].strip()
            if len(parts) > 1:
                bullets.extend([p.strip() for p in parts[1].split("\n") if p.strip()])
        else:
            for line in txt.split("\n"):
                line = line.strip()
                if line:
                    bullets.append(line)
    return title, bullets


def _load_font(size: int):
    """Intenta cargar una fuente decente; cae a default si falla."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_slide_png(slide_num: int, title: str, bullets: List[str]) -> "Image.Image":
    # Fondo degradado oscuro (el mismo esquema que la UI)
    img = Image.new("RGB", (SLIDE_W, SLIDE_H), (11, 15, 26))
    draw = ImageDraw.Draw(img)

    # barra decorativa superior
    for y in range(18):
        alpha = int(180 * (1 - y / 18))
        draw.line([(0, y), (SLIDE_W, y)], fill=(57, 160, 255, alpha))

    # número de slide
    num_font = _load_font(42)
    draw.text((SLIDE_W - 130, 50), f"#{slide_num:02d}",
              fill=(120, 140, 190), font=num_font)

    # título
    title_font = _load_font(84)
    if title:
        title = title.strip()[:80]
        draw.text((110, 120), title, fill=(240, 245, 255), font=title_font)
        # subrayado sutil
        try:
            bbox = draw.textbbox((110, 120), title, font=title_font)
            underline_y = bbox[3] + 8
            draw.rectangle([(110, underline_y), (110 + min(800, bbox[2] - bbox[0]),
                             underline_y + 4)], fill=(57, 160, 255))
        except AttributeError:
            pass

    # bullets
    bullet_font = _load_font(44)
    y = 320
    for b in bullets[:8]:
        lines = textwrap.wrap(b, width=70) or [b]
        for j, line in enumerate(lines):
            prefix = "•  " if j == 0 else "   "
            draw.text((140, y), prefix + line, fill=(210, 220, 240), font=bullet_font)
            y += 64
            if y > SLIDE_H - 120:
                break
        if y > SLIDE_H - 120:
            break

    # pie
    foot_font = _load_font(26)
    draw.text((110, SLIDE_H - 70), f"GestDeck · Modo Virtual · Slide {slide_num}",
              fill=(120, 135, 170), font=foot_font)

    return img


# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Uso: python -m core.slide_reader <archivo.pptx> [carpeta_salida]")
        sys.exit(2)
    pptx = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./sessions/_preview"
    imgs = convert_pptx_to_images(pptx, out_dir)
    for p in imgs:
        print(p)


if __name__ == "__main__":
    main()
