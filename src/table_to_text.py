#!/usr/bin/env python3
import fitz                   # PyMuPDF
import pdfplumber
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
PDF_DIR      = Path("/home/daniel/repos/RAG_in_mechanical_processing/src/data/training")  # folder with PDFs
OUT_BASE_DIR = Path("output")  # base output folder for all PDFs

# ------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def filter_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trims the table: keeps rows from the 4th onward and columns up to the 3rd.
    """
    return df.iloc[3:, :3]


def extract_image_and_table(pdf_path: Path, out_dir: Path):
    """
    Z pliku PDF wyciąga obraz nr 2 z pierwszej strony
    i pierwszą tabelę z pierwszej strony.
    Zapisuje je w out_dir pod tą samą nazwą bazową,
    różniącą się jedynie rozszerzeniem.
    """
    stem = pdf_path.stem
    ensure_dir(out_dir)

    # --- Obraz #2 ze strony 1 ---
    doc = fitz.open(pdf_path)
    page_index = 0
    images = doc.get_page_images(page_index, full=True)
    if len(images) >= 2:
        xref = images[1][0]  # indeks 1 -> drugie zdjęcie
        img_info = doc.extract_image(xref)
        ext = img_info.get("ext", "png")
        img_bytes = img_info.get("image")
        img_path = out_dir / f"{stem}.{ext}"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"Zapisano obraz: {img_path}")
    else:
        print(f"Brak drugiego obrazu na stronie 1 w: {pdf_path.name}")
    doc.close()

    # --- Pierwsza tabela ze strony 1 ---
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        tables = page.extract_tables()
        if tables and len(tables) >= 1:
            raw = tables[0]
            df = pd.DataFrame(raw[1:], columns=raw[0])
            df = filter_table(df)
            csv_path = out_dir / f"{stem}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Zapisano tabelę: {csv_path}")
        else:
            print(f"Brak tabeli nr 1 na stronie 1 w: {pdf_path.name}")


def process_all_pdfs(pdf_dir: Path, out_base_dir: Path):
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Brak plików PDF w folderze: {pdf_dir}")
        return

    for pdf_path in pdf_files:
        dest_dir = ensure_dir(out_base_dir / pdf_path.stem)
        print(f"Przetwarzam {pdf_path.name}...")
        extract_image_and_table(pdf_path, dest_dir)
        print()


def main():
    ensure_dir(OUT_BASE_DIR)
    process_all_pdfs(PDF_DIR, OUT_BASE_DIR)


if __name__ == "__main__":
    main()
