# tools/dump_pdf_lines.py
import sys
import pdfplumber

def dump(path, pages=3, maxlines=200):
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages[:pages]):
            print(f"\n--- Page {i+1} ---")
            txt = page.extract_text()
            if not txt:
                print("<no text>")
                continue
            lines = txt.splitlines()
            for j, L in enumerate(lines[:maxlines]):
                print(f"{j+1:03d}: {L}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/dump_pdf_lines.py <pdf_path>")
    else:
        dump(sys.argv[1], pages=5)
