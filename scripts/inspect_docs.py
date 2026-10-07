import sys
import pypdf
import zipfile
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

def extract_pdf_info(filename):
    print(f"\n=================== PDF: {filename} ===================")
    reader = pypdf.PdfReader(filename)
    num_pages = len(reader.pages)
    print(f"Total Pages: {num_pages}")
    first_pages_text = ""
    for i in range(min(3, num_pages)):
        first_pages_text += f"\n--- Page {i+1} ---\n" + (reader.pages[i].extract_text() or "")
    print(first_pages_text[:2000])

def extract_pptx_info(filename):
    print(f"\n=================== PPTX: {filename} ===================")
    with zipfile.ZipFile(filename, 'r') as z:
        slide_files = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
        print(f"Total Slides: {len(slide_files)}")
        # Sort slides numerically
        slide_files.sort(key=lambda s: int(''.join(filter(str.isdigit, s)) or 0))
        for sf in slide_files[:5]:
            xml_content = z.read(sf)
            tree = ET.fromstring(xml_content)
            texts = [node.text for node in tree.iter() if node.text]
            slide_text = " ".join(texts).strip()
            print(f"--- {sf} ---: {slide_text[:300]}")

for pdf in ['research1.pdf', 'research2.pdf', 'basepaper.pdf']:
    try:
        extract_pdf_info(pdf)
    except Exception as e:
        print(f"Error reading {pdf}: {e}")

for ppt in ['leukemia ppt.pptx', 'leukemia1 (1).pptx']:
    try:
        extract_pptx_info(ppt)
    except Exception as e:
        print(f"Error reading {ppt}: {e}")
