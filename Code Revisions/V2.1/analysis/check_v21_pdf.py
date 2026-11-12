"""Check final V2.1 TeX/PDF and create contact sheets for visual review."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'audit/v21_final'
log = (ROOT / 'output/pdf/Thesis_Revised.log').read_text(errors='replace')
biber = (ROOT / 'output/pdf/Thesis_Revised.blg').read_text(errors='replace')
warnings = re.findall(r'^.*(?:Overfull|Underfull|LaTeX Warning|Package .* Warning|undefined references).*$', log, re.M)
assert not warnings, warnings
assert not re.search(r'\b(?:WARN|ERROR)\b', biber)
sources = list((ROOT / 'manuscript').glob('*.tex')) + list((ROOT / 'manuscript/generated').glob('*.tex'))
text = '\n'.join(p.read_text(encoding='utf-8') for p in sources)
keys = set()
for match in re.finditer(r'\\(?:auto|text|foot|paren)?cite\w*(?:\[[^\]]*\])*\{([^}]+)\}', text):
    keys.update(key.strip() for key in match.group(1).split(','))
entries = set(re.findall(r'@\w+\{([^,]+),', (ROOT / 'manuscript/bibliography.bib').read_text(encoding='utf-8')))
assert keys == entries, {'uncited': sorted(entries-keys), 'missing': sorted(keys-entries)}
labels = re.findall(r'\\label\{([^}]+)\}', text)
refs = set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', text))
assert len(labels) == len(set(labels)), 'Duplicate labels'
assert refs <= set(labels), sorted(refs-set(labels))
assert not re.search(r'(?i)\b(?:TODO|FIXME|TBD|INSERT HERE)\b', text)
assert '\\tableofcontents' not in text
pages = (HERE / 'manuscript.txt').read_text(encoding='utf-8').split('\f')
if not pages[-1].strip(): pages.pop()
assert pages and all(p.strip() for p in pages)
assert not any('??' in p for p in pages)
namespace = {'x': 'http://www.w3.org/1999/xhtml'}
boxes = ET.parse(HERE / 'manuscript_bbox.html').findall('.//x:page', namespace)
assert len(boxes) == len(pages)
for number, page in enumerate(boxes, 1):
    width, height = float(page.attrib['width']), float(page.attrib['height'])
    for word in page.findall('.//x:word', namespace):
        b = {k: float(word.attrib[k]) for k in ('xMin','yMin','xMax','yMax')}
        assert 0 <= b['xMin'] <= b['xMax'] <= width, (number, word.text, b)
        assert 0 <= b['yMin'] <= b['yMax'] <= height, (number, word.text, b)
rendered = sorted((HERE / 'render').glob('page-*.png'))
assert len(rendered) == len(pages)
sheets = []
for start in range(0,len(rendered),9):
    sheet = Image.new('RGB',(1350,1800),'#dddddd')
    draw = ImageDraw.Draw(sheet)
    for offset,path in enumerate(rendered[start:start+9]):
        page = Image.open(path).convert('RGB'); page.thumbnail((436,560))
        col,row=offset%3,offset//3
        sheet.paste(page,(col*450+(450-page.width)//2,row*600+27))
        draw.text((col*450+12,row*600+8),f'PDF page {start+offset+1}',fill='black')
    out=HERE/f'contact_{start//9+1}.jpg'; sheet.save(out,quality=92); sheets.append(out.name)
inventory=[]
for n,page in enumerate(pages,1):
    lines=[line.strip() for line in page.splitlines() if line.strip()]
    inventory.append({'pdf_page':n,'start':' | '.join(lines[:3]),'end':' | '.join(lines[-3:])})
(HERE/'page_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
report={'pdf_pages':len(pages),'warnings':warnings,'bibliography_entries':len(entries),
        'cited_keys':len(keys),'resolved_labels':len(labels),'resolved_references':len(refs),
        'all_text_bounds_inside_pages':True,'blank_pages':0,'contact_sheets':sheets,
        'visual_review':'Pending separate image inspection'}
(HERE/'pdf_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
