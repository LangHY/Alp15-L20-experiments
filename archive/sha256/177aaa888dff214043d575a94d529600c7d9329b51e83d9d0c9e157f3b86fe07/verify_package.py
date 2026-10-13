"""Verify extracted project files against the handoff manifest; no network."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
m=json.loads((root/'handover/PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
errors=[]
for r in m['files']:
 p=root/r['file']
 if not p.is_file():errors.append('Missing: '+r['file']);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if p.stat().st_size!=r['bytes'] or h.hexdigest()!=r['sha256']:errors.append('Changed: '+r['file'])
print('Checked',len(m['files']),'files')
if errors:
 print('\n'.join(errors));sys.exit(1)
print('All recorded files match. Manifest itself is protected by the external ZIP SHA256.')
