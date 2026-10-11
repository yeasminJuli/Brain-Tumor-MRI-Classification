import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
text = Path(r"C:\user\juli\thesis\original_file\all_codes\manuscript_extracted.txt").read_text(encoding="utf-8")
split = re.search(r"\nREFERENCES\s*\n", text, flags=re.I)
if split:
    body, refs = text[:split.start()], text[split.end():]
else:
    first = text.find("[1]")
    body, refs = text[:first], text[first:]

for n in range(1, 63):
    count = len(re.findall(rf"\[{n}\]", body))
    print(f"[{n}] citations={count}")
