import re
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "EXPERT_BIBLE.md")
OUTPUT_FILE = os.path.join(BASE_DIR, "EXPERT_BIBLE.txt")

print("Reading EXPERT_BIBLE.md ...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()

# ─── The .md is stored as ONE long line with literal \n sequences ─────────────
# First replace the literal 4-char sequence  \  n  with a real newline
content = raw.replace('\\n', '\n')

# ─── STEP 1: Remove HTML comment padding lines only ──────────────────────────
# These are lines like: <!-- Extra Technical Context Padding Line N: ... -->
content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

# ─── STEP 2: Remove markdown heading markers (#  ##  ###) ────────────────────
content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)

# ─── STEP 3: Remove bold / italic markers ────────────────────────────────────
content = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', content, flags=re.DOTALL)
content = re.sub(r'\*\*(.+?)\*\*',     r'\1', content, flags=re.DOTALL)
content = re.sub(r'\*(.+?)\*',         r'\1', content, flags=re.DOTALL)
content = re.sub(r'___(.+?)___',       r'\1', content, flags=re.DOTALL)
content = re.sub(r'__(.+?)__',         r'\1', content, flags=re.DOTALL)
content = re.sub(r'_(.+?)_',           r'\1', content, flags=re.DOTALL)

# ─── STEP 4: Remove fenced code blocks ───────────────────────────────────────
content = re.sub(r'```[^\n]*\n(.*?)```', r'\1', content, flags=re.DOTALL)

# ─── STEP 5: Remove inline backtick formatting ───────────────────────────────
content = re.sub(r'`([^`]+)`', r'\1', content)

# ─── STEP 6: Remove markdown links [text](url) → just keep text ──────────────
content = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', content)

# ─── STEP 7: Replace markdown horizontal rules (---) with plain separator ────
SEPARATOR = '=' * 70
content = re.sub(r'^-{3,}\s*$', SEPARATOR, content, flags=re.MULTILINE)

# ─── STEP 8: Remove blockquote markers (> ...) ───────────────────────────────
content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)

# ─── STEP 9: Strip emoji / decorative unicode symbols ────────────────────────
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"   # emoticons
    u"\U0001F300-\U0001F5FF"   # symbols & pictographs
    u"\U0001F680-\U0001F6FF"   # transport & map
    u"\U0001F1E0-\U0001F1FF"   # flags
    u"\U00002702-\U000027B0"   # dingbats
    u"\U000024C2-\U0001F251"   # enclosed
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE
)
content = emoji_pattern.sub('', content)

# ─── STEP 10: Normalize list bullets to readable format ──────────────────────
content = re.sub(r'^([ \t]*)[-*]\s+', r'\1  * ', content, flags=re.MULTILINE)

# ─── STEP 11: Collapse 3+ blank lines → 2 blank lines max ────────────────────
content = re.sub(r'\n{4,}', '\n\n\n', content)

# ─── STEP 12: Strip leading/trailing whitespace ───────────────────────────────
content = content.strip()

# ─── STEP 13: Prepend a clean ASCII header ───────────────────────────────────
border = '=' * 70
header = (
    border + '\n'
    + '  EXPERT BIBLE: MULTIVARIATE DEMAND FORECASTING\n'
    + '  Master Technical Manual - Plain Text Edition\n'
    + border + '\n\n'
)
content = header + content

# ─── Write output ─────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

lines   = content.split('\n')
size_kb = os.path.getsize(OUTPUT_FILE) / 1024

print(f"Done!")
print(f"Output file  : {OUTPUT_FILE}")
print(f"Total lines  : {len(lines)}")
print(f"Size         : {size_kb:.1f} KB")
print()
print("--- PREVIEW (first 80 lines) ---")
for line in lines[:80]:
    print(line)
