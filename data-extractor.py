import re
import csv
import os
from datetime import datetime
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_FILE = "whatsapp_chat.csv"
SCRIPT_DIR  = Path(__file__).parent.resolve()

# Add any known long contact names you want to shorten
SENDER_MAP = {
    "Zain Aziz Chaudhary Talha's Contact Does Fyp Projects And Offloads Them": "Zain Aziz",
}

# ── REGEX ────────────────────────────────────────────────────────────────────
# Standard format:  D/M/YY, HH:MM - Sender: Message
MSG_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})(?:\s?[AP]M)?\s-\s([^:]+?):\s(.+)$'
)
# System line (no sender):  D/M/YY, HH:MM - Some system text
SYS_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})(?:\s?[AP]M)?\s-\s(.+)$'
)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def parse_dt(date_s: str, time_s: str) -> datetime | None:
    for fmt in ("%d/%m/%y %H:%M", "%m/%d/%y %H:%M",
                "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt)
        except ValueError:
            continue
    return None

def classify(text: str) -> str:
    t = text.strip()
    if t == "<Media omitted>":                          return "media"
    if "deleted this message" in t or t == "You deleted this message": return "deleted"
    if "<This message was edited>" in t:                return "text"   # edited text
    if re.match(r'https?://', t):                       return "link"
    return "text"

def strip_edited(text: str) -> tuple[str, bool]:
    if "<This message was edited>" in text:
        return text.replace("<This message was edited>", "").strip(), True
    return text.strip(), False

def clean(text: str) -> str:
    """Strip invisible Unicode control characters WhatsApp injects."""
    return re.sub(r'[\u200e\u200f\u202a-\u202e\ufeff]', '', text).strip()

def normalise(name: str) -> str:
    return SENDER_MAP.get(clean(name), clean(name))

def derive_chat_name(filepath: Path) -> str:
    """Extract meaningful chat name from the filename."""
    name = filepath.stem  # remove .txt
    # WhatsApp exports: "WhatsApp Chat with XYZ" or "WhatsApp-Chat-mit-XYZ" etc.
    for prefix in ["WhatsApp Chat with ", "WhatsApp Chat - "]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

# ── PARSER ───────────────────────────────────────────────────────────────────
def parse_file(filepath: Path, start_id: int = 1) -> list[dict]:
    records   = []
    msg_id    = start_id
    chat_name = derive_chat_name(filepath)
    current: dict | None = None

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    def flush():
        if current:
            records.append(current)

    for raw in lines:
        line = raw.rstrip("\n")

        # ── Message line ───────────────────────────────────────────────────
        m = MSG_RE.match(line)
        if m:
            flush()
            date_s, time_s, sender_raw, body = m.groups()
            dt             = parse_dt(date_s, time_s)
            sender         = normalise(sender_raw)
            body, edited   = strip_edited(clean(body))
            ctype          = classify(body)

            current = {
                "message_id"   : msg_id,
                "source_file"  : filepath.name,
                "chat_name"    : chat_name,
                "date"         : dt.strftime("%Y-%m-%d") if dt else date_s,
                "time"         : dt.strftime("%H:%M")    if dt else time_s,
                "datetime"     : dt.isoformat()          if dt else f"{date_s} {time_s}",
                "day_of_week"  : dt.strftime("%A")       if dt else "",
                "sender"       : sender,
                "message"      : body,
                "content_type" : ctype,
                "is_edited"    : edited,
                "is_system"    : False,
                "word_count"   : len(body.split()) if ctype == "text" else 0,
                "char_count"   : len(body)         if ctype == "text" else 0,
            }
            msg_id += 1
            continue

        # ── System / meta line ─────────────────────────────────────────────
        s = SYS_RE.match(line)
        if s:
            flush()
            date_s, time_s, sys_text = s.groups()
            dt = parse_dt(date_s, time_s)

            current = {
                "message_id"   : msg_id,
                "source_file"  : filepath.name,
                "chat_name"    : chat_name,
                "date"         : dt.strftime("%Y-%m-%d") if dt else date_s,
                "time"         : dt.strftime("%H:%M")    if dt else time_s,
                "datetime"     : dt.isoformat()          if dt else f"{date_s} {time_s}",
                "day_of_week"  : dt.strftime("%A")       if dt else "",
                "sender"       : "SYSTEM",
                "message"      : sys_text.strip(),
                "content_type" : "system",
                "is_edited"    : False,
                "is_system"    : True,
                "word_count"   : 0,
                "char_count"   : 0,
            }
            msg_id += 1
            continue

        # ── Continuation of previous message ───────────────────────────────
        if current and line.strip():
            current["message"] += " " + line.strip()
            if current["content_type"] == "text":
                current["word_count"] = len(current["message"].split())
                current["char_count"] = len(current["message"])

    flush()
    return records

# ── DEDUP ────────────────────────────────────────────────────────────────────
def load_existing_keys(filepath: Path) -> set[tuple]:
    """Return a set of (source_file, datetime, sender, message) tuples already in CSV."""
    keys = set()
    if not filepath.exists():
        return keys
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((
                row.get("source_file", ""),
                row.get("datetime",    ""),
                row.get("sender",      ""),
                row.get("message",     ""),
            ))
    return keys

def get_max_id(filepath: Path) -> int:
    """Return the highest message_id already in the CSV (0 if none)."""
    max_id = 0
    if not filepath.exists():
        return max_id
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row.get("message_id", 0)))
            except ValueError:
                pass
    return max_id

# ── WRITER ───────────────────────────────────────────────────────────────────
FIELDS = [
    "message_id", "source_file", "chat_name",
    "date", "time", "datetime", "day_of_week",
    "sender", "message", "content_type",
    "is_edited", "is_system", "word_count", "char_count",
]

def write_csv(records: list[dict], filepath: Path, append: bool) -> None:
    mode = "a" if append else "w"
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not append:
            f.write("\ufeff")   # UTF-8 BOM → Excel renders emojis & Urdu correctly
            writer.writeheader()
        writer.writerows(records)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
def print_summary(all_records: list[dict], files_processed: list[Path],
                  new_count: int, skipped: int) -> None:
    senders = {}
    chats   = {}
    ctypes  = {}
    for r in all_records:
        if not r["is_system"]:
            senders[r["sender"]]    = senders.get(r["sender"], 0) + 1
        chats[r["chat_name"]]       = chats.get(r["chat_name"], 0) + 1
        ctypes[r["content_type"]]   = ctypes.get(r["content_type"], 0) + 1

    w = 54
    print(f"\n{'='*w}")
    print(f"  WhatsApp → CSV  |  Quality Report")
    print(f"{'='*w}")
    print(f"  TXT files found   : {len(files_processed)}")
    print(f"  New rows added    : {new_count}")
    print(f"  Duplicate rows    : {skipped} (skipped)")
    print(f"\n  Content breakdown:")
    for k, v in sorted(ctypes.items(), key=lambda x: -x[1]):
        print(f"    {k:<20} {v}")
    print(f"\n  Messages per chat:")
    for k, v in sorted(chats.items(), key=lambda x: -x[1]):
        print(f"    {k[:40]:<42} {v}")
    print(f"\n  Messages per sender (non-system):")
    for k, v in sorted(senders.items(), key=lambda x: -x[1]):
        print(f"    {k[:40]:<42} {v}")
    print(f"{'='*w}\n")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    out_path = SCRIPT_DIR / OUTPUT_FILE

    # Find all TXT files in Dataset folder
    dataset_dir = SCRIPT_DIR / "Dataset"
    if not dataset_dir.exists():
        print(f"❌  Dataset folder not found at {dataset_dir}")
        return

    txt_files = [
        p for p in dataset_dir.glob("*.txt")
        if p.name != OUTPUT_FILE
    ]

    if not txt_files:
        print(f"❌  No .txt files found in {dataset_dir}")
        return

    print(f"\n📂  Found {len(txt_files)} TXT file(s):")
    for f in txt_files:
        print(f"     • {f.name}")

    # Load what's already in the CSV to avoid duplicates
    existing_keys = load_existing_keys(out_path)
    next_id       = get_max_id(out_path) + 1
    is_append     = out_path.exists()

    all_new_records = []
    total_skipped   = 0

    for txt in sorted(txt_files):
        raw_records = parse_file(txt, start_id=next_id)

        new_records = []
        for r in raw_records:
            key = (r["source_file"], r["datetime"], r["sender"], r["message"])
            if key in existing_keys:
                total_skipped += 1
            else:
                existing_keys.add(key)
                new_records.append(r)

        next_id += len(new_records)
        all_new_records.extend(new_records)
        print(f"  ✔  {txt.name}: {len(new_records)} new rows ({len(raw_records) - len(new_records)} dupes skipped)")

    if all_new_records:
        write_csv(all_new_records, out_path, append=is_append)
        print(f"\n✅  Output: {out_path}")
    else:
        print("\n⚠️   No new rows to add — CSV is already up to date.")

    print_summary(all_new_records, txt_files, len(all_new_records), total_skipped)

if __name__ == "__main__":
    main()