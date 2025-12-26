import os
import glob

LABEL_DIR = r"D:\projects\zeex-fire-detect\data\processed\fire_smoke_yolo\labels\train"

def clamp(value):
    """Ensure values are between 0 and 1."""
    try:
        v = float(value)
        if v < 0:
            return "0.0"
        if v > 1:
            return "0.999999"
        return str(v)
    except:
        return None

def fix_label_file(path):
    fixed_lines = []
    changed = False

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                # invalid label format → skip
                continue

            cls, x, y, w, h = parts
            x2 = clamp(x)
            y2 = clamp(y)
            w2 = clamp(w)
            h2 = clamp(h)

            if None in [x2, y2, w2, h2]:
                continue

            # detect changes
            if [x, y, w, h] != [x2, y2, w2, h2]:
                changed = True

            fixed_lines.append(f"{cls} {x2} {y2} {w2} {h2}")

    # write back only if needed
    if changed:
        backup = path + ".bak"
        os.rename(path, backup)

        with open(path, "w") as f:
            for l in fixed_lines:
                f.write(l + "\n")

        print(f"Fixed: {path} (backup saved as .bak)")
    else:
        print(f"OK: {path}")


def main():
    txt_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    print(f"Found {len(txt_files)} label files.")

    for path in txt_files:
        fix_label_file(path)

    print("✔ All label files checked and fixed.")

if __name__ == "__main__":
    main()
