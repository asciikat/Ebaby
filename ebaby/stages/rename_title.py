"""Final slug rename, moving every image into the renamed-output folder —
ported from ebay_Api_rename_new.py's plan_renames/apply_renames.

Unlike the original script (which re-reads barcode .txt files from disk),
this stage takes the already-known {set_key: barcode} and {set_key: title}
maps as arguments, since the barcode and eBay stages already ran earlier in
this batch. This is the LAST rename before cropping — the exact ordering the
app must guarantee.

Fallback ladder per set (all sets get moved, nothing is left behind):
  eBay match found     -> named after the eBay title slug
  barcode, no match    -> named after the barcode digits
  no barcode at all    -> named "NoBarcode_<key>"
Brand-new (number-keyed) sets get "_new" appended to the role.
"""
from ebaby.naming_utils import slugify_title


def plan_renames(sets, key_to_barcode, key_to_title, dest_dir):
    """sets: {set_key: [(role, path), ...]}
    key_to_barcode: {set_key: barcode_str_or_None}
    key_to_title: {set_key: ebay_title_str_or_empty}
    Returns (renames, notes):
      renames = [(src_path, dest_path), ...]
      notes   = [(set_key, human_reason), ...] for barcode/eBay misses.
    """
    renames = []
    notes = []
    used = set()
    for key, members in sets.items():
        is_new_stock = key.isdigit()
        barcode = key_to_barcode.get(key)

        if barcode:
            slug = slugify_title(key_to_title.get(key, ""), fallback="")
            if not slug:
                slug = barcode
                notes.append((key, f"no eBay match — named by barcode {barcode}"))
        else:
            slug = f"NoBarcode_{key}"
            notes.append((key, "no barcode found — moved with placeholder name, needs manual ID"))

        base, n = slug, 2
        while slug in used:
            slug = f"{base}_{n}"
            n += 1
        used.add(slug)

        for role, path in members:
            tagged_role = f"{role.lower()}_new" if is_new_stock else role.lower()
            dest = dest_dir / f"{slug}_{tagged_role}{path.suffix}"
            if dest != path:
                renames.append((path, dest))
    return renames, notes


def apply_renames(renames):
    """Moves files, skipping (not clobbering) any target that already
    exists. Returns the count actually moved."""
    dest_dirs = {dst.parent for _, dst in renames}
    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)
    done = 0
    for src, dst in renames:
        if dst.exists():
            continue
        src.rename(dst)
        done += 1
    return done
