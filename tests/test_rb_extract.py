"""Field-merge + anti-hallucination guards: the user's 'never guess' rule."""
from redboxflip import extract
from redboxflip.models import Face


def _back(**kw):
    base = {"upright": True, "all_text": "", "title": "", "low_confidence": []}
    base.update(kw)
    return base


def test_region_rejected_when_not_in_transcribed_text():
    # Model claims a region it did NOT actually transcribe -> must be dropped.
    per = {Face.BACK: _back(all_text="some synopsis text with no region marking",
                            region="Region 1 US/Canada")}
    scan = extract.merge_faces(per)
    assert not scan.region.known
    assert scan.region.needs_check


def test_region_accepted_when_present_in_text():
    per = {Face.BACK: _back(all_text="Disc two single sided REGION 4 PAL format",
                            region="Region 4")}
    scan = extract.merge_faces(per)
    assert scan.region.known
    assert "Region 4" in scan.region.value
    assert scan.region.confidence == "High"


def test_palntsc_only_from_literal_token():
    per = {Face.BACK: _back(all_text="this disc is PAL format region 4", pal_ntsc="PAL")}
    assert extract.merge_faces(per).pal_ntsc.value == "PAL"
    per2 = {Face.BACK: _back(all_text="no standard printed here", pal_ntsc="NTSC")}
    assert not extract.merge_faces(per2).pal_ntsc.known


def test_rating_must_be_valid_badge_and_in_text():
    per = {Face.BACK: _back(all_text="parental guidance PG recommended", rating="PG")}
    assert extract.merge_faces(per).rating.value == "PG"
    # 'M' for "movie" is not a real read of a rating not in the text
    per2 = {Face.BACK: _back(all_text="great film", rating="Drama")}
    assert not extract.merge_faces(per2).rating.known


def test_year_validated_to_four_digits_in_text():
    per = {Face.BACK: _back(all_text="copyright 2002 warner", release_year="2002")}
    assert extract.merge_faces(per).release_year.value == "2002"
    per2 = {Face.BACK: _back(all_text="no year here", release_year="2002")}
    assert not extract.merge_faces(per2).release_year.known


def test_barcode_never_comes_from_model():
    per = {Face.BACK: _back(all_text="9325336010945", barcode="9325336010945")}
    scan = extract.merge_faces(per)
    # merge_faces ignores barcode entirely; the pipeline sets it via a real decoder
    assert not scan.barcode.known
    assert scan.barcode.needs_check


def test_num_discs_word_to_number():
    per = {Face.FRONT: _back(all_text="two-disc special edition", num_discs="two")}
    assert extract.merge_faces(per).num_discs.value == "2"


def test_back_preferred_over_front_for_region():
    per = {
        Face.BACK: _back(all_text="region 4 pal", region="Region 4"),
        Face.FRONT: _back(all_text="region 2", region="Region 2"),
    }
    assert "4" in extract.merge_faces(per).region.value


def test_title_kept_even_if_not_in_transcription():
    # Stylised cover title may not appear in all_text; still keep it (Medium).
    per = {Face.FRONT: _back(all_text="director's cut special edition",
                             title="Amadeus")}
    scan = extract.merge_faces(per)
    assert scan.title.value == "Amadeus"
    assert scan.title.confidence in ("Medium", "High")


def test_suggested_title_assembles_known_fields():
    per = {Face.FRONT: _back(all_text="region 4 pal 2002", title="Amadeus"),
           Face.BACK: _back(all_text="region 4 pal 2002",
                            region="Region 4", pal_ntsc="PAL", release_year="2002")}
    scan = extract.merge_faces(per)
    assert scan.suggested_title.startswith("Amadeus")
    assert "Region 4" in scan.suggested_title
    assert "PAL" in scan.suggested_title


def test_suggested_title_capped_at_80_chars():
    per = {Face.FRONT: _back(
        all_text="x", title="A Very Long Movie Title That Goes On And On And On Indeed",
        edition="Ultimate Collector's Special Limited Numbered Edition Boxset")}
    assert len(extract.merge_faces(per).suggested_title) <= 80


def test_format_defaults_to_dvd_with_check_flag():
    per = {Face.BACK: _back(all_text="no format word visible")}
    fmt = extract.merge_faces(per).format
    assert fmt.value == "DVD"
    assert fmt.needs_check          # defaulted, so flag it


# --- mining structured fields out of the transcription --------------------- #

def test_mine_num_discs_from_transcription():
    # Model left num_discs blank but the text clearly says "2 DISC".
    per = {Face.INSIDE: _back(all_text="THE LIFE OF DAVID GALE DVD 2 DISC REGION 4")}
    assert extract.merge_faces(per).num_discs.value == "2"


def test_mine_subtitles_and_special_features():
    txt = ("SPECIAL FEATURES: Interactive Menus Scene Selection Deleted Scenes\n"
           "Subtitles: English, Chinese, Bahasa and Thai")
    scan = extract.merge_faces({Face.BACK: _back(all_text=txt)})
    assert "English" in scan.subtitles.value
    assert "Interactive Menus" in scan.special_features.value


def test_mine_region_ignores_letter_region_codes():
    # "REGION A" is a Blu-ray code; for a DVD that's a misread -> don't accept it.
    per = {Face.INSIDE: _back(all_text="DVD REGION A disc")}
    assert not extract.merge_faces(per).region.known


def test_mine_region_accepts_numeric():
    per = {Face.BACK: _back(all_text="this disc is region 4 only")}
    assert "4" in extract.merge_faces(per).region.value


def test_mine_palntsc_from_text():
    per = {Face.BACK: _back(all_text="encoded for NTSC televisions")}
    assert extract.merge_faces(per).pal_ntsc.value == "NTSC"


def test_mine_rating_ignores_stray_letters_in_prose():
    # Prose full of single capital letters must NOT yield a bogus rating.
    txt = ("A TWIST IN THE TALE that will send your head spinning. R MARIE "
           "CLAIRE great story G")
    assert not extract.merge_faces({Face.BACK: _back(all_text=txt)}).rating.known


def test_mine_rating_accepts_real_badge():
    txt = "PG PARENTAL GUIDANCE RECOMMENDED for persons under 15 years"
    assert extract.merge_faces({Face.FRONT: _back(all_text=txt)}).rating.value == "PG"


def test_special_features_stops_before_subtitles():
    txt = "SPECIAL FEATURES: Deleted Scenes Cast and Crew Subtitles: English Thai"
    sf = extract.merge_faces({Face.BACK: _back(all_text=txt)}).special_features.value
    assert "Deleted Scenes" in sf
    assert "Thai" not in sf


def test_merge_faces_survives_malformed_model_output():
    # Small models return odd shapes: null low_confidence, numeric/None fields,
    # a list where text was expected. None of it should crash the merge.
    bad = {
        Face.BACK: {"upright": True, "all_text": ["region 4", "pal"],
                    "title": 12345, "num_discs": 2, "region": None,
                    "low_confidence": None, "rating": ["PG"]},
        Face.FRONT: None,
    }
    scan = extract.merge_faces(bad)          # must not raise
    assert scan.to_dict()["title"]["value"] is not None
