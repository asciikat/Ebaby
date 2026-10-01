from ebaby.naming_utils import slugify_title


def test_basic_title():
    # "the" is NOT in STOP_WORDS in the proven script, so it is kept verbatim.
    assert slugify_title("The Matrix", fallback="000") == "The_Matrix"


def test_strips_stop_words_and_limits_to_three_words():
    # "of" is a stop word and is dropped; "the" is not a stop word and stays.
    assert slugify_title("Lord of the Rings: The Fellowship", fallback="000") == "Lord_The_Rings"


def test_dedupes_repeated_words():
    assert slugify_title("Alien vs Alien Redux", fallback="000") == "Alien_Vs_Redux"


def test_empty_title_uses_fallback():
    assert slugify_title("", fallback="883316276402") == "883316276402"
    assert slugify_title(None, fallback="883316276402") == "883316276402"


def test_non_ascii_is_stripped():
    assert slugify_title("Amélie", fallback="000") == "Amelie"
