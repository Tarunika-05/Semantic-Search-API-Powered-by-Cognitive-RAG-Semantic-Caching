from app.dataset import _clean_text, _compute_fingerprint

def test_clean_text_removes_urls():
    text = "Check this out https://example.com/foo and this http://bar.com"
    cleaned = _clean_text(text)
    assert "http" not in cleaned
    assert "Check this out" in cleaned

def test_clean_text_removes_emails():
    text = "Contact me at user@example.com for more info."
    cleaned = _clean_text(text)
    assert "@" not in cleaned
    assert "Contact me at" in cleaned

def test_clean_text_removes_file_paths():
    text = "The file is located at /usr/local/bin/python or C:\\Windows\\System32"
    cleaned = _clean_text(text)
    assert "usr" not in cleaned
    assert "Windows" not in cleaned

def test_clean_text_normalizes_whitespace():
    text = "This   has \n\n too much \t whitespace."
    cleaned = _clean_text(text)
    assert "This has \n\n too much whitespace." == cleaned

def test_compute_fingerprint_deterministic():
    text1 = "This is a test document."
    text2 = "This is a test document."
    assert _compute_fingerprint(text1) == _compute_fingerprint(text2)

def test_different_docs_different_fingerprint():
    text1 = "This is a test document."
    text2 = "This is a different document."
    assert _compute_fingerprint(text1) != _compute_fingerprint(text2)

def test_near_duplicates_same_fingerprint():
    # Only alphanumeric characters and lowercase matters for fingerprint
    text1 = "This is a test document!"
    text2 = "this IS a TEST document..."
    assert _compute_fingerprint(text1) == _compute_fingerprint(text2)
