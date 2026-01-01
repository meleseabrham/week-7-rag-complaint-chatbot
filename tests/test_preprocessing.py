from src.preprocessing import clean_text

def test_clean_text_lowercasing():
    assert clean_text("HELLO") == "hello"

def test_clean_text_boilerplate():
    assert "writing to file a complaint" not in clean_text("I am writing to file a complaint about my card.")

def test_clean_text_special_chars():
    assert clean_text("Hello@#$% World") == "hello world"

def test_clean_text_whitespace():
    assert clean_text("  hello    world  ") == "hello world"
