import spacy
import inflect


def vp_present_participle(phrase):
    nlp = spacy.load("en")
    doc = nlp(phrase)
    return ' '.join([
        inflection_engine.present_participle(token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
        for token in doc
    ])

def posessive(word):
    inflection_engine = inflect.engine()
    if inflection_engine.singular_noun(word) is False:
        return "have"
    else:
        return "has"

def article(word):
    return "an" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "a"

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start if start != -1 else None