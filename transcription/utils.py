from deepsegment import DeepSegment  # ML based segment detection
from spellchecker import SpellChecker


def split_phrase_into_sentences(phrase):
    """
    Splits long phrases such as
    "I eat food I run outside to" => ["I eat food", "I run outside"]
    """
    segmenter = DeepSegment("en")
    return segmenter.segment(phrase)


def spell_check_phrase(phrase):
    spell_checker = SpellChecker()
    words = spell_checker.split_words(phrase)
    corrected_words = []
    for word in words:
        if word in spell_checker:
            corrected_words.append(word)
        else:
            corrected_words.append(spell_checker.correction(word))
    return " ".join(corrected_words)


if __name__ == "__main__":
    #print("Spell check: ", spell_check_phrase("this is a corrreeeeeected phrase"))
    #print("Phrase : ", split_phrase_into_sentences("I eat food I run outside to"))
    pass