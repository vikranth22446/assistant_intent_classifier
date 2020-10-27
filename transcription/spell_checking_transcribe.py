from spellchecker import SpellChecker

spell = SpellChecker()
print(len(spell._word_frequency._dictionary.keys()))
# spell.load_words()
# # find those words that may be misspelled
# misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))
# updated_lines = []
# with open('test.txt') as f:
#     for line in f.readlines():
#         words = []
#         misspelled = spell.unknown(line.split(" "))
#         for word in misspelled:
#             corrected_word = spell.correction(word)
#             print(word, corrected_word)
#             # words.append(corrected_word)
#         # updated_lines.append(" ".join(words))
