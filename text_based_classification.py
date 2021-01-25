import tempfile
from transcription.process_pipeline import DeepspeechTranscription, process_pipeline_entry
from skills.general_classifier.general_classifier import GeneralClassifierSkill
from skills.shopping_classifier.shopping_skill import ShoppingSkill

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # cleanup error message items

skills = [GeneralClassifierSkill(), ShoppingSkill()]

def classify_text(skills, text):
    for skill in skills:
        print("Running classification for skill: ", skill.skill_name)
        print(skill.classify(text))
        if isinstance(skill, ShoppingSkill):
            print("Possible shopping item", skill.find_shopping_item(text))

def classify_file(skills, file):
    with open(file) as f:
        for text in f.readlines():
            if text:
                classify_text(skills, text)

if __name__ == "__main__":
    text = "We ran out of watermelons"
    classify_text(skills, text)
    with tempfile.TemporaryDirectory() as dirpath:
        transcript_file = process_pipeline_entry(
            "save_audio/savewav_2020-09-09_19-04-59_750112.wav",
            dirpath,
            DeepspeechTranscription(),
        )
        classify_file(skills, transcript_file)
    
    # general_classifier = GeneralClassifierSkill()
    # print(general_classifier.classify("Weather is nice outside"))

