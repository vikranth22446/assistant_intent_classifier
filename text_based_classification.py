from skills.basic_commands.basic_commands_skills import RecordingSkill
import tempfile
from transcription.process_pipeline import DeepspeechTranscription, Transcription, Wav2LetterTranscription, process_pipeline_entry
from skills.general_classifier.general_classifier import GeneralClassifierSkill
from skills.shopping_classifier.shopping_skill import ShoppingSkill
import click
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # cleanup error message items
# humanfriendly, click, padatious, psutil, memory_profiler, gdown, paramiko


skills = [GeneralClassifierSkill(), ShoppingSkill()]

def classify_text(skills, text):
    print("Classifying phrase: ", text)
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

def classify_audio_file(skills, audio_file, transcription_model: Transcription):
    with tempfile.TemporaryDirectory() as dirpath:
        transcript_file = process_pipeline_entry(
            audio_file,
            dirpath,
            transcription_model,
        )
        classify_file(skills, transcript_file)
    
@click.command()
@click.option("--text", default=None, help="Run classification on provided text")
@click.option("--audio_path", default=None, help="Transcribe audio then provide classification")
@click.pass_context
def cli(ctx, text, audio_path):
    if not text and not audio_path:
        click.echo(ctx.get_help())

    if text:
        classify_text(skills, text)
    if audio_path:
        classify_audio_file(skills, audio_path, DeepspeechTranscription())

if __name__ == "__main__":
    # cli()
    skills = [ShoppingSkill(), GeneralClassifierSkill(), RecordingSkill()]
    # classify_text(skills, "We ran out of watermelons")
    classify_audio_file(skills, "save_audio/savewav_2020-09-09_19-04-59_750112.wav", Wav2LetterTranscription())
    # general_classifier = GeneralClassifierSkill()
    # print(general_classifier.classify("Weather is nice outside"))
