from padatious import IntentContainer
from skills.skill_utils import Skill

class RecordingSkill(Skill):
    intent_cache = "intent_cache"
    cutoff_prob = .7
    skill_name = "Recording"

    def __init__(self, run_config=None, model_files_download_paths=None, model_folder=None):
        # super().__init__(run_config, model_files_download_paths=model_files_download_paths, model_folder=model_folder)
        self.init_models()

    def init_models(self):
        recording_container = IntentContainer(self.intent_cache)
        recording_container.add_intent("oos", [])
        recording_container.add_intent(
            "recording",
            ["start recording", "record this", "Record for {time}"],
        )
        recording_container.add_intent("stop", ["stop recording"])
        recording_container.add_intent("erase", ["Erase recording", "remove recording"])
        recording_container.train()
        self.recording_container = recording_container

    def classify(self, text):
        res = self.recording_container.calc_intent(text)
        prob = res.conf
        name = res.name
        if name == "recording" and prob > self.cutoff_prob:
            print("Saving recording for mic", res.matches)
        if name == "stop" and prob > self.cutoff_prob:
            print("stop recording from mic", res.matches)
        if name == "erase" and prob > self.cutoff_prob:
            print("erasing recording from mic", res.matches)
        if prob < self.cutoff_prob:
            name = "oos"
        return name, prob

class DismissAiSkill(Skill):
    intent_cache = "intent_cache"
    cutoff_prob = .7
    skill_name = "Dismiss"

    def __init__(self, run_config=None, model_files_download_paths=None, model_folder=None):
        # super().__init__(run_config, model_files_download_paths=model_files_download_paths, model_folder=model_folder)
        self.init_models()

    def init_models(self):
        self.dismiss_ai_container = IntentContainer("intent_cache")
        self.dismiss_ai_container.add_intent("oos", [])
        self.dismiss_ai_container.add_intent(
            "dismiss", [" Nevermind", "Dismissed", "Forget it", "Go away"]
        )
        self.dismiss_ai_container.train()

    def classify(self, text):
        res = self.dismiss_ai_container.calc_intent(text)
        prob = res.conf
        name = res.name
        if name == "dismiss":
            print("Stopping interaction", res.matches)
        if prob < self.cutoff_prob:
            name = "oos"
        return name, prob


class DateSkill(Skill):
    intent_cache = "intent_cache"
    cutoff_prob = .7
    skill_name = "Date"

    def __init__(self, run_config=None, model_files_download_paths=None, model_folder=None):
        # super().__init__(run_config, model_files_download_paths=model_files_download_paths, model_folder=model_folder)
        self.init_models()

    def init_models(self):
        self.date_container = IntentContainer("intent_cache")
        self.date_container.add_intent("oos", [])
        self.date_container.add_intent(
            "date",
            [
                "What time is it",
                "get the time",
                "get the date",
                "day of the week",
                "what day of the week is it",
                "what day is it",
            ],
        )
        self.date_container.train()

    def classify(self, text):
        res = self.date_container.calc_intent(text)
        prob = res.conf
        name = res.name
        if name == "date":
            print("Returning current date", res.matches)
        if prob < 0.7:
            name = "oos"
        return name, prob


class NewsSkill(Skill):
    intent_cache = "intent_cache"
    cutoff_prob = .7
    skill_name = "Date"

    def __init__(self, run_config=None, model_files_download_paths=None, model_folder=None):
        # super().__init__(run_config, model_files_download_paths=model_files_download_paths, model_folder=model_folder)
        self.init_models()

    def init_models(self):
        self.news_container = IntentContainer("intent_cache")
        self.news_container.add_intent("oos", [])
        self.news_container.add_intent(
            "play_news",
            ["what's on the news", "tell me the news", "play the bbc news", "restart news"],
        )
        self.news_container.train()



    def classify(self, text):
        res = self.news_container.calc_intent(text)
        prob = res.conf
        name = res.name
        if name == "play_news":
            print("Playing news", res.matches)
        if prob < 0.7:
            name = "oos"
        return name, prob



class AlarmSkill(Skill):
    intent_cache = "intent_cache"
    cutoff_prob = .7
    skill_name = "Alarm"

    def __init__(self, run_config=None, model_files_download_paths=None, model_folder=None):
        # super().__init__(run_config, model_files_download_paths=model_files_download_paths, model_folder=model_folder)
        self.init_models()

    def init_models(self):
        alarm_container = IntentContainer("intent_cache")
        alarm_container.add_intent("oos", [])
        alarm_container.add_intent(
            "reminder",
            [
                "Set alarms",
                "Set an alarm for {time}",
                "Set reminders",
                "remind me to {object} at {time}",
                "set {repeating} timer for {time}",
            ],
        )
        alarm_container.add_intent("timer", ["set timer for {minutes}"])
        alarm_container.train()

    def classify(self, text):
        res = self.alarm_container.calc_intent(text)
        prob = res.conf
        name = res.name
        if name == "reminder":
            print("creating alarm", res.matches)
        if name == "timer":
            print("Setting timer for", res.matches)
        if prob < 0.7:
            name = "oos"
        return name, prob