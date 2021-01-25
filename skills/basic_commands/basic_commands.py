from padatious import IntentContainer

recording_container = IntentContainer("intent_cache")
recording_container.add_intent("oos", [])
recording_container.add_intent(
    "recording",
    ["start recording", "record this", "Record for {time}"],
)
recording_container.add_intent("stop", ["stop recording"])
recording_container.add_intent("erase", ["Erase recording", "remove recording"])
recording_container.train()


def audio_recorder(text, oos=[]):

    res = recording_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "recording":
        print("Saving recording for mic", res.matches)
    if name == "stop":
        print("stop recording from mic", res.matches)
    if name == "erase":
        print("erasing recording from mic", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


dismiss_ai_container = IntentContainer("intent_cache")
dismiss_ai_container.add_intent("oos", [])
dismiss_ai_container.add_intent(
    "dismiss", [" Nevermind", "Dismissed", "Forget it", "Go away"]
)
dismiss_ai_container.train()


def dismiss_ai(text, oos=[]):
    res = dismiss_ai_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "dismiss":
        print("Stopping interaction", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


date_container = IntentContainer("intent_cache")
date_container.add_intent("oos", [])
date_container.add_intent(
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
date_container.train()


def question_time(text, oos=[]):
    res = date_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "date":
        print("Returning current date", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


news_container = IntentContainer("intent_cache")
news_container.add_intent("oos", [])
news_container.add_intent(
    "play_news",
    ["what's on the news", "tell me the news", "play the bbc news", "restart news"],
)
news_container.train()


def play_news(text, oos=[]):
    res = news_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "play_news":
        print("Playing news", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


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


def set_alarm_reminder(text, oos=[]):
    res = alarm_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "reminder":
        print("creating alarm", res.matches)
    if name == "timer":
        print("Setting timer for", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


weather_container = IntentContainer("intent_cache")
weather_container.add_intent("oos", [])
weather_container.add_intent(
    "weather", ["what is the weather", "weather conditions and forecasts", "forecast"]
)
weather_container.train()


def weather_query(text, oos=[]):
    res = weather_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "weather":
        print("getting weather", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


random_container = IntentContainer("intent_cache")
random_container.add_intent("oos", [])
random_container.add_intent(
    "roll_dice",
    [
        "Roll the dice",
        "roll dice",
    ],
)
random_container.add_intent(
    "flip_coin",
    [
        "Flip a coin",
        "flip coin",
    ],
)
random_container.train()


def random_intent(text, oos=[]):
    res = random_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "roll_dice":
        print("rolling dice", res.matches)
    if name == "flip_coin":
        print("Flipping coin", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


play_music_container = IntentContainer("intent_cache")
play_music_container.add_intent("oos", [])
play_music_container.add_intent(
    "music", ["Play {song} from {event}", "play {song}", "list to {song}"]
)
play_music_container.train()


def play_music(text, oos=[]):
    res = play_music_container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "music":
        print("playing music", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob
