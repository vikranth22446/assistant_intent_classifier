from padatious import IntentContainer


def schedule_meeting_intent(text, oos=[]):
    container1 = IntentContainer("intent_cache")
    container1.add_intent("oos", oos)
    container1.add_intent(
        "sched",
        [
            "I have a meeting at {time} on {day} for {what}.",
            "I have a meeting at {time} for {what}.",
            "I have a appointment at {time} on {day} for {what}.",
            "I have a {what} at {time} on {day}.",
            "I have a {what} at {time}.",
            "My meeting is at {time}",
            "We have to be there at {time}",
        ],
    )
    container1.train()
    res = container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "sched":
        print("Registering meeting with ", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob


def latest_time_up(text, oos=[]):
    container = IntentContainer("intent_cache")
    container.add_intent("oos", oos)
    container.add_intent(
        "howlate",
        ["How late was I up last night?", "When was my last command to you yesterday?"],
    )
    container.train()
    res = container.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "howlate":
        print("Finding How late last command", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob
