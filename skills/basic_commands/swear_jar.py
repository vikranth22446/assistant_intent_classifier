# print('-----------------------------')
from padatious import IntentContainer


def check_profanity(commandStr, swear_count=0, cutoff=0.75):
    print("Command feeded in: " + commandStr)
    # from profanity_check import predict_prob
    # includes_swear = predict_prob([commandStr]) > 0.75
    # if includes_swear:
    #     swear_count += 1
    #     print("Your swear count has gone up to " + str(swear_count))
    # return includes_swear, swear_count
    return "oos", 0


def resetSwear(commandStr, swear_count, currentContainer):
    print("Command feeded in: " + commandStr)
    resultDict = currentContainer.calc_intent(commandStr)
    if resultDict.name == "reset":
        print("Your swear count is back to 0")
        swear_count = 0
    return swear_count


remove_swear_intent_model = IntentContainer("intent_cache")
remove_swear_intent_model.add_intent(
    "reset", ["Reset my swear jar", "Restart my swear jar count"]
)
remove_swear_intent_model.train()


def remove_swear_intent(text, swear_count=0, oos=[], cutoff=0.7):

    # swear_count = resetSwear(
    #     "Put my swear jar count back to 0", swear_count, remove_swear_intent_model
    # )
    print("-----------------------------")
    res = remove_swear_intent_model.calc_intent(text)
    prob = res.conf
    name = res.name
    if name == "reset":
        print("Resetting swear count: ", res.matches)
    if prob < 0.7:
        name = "oos"
    return name, prob
