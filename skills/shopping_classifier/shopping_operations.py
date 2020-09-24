from padatious import IntentContainer
import yagmail


def add_to_shopping_list_intent(text, oos):
    def addToList(commandStr, shoppingList, currentContainer):
        resultDict = currentContainer.calc_intent(commandStr)
        print("Command feeded in: " + commandStr)
        # print(resultDict)
        if resultDict.name == "add":
            if len(resultDict.matches) >= 1:
                shoppingList.append(resultDict.matches["item"])
        shopStr = ", ".join(shoppingList)
        print("Your shopping list now comprises of " + shopStr)

    containerShoppingAdd = IntentContainer("intent_cache")
    containerShoppingAdd.add_intent("oos", x)
    containerShoppingAdd.add_intent(
        "add",
        [
            "Add {item} to my shopping list.",
            "I need to get {item} from the grocery store",
            "I need to remember to get {item} from the store",
            "Put {item} in my shopping list",
            "I need {item} in the shopping list",
        ],
    )
    containerShoppingAdd.train()
    shoppingList = []
    addToList(
        "I need to get eggs from the grocery store", shoppingList, containerShoppingAdd
    )
    addToList(
        "I need to get turnovers from the store", shoppingList, containerShoppingAdd
    )
    print("-----------------------------")


def removeList(commandStr, shoppingList, currentContainer):
    resultDict = currentContainer.calc_intent(commandStr)
    print("Command feeded in: " + commandStr)
    # print(resultDict)
    if resultDict.name == "remove":
        if len(resultDict.matches) >= 1:
            shoppingList.remove(resultDict.matches["item"])
    shopStr = ", ".join(shoppingList)
    if len(shoppingList) == 0:
        shopStr = "nothing"
    print("Your shopping list now comprises of " + shopStr)


def remove_from_shopping_list_intent(text, oos, shoppingList):
    containerShoppingRemove = IntentContainer("intent_cache")
    containerShoppingRemove.add_intent("oos", oos)
    containerShoppingRemove.add_intent(
        "remove",
        [
            "Remove {item} from my shopping list.",
            "I do not need to get {item} from the grocery store",
            "I do not need to remember to get {item} from the store",
            "Do not put {item} in my shopping list",
            "I do not need {item} in the shopping list",
        ],
    )
    containerShoppingRemove.train()
    removeList(
        "Remove eggs from my shopping list", shoppingList, containerShoppingRemove
    )
    removeList(
        "Remove turnovers from shopping list", shoppingList, containerShoppingRemove
    )
    print("-----------------------------")


def promptRanOutList(commandStr, shoppingList, ranOutContainer):
    resultDict = ranOutContainer.calc_intent(commandStr)
    print("Command feeded in: " + commandStr)
    if resultDict.name == "out":
        if len(resultDict.matches) >= 1:
            # speak("Should I add " + resultDict.matches['item'] + "?")
            print(
                "Spoken: Should I add "
                + resultDict.matches["item"]
                + " to the shopping list?"
            )


def ran_out_intent(text, oos, shoppingList):
    containerShoppingInitial = IntentContainer("intent_cache")
    containerShoppingInitial.add_intent("oos", oos)
    containerShoppingInitial.add_intent(
        "out",
        [
            "I just used the last {item}",
            "I think that was the last {item}",
            "We just finished the {item}",
            "We ran out of {item}",
        ],
    )
    containerShoppingInitial.train()

    promptRanOutList("We ran out of milk", shoppingList, containerShoppingInitial)


print("-----------------------------")


def ranOutList(commandStr, shoppingList, ranOutContainer, positiveContainer, talkArray):
    resultPositiveDict = positiveContainer.calc_intent(commandStr)
    print("Command feeded in: " + talkArray[len(talkArray) - 1])
    print("Command feeded in: " + commandStr)
    if resultPositiveDict.name == "positive":
        resultRanOutDict = ranOutContainer.calc_intent(talkArray[len(talkArray) - 1])
        print(resultRanOutDict)
        if resultRanOutDict.name == "out":
            if len(resultRanOutDict.matches) >= 1:
                shoppingList.append(resultRanOutDict.matches["item"])
    shopStr = ", ".join(shoppingList)
    if len(shoppingList) == 0:
        shopStr = "nothing"
    print("Your shopping list now comprises of " + shopStr)


def ran_out_item_intent(text, oos):
    containerShoppingOut = IntentContainer("intent_cache")
    containerShoppingOut.add_intent("oos", oos)
    containerShoppingOut.add_intent(
        "out",
        [
            "I just used the last {item}",
            "I think that was the last {item}",
            "We just finished the {item}",
            "We ran out of {item}",
        ],
    )
    containerShoppingOut.train()


def affirmation_intent(text, oos):
    containerShoppingPositive = IntentContainer("intent_cache")
    containerShoppingPositive.add_intent("oos", oos)
    containerShoppingPositive.add_intent(
        "positive", ["Yes", "Yea", "Definitely", "Please do", "Yup"]
    )
    containerShoppingPositive.train()


def emailList(commandStr, shoppingList, emailContainer, email):
    resultDict = emailContainer.calc_intent(commandStr)
    # print(resultDict)
    print("Command feeded in: " + commandStr)
    if resultDict.name == "email":
        yag = yagmail.SMTP("mycroftiscool", "mycroftisreallycool")
        yag.send(email, "Shopping List", shoppingList)
        print("Spoken: The list has been emailed to you")
        # speak("The list has been emailed to you")


def email_shopping_list_intent(text, oos):
    containerShoppingEmail = IntentContainer("intent_cache")
    containerShoppingEmail.add_intent("oos", oos)
    containerShoppingEmail.add_intent(
        "email",
        [
            "Please email me the shopping list",
            "I am going to the store",
            "I need the shopping list",
            "Send me the shopping list",
        ],
    )
    containerShoppingEmail.train()
