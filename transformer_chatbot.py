import transformers
nlp = transformers.pipeline("conversational", 
                            model="microsoft/DialoGPT-medium")

print("Start chatting with the bot (type 'quit' to stop)!")


while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
   

    chat = nlp(transformers.Conversation(user_input), pad_token_id=50256)
    res = str(chat)
    res = res[res.find("bot >> ")+6:].strip()


    print("Bot:",res )

