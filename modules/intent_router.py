from modules.llm import chat


def classify_intent(user_message):
    """Return 'heart_diagnosis' or 'general_query'."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. Read the user's message and reply with ONE word only.\n"
                "- Reply 'heart_diagnosis' if user wants to check/diagnose heart disease.\n"
                "- Reply 'general_query' for anything else (hospital info, doctor timings, etc.)\n"
                "Reply with one word only."
            )
        },
        {"role": "user", "content": user_message}
    ]
    response = chat(messages, max_tokens=10).lower()

    if "heart" in response:
        return "heart_diagnosis"
    return "general_query"