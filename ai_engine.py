import os
import asyncio
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD .ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        f"GEMINI_API_KEY not found in {ENV_FILE}"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

# Stores every sentence already used
USED_SENTENCES = set()


# ============================================================
# SENTENCE NORMALIZATION
# ============================================================

def normalize_sentence(sentence):
    """
    Converts sentences to a standard form so that
    small differences in capitalization/punctuation
    are still treated as duplicates.
    """

    return (
        sentence
        .strip()
        .lower()
        .replace('"', '')
        .replace("'", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .split()
    )


def sentence_key(sentence):
    return " ".join(normalize_sentence(sentence))


# ============================================================
# FALLBACK SENTENCES
# ============================================================

FALLBACK_SENTENCES = {

    "hello": [
        "Hello, it is nice to meet you.",
        "Hello, how are you doing today?",
        "Hi there, I hope you are well.",
        "Hello, I hope you are having a good day.",
        "Hi, it is good to see you.",
        "Hello, how has your day been?",
        "Hi there, how are things going?",
        "Hello, I am glad to see you.",
        "Hi, I hope everything is going well.",
        "Hello, nice to see you again.",
        "Hi there, I wanted to say hello.",
        "Hello, I hope you are feeling good.",
        "Hi, it is wonderful to see you.",
        "Hello, I am happy to meet you.",
        "Hi there, how are you feeling today?",
        "Hello, I hope your day is going well.",
        "Hi, I am pleased to see you.",
        "Hello, I just wanted to say hi.",
        "Hi there, it is good to see you.",
        "Hello, I hope everything is fine today."
    ],

    "helpme": [
        "Please help me with this.",
        "I need your help right now.",
        "Could you please help me?",
        "I need some assistance, please.",
        "Can you help me for a moment?",
        "Please give me some help.",
        "I would really appreciate your help.",
        "Could you assist me with this?",
        "I need someone to help me.",
        "Please help me with this situation.",
        "Can you please help me now?",
        "I could really use your help.",
        "Please assist me with this problem.",
        "I need your assistance right away.",
        "Could you give me a hand?",
        "Please help me figure this out.",
        "I would like some help, please.",
        "Can someone please help me?",
        "I need a little help here.",
        "Please help me with what I am doing."
    ],

    "stop": [
        "Please stop what you are doing.",
        "Stop right now, please.",
        "I need you to stop.",
        "Please stop for a moment.",
        "Could you please stop?",
        "I want you to stop now.",
        "Please stop immediately.",
        "You need to stop right now.",
        "Please stop doing that.",
        "I am asking you to stop.",
        "Please stop this right away.",
        "I really need you to stop.",
        "Could you stop for a moment?",
        "Please do not continue.",
        "I want this to stop now.",
        "You should stop what you are doing.",
        "Please stop and listen to me.",
        "I need this to stop immediately.",
        "Please stop right there.",
        "I am asking you to stop now."
    ],

    "thankyou": [
        "Thank you so much for helping me.",
        "I really appreciate your help.",
        "Thank you, that means a lot.",
        "Thanks so much for your support.",
        "I am very grateful for your help.",
        "Thank you for being there for me.",
        "I truly appreciate everything you did.",
        "Thanks a lot for helping me.",
        "Thank you, I really appreciate it.",
        "I am thankful for your support.",
        "Thank you for helping me today.",
        "I really appreciate what you did.",
        "Thanks for taking the time to help.",
        "Thank you for your kindness.",
        "I am grateful for everything you did.",
        "Thanks, I really needed your help.",
        "Thank you for supporting me.",
        "I truly appreciate your assistance.",
        "Thanks so much, I am grateful.",
        "Thank you, your help means a lot."
    ],

    "yes": [
        "Yes, that is correct.",
        "Yes, I completely agree.",
        "Yes, that sounds good to me.",
        "Yes, I would like that.",
        "Yes, that is exactly right.",
        "Yes, I agree with you.",
        "Yes, please go ahead.",
        "Yes, that works for me.",
        "Yes, I understand.",
        "Yes, I would be happy to.",
        "Yes, that sounds perfect.",
        "Yes, I think that is right.",
        "Yes, I agree with that.",
        "Yes, that is what I want.",
        "Yes, please continue.",
        "Yes, I understand your point.",
        "Yes, I would like to do that.",
        "Yes, that seems like a good idea.",
        "Yes, I am okay with that.",
        "Yes, I agree completely."
    ]
}


# ============================================================
# FALLBACK INDEX
# ============================================================

FALLBACK_INDEX = {
    "hello": 0,
    "helpme": 0,
    "stop": 0,
    "thankyou": 0,
    "yes": 0
}


# ============================================================
# GET UNIQUE FALLBACK
# ============================================================

def get_unique_fallback(anchor):

    anchor = anchor.lower()

    if anchor not in FALLBACK_SENTENCES:
        base = f"I want {anchor}."

        if sentence_key(base) not in USED_SENTENCES:
            USED_SENTENCES.add(sentence_key(base))
            return base

        # Make unknown anchors unique
        counter = 2

        while True:
            sentence = f"I want {anchor}, please, number {counter}."
            key = sentence_key(sentence)

            if key not in USED_SENTENCES:
                USED_SENTENCES.add(key)
                return sentence

            counter += 1

    sentences = FALLBACK_SENTENCES[anchor]

    start_index = FALLBACK_INDEX[anchor]

    for i in range(len(sentences)):

        index = (start_index + i) % len(sentences)

        sentence = sentences[index]

        key = sentence_key(sentence)

        if key not in USED_SENTENCES:

            USED_SENTENCES.add(key)

            FALLBACK_INDEX[anchor] = index + 1

            return sentence

    # --------------------------------------------------------
    # If all predefined fallbacks have been used
    # --------------------------------------------------------

    counter = FALLBACK_INDEX[anchor] + 1

    while True:

        sentence = (
            f"I would like to communicate about "
            f"{anchor}, please."
        )

        # Add a unique natural variation if necessary
        if counter > 2:
            sentence = (
                f"I would like to communicate about "
                f"{anchor}, please, for request {counter}."
            )

        key = sentence_key(sentence)

        if key not in USED_SENTENCES:

            USED_SENTENCES.add(key)

            FALLBACK_INDEX[anchor] = counter + 1

            return sentence

        counter += 1


# ============================================================
# GEMINI SENTENCE GENERATION
# ============================================================

def generate_sentence_with_gemini(anchor, emotion="neutral"):

    # Try Gemini several times if it produces duplicates
    for attempt in range(8):

        # Give Gemini some previously used sentences
        # so it knows what NOT to repeat.
        previous_sentences = list(USED_SENTENCES)[-15:]

        previous_text = "\n".join(
            f"- {sentence}"
            for sentence in previous_sentences
        )

        prompt = f"""
You are a communication assistant for an
EEG-based Brain-Computer Interface.

The EEG system detected this anchor word:

{anchor}

Detected emotion:

{emotion}

Generate ONE completely new natural spoken sentence
that expresses the meaning of the anchor word.

RULES:

1. Preserve the meaning of the anchor word.
2. Do not change the user's intention.
3. Do not introduce an unrelated topic.
4. Keep the sentence between 5 and 12 words.
5. Make it natural for spoken communication.
6. Use different wording from previous sentences.
7. NEVER return one of the previously used sentences.
8. Return ONLY the sentence.
9. Do not use quotation marks.
10. Do not explain anything.

Previously used sentences:

{previous_text}

Generate a NEW sentence now.
"""

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0
                )
            )

            sentence = response.text.strip()

            # Remove quotation marks
            sentence = sentence.strip('"').strip("'")

            key = sentence_key(sentence)

            # ------------------------------------------------
            # CHECK DUPLICATE
            # ------------------------------------------------

            if key not in USED_SENTENCES:

                USED_SENTENCES.add(key)

                print(
                    f"\nGemini sentence "
                    f"[{len(USED_SENTENCES)}]: {sentence}"
                )

                return sentence

            print(
                f"\nDuplicate detected."
                f" Retrying Gemini "
                f"({attempt + 1}/8)..."
            )

        except Exception as e:

            print("\nGemini error:", e)
            break

    # ========================================================
    # GEMINI FAILED / REPEATED
    # ========================================================

    sentence = get_unique_fallback(anchor)

    print(
        f"\nUsing unique fallback "
        f"[{len(USED_SENTENCES)}]: {sentence}"
    )

    return sentence


# ============================================================
# PROCESS PREDICTION
# ============================================================

def process_prediction(
    anchor,
    emotion="neutral",
    confidence=1.0
):

    if confidence < CONFIDENCE_THRESHOLD:

        return "Uncertain prediction"

    return generate_sentence_with_gemini(
        anchor,
        emotion
    )


# ============================================================
# EDGE TTS
# ============================================================

def text_to_wav(
    text,
    filename="output.mp3"
):

    async def generate():

        communicate = edge_tts.Communicate(
            text,
            "en-US-AriaNeural",
            rate="+20%"
        )

        await communicate.save(filename)

    asyncio.run(generate())