import google.generativeai as genai

from src.prompt_engine import build_prompt


def configure_gemini(api_key):

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(

        "gemini-2.5-flash"

    )


def ask_gemini(

    api_key,

    df,

    question

):

    model = configure_gemini(

        api_key

    )

    prompt = build_prompt(

        df,

        question

    )

    response = model.generate_content(

        prompt

    )

    return response.text