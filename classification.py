import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def classify_emotion(user_input):
    model = ChatOpenAI(model="gpt-3.5-turbo")
    # model = ChatOpenAI(model="gpt-4-1106-preview").bind(
    #     response_format={"type": "json_object"}
    # )

    class Emotion(BaseModel):
        emotion: str = Field(description="Emotion detected in the user input, whether sad, happy, or neutral")
        justification: str = Field(description="Justification for emotion detected")

    parser = JsonOutputParser(pydantic_object=Emotion)

    prompt = PromptTemplate(
        template="""
            Decide the emotion of the user input and provide a justification for \
            the emotion detected in the user input whether sad, happy, or neutral.
            User input: "{user_input}"
            Your answer must be in JSON format: {format_instructions}
        """,
        input_variables=["user_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    output = chain.invoke({"user_input": user_input})
    return output