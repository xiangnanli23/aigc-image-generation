"""
    https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o
    
    openai
        0.28.0
        1.35.1 
    ENGINE
        gpt4-turbo
        gpt4-o

"""

import os
from openai import OpenAI 



BASE_URL       = "http://ml-llm-adaptor-offline.prod-kubeflow.soulapp-inc.cn"
OPENAI_API_KEY = "ysj_aigc_lixiangnan_2024_04_12"
MODEL          = "gpt-4o"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY),
    base_url=BASE_URL,
    )




def chat_completions():
    """ 
    https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    
    Chat Completions API
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
            {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
        ],
        seed=200,
    )
    print("Assistant: " + completion.choices[0].message.content)
    
    # # streaming
    # stream = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[
    #         # {"role": "user", "content": "Say this is a test"},
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Who won the world series in 2020?"},
    #         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #         {"role": "user", "content": "Where was it played?"}
    #     ],
    #     stream=True,
    # )
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")


def get_embeddings():
    response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="The food was delicious and the waiter..."
    )

    print(response)


def image_generation():
    """"""
    response = client.images.generate(
        model="dall-e-3",
        prompt="1 handsome British male, angel, half-body portrait, (messy long hair, sweat-soaked hair, golden hair:1.6), (expression of fear, frown:1.6), (skinny body), (white dirty hemp robe, sweat-soaked robe:1.6), (pitch-dark dungeon, kneeling on the ground:1.6)",
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    print(image_url)




################################################################################
def get_chatgpt_response(
    instructions: str = "",
    ) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instructions}, # <-- This is the system message that provides context to the model
            {"role": "user", "content": ""}  # <-- This is the user message for which the model will generate a response
        ],
    )
    res = completion.choices[0].message.content
    return res

def get_chatgpt_response_v2(
    prompt: str = "",
    instructions: str = "",
    ) -> str:
    """ refined by jinyongyi """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instructions}, # <-- This is the system message that provides context to the model
            {"role": "user", "content": prompt}  # <-- This is the user message for which the model will generate a response
        ],
    )
    res = completion.choices[0].message.content
    return res



if __name__ == "__main__":
    chat_completions()
    # get_embeddings()
    # image_generation()
    
    pass