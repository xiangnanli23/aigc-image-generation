import openai

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


client = openai.OpenAI(
    api_key="soul_aigc_prompt_refine_2024_0827",
    base_url="http://test-claude-llm-service-adapter.qa-kubeflow.soulapp-inc.cn"
)


def get_claude_response(
    instructions: str = "",
    ) -> str:
    completion = client.chat.completions.create(
        model="anthropic.claude-3-haiku-20240307-v1:0", 
        messages=[
            {"role": "user", "content": instructions}, # <-- This is the system message that provides context to the model
            # {"role": "user", "content": ""}  # <-- This is the user message for which the model will generate a response
        ],
    )

    res = completion.choices[0].message.content
    return res


def chat_completions():
    """ 
    https://platform.openai.com/docs/guides/text-generation/chat-completions-api
    
    Chat Completions API
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
        ],
    )
    print("Assistant: " + completion.choices[0].message.content)


if __name__ == '__main__':
    print(get_claude_response("who are you?"))
    # chat_completions()