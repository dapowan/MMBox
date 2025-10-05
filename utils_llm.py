import types
from math import ceil
import httpx
import yaml
import tiktoken
import base64
import openai
from openai import OpenAI

base_url = 'https://api.bltcy.ai/v1'

class LLMProxy:

    def __init__(self, model, key, chat_config=None, running_mode='real'):
        self.client = OpenAI(api_key=key, base_url=base_url) #
        self.running_mode = running_mode
        self.model = model
        self.seed = chat_config.seed if chat_config is not None else None
        self.chat_config = chat_config

    def generate_text(self, input_text, system_text=None):
        input_text_gpt = self.wrap_input(input_text, system_text)
        if self.running_mode == 'test':
            response = "test"
            return response, (0, 0)
        try:
            response = self.client.chat.completions.create(model=self.model, messages=input_text_gpt)
            return response.choices[0].message.content, self.extract_usage(response.usage)
        except Exception as e:
            print(e)
            return None, (0, 0)

    def extract_usage(self, usage):
        return usage.prompt_tokens, usage.completion_tokens

    def wrap_input(self, input_text, system_text):
        if system_text:
            input_text_gpt = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": input_text}
            ]
        else:
            input_text_gpt = [
                {"role": "user", "content": input_text}
            ]
        return input_text_gpt


if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06'
    # model_config = types.SimpleNamespace(**{'name': name})
    model = LLMProxy(name, key, None, running_mode='real')
    response, usage = model.generate_text('Hello', 'you are a helpful assistant')
