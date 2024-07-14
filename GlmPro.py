from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import zhipuai
zhipuai.api_key = "bca178f6617ef3f19a9ceae0df29bc71.mrFwlk1gENNmQyjX"
class Chatglmpro(LLM):
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Chatglmpro"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        response = zhipuai.model_api.invoke(
            model="chatglm_pro",
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=0.9, )

        if response['code'] != 200:
            return "error"
        # resp = response.json()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response.get('data').get('choices')[0].get('content')][1:-1]]
        return response.get('data').get('choices')[0].get('content')[1:-1]