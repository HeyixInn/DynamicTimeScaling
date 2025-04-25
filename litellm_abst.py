import litellm
import threading
import time
from tqdm import tqdm
import json


class AbstLiteLLM:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model = model_name
        self.MAX_RETRIES = 3

        # file_path = '/home/sc5687/2_Project/CodeSampling/KEYS/vertexAI.json'
        # with open(file_path, 'r') as file:
        #     vertex_credentials = json.load(file)
        # self.vertex_credentials_json = json.dumps(vertex_credentials)

        self.kwargs = {
            # "vertex_credentials": self.vertex_credentials_json,
            # "logprobs": 0,
            "drop_params": True
        }

    def _invoke_model(self, message, result_list, index, try_num=0):
        if try_num >= self.MAX_RETRIES:
            result_list[index] = None
            return
        try:
            response = litellm.completion(
                model=f"{self.provider}/{self.model}",
                messages=message,
                temperature=0,
                max_tokens=2048,
                top_p=1.0,
            )
            result_list[index] = response
        except Exception as e:
            print(e)
            time.sleep(60)
            self._invoke_model(message, result_list, index, try_num+1)

    def _query(self, messages, batch_size=100):
        responses = []
        for i in tqdm(range(0, len(messages), batch_size)):
            batch = messages[i:i + batch_size]

            all_res = [None] * len(batch)  # Initialize result list with None
            threads = []

            for j, payload in enumerate(batch):
                thread = threading.Thread(target=self._invoke_model, args=(payload, all_res, j))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            all_res = [d for d in all_res if d is not None]
            responses.extend(all_res)
            if i + batch_size < len(messages):
                time.sleep(60)

        return responses

    def chat_llm(self, messages: list):
        # messages = [[{"content": pt, "role": "user"}] for pt in prompts]

        responses = self._query(messages)

        responses = [r if isinstance(r, litellm.ModelResponse) else None for r in responses]

        return responses



    @staticmethod
    def extract_text_logprobs(model_pred):
        def extract_token_prob(prob_list):
            res = []
            for d in prob_list:
                res.append([d.token, d.logprob])
            return res

        num_of_gen = len(model_pred.choices)
        assert num_of_gen == 1

        pred_text = model_pred.choices[0].message.content
        if hasattr(model_pred.choices[0], 'logprobs'):
            logits = extract_token_prob(model_pred.choices[0].logprobs.content)
        else:
            logits = None

        return pred_text, logits