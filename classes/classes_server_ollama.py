import os
import uuid
import datetime
import re
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ollama

# ===========================
# LLM Response & Role Classes
# ===========================
class ResponseType:
    GENERATED = "generated"
    ERROR = "error"

class LLMResponse:
    def __init__(self, prompt_id, raw_text, timestamp, response_type):
        self.prompt_id = prompt_id
        self.raw_text = raw_text
        self.timestamp = timestamp
        self.response_type = response_type

# ===========================
# Ollama Server
# ===========================
class OllamaServer:
    def __init__(self, client=None):
        if client is None:
            self.client = ollama.Client(
                host="https://mnj9tgczbpbez1-11434.proxy.runpod.net"
            )
        else:
            self.client = client

    def get_models_list(self):
        try:
            models = self.client.list()
            return [m.model for m in models.get("models", [])]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def download_model_if_not_exists(self, model_name):
        print(f"Using remote model: {model_name}")  # no local download needed

# ===========================
# Ollama Chat
# ===========================
class OllamaChat:
    USER = "user"
    ASSISTANT = "assistant"

    def __init__(self, server: OllamaServer, model: str):
        self.server = server
        self.model = model
        self.messages = []
        self.server.download_model_if_not_exists(model)

    def add_history(self, content: str, role: str):
        self.messages.append({"role": role, "content": content})

    def clear_history(self):
        self.messages = []

    def send_prompt(self, prompt: str, prompt_uuid: str = None, use_history=False, stream=False, max_retries=3):
        if prompt_uuid is None:
            prompt_uuid = str(uuid.uuid4())

        messages = self.messages + [{"role": self.USER, "content": prompt}] if use_history else [{"role": self.USER, "content": prompt}]

        retries = 0
        while retries < max_retries:
            try:
                response = self.server.client.chat(model=self.model, messages=messages, stream=stream)

                complete_message = ""
                if stream:
                    for line in response:
                        complete_message += line["message"]["content"]
                        print(line["message"]["content"], end="", flush=True)
                else:
                    complete_message = response.get("message", {}).get("content", "").strip()

                if use_history:
                    self.add_history(prompt, self.USER)
                    self.add_history(complete_message, self.ASSISTANT)

                return LLMResponse(
                    prompt_id=prompt_uuid,
                    raw_text=complete_message,
                    timestamp=datetime.datetime.now(),
                    response_type=ResponseType.GENERATED
                )

            except Exception as e:
                retries += 1
                print(f"\n⚠️ Error sending prompt (attempt {retries}/{max_retries}): {e}")
                time.sleep(5)
                if retries >= max_retries:
                    print("❌ Maximum retries reached. Model seems disconnected.")
                    input("Fix the connection and press ENTER to continue...")
                    retries = 0

