import os
import json
from typing import Any

from ..lib.exception import UnexpectedResult

from .base import Base
from .google import GoogleTranslate
from .genai import GenAI
from .languages import gemini as vertexai_gemini  # Reuse Gemini's language list


load_translations()


class VertexAITranslate(GoogleTranslate, GenAI):
    name = "VertexAI"
    alias = "Vertex AI (Gemini)"
    lang_codes = GenAI.load_lang_codes(vertexai_gemini)
    need_api_key = False
    using_tip = _(
        "This engine uses Google Application Default Credentials (ADC) and Function Calling for robust, structured output. You can authenticate by:\n"
        "1. Providing the path to your service account credential JSON file below.\n"
        "2. Or, by running `gcloud auth application-default login` in your terminal.\n"
        "You can create a service account and get the JSON key file from the "
        '<a href="https://console.cloud.google.com/iam-admin/serviceaccounts">Google Cloud Console</a>.'
    ).replace("\n", "<br />")

    samplings = ["temperature", "top_p"]
    sampling = "temperature"
    stream = False

    request_timeout: float = 180.0

    _tool_decls = {
        "function_declarations": [
            {
                "name": "translation_output",
                "description": "This is the translated text.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "translation": {
                            "type": "STRING",
                            "description": "The final translated text.",
                        }
                    },
                    "required": ["translation"],
                },
            }
        ]
    }

    # prompt = (
    #     "You are tasked with translating an <slang> e-book passage into <tlang>, but rather than a literal translation, focus on capturing the essence of the author's words in a humorous style akin to Taiwanese PTT users.\n"
    #     "Your translation should:\n"
    #     "- Stay true to the author's intended meaning and message.\n"
    #     "- Use expressive language that makes readers feel immersed in the content.\n"
    #     "- Apply a humorous, lively style resembling the discourse of Taiwanese PTT.\n"
    # )

    prompt = (
        "你是一名負責翻譯英文電子書的專家，但你的翻譯不僅僅是逐字翻譯，而是要用自己的話來表達作者的意圖，並且保持與作者旨意一致。\n"
        "你的文筆需要如台灣作家九把刀一般，富有情感且直擊人心。翻譯時要讓讀者感受到文字的生命力和感染力。 並確保符合九把刀的文風和提示中的要求\n。"
        "You are tasked with translating an <slang> e-book passage into <tlang>, but rather than a literal translation"
    )
    temperature: float = 0.5
    top_p: float = 1.0
    top_k = 1
    # Provide a list of common Vertex AI models
    models: list[str] = [
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-pro",
    ]
    model: str | None = models[0]

    _cached_config: dict[str, Any] = {}

    def __init__(self):
        super().__init__()
        self.prompt = self.config.get("prompt", self.prompt)
        self.temperature = self.config.get("temperature", self.temperature)
        self.top_k = self.config.get("top_k", self.top_k)
        self.top_p = self.config.get("top_p", self.top_p)
        self.model = self.config.get("model", self.model)

        request_timeout_config = self.config.get("request_timeout")
        if request_timeout_config is not None:
            self.request_timeout = request_timeout_config

    # Implement the abstract method from GenAI
    def get_models(self) -> list[str]:
        """
        Returns a hardcoded list of common Vertex AI models.
        A dynamic discovery API call is too complex for this context.
        """
        return self.models

    def _get_config_from_file(self) -> dict:
        """Reads and caches project_id and location from the credential file."""
        cred_path = self.config.get("credential_path")
        if not cred_path or not os.path.exists(cred_path):
            return {}

        if cred_path in self._cached_config:
            return self._cached_config[cred_path]

        with open(cred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            config = {
                "project_id": data.get("project_id"),
                "location": self.config.get("location", "us-central1"),
            }
            self._cached_config[cred_path] = config
            return config

    def _get_project_id(self):
        """Overrides GoogleTranslate's method to prioritize the credential file."""
        file_config = self._get_config_from_file()
        if file_config.get("project_id"):
            return file_config.get("project_id")
        return super()._get_project_id()

    def _get_credential(self):
        """Overrides GoogleTranslate's method to temporarily set the credential env var."""
        cred_path = self.config.get("credential_path")
        if cred_path and os.path.exists(cred_path):
            old_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
            try:
                return super()._get_credential()
            finally:
                if old_env is None:
                    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                else:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_env

        return super()._get_credential()

    def get_endpoint(self):
        project_id = self._get_project_id()
        # 從設定檔讀取 location，預設為 us-central1
        location = self.config.get("location", "us-central1")

        # --- 新的 URL 格式判斷邏輯 ---
        if location == "global":
            # 使用 global 端點格式
            base_url = "https://aiplatform.googleapis.com/v1"
        else:
            # 使用 region-specific 端點格式
            base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        # --- 邏輯結束 ---

        model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{self.model}"
        print(f"{base_url}/{model_path}:generateContent")

        return f"{base_url}/{model_path}:generateContent"

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % self._get_credential(),
        }

    def _get_system_prompt(self):
        prompt = self.prompt.replace("<tlang>", self.target_lang)
        if self._is_auto_lang():
            prompt = prompt.replace("<slang>", "the detected source language")
        else:
            prompt = prompt.replace("<slang>", self.source_lang)
        if self.merge_enabled:
            prompt += (
                " Ensure that placeholders matching the pattern {{id_\\d+}} "
                "in the content are retained."
            )
            # 新增關於分隔符的指示
            prompt += (
                " The input text may contain multiple paragraphs separated by "
                f'double newlines ("{self.separator.replace(chr(10), r"n")}"). '
                "It is crucial that you preserve these separators in the "
                "translated output to maintain the original document structure."
            )
        return prompt

    def get_body(self, text):
        return json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "system_instruction": {"parts": [{"text": self._get_system_prompt()}]},
                "generationConfig": {
                    "temperature": self.temperature,
                    "topP": self.top_p,
                    "topK": self.top_k,
                },
                "tools": [self._tool_decls],
                # Force the model to use our defined function for the output
                "tool_config": {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": ["translation_output"],
                    }
                },
            }
        )

    def get_result(self, response):
        try:
            data = json.loads(response)
            if "candidates" not in data or not data["candidates"]:
                error_info = data.get("error", "No candidates in response")
                raise UnexpectedResult("Vertex AI Error: " + str(error_info))

            parts = data["candidates"][0]["content"]["parts"]

            # The result should be in a functionCall
            if parts and "functionCall" in parts[0]:
                func_call = parts[0]["functionCall"]
                if (
                    func_call.get("name") == "translation_output"
                    and "args" in func_call
                ):
                    return func_call["args"].get("translation", "")

            # Fallback for unexpected response structure
            raise UnexpectedResult(
                "Vertex AI response did not contain the expected function call."
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise UnexpectedResult(
                f"Failed to parse Vertex AI response: {e}\nRaw response: {response}"
            )
