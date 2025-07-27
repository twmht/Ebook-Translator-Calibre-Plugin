import os
import json
from typing import Any

from ..lib.exception import UnexpectedResult

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

    #     prompt = """As a professional translator, your task is to translate and interpret the book "Why Don't We Learn From History?" by Liddell Hart, ensuring that the translation reflects the original author's intent and meaning, allowing the reader to understand clearly without awkward literal translations.
    #
    # You are tasked with translating an <slang> e-book passage into <tlang>
    #
    # Your writing style must showcase the following traits:
    # - **Clarity, Conciseness, and Impactful Precision:** Strive for clear, concise, and powerful translation, avoiding lengthy or vague expressions. Hart emphasizes that "hard writing makes easy reading," signaling the need for refining complex ideas into easily understandable language while ensuring information accuracy.
    # - **Profound Explanation with Accessible Insights:** Translation should go beyond word conversion and effectively "bring out the meaning." Hart criticizes overly scientific historical writing that lacks conclusions and significance. Therefore, adeptly extract key concepts, historical lessons, and causal relationships from the book, offering explanations, details, and insights that surpass surface summaries. The goal is for the reader to gain "practical value" and "personal philosophy," not merely a dry stack of facts.
    # - **Engaging Narrative Style:** Hart was akin to a "professional journalist, even a popular journalist," likening historical research to an "unending detective story." Borrowing this style, use lively language and concrete examples to enhance the readability of the translation, drawing attention from Taiwanese readers and avoiding the stiffness common in academic works. This aids in more effectively spreading Hart's profound thoughts.
    # - **Embodiment of Objective Rationality and Critical Thinking:** Although allowing free interpretation, maintain alignment with the author's pursuit of "rigorous objectivity" and "intellectual detachment." When understanding and communicating the original text, uphold scientific inquiry and guided by "critical skepticism." Faithfully present Hart's deep insights on history, including detecting "unrealities" and criticizing "self-deception," avoiding personal bias impacting message transmission.
    # - **Emphasis on Relevance and Contemporary Revelation:** Hart emphasizes history's "practical value" and advocates "deriving future" from past experiences. In translation, focus on linking historical lessons to Taiwan's contemporary social and political realities. Subtle wordings can make readers feel history is not distant but wisdom closely related to their lives. For instance, integrating strategies from the book on "truth broadcasting" (like "avoiding direct attacks, turning to flanking maneuvers" or "packing new ideas as a modern revival of traditional wisdom") in translation tone to closely fit Taiwanese readers' thinking patterns.
    # - **Fidelity to Author's Core Philosophy:** The core requirement is "complete consistency with the author's main ideas and meaning." It necessitates understanding Hart's pursuit of "truth," insistence on "independent inquiry," severe criticism of "blind obedience" and "corruption of power," and emphasis on "freedom" and "restraint." The "art" of translation is ensuring "bringing out meaning" while not "diverging from the truth," conveying the author's profound thoughts.
    #
    # Overall, the Taiwanese translator should act as a disseminator of thought with both scholarly depth and journalistic acuity, transforming Liddell Hart's insights into a voice of wisdom—both faithful to the original intent and highly inspiring—in a manner most acceptable to Taiwanese readers."""

    #     prompt = """You are entrusted with translating and interpreting the book "Running Toward Mystery" with an aim to achieve alignment with the author's core themes and intentions, without translating in a rigid manner. The text should be relatable and immediately understandable.
    #
    # You are tasked with translating an <slang> e-book passage into <tlang>
    #
    # Guidelines for your translation style:
    # * **Authentic and Personal Voice**: Maintain the autobiographical narrative style with vivid storytelling, allowing your personal insights to breathe life into the translation while preserving the content's integrity.
    # * **Clarity and Conciseness**: Eliminate redundant phrases, distilling complex ideas into direct, impactful statements. Use metaphors and concrete examples to aid comprehension of abstract philosophical concepts.
    # * **Emphasis on Practice and Experience**: Highlight the significance of practice over mere knowledge, reflecting the author's emphasis on practical application through relatable examples.
    # * **Balanced Critique and Acceptance**: Present the author's respect for tradition critically yet gently, challenging readers' perceptions without sounding authoritative.
    # * **Conveying "Joy of Disillusionment" and Transcending Dualities**: Capture the paradoxical experiences and transcendental themes inherently described in the text.
    # * **Localization and Universal Appeal**: Utilize expressions familiar to Taiwanese readers while preserving the text’s universal significance. Adapt cultural elements to be understood and appreciated in the Taiwanese context.
    # * **Rich Emotional Spectrum**: Craft translations that intricately convey the author's emotional spectrum from struggle to serenity, in a naturally profound manner.
    # * **Strategic Use of Section Titles and Paragraphing**: Adopt concise and suggestive chapter titles; use heading hierarchies and text formatting to enhance readability and comprehension.
    #
    # The goal is to transform the author's profound insights into reader-friendly and moving text, akin to a storyteller's engaging and compassionate tone."""

    #     prompt = """Translate the book "Broken Money" by Lyn Alden while maintaining the author's original intent and clarity.
    #
    #     You are tasked with translating an <slang> e-book passage into <tlang>
    # The style should be:
    # - Plain and conversational, using everyday language familiar to Taiwanese readers.
    # - Storytelling in approach, making complex concepts accessible.
    # - Concise, with clear logical progression.
    # - Providing in-depth insights, explaining underlying reasons and encouraging critical thinking.
    # - Focusing on a technological perspective, showing component interactions in the financial system.
    # - Appropriately localized, using terms familiar to the Taiwanese audience.
    #
    # Ensure that the translation is not just a conversion of language but a re-creation of knowledge tailored to the target audience."""

    #     prompt = """You are tasked with translating and interpreting the book "Psych" for Taiwanese readers. Your translation should achieve alignment with the main ideas and intentions of the author, Paul Bloom, without rigid word-for-word translation, ensuring that readers immediately understand the content.
    #
    # In your translation, exhibit the following characteristics:
    #
    # 1. **Simple and Understandable Language:**
    #    - Avoid overly rigid or academic vocabulary unless absolutely necessary, and provide clear explanations when required to ensure readers can "immediately understand."
    #    - Your writing style should reflect the original author Paul Bloom's succinct and powerful way of addressing university students, with a "family-friendly" approach. Avoid lengthy or ineffective expressions, just as the author critiques theories for "explaining everything, thus explaining nothing."
    #
    # 2. **Effective Use of Taiwanese Localization Examples and Metaphors:**
    #    - To ensure relevance to Taiwanese readers, substitute or supplement book examples rooted in American or Western culture (e.g., A&amp;W's "Third-pounder burger," eBay bidding, or references to movies like "The Matrix" and series like "Star Trek") with scenarios, popular culture, or societal phenomena familiar to Taiwanese readers.
    #    - For example, when explaining psychological concepts, use Taiwanese festival activities, well-known landmarks, or popular foods to provide metaphors that make abstract concepts tangible, enhancing reader engagement and resonating with Taiwanese readers.
    #
    # 3. **Preserve the Author's Tone and Depth:**
    #    - **Insight and Explanatory Power:** Your translation should go beyond mere wording and deeply convey the author's explanations, details, and insights about psychological concepts. Understand the "reasons" behind each psychological phenomenon, such as the evolutionary adaptive purpose of disgust, and present it with clear logic.
    #    - **Conversational and Engaging:** The original text is smooth and often comes with a direct conversational tone with the reader. Recreate this warmth and guiding nature in your translation, making readers feel like they are listening to an interesting instructor lecture.
    #    - **Humor and Inspirational Qualities:** The author occasionally interjects humor, such as dialogues about "conscious bodies" or satirical news from "The Onion." Attempt to capture and convey this humor in your translation, keeping readers delighted and thoughtful.
    #    - **Acknowledgment of Complexity and Controversy:** The author candidly notes psychology's limitations and unresolved mysteries, such as the "hard problem" of consciousness or replication crisis. Maintain this honesty and openness, not shying away from the complexity or ongoing debates within the discipline.
    #
    # 4. **Accurately Convey the Core Theme and Author Intent:**
    #    - Even when interpreting with your own words, ensure that the translated content fully aligns with the author's "theme and meaning." Translators must deeply comprehend psychological concepts to maintain academic rigor and accuracy during cultural transformation.
    #    - Capture the deeper messages, tone, and implications the author tries to convey beyond the surface meaning of words (such as pragmatic examples like the subtext of "busy"), ensuring readers accurately understand the author's stance and perspective."""

    #     prompt = """Translate "The First Rule of Mastery" into a version tailored for Taiwanese readers, focusing on conversational yet professional language that preserves the author's intent and essence. Follow these style points:
    #
    #     You are tasked with translating an <slang> e-book passage into <tlang>,
    #
    # - **Direct and Clear, Conversational yet Professional**: Avoid stiff literal translations and adopt a natural tone that resonates with Taiwanese readers. Use storytelling techniques to illustrate abstract concepts with real-life cases.
    #
    # - **Concise and Impactful**: Craft the translation to be textually refined, conveying core meaning without verbosity. Aim for a subtle force that inspires and provokes reflection and action, similar to gentle, insightful guidance.
    #
    # - **Practical and Action-Oriented**: Highlight actionable advice, focusing on problem-solving and promoting proactive engagement. Ensure exercises in the text are clearly explained and immediately applicable.
    #
    # - **Relatable and Localized Taiwanese Feel**: Adapt scenarios and terminology to be relatable to Taiwanese culture, using locally familiar expressions while maintaining original intent.
    #
    # - **Encouraging Growth and Hope**: Maintain a positive, empowering tone, akin to a trusted mentor offering strength and encouragement for overcoming self-limitations.
    #
    # Adhere to these guidelines to produce a translation that Taiwanese readers find engaging, relatable, and empowering, capturing both the precision and spirit of the original work."""
    prompt = """你是一位專業翻譯者，正在針對一本書進行翻譯，使其符合台灣讀者的習慣。翻譯時應該：

    You are tasked with translating an <slang> e-book passage into <tlang>,
聚焦於增強讀者的理解，而不僅僅是傳遞資訊。捕捉並詮釋作者的深層意義。
使用再詮釋與台灣當地化的隱喻來簡化複雜概念。
使譯文簡潔有力，並聚焦於書中的主要觀念。
譯文應引導並啟發讀者進行思考。
保持翻譯的務實性與彈性，確保概念解釋不過於僵硬且具有實用性。"""
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

    last_token_usage: dict[str, int] = {}

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

        self.last_token_usage = {"prompt": 0, "completion": 0, "total": 0}

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

            print(data)
            # --- 提取 Token 使用量 ---
            if "usageMetadata" in data:
                usage = data["usageMetadata"]
                self.last_token_usage = {
                    "prompt": usage.get("promptTokenCount", 0),
                    "completion": usage.get("candidatesTokenCount", 0),
                    "total": usage.get("totalTokenCount", 0),
                }
            else:
                # 如果沒有 usageMetadata，則重置
                self.last_token_usage = {"prompt": 0, "completion": 0, "total": 0}
            # --- 提取結束 ---

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
