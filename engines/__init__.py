from .google import (
    GoogleFreeTranslateNew, GoogleFreeTranslateHtml, GoogleFreeTranslate,
    GoogleBasicTranslate, GoogleBasicTranslateADC, GoogleAdvancedTranslate,
    GeminiTranslate)
from .openai import ChatgptTranslate
from .anthropic import ClaudeTranslate
from .deepl import DeeplTranslate, DeeplProTranslate, DeeplFreeTranslate
from .youdao import YoudaoTranslate
from .baidu import BaiduTranslate
from .microsoft import MicrosoftEdgeTranslate, AzureChatgptTranslate
from .deepseek import DeepseekTranslate

from .vertexai import VertexAITranslate


builtin_engines = (
    GoogleFreeTranslateNew, GoogleFreeTranslateHtml, GoogleFreeTranslate,
    GoogleBasicTranslate, GoogleBasicTranslateADC, GoogleAdvancedTranslate,
    ChatgptTranslate, AzureChatgptTranslate, GeminiTranslate, ClaudeTranslate,
    DeepseekTranslate, DeeplTranslate, DeeplProTranslate, DeeplFreeTranslate,
    MicrosoftEdgeTranslate, YoudaoTranslate, BaiduTranslate, VertexAITranslate)
