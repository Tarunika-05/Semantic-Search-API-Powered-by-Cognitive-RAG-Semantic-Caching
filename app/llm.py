import logging
import google.generativeai as genai

# Configure logger
logger = logging.getLogger(__name__)

class LLMProvider:
    def generate(self, prompt: str) -> str | None:
        raise NotImplementedError("Subclasses must implement generate()")


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, endpoint: str | None = None, deployment: str | None = None):
        from app.config import settings
        self.api_key = api_key or settings.azure_openai_api_key
        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.deployment = deployment or settings.azure_openai_model_name
        self.api_version = settings.azure_openai_api_version
        
        if not self.api_key or not self.endpoint:
            logger.warning("Missing Azure OpenAI credentials. LLM generation will fail.")

    def generate(self, prompt: str) -> str | None:
        if not self.api_key or not self.endpoint:
            return None
            
        import urllib.request
        import json
        
        # Format: {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api-version}
        base_url = self.endpoint.rstrip("/")
        url = f"{base_url}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return str(content).strip() if content else None
        except Exception as e:
            logger.error(f"Error generating Azure OpenAI content: {e}")
            try:
                error_body = e.read().decode('utf-8')
            except:
                error_body = str(e)
            return f"AZURE ERROR: {error_body}"


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        from app.config import settings
        key = api_key or settings.gemini_api_key
        if not key:
            logger.warning("No GEMINI_API_KEY provided. LLM generation will be disabled.")
            self.model = None
            return

        try:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.model = None

    def generate(self, prompt: str) -> str | None:
        if not self.model:
            return None
        try:
            response = self.model.generate_content(prompt)
            return str(response.text).strip() if response.text else None
        except Exception as e:
            logger.error(f"Error generating LLM content: {e}")
            return None

class DummyProvider(LLMProvider):
    def generate(self, prompt: str) -> str | None:
        import re
        titles = re.findall(r"--- Document \[\d+\]: (.*?) ---", prompt)
        
        response = "### **[SIMULATED AI ANSWER]**\n\n"
        response += "Based on the retrieved academic papers, here is a synthesized answer to your query. *(Note: The Gemini API key hit a quota limit, so this is a beautifully formatted fallback simulation!)*\n\n"
        
        if titles:
            response += "### Key Discoveries\n"
            response += "The literature discusses several key concepts that address your query directly. "
            for i, t in enumerate(titles, 1):
                response += f"For instance, it is highlighted that **{t.split('(')[0].strip()}** offers significant improvements to the baseline metrics [{i}]. "
            
            response += "\n\n### Methodological Shifts\n"
            response += "Recent advancements show a clear shift towards more robust optimization strategies. By leveraging these techniques, models can achieve better convergence rates while maintaining high accuracy.\n\n"
            
            response += "### Conclusion\n"
            response += "In summary, the provided context strongly indicates that this methodology is highly effective for the specified machine learning tasks."
        else:
            response += "No relevant documents were found to synthesize an answer."
            
        return response

def sanitize_query(query: str) -> str:
    # Basic prompt injection guard: remove common injection patterns
    forbidden_phrases = ["ignore previous instructions", "system prompt", "you are a", "forget"]
    sanitized = query
    for phrase in forbidden_phrases:
        sanitized = sanitized.replace(phrase, "")
    return sanitized.strip()

def build_rag_prompt(query: str, retrieved_docs: list[str]) -> str:
    # Format retrieved docs with citations
    docs_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        # Extract title (assume first line is title)
        title = doc.split('\n')[0][:100]
        docs_text += f"--- Document [{i}]: {title} ---\n{doc}\n\n"
        
    sanitized_query = sanitize_query(query)
    prompt = f"""You are an expert AI research assistant. A user has asked the following technical query:
"{sanitized_query}"

Below are excerpts from relevant ArXiv Machine Learning research papers retrieved by our search engine:

{docs_text}
CRITICAL INSTRUCTIONS:
1. Synthesize a comprehensive, highly readable, and technical answer to the user's query using ONLY the provided excerpts.
2. DO NOT start your answer with phrases like "Based on the provided papers" or "According to the documents". Jump directly into the answer.
3. Group related concepts together into short, digestible paragraphs under Markdown headers (###).
4. Use bold text to highlight key terms.
5. You MUST include inline citations in the format [1], [2], etc., corresponding to the Document number. For example: "CNNs are highly effective at image recognition [2]."
6. Do NOT just list out bullet points summarizing each document.
7. If the excerpts do not contain enough information to fully answer the query, provide the best partial answer possible and briefly note the limitations of the context at the very end.
8. Do not hallucinate or use outside knowledge.
"""
    return prompt

def generate_answer(query: str, retrieved_docs: list[str], provider: LLMProvider | None = None) -> str | None:
    if not provider:
        provider = DummyProvider()
    
    prompt = build_rag_prompt(query, retrieved_docs)
    return provider.generate(prompt)
