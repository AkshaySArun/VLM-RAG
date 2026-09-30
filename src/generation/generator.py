import os
import base64
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


class MultimodalGenerator:
    def __init__(
        self,
        model_name: Optional[str] = None
    ):
        """
        Initializes the generator using Groq via LangChain.
        Configurable via environment variable LLM_MODEL.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.llm = None

        if not self.api_key:
            print("[!] Warning: GROQ_API_KEY not found in environment.")
        else:
            self._init_llm()

    def _init_llm(self):
        try:
            self.api_key = os.getenv("GROQ_API_KEY")
            self.model_name = os.getenv("LLM_MODEL", self.model_name or "openai/gpt-oss-120b")
            if self.api_key:
                self.llm = ChatGroq(
                    model=self.model_name,
                    groq_api_key=self.api_key,
                    temperature=0.1
                )
                print(f"[+] Generator initialized for Groq model: {self.model_name}")
        except Exception as e:
            print(f"[!] Failed to initialize ChatGroq with model {self.model_name}: {e}")
            self.llm = None

    def generate_answer(
        self,
        query: str,
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer using retrieved text, table and image/OCR context.
        """
        print(f"[*] Generating answer with Groq (Model: {self.model_name}) for query: '{query}'")

        text_context = []
        source_refs = []

        for item in context_items:
            metadata = item.get("metadata", {})
            source_refs.append(metadata)

            content_type = metadata.get("content_type", "text")
            source = metadata.get("filename") or metadata.get("source", "Uploaded Paper")
            page = metadata.get("page_number", "N/A")

            if content_type == "text":
                content = item.get("content", "")
                text_context.append(
                    f"TEXT SOURCE\nSource: {source}\nPage: {page}\nContent:\n{content}"
                )
            elif content_type == "table":
                content = item.get("content", "")
                text_context.append(
                    f"TABLE SOURCE\nSource: {source}\nPage: {page}\nTable Data:\n{content}"
                )
            elif content_type == "image":
                ocr_text = metadata.get("ocr_text", "")
                if not ocr_text:
                    ocr_text = "No readable text detected in image/figure."
                image_path = metadata.get("image_path", "")
                text_context.append(
                    f"FIGURE/IMAGE SOURCE\nSource: {source}\nPage: {page}\nOCR Text:\n{ocr_text}\nImage Path: {image_path}"
                )

        if not text_context:
            return {
                "answer": "The paper does not contain enough information to answer this question because no relevant context was found.",
                "sources": []
            }

        context_str = "\n\n-------------------\n\n".join(text_context)

        system_prompt = SystemMessage(
            content="""You are a world-class AI Research Paper Assistant.
Your job is to answer user questions grounded ONLY in the retrieved paper context provided below.

RULES FOR ANSWERING:
1. GROUNDED IN CONTEXT: Prioritize the retrieved paper context above all else. Use specific details, numerical values, equations, architecture descriptions, and experimental results from the context.
2. CITATIONS: State the exact page number(s) and source filename for your statements (e.g., "According to page 3...", "[Page 4]").
3. FIGURES & TABLES: Refer to extracted OCR text or table structures whenever relevant.
4. MATHEMATICAL FORMULAS: Render math and formulas using LaTeX notation where appropriate.
5. NO HALLUCINATION / UNFOUNDED CLAIMS: If the provided context does NOT contain enough information to answer the question, explicitly state: "The provided paper does not contain information about..." Do not invent facts not supported by the paper.
6. DOMAIN FOCUS: Provide clear, technical, structured markdown answers.
"""
        )

        human_prompt = f"""RETRIEVED PAPER CONTEXT:
=======================
{context_str}

USER QUESTION:
==============
{query}

INSTRUCTIONS:
Provide a clear, grounded answer citing page numbers and figures where relevant. If the context does not contain the answer, state that clearly.

Answer:"""

        messages = [system_prompt, HumanMessage(content=human_prompt)]

        # Ensure LLM instance is up-to-date
        if not self.llm:
            self._init_llm()

        if not self.api_key:
            return {
                "answer": "Error: GROQ_API_KEY environment variable is missing. Please set it in your .env file.",
                "sources": source_refs
            }

        if not self.llm:
            return {
                "answer": f"Error: Could not initialize Groq LLM model '{self.model_name}'. Check your model configuration and GROQ_API_KEY.",
                "sources": source_refs
            }

        try:
            response = self.llm.invoke(messages)
            return {
                "answer": response.content,
                "sources": source_refs
            }
        except Exception as e:
            print(f"[!] Groq LLM generation error: {e}")
            return {
                "answer": f"Generation Error (Model: {self.model_name}): {str(e)}",
                "sources": source_refs
            }

if __name__ == "__main__":
    generator = MultimodalGenerator()
    print("MultimodalGenerator module loaded.")