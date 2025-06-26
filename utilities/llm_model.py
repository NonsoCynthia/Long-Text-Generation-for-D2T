import os, getpass
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Optional, Text, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())  # Load environment variables

# === Base Interface ===
class ModelBase:
    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        raise NotImplementedError

    def raw_model(self):
        raise NotImplementedError


# === Ollama Model ===
class OllamaModel(ModelBase):
    def __init__(self, model_name: str = "llama3.2", temperature: float = 1.0):
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm


# === OpenAI Model ===
class OpenAIModel(ModelBase):
    def __init__(self, model_name: str = "gpt-4", temperature: float = 1.0, api_key: Optional[str] = None):
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_API_KEY") or api_key
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_key)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm


# === Anthropic Model ===
class AnthropicModel(ModelBase):
    def __init__(self, model_name: str = "claude-3-5-sonnet-latest", temperature: float = 1.0, api_key: Optional[str] = None):
        from langchain_anthropic import ChatAnthropic
        claude_key = os.environ.get("ANTHROPIC_API_KEY") or api_key
        self.llm = ChatAnthropic(model=model_name, temperature=temperature, api_key=claude_key, max_tokens=4024)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm
    

# === Groq Model ===
class GroqModel(ModelBase):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 1.0, api_key: Optional[str] = None):
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY") or api_key
        os.environ["GROQ_API_KEY"] = groq_key
        self.llm = ChatGroq(model=model_name, temperature=temperature, api_key=groq_key)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm

# === aiXplain Model ===
class AiXplainModel(ModelBase):
    def __init__(self, model_id: str = "640b517694bf816d35a59125", temperature: float = 1.0, api_key: Optional[str] = None):
        from aixplain.factories import ModelFactory
        os.environ["TEAM_API_KEY"] = os.getenv("TEAM_API_KEY") or api_key
        self.llm = ModelFactory.get(model_id)
        self.temperature = temperature  # store if you need to use it in prompts

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm


# === HuggingFace Model ===
class HFModel(ModelBase):
    def __init__(
        self,
        model_name: str = "../finetune/llama_en_lora",
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = 1024,   # None → no length cap
        quant: str = "8bit",                    # 8bit | 4bit | none
        api_key: Optional[str] = None,
    ):
        from pathlib import Path
        import warnings
        import os

        # --- heavy imports are lazy here -----------------------------------
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from langchain_huggingface import HuggingFacePipeline

        try:
            from bitsandbytes import BitsAndBytesConfig
            _BNB = True
        except ImportError:
            _BNB = False

        self.chatlike = False   # ← NEW: to control prompt behavior

        # -------------------------------------------------------------------
        # ➊  Local LoRA adapter directory
        # -------------------------------------------------------------------
        local_path = Path(os.path.join("/home/cosuji/spinning-storage/cosuji/NLG_Exp/Long-Text-Generation-for-D2T/finetune", model_name)).expanduser().resolve()
        if local_path.is_dir():
            adapter_dir = local_path

            # tokenizer ships with the adapter
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)

            # detect backbone from adapter_config.json
            base_id = "meta-llama/Llama-2-13b-chat-hf"        # fallback
            cfg_file = adapter_dir / "adapter_config.json"
            if cfg_file.exists():
                import json, io
                with io.open(cfg_file, "r", encoding="utf-8") as fh:
                    base_id = json.load(fh)["base_model_name_or_path"]

            # choose precision
            quant = quant.lower()
            if quant == "none" or not _BNB:
                backbone = AutoModelForCausalLM.from_pretrained(
                    base_id, torch_dtype="auto", device_map="auto"
                )
            else:
                try:
                    if quant == "8bit":
                        qcfg = BitsAndBytesConfig(load_in_8bit=True)
                    elif quant == "4bit":
                        qcfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                    else:
                        raise ValueError("--quant must be 8bit | 4bit | none")
                    backbone = AutoModelForCausalLM.from_pretrained(
                        base_id, quantization_config=qcfg, device_map="auto"
                    )
                except RuntimeError as err:
                    warnings.warn(
                        f"bitsandbytes backend unavailable ({err}); falling back to fp16"
                    )
                    backbone = AutoModelForCausalLM.from_pretrained(
                        base_id, torch_dtype="auto", device_map="auto"
                    )

            model = PeftModel.from_pretrained(backbone, adapter_dir)

            # kwargs let us omit max_new_tokens when you want “unlimited”
            pipe_kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
            if max_new_tokens is not None:
                pipe_kwargs["max_new_tokens"] = max_new_tokens

            pipe = pipeline("text-generation", **pipe_kwargs)
            self.llm = HuggingFacePipeline(pipeline=pipe)
            # self.llm = pipeline("text-generation", **pipe_kwargs)

        # -------------------------------------------------------------------
        # ➋  Plain HF Hub model
        # -------------------------------------------------------------------
        else:
            from langchain_huggingface import ChatHuggingFace
            hf_token = os.getenv("HF_TOKEN") or api_key
            self.llm = ChatHuggingFace(
                                llm=model_name,
                                temperature=temperature,
                                huggingfacehub_api_token=hf_token
                            )

            self.chatlike = True   # ← will need ChatPromptTemplate

    def model_(self, agent_prompts: Optional[Text]):
        if self.chatlike:
            prompt = ChatPromptTemplate.from_messages([
                ("system", agent_prompts),
                ("human", "{input}")
            ])
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=f"{agent_prompts}\n\n{{input}}"
            )
        return prompt | self.llm

    def raw_model(self):
        return self.llm



# === Unified Factory ===
class UnifiedModel:
    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()

        if provider == "ollama":
            self.model = OllamaModel(**kwargs)

        elif provider == "openai":
            kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
            if not kwargs["api_key"]:
                raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass `api_key`.")
            self.model = OpenAIModel(**kwargs)
            
        elif provider == "anthropic":
            kwargs.setdefault("api_key", os.getenv("ANTHROPIC_API_KEY"))
            if not kwargs["api_key"]:
                raise ValueError("ANTHROPIC_API_KEY not found. Set it in .env or pass `api_key`.")
            self.model = AnthropicModel(**kwargs)

        elif provider == "groq":
            kwargs.setdefault("api_key", os.getenv("GROQ_API_KEY"))
            if not kwargs["api_key"]:
                raise ValueError("GROQ_API_KEY not found. Set it in .env or pass `api_key`.")
            self.model = GroqModel(**kwargs)

        elif provider in ["hf", "huggingface"]:
            # kwargs.setdefault("hf_token", os.getenv("HF_TOKEN"))
            self.model = HFModel(**kwargs)

        elif provider == "aixplain":
            kwargs.setdefault("model_id", "640b517694bf816d35a59125")
            self.model = AiXplainModel(**kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def model_(self, agent_prompts: Optional[Text]):
        return self.model.model_(agent_prompts)

    def raw_model(self):
        return self.model.raw_model()


model_name = {
    "ollama": {"model_name": "llama3.2", "temperature": 1.0},
    "openai": {"model_name": "gpt-4.1", "temperature": 1.0},
    "anthropic": {"model_name": "claude-3-haiku-latest", "temperature": 1.0},
    "groq": {"model_name": "deepseek-r1-distill-llama-70b", "temperature": 1.0},
    "hf": {"model_name": "../finetune/llama_en_lora", "max_new_tokens": None, "quant": "8bit", "temperature": 1.0},
    "aixplain": {"model_id": "640b517694bf816d35a59125", "temperature": 1.0},
}#.get(provider.lower())
