#!/usr/bin/env python3
"""
Fine-tuning trainer using Unsloth for ultra-fast training with GPT-OSS 20B
Dataset: chizhikchi/CARES from HuggingFace
"""

import os
import sys
import types

# -----------------------------------------------------------------------------
# Compatibility shim
# Torch 2.4+ may not expose `config` as an attribute on `torch._inductor` even
# though `torch._inductor.config` is importable. Some Unsloth deps (e.g.
# `unsloth_zoo`) expect `torch._inductor.config` attribute to exist.
# -----------------------------------------------------------------------------
try:
    import importlib
    import torch  # noqa: F401

    _inductor = importlib.import_module("torch._inductor")
    if not hasattr(_inductor, "config"):
        _inductor.config = importlib.import_module("torch._inductor.config")
except Exception:
    # If Torch/Inductor isn't available, leave as-is.
    pass

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset
from huggingface_hub import login, HfFolder
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# Add local paths for utilities
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils import build_cwlc_dataset_from_zip, build_radgraph_dataset_from_jsonl
except Exception:
    build_cwlc_dataset_from_zip = None
    build_radgraph_dataset_from_jsonl = None




def setup_logging():
    """Configure logging with colors if available"""
    
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()


def authenticate_huggingface():
    """Authenticate with HuggingFace Hub"""
    hf_token = None
    
    # 1. Try environment variable
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # 2. Try .env file
    if not hf_token:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
    
    # 3. Try saved token
    if not hf_token:
        try:
            hf_token = HfFolder.get_token()
        except:
            pass
    
    # 4. Attempt login
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            logger.info("✓ HuggingFace authentication successful")
            return True
        except Exception as e:
            logger.warning(f"Could not authenticate with provided token: {e}")
    else:
        logger.info("No HuggingFace token found")
        logger.info("To use private models or avoid rate limits, set HF_TOKEN:")
        logger.info("  export HF_TOKEN='your_token_here'")
        logger.info("  or create a .env file with: HF_TOKEN=your_token_here")
        logger.info("Continuing without authentication...")
    
    return False


class UnslothFineTuner:
    """Fine-tuning class for GPT-OSS models with Unsloth"""
    
    def __init__(
        self,
        model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        dataset_name: str = "chizhikchi/CARES",
        output_dir: str = "./output",
        max_seq_length: int = 1024,
        batch_size: int = 2,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        hf_token: Optional[str] = None,
        reasoning_effort: str = "medium",
        cwlc_zip: Optional[str] = None,
        radgraph_jsonl: Optional[List[str]] = None,
        mimic_json: Optional[str] = None,
        train_pct: int = 100,
        resume_adapter: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.reasoning_effort = reasoning_effort
        self.cwlc_zip = cwlc_zip
        self.radgraph_jsonl = radgraph_jsonl
        self.mimic_json = mimic_json
        self.train_pct = max(1, min(100, train_pct))
        self.resume_adapter = resume_adapter
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Authenticate with HuggingFace if token provided
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        authenticate_huggingface()
        
        # Configure device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_dataset_from_hf(self) -> Dict:
        """Load dataset from HuggingFace or local CWLC ZIP, RadGraph JSONL, or MIMIC JSON"""
        if self.mimic_json:
            logger.info(f"Loading MIMIC dataset from JSON: {self.mimic_json}")
            mimic_path = Path(self.mimic_json)
            if not mimic_path.exists():
                raise FileNotFoundError(f"MIMIC JSON file not found: {self.mimic_json}")
            with open(mimic_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            logger.info(f"MIMIC dataset loaded: {len(records)} examples")
            from datasets import Dataset as HFDataset
            ds = HFDataset.from_list(records)
            logger.info(f"Columns: {ds.column_names}")
            return {"train": ds}

        if self.radgraph_jsonl:
            if build_radgraph_dataset_from_jsonl is None:
                raise RuntimeError("utils.build_radgraph_dataset_from_jsonl is not available")
            logger.info(f"Building RadGraph dataset from JSONL files: {self.radgraph_jsonl}")
            ds = build_radgraph_dataset_from_jsonl(self.radgraph_jsonl)
            logger.info(f"RadGraph dataset built: {len(ds)} examples")
            logger.info(f"Columns: {ds.column_names}")
            return {"train": ds}

        if self.cwlc_zip:
            if build_cwlc_dataset_from_zip is None:
                raise RuntimeError("utils.build_cwlc_dataset_from_zip is not available")
            logger.info(f"Building CWLC dataset from ZIP: {self.cwlc_zip}")
            ds = build_cwlc_dataset_from_zip(self.cwlc_zip)
            logger.info(f"CWLC dataset built: {len(ds)} examples")
            logger.info(f"Columns: {ds.column_names}")
            return {"train": ds}
        
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Available splits: {list(dataset.keys())}")
            
            # Show dataset information
            for split_name, split_data in dataset.items():
                logger.info(f"{split_name}: {len(split_data)} examples")
                if len(split_data) > 0:
                    logger.info(f"Columns: {split_data.column_names}")
            
            return dataset
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def convert_cares_to_chat_format(self, dataset):
        """Convert CARES dataset to chat format for GPT-OSS"""
        logger.info("Converting CARES dataset to chat format...")
        
        def format_cares_example(example):
            """
            Convert CARES example to GPT-OSS Harmony format:
            Input: ICD-10 codes
            Output: Clinical note with proper channel specification
            """
            # Get ICD-10 codes and clinical text
            icd10_sub_codes = example.get('icd10', [])
            clinical_text = example.get('full_text', '')
            icd10_codes = [code.split('.')[0] for code in icd10_sub_codes]
            # Remove 10 or fewer repeated icd10 codes (are in /home/lmiranda/ehr-synthetic-bmc-2026/data/cares_icd10_repetitions.txt)
            # en este txt los datos vienen por ejemplo como
            # E11.9 : 13
            # I10 : 9
            # .. 
            # por lo que se entiende que si la cantidad es <=10 se elimina
            with open(Path(__file__).parent.parent / "data" / "cares_icd10_repetitions.txt", "r") as f:
                lines = f.readlines()
                codes_to_remove = set()
                for line in lines:
                    parts = line.split(":")
                    if len(parts) == 2:
                        code = parts[0].strip()
                        count = int(parts[1].strip())
                        if count < 10:
                            codes_to_remove.add(code)

            icd10_codes = [code for code in icd10_codes if code not in codes_to_remove]
            # Incluye la descripcion de los codigos ICD-10 que estan en /home/lmiranda/ehr-synthetic-bmc-2026/data/cares_icd10_descriptions_es.json
            # el formato de este json es
            # {
            #   "E11.9": "Diabetes mellitus tipo 2 sin complicaciones",
            #   "I10": "Hipertensión esencial (primaria)",
            #   ...
            # }
            with open(Path(__file__).parent.parent / "data" / "cares_icd10_descriptions_es.json", "r") as f:
                icd10_descriptions = json.load(f)
            icd10_codes_with_desc = []
            for code in icd10_codes:
                description = icd10_descriptions.get(code, "Descripción no disponible")
                icd10_codes_with_desc.append(f"{code} ({description})")
            icd10_codes = icd10_codes_with_desc
            #
            ## If text does not contain icd10 codes, ignore example
            #if not any(code.split(" ")[0] in clinical_text for code in icd10_codes):
            #    logger.warning(f"Example ignored: clinical text does not contain any of the ICD-10 codes")
            #    return {"messages": []}


            # Convert ICD-10 list to comma-separated string
            if isinstance(icd10_codes, list):
                icd10_str = ', '.join(icd10_codes)
            else:
                icd10_str = str(icd10_codes)
            
            # Create chat format messages using GPT-OSS format
            messages = [
                {
                    "role": "developer",
                    "content": f"reasoning language: Spanish\n\nYou are a medical assistant that generates clinical notes from ICD-10 diagnosis codes."
                },
                {
                    "role": "user",
                    "content": f"Generate a detailed clinical note for the following ICD-10 codes: {icd10_str}"
                },
                {
                    "role": "assistant",
                    "content": clinical_text
                }
            ]
            
            return {"messages": messages}
        
        # Apply conversion to all splits
        converted_dataset = {}
        for split_name, split_data in dataset.items():
            logger.info(f"Processing split: {split_name}")
            converted_dataset[split_name] = split_data.map(
                format_cares_example,
                desc=f"Converting {split_name} to chat format"
            )
        
        logger.info("Dataset conversion completed")
        return converted_dataset
    
    def convert_cantemist_to_chat_format(self, dataset):
        """Convert CANTEMIST-like SMC dataset (somosnlp/SMC) to chat format.
        - Filtra por source == "1" (según especificación del usuario).
        - Agrupa por document_id para reunir todos los tópicos (CIE-O 3) de ese documento.
        - Crea mensajes en formato GPT-OSS donde el asistente es un experto oncológico que genera la nota clínica
          a partir de los tópicos CIE-O-3, y el contenido del asistente es el `raw_text`.
        """
        logger.info("Converting SMC (CANTEMIST-like) dataset to chat format...")

        def build_grouped_dataset(ds_split):
            # Filtrar por source == "1"
            try:
                filtered = ds_split.filter(lambda x: str(x.get("source")) == "1")
            except Exception as e:
                logger.warning(f"Could not filter by source == '1': {e}. Proceeding without filter.")
                filtered = ds_split

            # Convertir a pandas para agrupar por document_id
            try:
                df = filtered.to_pandas()
            except Exception as e:
                logger.error(f"Failed to convert split to pandas: {e}")
                raise

            # Asegurar columnas esperadas
            required_cols = [
                "raw_text", "topic", "document_id", "source", "country",
                "topic_type", "raw_text_type", "speciallity"
            ]
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in split. Detected columns: {list(df.columns)}")

            # Agrupar por document_id: reunir topics (lista) y elegir un raw_text representativo (primero no nulo)
            def agg_raw_text(series):
                for v in series:
                    if isinstance(v, str) and len(v.strip()) > 0:
                        return v
                return series.iloc[0] if len(series) > 0 else ""

            grouped = df.groupby("document_id", dropna=False).agg({
                "topic": lambda s: [t for t in s if isinstance(t, str)],
                "raw_text": agg_raw_text,
                # Retener metadatos útiles (tomar el primer valor)
                "source": "first",
                "country": "first",
                "topic_type": "first",
                "raw_text_type": "first",
                "speciallity": "first",
            }).reset_index()

            # Construir mensajes por fila agrupada
            def build_messages(row):
                topics = row.get("topic", [])
                topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
                clinical_text = row.get("raw_text", "")

                messages = [
                    {
                        "role": "developer",
                        "content": (
                            "reasoning language: Spanish\n\n"
                            "Eres un experto médico oncológico. Genera una nota clínica detallada y coherente a partir de los tópicos CIE-O-3 suministrados."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Genera una nota clínica basada en los siguientes tópicos CIE-O-3 (del mismo documento): {topics_str}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": clinical_text,
                    },
                ]
                return {"messages": messages}

            grouped["messages"] = grouped.apply(build_messages, axis=1)
            # Reducir a solo la columna messages para el entrenamiento
            out_df = grouped[["messages"]]

            from datasets import Dataset
            return Dataset.from_pandas(out_df, preserve_index=False)

        # Aplicar conversión a cada split
        converted_dataset = {}
        for split_name, split_data in dataset.items():
            logger.info(f"Processing split: {split_name}")
            converted_dataset[split_name] = build_grouped_dataset(split_data)

        logger.info("SMC dataset conversion to chat format completed")
        return converted_dataset

    def convert_cwlc_to_chat_format(self, dataset: Dict[str, Any]):
        """Convert CWLC dataset (with columns text, tags) to chat format.
        User message: "A partir de las siguiente Entidades Médicas [ (entidad, valor), ... ] debes redactar una nota clínica que las contenga"
        Assistant message: texto clínico (columna 'text').
        """
        logger.info("Converting CWLC dataset to chat format...")

        def to_messages(example: Dict):
            tags = example.get("tags", []) or []
            # tags expected as list of [label, value]
            pairs = []
            for item in tags:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pairs.append(f"({item[0]}, {item[1]})")
                except Exception:
                    continue
            pairs_str = ", ".join(pairs)
            text = example.get("text", "")
            messages = [
                {
                    "role": "user",
                    "content": (
                        "A partir de las siguiente Entidades Médicas [" + pairs_str + 
                        "] debes redactar una nota clínica que las contenga"
                    ),
                },
                {
                    "role": "assistant",
                    "content": text,
                },
            ]
            return {"messages": messages}

        converted: Dict[str, Any] = {}
        for split_name, split_data in dataset.items():
            converted[split_name] = split_data.map(
                to_messages,
                desc=f"Converting {split_name} CWLC to chat format",
            )
        logger.info("CWLC conversion completed")
        return converted

    def convert_radgraph_to_chat_format(self, dataset: Dict[str, Any]):
        """Convert RadGraph-XL dataset to chat format for GPT-OSS.
        
        Uses NER entities as input prompt and full radiology report as target output.
        Input: List of NER entities with their labels (Anatomy, Observation, etc.)
        Output: Full radiology report text
        """
        logger.info("Converting RadGraph-XL dataset to chat format...")

        def to_messages(example: Dict):
            ner_entities = example.get("ner_entities", []) or []
            text = example.get("text", "")
            
            # Build entity pairs from NER data
            # Format: (label, entity_text) pairs
            pairs = []
            for entity in ner_entities:
                if isinstance(entity, dict):
                    label = entity.get("label", "Unknown")
                    entity_text = entity.get("text", "")
                    if entity_text:
                        pairs.append(f"({label}, {entity_text})")
            
            pairs_str = ", ".join(pairs)
            
            # Create chat format messages
            messages = [
                {
                    "role": "developer",
                    "content": (
                        "reasoning language: English\n\n"
                        "You are a radiology expert. Generate a detailed radiology report from the given medical entities. "
                        "The entities include anatomical structures and clinical observations with their presence status."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate a complete radiology report based on the following medical entities: [{pairs_str}]"
                    ),
                },
                {
                    "role": "assistant",
                    "content": text,
                },
            ]
            return {"messages": messages}

        converted: Dict[str, Any] = {}
        for split_name, split_data in dataset.items():
            converted[split_name] = split_data.map(
                to_messages,
                desc=f"Converting {split_name} RadGraph to chat format",
            )
        logger.info("RadGraph conversion completed")
        return converted

    def convert_mimic_to_chat_format(self, dataset: Dict[str, Any]):
        """Convert MIMIC Alpaca-format dataset (instruction/input/output) to chat format."""
        logger.info("Converting MIMIC dataset to chat format...")

        def to_messages(example: Dict):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output_text = example.get("output", "")

            # Build user prompt combining instruction and input
            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"
            else:
                user_content = instruction

            messages = [
                {
                    "role": "developer",
                    "content": (
                        "reasoning language: English\n\n"
                        "You are a medical expert that generates clinical discharge summaries "
                        "from procedure and diagnosis code descriptions."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": output_text,
                },
            ]
            return {"messages": messages}

        converted: Dict[str, Any] = {}
        for split_name, split_data in dataset.items():
            converted[split_name] = split_data.map(
                to_messages,
                desc=f"Converting {split_name} MIMIC to chat format",
            )
        logger.info("MIMIC conversion completed")
        return converted

    def prepare_model_and_tokenizer(self):
        """Load GPT-OSS model with Unsloth optimizations"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info("Using Unsloth for 2x faster training!")
        
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Load model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=True,  # Use 4-bit quantization
            device_map={'': local_rank},
        )
        
        if self.resume_adapter:
            # Resume from existing LoRA adapter (continued finetuning)
            logger.info(f"Resuming from existing adapter: {self.resume_adapter}")
            from peft import PeftModel
            # Load on the current local rank device assigned by torchrun
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = PeftModel.from_pretrained(
                self.model,
                self.resume_adapter,
                is_trainable=True,
                device_map={'': local_rank},
            )
            # Enable gradient checkpointing for memory savings
            self.model.enable_input_require_grads()
            logger.info("Existing adapter loaded — continuing training")
        else:
            # Apply new LoRA with Unsloth
            logger.info(f"Configuring LoRA (r={self.lora_r}, alpha={self.lora_alpha})...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_r,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=self.lora_alpha,
                lora_dropout=0,  # Optimized
                bias="none",  # Optimized
                use_gradient_checkpointing="unsloth",  # 30% less VRAM!
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        
        logger.info("Model prepared successfully with Unsloth optimizations")
    
    def format_prompts(self, examples):
        """Format examples using GPT-OSS chat template"""
        convos = examples["messages"]

        # Aplicar chat template con razonamiento, soportando filas donde
        # cada elemento puede ser una lista de mensajes o un dict {"messages": [...]}
        # Además, proteger contra filas vacías / mal formadas.
        texts = []
        for convo in convos:
            try:
                msgs = convo["messages"] if isinstance(convo, dict) and "messages" in convo else convo

                # `apply_chat_template` asume que `msgs` es una lista no vacía.
                if not isinstance(msgs, list) or len(msgs) == 0:
                    texts.append("")
                    continue

                text = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    reasoning_effort=self.reasoning_effort,
                )
                texts.append(text)
            except Exception as e:
                logger.warning(f"Skipping malformed conversation in format_prompts: {e}")
                texts.append("")

        return {"text": texts}

    def train(self):
        """Execute fine-tuning process with Unsloth"""
        logger.info("="*70)
        logger.info("STARTING FINE-TUNING WITH UNSLOTH + GPT-OSS 20B")
        logger.info("="*70)

        # 1. Load dataset
        step1_name = "Loading dataset"
        logger.info("\n" + "="*70)
        logger.info(f"STEP 1: {step1_name}")
        logger.info("="*70)
        dataset = self.load_dataset_from_hf()

        # 2. Convert to chat format
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Converting to chat format")
        logger.info("="*70)

        if self.mimic_json:
            dataset = self.convert_mimic_to_chat_format(dataset)
        elif self.radgraph_jsonl:
            dataset = self.convert_radgraph_to_chat_format(dataset)
        elif self.cwlc_zip:
            dataset = self.convert_cwlc_to_chat_format(dataset)
        elif self.dataset_name == "chizhikchi/CARES":
            dataset = self.convert_cares_to_chat_format(dataset)
        elif self.dataset_name == "somosnlp/SMC":
            dataset = self.convert_cantemist_to_chat_format(dataset)

        # Filter out empty / invalid conversations early to avoid chat-template errors.
        def _has_messages(example: Dict[str, Any]) -> bool:
            msgs = example.get("messages")
            if isinstance(msgs, dict) and "messages" in msgs:
                msgs = msgs.get("messages")
            return isinstance(msgs, list) and len(msgs) > 0

        try:
            dataset["train"] = dataset["train"].filter(_has_messages, desc="Filtering empty conversations")
        except Exception as e:
            logger.warning(f"Could not filter empty conversations: {e}")

        # If we filtered everything out, stop with a clear message.
        try:
            if len(dataset["train"]) == 0:
                raise RuntimeError(
                    "Training dataset is empty after filtering invalid/empty 'messages'. "
                    "This typically means your dataset mapping produced empty conversations."
                )
        except TypeError:
            # Some datasets might not support len() at this point; ignore.
            pass

        # Subsample training set if train_pct < 100
        if self.train_pct < 100:
            import random
            total = len(dataset["train"])
            n_samples = max(1, int(total * self.train_pct / 100))
            rng = random.Random(3407)
            indices = sorted(rng.sample(range(total), n_samples))
            dataset["train"] = dataset["train"].select(indices)
            logger.info(f"Subsampled training set: {n_samples}/{total} examples ({self.train_pct}%)")
        else:
            logger.info(f"Using 100% of training set: {len(dataset['train'])} examples")

        # 3. Prepare model and tokenizer
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Preparing model and tokenizer")
        logger.info("="*70)
        self.prepare_model_and_tokenizer()

        # 4. Format prompts
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Formatting prompts")
        logger.info("="*70)
        train_dataset = dataset["train"].map(
            self.format_prompts,
            batched=True,
            desc="Formatting prompts",
        )

        # Drop any rows that still ended up with empty text.
        try:
            train_dataset = train_dataset.filter(
                lambda x: isinstance(x.get("text"), str) and len(x["text"].strip()) > 0,
                desc="Filtering empty formatted texts",
            )
        except Exception as e:
            logger.warning(f"Could not filter empty formatted texts: {e}")

        # Unsloth's SFTTrainer will call next(iter(train_dataset)) during init.
        # Ensure we never pass an empty dataset to it.
        try:
            if len(train_dataset) == 0:
                raise RuntimeError(
                    "Training dataset is empty after prompt formatting. "
                    "Likely causes: empty 'messages', missing assistant content, or failed chat template."
                )
        except TypeError:
            pass

        # 5. Configure training
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Configuring training")
        logger.info("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_output_dir = self.output_dir / f"checkpoint_{timestamp}"
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            packing=False,  # Can make training 5x faster for short sequences
            args=SFTConfig(
                dataset_num_proc=2,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=str(training_output_dir),
                report_to="tensorboard",
                save_strategy="steps",
                save_steps=100,
                save_total_limit=3,
            ),
        )
        
        # Train only on assistant responses (ignore system and user messages)
        logger.info("Configuring training on responses only...")
        gpt_oss_kwargs = dict(
            instruction_part="<|start|>user<|message|>",
            response_part="<|start|>assistant<|message|>"
        )
        
        trainer = train_on_responses_only(
            trainer,
            **gpt_oss_kwargs,
        )
        
        # 6. Train
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Starting training")
        logger.info("="*70)
        
        # Show memory stats before training
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        trainer_stats = trainer.train()
        
        # Show memory stats after training
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING STATISTICS")
        logger.info("="*70)
        logger.info(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
        logger.info(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
        logger.info(f"Peak reserved memory: {used_memory} GB")
        logger.info(f"Peak reserved memory for training: {used_memory_for_lora} GB")
        logger.info(f"Peak reserved memory % of max: {used_percentage}%")
        logger.info(f"Peak reserved memory for training % of max: {lora_percentage}%")
        
        # 7. Save model
        logger.info("\n" + "="*70)
        logger.info("STEP 7: Saving model")
        logger.info("="*70)
        
        final_model_path = self.output_dir / "final_model"
        self.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save metrics
        metrics = trainer_stats.metrics
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("="*70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Model saved at: {final_model_path}")
        logger.info(f"Metrics saved at: {metrics_file}")
        logger.info(f"Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info("="*70)
        
        return final_model_path


def main():
    """Main function for CLI execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-OSS 20B with Unsloth on the CARES dataset"
    )
    
    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Model name (use Unsloth 4bit quantized version)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chizhikchi/CARES",
        help="Dataset name from HuggingFace"
    )
    
    # Training configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/gpt-oss-20b",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs if set)"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    # GPT-OSS specific
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for GPT-OSS"
    )
    parser.add_argument(
        "--cwlc-zip",
        type=str,
        default=None,
        help="Path to CWLC ZIP file (overrides --dataset when provided)"
    )
    parser.add_argument(
        "--radgraph-jsonl",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to RadGraph-XL JSONL files (overrides --dataset when provided)"
    )
    parser.add_argument(
        "--mimic-json",
        type=str,
        default=None,
        help="Path to MIMIC Alpaca-format JSON file (overrides --dataset when provided)"
    )
    parser.add_argument(
        "--train-pct",
        type=int,
        default=100,
        help="Percentage of training set to use (1-100, default: 100)"
    )
    parser.add_argument(
        "--resume-from-adapter",
        type=str,
        default=None,
        help="Path to existing LoRA adapter directory for continued finetuning (e.g. output/CWLC/gpt-oss-20b_100pct_.../final_model)"
    )
    
    # Testing options
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (100 samples, 30 steps)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("FINE-TUNING GPT-OSS 20B WITH UNSLOTH")
    logger.info("="*70)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Dataset: {args.dataset}")
    if args.cwlc_zip:
        logger.info(f"  CWLC ZIP: {args.cwlc_zip}")
    if args.radgraph_jsonl:
        logger.info(f"  RadGraph JSONL: {args.radgraph_jsonl}")
    if args.mimic_json:
        logger.info(f"  MIMIC JSON: {args.mimic_json}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Train %: {args.train_pct}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Max length: {args.max_seq_length}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LoRA r: {args.lora_r}")
    logger.info(f"  LoRA alpha: {args.lora_alpha}")
    logger.info(f"  Reasoning effort: {args.reasoning_effort}")
    if args.max_steps:
        logger.info(f"  Max steps: {args.max_steps}")
    if args.test_mode:
        logger.info(f"  🧪 TEST MODE ENABLED")
    if args.resume_from_adapter:
        logger.info(f"  Resume adapter: {args.resume_from_adapter}")
    logger.info("")
    
    try:
        # Create fine-tuner
        fine_tuner = UnslothFineTuner(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs if not args.max_steps else 1,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            reasoning_effort=args.reasoning_effort,
            cwlc_zip=args.cwlc_zip,
            radgraph_jsonl=args.radgraph_jsonl,
            mimic_json=args.mimic_json,
            train_pct=args.train_pct,
            resume_adapter=args.resume_from_adapter,
        )
        
        # Train
        final_model_path = fine_tuner.train()
        
        logger.info("\n" + "="*70)
        logger.info("🎉 SUCCESS! Model training completed")
        logger.info("="*70)
        logger.info(f"\nNext steps:")
        logger.info(f"1. Generate text: python src/generator_unsloth.py --model-path {final_model_path}")
        logger.info(f"2. View TensorBoard: tensorboard --logdir {args.output_dir}/logs")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("="*70)
        logger.error(f"❌ ERROR DURING TRAINING: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
