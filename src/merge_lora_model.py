#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for vLLM inference.

This script loads a LoRA adapter fine-tuned with Unsloth/PEFT and merges it
with the base model, saving the result in FP16 format compatible with vLLM.

Usage:
    python src/merge_lora_model.py \
        --lora-path output/gpt-oss-20b-mimic-small_20260210_164243/final_model \
        --output-path output/gpt-oss-20b-mimic-merged \
        --base-model unsloth/gpt-oss-20b

Note: The base model should be the FP16 version, not the 4-bit quantized one.
If the original was trained with unsloth/model-bnb-4bit, use unsloth/model instead.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for vLLM"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for merged model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (auto-detected from adapter_config.json if not specified)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype for merged model"
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="10GB",
        help="Maximum shard size for saved model"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "cpu", "sequential"],
        help="Device map for loading model. Use 'cpu' if running out of GPU memory"
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Use low CPU memory mode (slower but uses less RAM)"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Validate inputs
    # =========================================================================
    
    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        print(f"❌ ERROR: LoRA path not found: {lora_path}")
        sys.exit(1)
    
    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"❌ ERROR: adapter_config.json not found in {lora_path}")
        print("   This does not appear to be a LoRA adapter directory")
        sys.exit(1)
    
    # =========================================================================
    # Detect base model
    # =========================================================================
    
    print("=" * 70)
    print("LoRA Merge for vLLM")
    print("=" * 70)
    print(f"LoRA path: {lora_path}")
    print(f"Output: {args.output_path}")
    
    if args.base_model:
        base_model = args.base_model
    else:
        # Read from adapter_config.json
        with open(adapter_config_path) as f:
            # Handle JSON with comments (some configs have them)
            content = f.read()
            # Remove single-line comments
            import re
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            adapter_config = json.loads(content)
        
        base_model = adapter_config.get("base_model_name_or_path", "")
        
        if not base_model:
            print("❌ ERROR: Could not detect base model from adapter_config.json")
            print("   Please specify --base-model manually")
            sys.exit(1)
        
        # Convert 4-bit model references to FP16 equivalents
        # Example: unsloth/gpt-oss-20b-unsloth-bnb-4bit -> unsloth/gpt-oss-20b
        if "-bnb-4bit" in base_model or "-4bit" in base_model:
            original = base_model
            base_model = base_model.replace("-unsloth-bnb-4bit", "")
            base_model = base_model.replace("-bnb-4bit", "")
            base_model = base_model.replace("-4bit", "")
            print(f"⚠️  Detected 4-bit model: {original}")
            print(f"   Converting to FP16: {base_model}")
    
    print(f"Base model: {base_model}")
    print(f"Output dtype: {args.dtype}")
    print(f"Device map: {args.device_map}")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Load model and merge
    # =========================================================================
    
    print("1. Loading base model...")
    if args.device_map == "cpu":
        print("   ⚠️ Using CPU - this will be slow but avoids GPU OOM")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load base model in target dtype
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    
    print(f"   ✓ Loaded base model: {base_model}")
    print(f"   ✓ Parameters: {model.num_parameters():,}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(lora_path),  # Use tokenizer from LoRA if available
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # =========================================================================
    # Load and merge LoRA
    # =========================================================================
    
    print("\n2. Loading LoRA adapter...")
    
    model = PeftModel.from_pretrained(
        model,
        str(lora_path),
        torch_dtype=torch_dtype,
    )
    
    print(f"   ✓ Loaded LoRA from: {lora_path}")
    
    print("\n3. Merging LoRA weights...")
    
    model = model.merge_and_unload()
    
    print("   ✓ LoRA weights merged")
    
    # =========================================================================
    # Save merged model
    # =========================================================================
    
    print(f"\n4. Saving merged model to {args.output_path}...")
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(
        str(output_path),
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )
    
    tokenizer.save_pretrained(str(output_path))
    
    print(f"   ✓ Model saved")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("✅ Merge Complete")
    print("=" * 70)
    print(f"Merged model saved to: {args.output_path}")
    print()
    print("You can now use this model with vLLM:")
    print(f"  --model-path {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
