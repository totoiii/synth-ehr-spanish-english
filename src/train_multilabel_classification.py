#!/usr/bin/env python3
"""train_multilabel_classification.py

Entrena un clasificador multi-label de códigos ICD-10 (o capítulos) sobre el corpus CARES.

Soporta 2 fuentes de entrenamiento:
- JSONL generado por `src/generate_all.py` (campos: icd10, original_text, generated_text)
- (Opcional) HF CARES train split (por defecto evaluamos siempre en test)

Evalúa en el split `test` del dataset CARES (HF) y reporta métricas multi-label.

Soporta dos tipos de etiquetas:
- `icd10`: 156 códigos ICD-10 individuales (más difícil)
- `chapters`: 16 capítulos ICD-10 (como en el paper CARES original, ~88% F1)

Ejemplos:
  # Usar chapters (16 clases) como el paper CARES original:
  python src/train_multilabel_classification.py \
    --train-jsonl output/generated_cares.jsonl \
    --train-text-source original \
    --label-type chapters \
    --model PlanTL-GOB-ES/bsc-bio-ehr-es \
    --output-dir output/icd10_clf/roberta_chapters \
    --num-train-epochs 30

  # Usar códigos ICD-10 individuales (156 clases):
  python src/train_multilabel_classification.py \
    --train-jsonl output/generated_cares.jsonl \
    --label-type icd10 \
    --model PlanTL-GOB-ES/bsc-bio-ehr-es \
    --output-dir output/icd10_clf/roberta_icd10

Notas:
- La lista de etiquetas se construye automáticamente según --label-type
- Los ejemplos cuya lista de etiquetas es vacía se filtran.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# ICD-10 CODE TO CHAPTER MAPPING
# Los capítulos del ICD-10 usados en CARES (16 capítulos)
# =============================================================================
ICD10_CHAPTER_RANGES = {
    "1": ("A00", "B99"),   # Certain infectious and parasitic diseases
    "2": ("C00", "D48"),   # Neoplasms
    "4": ("E00", "E90"),   # Endocrine, nutritional and metabolic diseases
    "6": ("G00", "G99"),   # Diseases of the nervous system
    "7": ("H00", "H59"),   # Diseases of the eye and adnexa
    "8": ("H60", "H95"),   # Diseases of the ear and mastoid process
    "9": ("I00", "I99"),   # Diseases of the circulatory system
    "10": ("J00", "J99"),  # Diseases of the respiratory system
    "11": ("K00", "K93"),  # Diseases of the digestive system
    "12": ("L00", "L99"),  # Diseases of the skin and subcutaneous tissue
    "13": ("M00", "M99"),  # Diseases of the musculoskeletal system
    "14": ("N00", "N99"),  # Diseases of the genitourinary system
    "17": ("Q00", "Q99"),  # Congenital malformations
    "18": ("R00", "R99"),  # Symptoms, signs and abnormal findings
    "19": ("S00", "T98"),  # Injury, poisoning and external causes
    "21": ("Z00", "Z99"),  # Factors influencing health status
}

# Lista ordenada de capítulos (como strings)
CHAPTER_LABELS = sorted(ICD10_CHAPTER_RANGES.keys(), key=int)


def _icd10_to_chapter(code: str) -> Optional[str]:
    """Mapea un código ICD-10 a su capítulo correspondiente.
    
    Args:
        code: Código ICD-10 normalizado (ej: "C71", "Z03", "M51")
        
    Returns:
        String del capítulo ("1", "2", ..., "21") o None si no se encuentra
    """
    code = code.strip().upper()
    if not code:
        return None
    
    # Extraer la letra inicial y los primeros dígitos
    letter = code[0] if code else ""
    
    for chapter, (start, end) in ICD10_CHAPTER_RANGES.items():
        start_letter, start_num = start[0], int(start[1:])
        end_letter, end_num = end[0], int(end[1:])
        
        if letter < start_letter or letter > end_letter:
            continue
            
        # Extraer número del código
        try:
            code_num = int(code[1:3]) if len(code) >= 3 else int(code[1:2]) if len(code) >= 2 else 0
        except ValueError:
            code_num = 0
        
        # Verificar si está en el rango
        if letter == start_letter and letter == end_letter:
            if start_num <= code_num <= end_num:
                return chapter
        elif letter == start_letter:
            if code_num >= start_num:
                return chapter
        elif letter == end_letter:
            if code_num <= end_num:
                return chapter
        elif start_letter < letter < end_letter:
            return chapter
    
    return None


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            ln = line.strip()
            if not ln:
                continue
            # Permitir comentarios tipo // ... (algunos dumps incluyen "// filepath: ...")
            if ln.startswith("//"):
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                # Dar contexto claro para depurar sin imprimir el archivo completo
                preview = ln[:200]
                raise json.JSONDecodeError(
                    f"JSON inválido en {path}:{line_no}: {e.msg}. Preview: {preview}",
                    ln,
                    e.pos,
                ) from e
    return rows


def _normalize_icd10(code: str) -> str:
    # Normalización mínima (mantener formato CARES, p.ej., "C72" o "E11.9")
    # Quitar lo que viene después del punto, si existe (p.ej., "C72.1" -> "C72")
    if "." in code:
        code = code.split(".")[0]
    return str(code).strip()


def _build_label_space(
    cares_splits: Sequence[Dataset],
    extra_rows: Optional[Iterable[Dict]] = None,
    label_type: str = "icd10",
) -> List[str]:
    labels = set()
    for ds in cares_splits:
        if ds is None:
            continue
        for ex in ds:
            for c in ex.get("icd10", []) or []:
                if label_type == "icd10":
                    labels.add(_normalize_icd10(c))
                elif label_type == "chapters":
                    chapter = _icd10_to_chapter(_normalize_icd10(c))
                    if chapter:
                        labels.add(chapter)

    if extra_rows is not None:
        for r in extra_rows:
            for c in r.get("icd10", []) or []:
                if label_type == "icd10":
                    labels.add(_normalize_icd10(c))
                elif label_type == "chapters":
                    chapter = _icd10_to_chapter(_normalize_icd10(c))
                    if chapter:
                        labels.add(chapter)

    # Orden estable
    return sorted(labels)


def _multi_hot(labels: Sequence[str], label2id: Dict[str, int]) -> List[float]:
    y = [0.0] * len(label2id)
    for c in labels or []:
        c = _normalize_icd10(c)
        if c in label2id:
            y[label2id[c]] = 1.0
    return y


def _prepare_train_dataset_from_jsonl(
    jsonl_path: str,
    text_source: str,
    label2id: Dict[str, int],
    label_type: str = "icd10",
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    rows = _read_jsonl(jsonl_path)

    # Filtrar a modo CARES (por seguridad) y ejemplos con labels
    filtered: List[Dict] = []
    for r in rows:
        if r.get("mode") not in {None, "cares"}:
            continue
        icd10 = r.get("icd10", []) or []
        if len(icd10) == 0:
            continue

        if text_source == "original":
            text = r.get("original_text")
        elif text_source == "generated":
            text = r.get("generated_text")
        elif text_source == "both_concat":
            t1 = r.get("original_text") or ""
            t2 = r.get("generated_text") or ""
            text = (t1 + "\n\n[SYNTHETIC]\n\n" + t2).strip()
        else:
            raise ValueError(f"Invalid --train-text-source: {text_source}")

        if not isinstance(text, str) or not text.strip():
            continue

        if label_type == "icd10":
            labels = _multi_hot(icd10, label2id)
        elif label_type == "chapters":
            chapters = [_icd10_to_chapter(_normalize_icd10(c)) for c in icd10]
            labels = _multi_hot([c for c in chapters if c], label2id)

        filtered.append(
            {
                "text": text,
                "labels": labels,
            }
        )

    ds = Dataset.from_list(filtered)

    # Submuestreo opcional (shuffle determinístico)
    if max_train_samples is not None and max_train_samples > 0 and len(ds) > max_train_samples:
        ds = ds.shuffle(seed=seed).select(range(max_train_samples))

    return ds


def _prepare_test_dataset_from_cares(
    cares_test: Dataset,
    label2id: Dict[str, int],
    label_type: str = "icd10",
    max_eval_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    def to_row(ex: Dict) -> Dict:
        icd10 = ex.get("icd10", []) or []
        if label_type == "icd10":
            labels = _multi_hot(icd10, label2id)
        elif label_type == "chapters":
            chapters = [_icd10_to_chapter(_normalize_icd10(c)) for c in icd10]
            labels = _multi_hot([c for c in chapters if c], label2id)
        return {
            "text": ex.get("full_text", ""),
            "labels": labels,
        }

    ds = cares_test.map(to_row, remove_columns=cares_test.column_names)
    ds = ds.filter(lambda r: isinstance(r.get("text"), str) and len(r["text"].strip()) > 0)

    if max_eval_samples is not None and max_eval_samples > 0 and len(ds) > max_eval_samples:
        ds = ds.shuffle(seed=seed).select(range(max_eval_samples))
    return ds


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _multilabel_example_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Métricas por-ejemplo (no flatten), útiles en multilabel desbalanceado."""
    preds = preds.astype(int)
    labels = labels.astype(int)

    # exact match / subset accuracy
    exact_match = float(np.mean(np.all(preds == labels, axis=1)))

    # hamming loss: fracción de etiquetas incorrectas
    hamming = float(np.mean(preds != labels))

    # Jaccard/IoU por ejemplo (promedio)
    inter = np.logical_and(preds == 1, labels == 1).sum(axis=1)
    union = np.logical_or(preds == 1, labels == 1).sum(axis=1)
    jacc = float(np.mean([_safe_div(i, u) for i, u in zip(inter, union)]))

    # Sample-based precision/recall/f1 promedio por ejemplo
    tp = (np.logical_and(preds == 1, labels == 1)).sum(axis=1)
    fp = (np.logical_and(preds == 1, labels == 0)).sum(axis=1)
    fn = (np.logical_and(preds == 0, labels == 1)).sum(axis=1)

    prec_s = float(np.mean([_safe_div(t, t + f) for t, f in zip(tp, fp)]))
    rec_s = float(np.mean([_safe_div(t, t + f) for t, f in zip(tp, fn)]))
    f1_s = float(np.mean([_safe_div(2 * t, 2 * t + f + n) for t, f, n in zip(tp, fp, fn)]))

    return {
        "subset_accuracy": exact_match,
        "hamming_loss": hamming,
        "jaccard_iou": jacc,
        "precision_sample": prec_s,
        "recall_sample": rec_s,
        "f1_sample": f1_s,
    }


def _per_label_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[str],
) -> List[Dict[str, float]]:
    """Reporte por etiqueta (estilo classification_report) sin sklearn.

    Devuelve una lista de filas con: label, support, tp, fp, fn, tn, precision, recall, f1, accuracy.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes incompatibles y_true={y_true.shape} vs y_pred={y_pred.shape}")

    n_examples, n_labels = y_true.shape
    if n_labels != len(label_list):
        raise ValueError(f"n_labels={n_labels} != len(label_list)={len(label_list)}")

    rows: List[Dict[str, float]] = []

    for j, code in enumerate(label_list):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        tp = int(np.logical_and(yp == 1, yt == 1).sum())
        fp = int(np.logical_and(yp == 1, yt == 0).sum())
        fn = int(np.logical_and(yp == 0, yt == 1).sum())
        tn = int(np.logical_and(yp == 0, yt == 0).sum())

        support = int(yt.sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) != 0 else 0.0
        acc = _safe_div(tp + tn, tp + tn + fp + fn)

        rows.append(
            {
                "label": code,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(acc),
                "prevalence": float(support) / float(n_examples) if n_examples > 0 else 0.0,
            }
        )

    # Orden: primero etiquetas más frecuentes
    rows.sort(key=lambda r: (r["support"], r["label"]), reverse=True)
    return rows


def _write_per_label_report(out_dir: Path, rows: List[Dict[str, float]]) -> None:
    out_json = out_dir / "per_label_report.json"
    out_csv = out_dir / "per_label_report.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_label": rows}, f, ensure_ascii=False, indent=2)

    fieldnames = ["label", "support", "prevalence", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _write_eval_predictions_jsonl(
    out_path: Path,
    texts: Sequence[str],
    true_multi_hot: np.ndarray,
    pred_multi_hot: np.ndarray,
    label_list: List[str],
    max_rows: Optional[int] = None,
) -> None:
    """Guarda por-ejemplo: original_text + true_labels + predicted_labels.

    - true_multi_hot/pred_multi_hot: arrays shape [n_examples, n_labels] con 0/1.
    - label_list: id -> label.

    Se escribe JSONL para evitar archivos gigantes en memoria.
    """
    n = int(true_multi_hot.shape[0])
    if pred_multi_hot.shape[0] != n:
        raise ValueError("pred_multi_hot y true_multi_hot deben tener misma cantidad de ejemplos")

    if max_rows is not None:
        n = min(n, int(max_rows))

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            yt = true_multi_hot[i]
            yp = pred_multi_hot[i]

            true_labels = [label_list[j] for j in np.where(yt == 1)[0].tolist()]
            pred_labels = [label_list[j] for j in np.where(yp == 1)[0].tolist()]

            rec = {
                "original_text": texts[i],
                "true_labels": true_labels,
                "predicted_labels": pred_labels,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class WeightedMultilabelTrainer(Trainer):
    """Trainer that replaces BCEWithLogitsLoss with a pos_weight-aware version.

    pos_weight[i] = (n_negative_i / n_positive_i) per class, capped at
    `max_pos_weight` to avoid extreme gradients for classes with only a handful
    of positive examples.  Passed as a buffer so it moves to the right device
    automatically.
    """

    def __init__(self, *args, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer_called = False
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self._pos_weight.to(logits.device)
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def _compute_pos_weight(train_ds, n_labels: int, max_pos_weight: float = 50.0) -> torch.Tensor:
    """Compute per-class pos_weight = n_neg / n_pos from training labels, capped."""
    label_matrix = np.array([ex["labels"] for ex in train_ds], dtype=np.float32)
    n = len(label_matrix)
    n_pos = label_matrix.sum(axis=0).clip(min=1)       # avoid /0
    n_neg = n - n_pos
    pos_weight = np.clip(n_neg / n_pos, 1.0, max_pos_weight)
    return torch.tensor(pos_weight, dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser()

    # Device / CUDA selection
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help=(
            "Opcional: fuerza CUDA_VISIBLE_DEVICES dentro del script (ej: '2' o '1,2'). "
            "Si no se setea, se respeta el entorno."
        ),
    )
    parser.add_argument(
        "--force-single-gpu",
        action="store_true",
        help=(
            "Fuerza uso de una sola GPU desde Transformers (evita DataParallel/NCCL). "
            "Útil si al dejar >1 GPU visible aparece 'torch.nn.parallel.data_parallel' en el stacktrace."
        ),
    )

    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=None,
        help="Path al JSONL generado (output/generated_gpt_oss_*.jsonl).",
    )
    parser.add_argument(
        "--train-text-source",
        type=str,
        default="generated",
        choices=["original", "generated", "both_concat"],
        help="Qué campo del JSONL usar como texto de entrenamiento.",
    )

    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=None,
        help="Path al JSONL de evaluación. Si se provee, se usará en lugar de --hf-dataset.",
    )
    parser.add_argument(
        "--eval-text-source",
        type=str,
        default="original",
        choices=["original", "generated", "both_concat"],
        help="Qué campo del JSONL de evaluación usar como texto.",
    )

    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="chizhikchi/CARES",
        help="Dataset HF de evaluación (CARES). Solo usado si no se provee --eval-jsonl.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PlanTL-GOB-ES/bsc-bio-ehr-es",
        help="Checkpoint HF base para SequenceClassification. Default: RoBERTa biomédico-clínico español.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directorio de salida")

    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.006)
    parser.add_argument("--num-train-epochs", type=float, default=40)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Paciencia para early stopping. 0 desactiva.")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold para binarizar sigmoid(logits)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)

    parser.add_argument(
        "--save-eval-predictions",
        action="store_true",
        help="Si se setea, guarda output-dir/eval_predictions.jsonl con texto + true_labels + predicted_labels.",
    )
    parser.add_argument(
        "--max-eval-predictions",
        type=int,
        default=None,
        help="Opcional: limita cuántos ejemplos escribir en eval_predictions.jsonl.",
    )

    parser.add_argument(
        "--label-type",
        type=str,
        default="chapters",
        choices=["icd10", "chapters"],
        help="Tipo de etiquetas: 'chapters' (16 clases, como paper CARES ~88%% F1) o 'icd10' (156 códigos, más difícil).",
    )

    args = parser.parse_args()

    # ---- Device selection (must happen ASAP) ----
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Si quedan múltiples GPUs visibles, HF Trainer puede caer en DataParallel.
    # En algunos entornos eso dispara NCCL errors. Forzamos single GPU.
    if args.force_single_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        # Si el usuario dejó múltiples GPUs visibles, nos quedamos con la primera visible
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0].strip()
        # Asegura que Accelerate/Trainer no intente distributed
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "-1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =====================
    # 1) Load CARES for eval + label space
    # =====================
    cares_splits = []
    cares_test = None
    if args.eval_jsonl is None:
        cares = load_dataset(args.hf_dataset)
        if "test" not in cares:
            raise RuntimeError(f"Dataset {args.hf_dataset} no tiene split 'test'. Splits: {list(cares.keys())}")
        cares_train = cares["train"] if "train" in cares else None
        cares_test = cares["test"]
        cares_splits = [s for s in [cares_train, cares_test] if s is not None]

    extra_rows = []
    if args.train_jsonl:
        extra_rows.extend(_read_jsonl(args.train_jsonl))
    if args.eval_jsonl:
        extra_rows.extend(_read_jsonl(args.eval_jsonl))

    label_list = _build_label_space(cares_splits, extra_rows=extra_rows if extra_rows else None, label_type=args.label_type)
    if len(label_list) == 0:
        raise RuntimeError("No se encontraron etiquetas ICD-10 para construir el label space.")

    label2id = {c: i for i, c in enumerate(label_list)}
    id2label = {i: c for c, i in label2id.items()}

    # Persist label space
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": label_list}, f, ensure_ascii=False, indent=2)

    # =====================
    # 2) Build datasets
    # =====================
    if not args.train_jsonl:
        raise RuntimeError("Por ahora, --train-jsonl es obligatorio para entrenar desde tus textos sintéticos/originales.")

    train_ds = _prepare_train_dataset_from_jsonl(
        jsonl_path=args.train_jsonl,
        text_source=args.train_text_source,
        label2id=label2id,
        label_type=args.label_type,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset vacío. Revisa --train-jsonl y --train-text-source.")

    if args.eval_jsonl:
        # Reutilizamos la función de Jsonl (que procesa texts) para evaluación
        eval_ds = _prepare_train_dataset_from_jsonl(
            jsonl_path=args.eval_jsonl,
            text_source=args.eval_text_source,
            label2id=label2id,
            label_type=args.label_type,
            max_train_samples=args.max_eval_samples,
            seed=args.seed,
        )
    else:
        eval_ds = _prepare_test_dataset_from_cares(
            cares_test=cares_test,
            label2id=label2id,
            label_type=args.label_type,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
        )

    if len(eval_ds) == 0:
        raise RuntimeError(
            "Eval dataset quedó vacío (CARES test). Revisa que el split tenga columna 'full_text' y ejemplos no vacíos."
        )

    # =====================
    # DIAGNÓSTICO: Verificar labels
    # =====================
    print("\n" + "="*60)
    print(f"CONFIGURACIÓN DE LABELS")
    print("="*60)
    print(f"  --label-type: {args.label_type}")
    print(f"  Número de labels: {len(label_list)}")
    print(f"  Labels: {label_list}")
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Eval examples: {len(eval_ds)}")
    
    # Verificar que hay positivos en train
    train_labels_sum = sum(sum(ex["labels"]) for ex in train_ds)
    eval_labels_sum = sum(sum(ex["labels"]) for ex in eval_ds)
    print(f"  Total positivos en train: {train_labels_sum:.0f}")
    print(f"  Total positivos en eval: {eval_labels_sum:.0f}")
    print(f"  Promedio labels/ejemplo (train): {train_labels_sum/len(train_ds):.2f}")
    print(f"  Promedio labels/ejemplo (eval): {eval_labels_sum/len(eval_ds):.2f}")
    
    # Mostrar ejemplo de labels
    sample_labels = train_ds[0]["labels"]
    active_labels = [label_list[i] for i, v in enumerate(sample_labels) if v == 1.0]
    print(f"  Ejemplo train[0] labels activos: {active_labels}")
    print("="*60 + "\n")

    # =====================
    # 3) Tokenize
    # =====================
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch: Dict) -> Dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(args.max_length),
        )

    train_tok = train_ds.map(tok, batched=True)
    eval_tok = eval_ds.map(tok, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # =====================
    # 4) Model + metrics
    # =====================
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    # Multi-label metrics: micro/macro F1, precision, recall + accuracy (flatten)
    f1_micro = evaluate.load("f1")
    f1_macro = evaluate.load("f1")
    precision_micro = evaluate.load("precision")
    recall_micro = evaluate.load("recall")
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = _sigmoid(np.asarray(logits))
        preds = (probs >= float(args.threshold)).astype(int)

        y_true = np.asarray(labels).astype(int)

        # ----- (A) Métricas flatten (útiles pero pueden inflarse por sparsity) -----
        flat_preds = preds.reshape(-1)
        flat_labels = y_true.reshape(-1)

        metrics: Dict[str, float] = {
            "accuracy_flat": accuracy.compute(predictions=flat_preds, references=flat_labels)["accuracy"],
            "f1_micro_flat": f1_micro.compute(predictions=flat_preds, references=flat_labels, average="micro")["f1"],
            "f1_macro_flat": f1_macro.compute(predictions=flat_preds, references=flat_labels, average="macro")["f1"],
            "precision_micro_flat": precision_micro.compute(predictions=flat_preds, references=flat_labels, average="micro")[
                "precision"
            ],
            "recall_micro_flat": recall_micro.compute(predictions=flat_preds, references=flat_labels, average="micro")[
                "recall"
            ],
        }

        # ----- (B) Métricas por-ejemplo (recomendadas para interpretar) -----
        metrics.update(_multilabel_example_metrics(preds=preds, labels=y_true))

        # ----- (C) Contexto del dataset (sparsity / prevalencia) -----
        # Promedio de etiquetas positivas por ejemplo, y densidad total
        positives_per_example = float(np.mean(y_true.sum(axis=1)))
        label_density = float(np.mean(y_true))  # fraction of positives in full matrix
        metrics["avg_labels_per_example"] = positives_per_example
        metrics["label_density"] = label_density

        return metrics

    # Si hay evaluación, Transformers prefija con 'eval_' automáticamente.
    metric_for_best_model = "eval_f1_macro_flat"  # macro F1 como en paper CARES

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        num_train_epochs=float(args.num_train_epochs),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        warmup_steps=int(args.warmup_steps),
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        save_total_limit=3,
        load_best_model_at_end=(args.eval_strategy != "no" and args.save_strategy != "no"),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_steps=int(args.logging_steps),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        report_to=["none"],
        seed=int(args.seed),
        # Importante: evita DataParallel cuando >1 GPU visible
        no_cuda=False,
    )

    # Early stopping callback (como en CARES original)
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    pos_weight = _compute_pos_weight(train_tok, n_labels=len(label_list))
    print("\npos_weight per class (capped at 50):")
    for lbl, pw in zip(label_list, pos_weight.tolist()):
        print(f"  {lbl:>4s}  {pw:.1f}x")

    trainer = WeightedMultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        pos_weight=pos_weight,
    )

    # Persist training config
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Threshold used: {args.threshold}")

    trainer.train()

    #print threshold used

    # Always do a final evaluation
    metrics = trainer.evaluate()

    #print threshold used

    # Per-label report (post-eval): run predictions once to get y_true/y_pred
    pred_out = trainer.predict(eval_tok)
    probs = _sigmoid(np.asarray(pred_out.predictions))
    y_pred = (probs >= float(args.threshold)).astype(int)
    y_true = np.asarray(pred_out.label_ids).astype(int)

    report_rows = _per_label_report(y_true=y_true, y_pred=y_pred, label_list=label_list)
    _write_per_label_report(out_dir=out_dir, rows=report_rows)

    # Guardar predicciones por ejemplo (para inspección)
    if args.save_eval_predictions:
        eval_pred_path = out_dir / "eval_predictions.jsonl"
        eval_texts = eval_ds["text"]
        _write_eval_predictions_jsonl(
            out_path=eval_pred_path,
            texts=eval_texts,
            true_multi_hot=y_true,
            pred_multi_hot=y_pred,
            label_list=label_list,
            max_rows=args.max_eval_predictions,
        )
        print(f"\nSaved eval predictions to: {eval_pred_path}")

    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nFinal metrics:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]}")

    print(f"\nPer-label report saved to: {out_dir / 'per_label_report.json'} and {out_dir / 'per_label_report.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
