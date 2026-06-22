"""
BƯỚC 10: Fine-tuning LoRA  (Bài 4)
====================================
10a. Tạo Q&A dataset từ chunks bằng Gemini
10b. Lưu dataset dạng JSONL (Hugging Face format)
10c. Script train LoRA (chạy trên Colab)
10d. Load adapter đã train để inference

Lưu ý: Bước train thực tế cần GPU — chạy trên Colab.
        Bước này chỉ tạo dataset local + hướng dẫn Colab.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR  = Path("data/dataset")
QA_FILE      = DATASET_DIR / "qa_pairs.jsonl"
ADAPTER_DIR  = Path("models/lora_adapter")
BASE_MODEL   = "vinai/phobert-base-v2"


# ================================================================ #
#  10a. Tạo Q&A dataset bằng Gemini                               #
# ================================================================ #

def generate_qa_dataset(chunks: list[dict], n_per_chunk: int = 1,
                        max_chunks: int = 100,
                        force_rerun: bool = False) -> list[dict]:
    """Dùng Gemini sinh câu hỏi - câu trả lời từ các chunk luật.

    Args:
        chunks:        List chunk dạng {"text":..., "article_number":...}.
        n_per_chunk:   Số cặp Q&A mỗi chunk.
        max_chunks:    Giới hạn số chunk xử lý (tránh tốn quota).
        force_rerun:   Nếu False và file đã tồn tại thì skip.

    Returns:
        List dict {"instruction": ..., "input": "", "output": ...}
    """
    if QA_FILE.exists() and not force_rerun:
        print(f"  ○ Đã có dataset: {QA_FILE}")
        return [json.loads(l) for l in QA_FILE.read_text(encoding="utf-8").splitlines() if l]

    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    qa_pairs = []
    selected = [c for c in chunks if c.get("method") == "article"][:max_chunks]

    print(f"\n{'='*55}")
    print(f"  BƯỚC 10a: SINH Q&A DATASET")
    print(f"{'='*55}")
    print(f"  Số chunk xử lý: {len(selected)} / {len(chunks)}")
    print(f"  Q&A mỗi chunk : {n_per_chunk}")
    print()

    for i, chunk in enumerate(selected):
        text = chunk["text"][:1500]
        article = chunk.get("article_number", "?")

        prompt = (
            f"Dựa vào đoạn luật sau (Điều {article}), hãy tạo {n_per_chunk} cặp "
            f"câu hỏi - câu trả lời ngắn gọn bằng tiếng Việt.\n\n"
            f"ĐOẠN LUẬT:\n{text}\n\n"
            "Trả về JSON array, mỗi phần tử: {\"question\": \"...\", \"answer\": \"...\"}\n"
            "Chỉ trả về JSON, không giải thích."
        )

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview", contents=prompt
            )
            raw = response.text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            pairs = json.loads(raw.strip())
            for p in pairs:
                qa_pairs.append({
                    "instruction": p["question"],
                    "input": "",
                    "output": p["answer"],
                    "article_number": article,
                })
        except Exception as e:
            print(f"  ⚠ Chunk {i} (Điều {article}) lỗi: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}] {len(qa_pairs)} Q&A pairs tạo được...")

    # Lưu JSONL
    with open(QA_FILE, "w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Đã tạo {len(qa_pairs)} Q&A pairs → {QA_FILE}")
    return qa_pairs


# ================================================================ #
#  10b. Huấn luyện LoRA (script cho Colab)                        #
# ================================================================ #

COLAB_SCRIPT = '''
# ============================================================
# CHẠY TRÊN GOOGLE COLAB (GPU T4 hoặc cao hơn)
# ============================================================
# 1. Upload file qa_pairs.jsonl lên Colab
# 2. Chạy cell sau:

!pip install -q transformers peft trl datasets accelerate bitsandbytes

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# Load dataset
dataset = load_dataset("json", data_files="qa_pairs.jsonl", split="train")

def format_example(example):
    return {"text": f"### Câu hỏi: {example['instruction']}\\n### Trả lời: {example['output']}"}

dataset = dataset.map(format_example)

model_name = "vilm/vistral-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Vistral/LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
trainer.model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
print("LoRA adapter saved!")
# Download lora_adapter/ folder và đặt vào models/lora_adapter/
'''


def save_colab_script():
    """Lưu script Colab ra file để tham khảo."""
    script_path = ADAPTER_DIR.parent / "colab_finetune.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(COLAB_SCRIPT, encoding="utf-8")
    print(f"  ✓ Colab script: {script_path}")
    return script_path


# ================================================================ #
#  10c. Inference với adapter đã train                             #
# ================================================================ #

def load_lora_model(base_model: str = "gpt2"):
    """Load model + LoRA adapter (nếu đã train).

    Args:
        base_model: Tên base model (phải khớp với model dùng khi train).

    Returns:
        (model, tokenizer) tuple hoặc None nếu chưa có adapter.
    """
    adapter_path = ADAPTER_DIR / "adapter_config.json"
    if not adapter_path.exists():
        print(f"  ⚠ Chưa có LoRA adapter tại {ADAPTER_DIR}/")
        print("    Hãy train trên Colab trước (xem models/colab_finetune.py)")
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        print(f"  Đang load {base_model} + LoRA adapter...")
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model.eval()
        print("  ✓ Model + adapter sẵn sàng")
        return model, tokenizer
    except Exception as e:
        print(f"  ⚠ Load model lỗi: {e}")
        return None, None


def lora_generate(question: str, model, tokenizer, max_new_tokens: int = 200) -> str:
    """Sinh câu trả lời bằng LoRA fine-tuned model.

    Args:
        question:       Câu hỏi.
        model:          Fine-tuned model.
        tokenizer:      Tokenizer.
        max_new_tokens: Độ dài tối đa output.

    Returns:
        Câu trả lời.
    """
    import torch
    prompt = f"### Câu hỏi: {question}\n### Trả lời:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("### Trả lời:")[-1].strip()
    return answer


# ================================================================ #
#  Run                                                             #
# ================================================================ #

def run_finetune(chunks_article: list[dict] | None = None,
               force_rerun: bool = False):
    """
    Returns:
        qa_pairs (list[dict]) — dataset Q&A đã tạo
    """
    print(f"\n{'='*55}")
    print(f"  BƯỚC 10: FINE-TUNING LoRA")
    print(f"{'='*55}\n")

    if chunks_article is None:
        from chunking import load_chunks
        _, chunks_article = load_chunks()

    # Sinh dataset
    qa_pairs = generate_qa_dataset(
        chunks_article, n_per_chunk=5, max_chunks=687,
        force_rerun=force_rerun,
    )

    # Lưu Colab script
    save_colab_script()

    # Thử load adapter (nếu có)
    model, tokenizer = load_lora_model()
    if model is not None:
        demo_q = "Quyền dân sự là gì?"
        print(f"\n  DEMO LoRA Inference:")
        print(f"  Câu hỏi: {demo_q}")
        answer = lora_generate(demo_q, model, tokenizer)
        print(f"  Trả lời: {answer}")

    print(f"\n  Dataset: {len(qa_pairs)} cặp Q&A  → {QA_FILE}")
    print(f"  Script Colab: models/colab_finetune.py")
    print(f"  → Train trên Colab rồi copy models/lora_adapter/ về đây\n")
    print(f"  ✓ Bước 10 hoàn tất!\n")
    return qa_pairs
