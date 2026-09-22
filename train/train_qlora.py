"""QLoRA supervised fine-tuning for the MyCents engine — one script, both tasks.

Reads a `messages` JSONL, applies the tokenizer's own chat template, trains a
LoRA adapter on a 4-bit base, writes checkpoints. It is deliberately
task-agnostic: the system prompt lives inside every record (built by
prompts.py for extraction, categorize_prompts.py for categorization), so this
script never needs to know which task it is training. Point it at different
data and a different output directory and it trains the other one.

    # categorization (CATEGORIZATION.md section 3)
    python train_qlora.py \
      --base Qwen/Qwen3.5-2B \
      --train data/categorize_train.jsonl \
      --val   data/categorize_val.jsonl \
      --out   checkpoints/qwen3.5-2b-categorize \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
      --epochs 2 --batch 4 --grad-accum 4

Run it with --dry-run first. That loads no model and touches no GPU: it reads
the data, renders one example through the chat template, and prints exactly
which tokens the loss will be computed on. Getting that mask wrong is the
failure this script is most likely to suffer and the hardest to notice
afterwards -- the run completes, the loss curve looks healthy, and the model
has spent its capacity learning to recite the system prompt.

Why this file exists: the original train_qlora.py that produced
checkpoints/qwen3.5-2b-qlora/checkpoint-550 was never committed, so the one
reproducible artifact of that run was the checkpoint itself. The defaults here
are reconstructed from what that checkpoint recorded -- see DEFAULTS below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults reconstructed from checkpoints/qwen3.5-2b-qlora/checkpoint-550
# --------------------------------------------------------------------------
# trainer_state.json records the learning rate at every 5th step. Fitting those
# points: lr 4e-05 at step 5 and 1.0831e-05 at step 550 of 642 total is a
# COSINE decay from 2e-4 with a 25-step warmup (linear decay would have left
# 2.98e-05 at step 550; cosine gives 1.09e-05). That is why the scheduler
# default below is cosine and not the more common linear.
#
# The extraction run also used r=32 / alpha=64 / dropout=0.0. The defaults here
# are CATEGORIZATION.md section 3's r=16 / alpha=32 / dropout=0.05 instead:
# eight-way classification over a fixed 47-subcategory taxonomy needs less
# capacity than free-form JSON extraction, and a smaller rank is less able to
# memorise the 5,200 catalog rows the synthetic baskets are drawn from. Pass
# --lora-r 32 --lora-alpha 64 --lora-dropout 0.0 to reproduce the extraction run.

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Qwen3.5's chat template renders every turn as
#   <|im_start|>{role}\n{content}<|im_end|>\n
# and an assistant turn additionally carries an empty reasoning block, so the
# text that immediately precedes the answer is:
#   <|im_start|>assistant\n<think>\n\n</think>\n\n
# Including that block in the response marker means the loss starts at the
# first character of the JSON. It also makes the marker byte-identical to what
# app/extraction.py:build_prompt produces at serving time with
# enable_thinking=False -- the training render and the serving prompt share a
# prefix exactly, which is the property that keeps the two from drifting.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("data and model")
    g.add_argument("--base", default="Qwen/Qwen3.5-2B",
                   help="base model repo id or local path (default: %(default)s)")
    g.add_argument("--train", required=True, type=Path, help="training JSONL")
    g.add_argument("--val", type=Path, default=None, help="validation JSONL (optional)")
    g.add_argument("--out", required=True, type=Path, help="checkpoint output directory")
    g.add_argument("--tokenizer", default=None,
                   help="where to read the chat template from for --dry-run "
                        "(default: --base). A local checkpoint works offline.")

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)

    g = p.add_argument_group("optimisation")
    g.add_argument("--max-seq-length", type=int, default=2048)
    g.add_argument("--epochs", type=float, default=2.0)
    g.add_argument("--max-steps", type=int, default=-1,
                   help="stop after N optimizer steps; overrides --epochs")
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--warmup-ratio", type=float, default=0.03)
    g.add_argument("--scheduler", default="cosine",
                   choices=["cosine", "linear", "constant"])
    g.add_argument("--batch", type=int, default=4, help="per-device batch size")
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--seed", type=int, default=3407)

    g = p.add_argument_group("checkpointing and logging")
    g.add_argument("--eval-steps", type=int, default=100)
    g.add_argument("--save-steps", type=int, default=100)
    g.add_argument("--logging-steps", type=int, default=5)
    g.add_argument("--save-total-limit", type=int, default=None,
                   help="keep only the N most recent checkpoints "
                        "(default: keep all, so any can be picked by eval)")
    g.add_argument("--resume", action="store_true",
                   help="resume from the last checkpoint in --out")
    g.add_argument("--report-to", default="none",
                   help="trainer reporting backend (default: none)")

    g = p.add_argument_group("safety")
    g.add_argument("--dry-run", action="store_true",
                   help="inspect the data and the loss mask, then exit. "
                        "Loads no model and uses no GPU.")
    g.add_argument("--no-4bit", action="store_true",
                   help="load the base in bf16 instead of 4-bit (needs far more VRAM)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Read a messages JSONL, failing loudly on the shapes that would otherwise
    train silently and wrongly."""
    if not path.exists():
        sys.exit(f"error: no such file: {path}")

    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"error: {path}:{lineno}: {exc}")
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                sys.exit(f"error: {path}:{lineno}: no 'messages' list")
            if messages[-1].get("role") != "assistant":
                # Without an assistant turn there is nothing to compute a loss
                # on, and the example would contribute nothing but tokens.
                sys.exit(f"error: {path}:{lineno}: last message is not the assistant turn")
            for i, message in enumerate(messages):
                if not isinstance(message, dict) or "role" not in message or "content" not in message:
                    sys.exit(f"error: {path}:{lineno}: messages[{i}] needs 'role' and 'content'")
            records.append(record)

    if not records:
        sys.exit(f"error: {path} has no records")
    return records


def describe(records: list[dict], label: str) -> None:
    kinds: dict[str, int] = {}
    for record in records:
        kind = (record.get("meta") or {}).get("kind", "-")
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"  {label}: {len(records)} records", end="")
    if len(kinds) > 1 or "-" not in kinds:
        spread = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"  [{spread}]")
    else:
        print()


def to_dataset(records: list[dict], render):
    """Render each record's `messages` through the chat template into a single
    `text` column.

    Unsloth's SFTTrainer does not apply the chat template itself -- handed a
    `messages` column it raises "You must specify a `formatting_func`". Doing
    the render here rather than through a formatting_func keeps it in one
    place, and it is the same `apply_chat_template` call --dry-run inspects,
    so what the dry run reports is what actually gets trained.

    `meta` is dropped: it is the data builder's bookkeeping, and leaving it on
    the dataset makes the collator try to tensorise it.
    """
    from datasets import Dataset
    return Dataset.from_list([{"text": render(r["messages"])} for r in records])


# --------------------------------------------------------------------------
# Dry run: prove the loss mask before spending an hour on the GPU
# --------------------------------------------------------------------------

def dry_run(args: argparse.Namespace, train_records, val_records) -> None:
    from transformers import AutoTokenizer

    source = args.tokenizer or args.base
    print(f"\nchat template from: {source}")
    tok = AutoTokenizer.from_pretrained(source)
    tok = getattr(tok, "tokenizer", tok)

    lengths = []
    for record in train_records:
        text = tok.apply_chat_template(record["messages"], tokenize=False)
        lengths.append(len(tok(text)["input_ids"]))
    lengths.sort()
    n = len(lengths)
    over = sum(1 for length in lengths if length > args.max_seq_length)
    print(f"\ntoken lengths (train): min={lengths[0]} median={lengths[n // 2]} "
          f"p99={lengths[int(n * 0.99)]} max={lengths[-1]}")
    print(f"over --max-seq-length {args.max_seq_length}: {over} "
          f"({over / n:.2%})" + ("  <-- these would be TRUNCATED" if over else "  ok"))

    rendered = tok.apply_chat_template(train_records[0]["messages"], tokenize=False)
    if RESPONSE_PART not in rendered:
        print(f"\n!! the response marker {RESPONSE_PART!r} does not appear in the "
              f"rendered example.\n   The loss mask would cover nothing. This "
              f"script's markers are written for Qwen3.5's\n   chat template; a "
              f"different base needs different ones.")
        sys.exit(1)

    head, _, answer = rendered.partition(RESPONSE_PART)
    print(f"\nloss mask check on record 1:")
    print(f"  prompt  {len(tok(head)['input_ids']):>5} tokens  (masked out, no loss)")
    print(f"  answer  {len(tok(answer)['input_ids']):>5} tokens  (trained on)")
    print(f"\n  ...prompt ends: {head[-60:]!r}")
    print(f"  answer begins : {answer[:180]!r}")

    # Recompute over the same records rather than reusing `lengths`, which was
    # sorted above and no longer lines up with train_records.
    sample = train_records[:200]
    masked = total = 0
    for record in sample:
        text = tok.apply_chat_template(record["messages"], tokenize=False)
        total += len(tok(text)["input_ids"])
        masked += len(tok(text.partition(RESPONSE_PART)[0])["input_ids"])
    print(f"\n  across the first {len(sample)} records, {masked / total:.1%} of "
          f"tokens are prompt and carry no loss.")
    print("  (that is expected and correct -- the system prompt is identical in "
          "every record)")

    if val_records:
        real = [r for r in val_records if (r.get("meta") or {}).get("kind") == "real"]
        print(f"\nvalidation: {len(val_records)} records, {len(real)} of kind 'real'")
        if real:
            print("  these will be evaluated separately as eval_real_loss")

    print("\ndry run only -- no model loaded, nothing trained.")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(args: argparse.Namespace, train_records, val_records) -> None:
    # unsloth must be imported before transformers/trl: it patches them on
    # import. Importing it here rather than at module scope keeps --dry-run
    # off the GPU entirely.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from trl import SFTConfig, SFTTrainer

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\nloading {args.base} (4bit={not args.no_4bit})")
    started = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_length,
        dtype=None,                       # let unsloth pick bf16 where supported
        load_in_4bit=not args.no_4bit,
    )
    print(f"base ready in {time.time() - started:.0f}s")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # For Qwen3.5 unsloth returns a multimodal processor, not a plain
    # tokenizer. Chat templating goes through the processor (as
    # app/extraction.py:build_prompt does at serving time); padding and
    # tokenization go through the inner text tokenizer, which is what the
    # trainer and the loss mask need.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def render(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False)

    train_ds = to_dataset(train_records, render)
    # Report validation on the real receipts separately. They are the only rows
    # that are not synthetic baskets, there are very few of them, and pooling
    # them into one number lets 279 synthetic records hide what the 15 real
    # ones are doing. CATEGORIZATION.md section 3: "watch val/real separately".
    eval_ds = None
    if val_records:
        real = [r for r in val_records if (r.get("meta") or {}).get("kind") == "real"]
        eval_ds = {"all": to_dataset(val_records, render)}
        if real:
            eval_ds["real"] = to_dataset(real, render)

    config = SFTConfig(
        output_dir=str(args.out),
        max_length=args.max_seq_length,
        dataset_text_field="text",        # to_dataset() rendered it already
        packing=False,                    # per-example, so the mask stays aligned
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps if eval_ds else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        dataset_num_proc=1,               # Windows: >1 deadlocks on spawn
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=text_tokenizer,
    )

    # Mask the prompt out of the loss. Without this the model spends its
    # capacity learning to reproduce the system prompt -- which is byte
    # identical in every record and never needs to be generated.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    _report_mask(trainer, tokenizer)

    # Record how this run was configured, next to its checkpoints. The
    # extraction run left no such file, which is why its hyperparameters had to
    # be reverse-engineered from trainer_state.json to write this script.
    (args.out / "train_config.json").write_text(
        json.dumps({**{k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()},
                    "target_modules": TARGET_MODULES,
                    "response_part": RESPONSE_PART,
                    "train_records": len(train_records),
                    "val_records": len(val_records or []),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                   indent=2),
        encoding="utf-8",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    minutes = (time.time() - started) / 60

    model.save_pretrained(str(args.out / "final"))
    tokenizer.save_pretrained(str(args.out / "final"))

    peak = (torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available() else 0.0)
    print(f"\ndone in {minutes:.0f} min, peak VRAM {peak:.1f} GiB")
    print(f"adapter written to {args.out / 'final'}")
    print("\nPick the checkpoint on real-receipt accuracy, not on loss "
          "(CATEGORIZATION.md section 4).")


def _report_mask(trainer, tokenizer) -> None:
    """Decode one masked example and show what survived.

    train_on_responses_only fails quietly when its markers do not match the
    chat template: every label stays -100, the loss is empty, and training
    "succeeds" while learning nothing. Checking one example costs nothing.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        example = trainer.train_dataset[0]
        labels = example["labels"]
    except (KeyError, IndexError, TypeError):
        print("could not inspect the loss mask; check it with --dry-run")
        return

    kept = [t for t in labels if t != -100]
    if not kept:
        sys.exit("error: the loss mask covers no tokens -- the response marker "
                 "did not match the chat template. Run --dry-run.")
    print(f"\nloss mask: {len(kept)}/{len(labels)} tokens trained on "
          f"({len(kept) / len(labels):.0%})")
    print(f"  first trained tokens: {tok.decode(kept[:40])!r}")


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"train: {args.train}")
    train_records = load_records(args.train)
    describe(train_records, "train")

    val_records = None
    if args.val:
        print(f"val:   {args.val}")
        val_records = load_records(args.val)
        describe(val_records, "val")

    if args.dry_run:
        dry_run(args, train_records, val_records)
        return

    train(args, train_records, val_records)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
