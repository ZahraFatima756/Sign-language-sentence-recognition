"""Step 3 - train the sentence recogniser (CTC).

Trains on the stitched TRAIN transcripts and checks word error rate (WER) on the
stitched VAL transcripts after every epoch. The best checkpoint is kept and
training stops early when WER stops improving.

Outputs:
  runs/model/best.pt      weights + vocabulary + settings
  runs/model/vocab.json   sign list in index order
  runs/model/train_log.csv
"""
import argparse
import csv
import json
import os
import time

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR, PATIENCE, SEED
from psl.data import ClipStore, StitchedDataset, build_vocab, collate, load_transcripts
from psl.model import SignSentenceNet, greedy_decode
from psl.urdu import word_error_rate


@torch.no_grad()
def evaluate(model, loader, vocab, criterion):
    model.eval()
    total_loss, refs, hyps = 0.0, [], []
    ds = loader.dataset
    for x, y, lengths, target_lengths in loader:
        logits = model(x)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        total_loss += criterion(log_probs, y, lengths, target_lengths).item() * len(x)
        hyps += greedy_decode(logits, lengths.tolist(), vocab)
    for t in ds.transcripts:
        refs.append(t["signs"])
    exact = sum(r == h for r, h in zip(refs, hyps)) / max(len(refs), 1)
    return total_loss / len(ds), word_error_rate(refs, hyps), exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit-train", type=int, default=0, help="use only N train variants (smoke test)")
    ap.add_argument("--limit-val", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 1)
    store = ClipStore()
    train_tx = load_transcripts("train")
    val_tx = load_transcripts("val")
    if args.limit_train:
        train_tx = train_tx[:args.limit_train]
    if args.limit_val:
        val_tx = val_tx[:args.limit_val]
    vocab = build_vocab(store, train_tx + val_tx)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")

    train_ds = StitchedDataset(train_tx, store, vocab, train=True, seed=SEED)
    val_ds = StitchedDataset(val_tx, store, vocab, train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = SignSentenceNet(len(vocab))
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    print(f"Vocabulary ({len(vocab)}): {vocab}")
    print(f"Train variants: {len(train_ds)}   Val variants: {len(val_ds)}   "
          f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_wer, best_epoch, bad_epochs = float("inf"), 0, 0
    log_path = MODEL_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_loss", "val_wer", "val_exact", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running = 0.0
            for x, y, lengths, target_lengths in train_dl:
                logits = model(x)
                log_probs = logits.log_softmax(-1).transpose(0, 1)
                loss = criterion(log_probs, y, lengths, target_lengths)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
                running += loss.item() * len(x)
            train_loss = running / len(train_ds)
            val_loss, val_wer, val_exact = evaluate(model, val_dl, vocab, criterion)
            secs = time.time() - t0
            log.writerow([epoch, round(train_loss, 4), round(val_loss, 4), round(val_wer, 4), round(val_exact, 4), round(secs, 1)])
            fh.flush()
            print(f"epoch {epoch:2d}  train_loss {train_loss:7.3f}  val_loss {val_loss:7.3f}  "
                  f"val_WER {val_wer:6.3f}  val_exact {val_exact:6.3f}  ({secs:.0f}s)", flush=True)

            if val_wer < best_wer:
                best_wer, best_epoch, bad_epochs = val_wer, epoch, 0
                torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                            "epoch": epoch, "val_wer": val_wer}, MODEL_DIR / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print(f"Early stopping: no WER improvement for {PATIENCE} epochs.")
                    break
    print(f"Best validation WER {best_wer:.3f} at epoch {best_epoch}. Saved to {MODEL_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
