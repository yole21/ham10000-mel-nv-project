import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True, help="Prediction CSV path")
    parser.add_argument("--out_dir", type=str, default="outputs/error_analysis", help="Output directory")
    parser.add_argument("--top_k", type=int, default=2, help="Number of examples per category")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)

    required_cols = ["image_path", "image_id", "true_label", "pred_label", "prob_mel"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Four groups
    tp = df[(df["true_label"] == 1) & (df["pred_label"] == 1)].copy()
    tn = df[(df["true_label"] == 0) & (df["pred_label"] == 0)].copy()
    fp = df[(df["true_label"] == 0) & (df["pred_label"] == 1)].copy()
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)].copy()

    # Sort by informativeness
    tp = tp.sort_values("prob_mel", ascending=False)   # most confident correct melanoma
    tn = tn.sort_values("prob_mel", ascending=True)    # most confident correct benign
    fp = fp.sort_values("prob_mel", ascending=False)   # most confident false alarm
    fn = fn.sort_values("prob_mel", ascending=True)    # most confident missed melanoma

    tp_sel = tp.head(args.top_k).copy()
    tn_sel = tn.head(args.top_k).copy()
    fp_sel = fp.head(args.top_k).copy()
    fn_sel = fn.head(args.top_k).copy()

    tp_sel["case_type"] = "TP"
    tn_sel["case_type"] = "TN"
    fp_sel["case_type"] = "FP"
    fn_sel["case_type"] = "FN"

    selected = pd.concat([tp_sel, tn_sel, fp_sel, fn_sel], axis=0)
    selected = selected[
        ["case_type", "image_id", "image_path", "true_label", "pred_label", "prob_mel"]
    ].reset_index(drop=True)

    base_name = os.path.splitext(os.path.basename(args.pred_csv))[0]
    out_csv = os.path.join(args.out_dir, f"{base_name}_selected_error_cases.csv")
    selected.to_csv(out_csv, index=False)

    print("\n=== Selected Cases ===")
    print(selected.to_string(index=False))
    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()