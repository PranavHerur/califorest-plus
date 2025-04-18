from califorest import metrics as em
from sklearn.metrics import roc_auc_score


def metrics(y_test, y_pred, random_seed) -> dict:
    score_auc = roc_auc_score(y_test, y_pred)
    score_hl = em.hosmer_lemeshow(y_test, y_pred)
    score_sh = em.spiegelhalter(y_test, y_pred)
    score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
    rel_small, rel_large = em.reliability(y_test, y_pred)

    results = {
        "random_seed": random_seed,
        "auc": score_auc,
        "brier": score_b,
        "brier_scaled": score_bs,
        "hosmer_lemshow": score_hl,
        "speigelhalter": score_sh,
        "reliability_small": rel_small,
        "reliability_large": rel_large,
    }

    # Print results
    print(f"  AUC: {score_auc:.4f}")
    print(f"  Brier Score: {score_b:.4f}")
    print(f"  Scaled Brier Score: {score_bs:.4f}")
    print(f"  Hosmer-Lemeshow p-value: {score_hl:.4f}")
    print(f"  Spiegelhalter p-value: {score_sh:.4f}")
    print(f"  Reliability-in-the-small: {rel_small:.6f}")
    print(f"  Reliability-in-the-large: {rel_large:.6f}")
    print()

    return results
