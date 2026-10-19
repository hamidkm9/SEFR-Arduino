import os, subprocess, sys, tempfile
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine


def sefr(X, y, K):
    n = np.bincount(y, minlength=K).astype(float)
    lo, hi = X.min(0), X.max(0)
    rng = hi - lo
    ok = rng > 0
    W = np.zeros((K, X.shape[1]))
    b = np.zeros(K)
    for c in range(K):
        mp = np.where(ok, (X[y == c].mean(0) - lo) / np.where(ok, rng, 1), 0)
        mn = np.where(ok, (X[y != c].mean(0) - lo) / np.where(ok, rng, 1), 0)
        w = np.where(ok, (mp - mn) / (mp + mn + 1e-7), 0)
        t = ((len(y) - n[c]) * (w @ mp) + n[c] * (w @ mn)) / len(y)
        W[c] = np.where(ok, w / np.where(ok, rng, 1), 0)
        b[c] = -(t + W[c] @ lo)
    return W, b


def lda(X, y, K, W, b):
    S = X @ W.T + b
    means = np.array([S[y == c].mean(0) for c in range(K)])
    R = S - means[y]
    C = R.T @ R / len(y)
    C += (1e-6 * np.trace(C) / K + 1e-12) * np.eye(K)
    A = np.linalg.solve(C, means.T).T
    prior = np.log(np.bincount(y, minlength=K) / len(y))
    nb = prior + A @ b - 0.5 * np.einsum("ck,ck->c", A, means)
    return A @ W, nb


exe = sys.argv[1]
ok_all = True
for name, loader in [("iris", load_iris), ("wine", load_wine), ("cancer", load_breast_cancer), ("digits", load_digits)]:
    X, y = loader(return_X_y=True)
    X = X.astype(np.float32).astype(float)
    K = y.max() + 1
    d = tempfile.mkdtemp()
    np.savetxt(f"{d}/data.csv", np.column_stack([y, X]), delimiter=",", fmt=["%d"] + ["%.9g"] * X.shape[1])
    for use_lda in [0, 1]:
        rc = subprocess.run([exe, f"{d}/data.csv", f"{d}/out.csv", str(use_lda), name]).returncode
        assert rc == 0, (name, use_lda, rc)
        lines = open(f"{d}/out.csv").read().split()
        params = np.array([[float(v) for v in l.split(",")] for l in lines[:K]])
        pred = np.array([int(v) for v in lines[K:]])
        W, b = sefr(X, y, K)
        if use_lda:
            W, b = lda(X, y, K, W, b)
        ref = (X @ W.T + b).argmax(1)
        scores = X @ W.T + b
        top2 = np.sort(scores, 1)[:, -2:]
        clear = (top2[:, 1] - top2[:, 0]) > 1e-3 * np.abs(scores).max()
        agree = (pred == ref)[clear].mean()
        werr = np.abs(params[:, :-1] - W).max() / np.abs(W).max()
        acc = (pred == y).mean()
        ok = agree == 1.0 and werr < 1e-3
        ok_all &= ok
        print(f"{name:7s} lda={use_lda}  train acc {100*acc:5.1f}%  agree with numpy {100*agree:5.1f}%  max weight rel err {werr:.1e}  {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok_all else 1)
