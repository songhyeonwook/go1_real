#!/usr/bin/env python3
"""LSTM hidden state → 부상 파라미터 선형 probe 학습 (시뮬 PC 전용).

phase3 student 는 부상 정보를 입력받지 않고 LSTM hidden(256) 안에 스스로
추론합니다 (MODELS_3PARADIGM.md 의 R² 표). 이 스크립트는 정답 라벨이 있는
시뮬 롤아웃에서 그 hidden 을 해독하는 ridge 선형 probe 를 학습해, 실기
deploy.py 가 읽는 injury_probe.npz 를 만듭니다. 파일을 모델 폴더
(예: sdk_deploy/model/phase3_antalgic_s42/)에 두면 walk/hang 중 1초마다
[EST] peg_FL=.. splint_len=.. friction=.. 형태로 실시간 표시됩니다.

입력 데이터 형식 — 시뮬 롤아웃을 .npz 로 저장 (여러 파일 가능):
  h      (T, 256)  스텝별 LSTM hidden (정책과 동일한 것)
  labels (T, K)    스텝별 정답 [예: peg_FL, peg_FR, peg_RL, peg_RR,
                   splint_len, friction] — 부상 없는 에피소드는 0/기본값
  names  (K,)      라벨 이름 문자열 배열 (모든 파일에서 동일해야 함)

go1_compar 쪽에서 play.py 롤아웃 중 policy 의 hidden 과 env 의 부상
파라미터를 함께 덤프하면 됩니다. 리셋 직후 스텝(부상 추론 수렴 전)은
--skip-steps 로 제외하는 것이 좋습니다 (기본 50 = 1초).

사용:
  python3 scripts/train_injury_probe.py rollouts/*.npz \
      -o sdk_deploy/model/phase3_antalgic_s42/injury_probe.npz
"""
import argparse
import glob
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", nargs="+", help="h/labels/names 를 담은 .npz 들")
    ap.add_argument("-o", "--output", required=True, help="injury_probe.npz 출력 경로")
    ap.add_argument("--ridge", type=float, default=1.0, help="ridge 정칙화 계수")
    ap.add_argument("--skip-steps", type=int, default=50,
                    help="각 롤아웃 시작에서 제외할 스텝 수 (LSTM 수렴 전)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="검증용으로 떼어둘 비율 (파일 단위가 아니라 셔플)")
    args = ap.parse_args()

    paths = []
    for pat in args.rollouts:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.exit("입력 롤아웃이 없습니다.")

    H_list, Y_list, names = [], [], None
    for p in paths:
        d = np.load(p)
        if "h" not in d or "labels" not in d:
            print("  skip (h/labels 없음):", p)
            continue
        h, y = d["h"], d["labels"]
        if names is None:
            names = [str(n) for n in d["names"]]
        elif [str(n) for n in d["names"]] != names:
            sys.exit(f"{p} 의 names 가 다른 파일과 다릅니다.")
        h, y = h[args.skip_steps:], y[args.skip_steps:]
        if len(h) != len(y):
            sys.exit(f"{p}: h {len(h)} != labels {len(y)}")
        H_list.append(np.asarray(h, dtype=np.float64))
        Y_list.append(np.asarray(y, dtype=np.float64))
        print("  loaded %s: %d steps" % (p, len(h)))

    H = np.concatenate(H_list)
    Y = np.concatenate(Y_list)
    print("total %d steps, hidden %d, labels %d (%s)" %
          (len(H), H.shape[1], Y.shape[1], ", ".join(names)))

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(H))
    n_val = int(len(H) * args.val_frac)
    vi, ti = idx[:n_val], idx[n_val:]

    # ridge 닫힌형: W = (X^T X + lam I)^-1 X^T Y, X 에 bias 열 추가
    X = np.hstack([H[ti], np.ones((len(ti), 1))])
    lam = args.ridge * np.eye(X.shape[1])
    lam[-1, -1] = 0.0  # bias 는 정칙화 제외
    Wb = np.linalg.solve(X.T @ X + lam, X.T @ Y[ti])
    W, b = Wb[:-1].T, Wb[-1]

    # 검증 R²
    Xv = np.hstack([H[vi], np.ones((len(vi), 1))])
    pred = Xv @ Wb
    ss_res = ((Y[vi] - pred) ** 2).sum(axis=0)
    ss_tot = ((Y[vi] - Y[vi].mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    for n, r in zip(names, r2):
        print("  val R² %-12s %+.3f" % (n, r))

    np.savez(args.output,
             W=W.astype(np.float32), b=b.astype(np.float32),
             names=np.array(names))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
