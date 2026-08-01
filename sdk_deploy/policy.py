"""내보낸 정책 로더 (NumPy .npz / ONNX / TorchScript).

scripts/rsl_rl/export_policy_onnx.py 가 만든 policy.onnx / policy.pt,
또는 scripts/export_policy_numpy.py 가 만든 policy_numpy.npz 를 받습니다.
입력 (1, OBS_DIM) float32 → 출력 (1, 12) raw action.

Go1 온보드 NX 에는 onnxruntime / torch 가 없으므로(Python 3.6.9, 오프라인)
NX 에서는 .npz 백엔드를 사용합니다. 개발 PC 에서는 셋 다 동작합니다.
"""

import json
import os

import numpy as np

import config as C


class Policy:
    def __init__(self, path: str):
        self._path = path
        if path.endswith(".npz"):
            self._load_numpy(path)
        elif path.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._sess.get_inputs()[0].name
            self._backend = "onnx"
        else:
            import torch

            self._torch = torch
            self._module = torch.jit.load(path, map_location="cpu")
            self._module.eval()
            self._backend = "torchscript"
        self._verify_reference_io()

    def _load_numpy(self, path):
        data = np.load(path)
        if "lstm_weight_ih" in data:
            raise ValueError(
                "recurrent .npz bundle; sdk_deploy 는 feed-forward phase1 전용입니다 "
                "(LSTM student 는 ROS 스택 deploy_policy.py 로 실행)"
            )
        bundle_obs = int(data["obs_dim"]) if "obs_dim" in data else None
        if bundle_obs is not None and bundle_obs != C.OBS_DIM:
            raise ValueError(
                f".npz obs dim {bundle_obs} != config OBS_DIM {C.OBS_DIM}; "
                "다른 export 의 번들이 아닌지 확인하세요"
            )
        n = int(data["num_mlp_layers"]) if "num_mlp_layers" in data else 4
        self._layers = [
            (data[f"{2 * i}_weight"].astype(np.float32),
             data[f"{2 * i}_bias"].astype(np.float32))
            for i in range(n)
        ]
        self._backend = "numpy"

    def _verify_reference_io(self):
        """모델 옆의 reference_io.json (개발 PC onnxruntime 출력)과 대조.

        파일이 없으면 조용히 통과. 있는데 안 맞으면 모터를 건드리기 전에
        즉시 중단합니다 — 잘못된/오염된 export 를 하드웨어에 올리는 것을 방지.
        """
        ref_path = os.path.join(os.path.dirname(os.path.abspath(self._path)),
                                "reference_io.json")
        if not os.path.exists(ref_path):
            return
        with open(ref_path) as f:
            ref = json.load(f)
        worst = 0.0
        for pair in ref["pairs"]:
            out = self(np.array(pair["obs"], dtype=np.float32))
            worst = max(worst, float(np.max(np.abs(out - np.array(pair["action"])))))
        if worst > 1e-3:
            raise ValueError(
                f"policy self-test MISMATCH: max|out-ref| = {worst:.3e} "
                f"(reference: {ref_path})"
            )
        print(f"policy self-test OK: {len(ref['pairs'])} pairs, "
              f"max|out-ref| = {worst:.2e} [{self._backend}]")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.reshape(1, -1).astype(np.float32)
        if x.shape[1] != C.OBS_DIM:
            raise ValueError(
                f"obs dim {x.shape[1]} != {C.OBS_DIM}; "
                "export 시 GO1_ABS_JOINT_OBS=1 이었는지 확인하세요"
            )
        if self._backend == "numpy":
            h = x[0]
            last = len(self._layers) - 1
            for i, (w, b) in enumerate(self._layers):
                h = h @ w.T + b
                if i < last:
                    h = np.where(h > 0.0, h, np.exp(h) - 1.0)  # ELU
            out = h
        elif self._backend == "onnx":
            (out,) = self._sess.run(None, {self._input_name: x})
        else:
            with self._torch.no_grad():
                out = self._module(self._torch.from_numpy(x)).numpy()
        action = np.asarray(out, dtype=np.float64).reshape(-1)
        if action.shape[0] != C.NUM_ACTIONS:
            raise ValueError(f"action dim {action.shape[0]} != 12")
        return action
