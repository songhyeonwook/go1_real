"""내보낸 정책 로더 (NumPy .npz / ONNX / TorchScript).

scripts/rsl_rl/export_policy_onnx.py 가 만든 policy.onnx / policy.pt,
또는 scripts/export_policy_numpy.py 가 만든 policy_numpy.npz 를 받습니다.

지원 모델:
  * phase1 teacher : feed-forward MLP, 입력 59 (policy 52 + privileged 7)
  * phase3 student : LSTM(1층) + MLP, 입력 52 (policy 그룹만 — privileged 는
    teacher 전용이므로, deploy 가 조립한 59차원 관측의 앞 52차원만 소비)

recurrent 정책은 hidden/cell 상태를 스텝 간에 이어가므로, 에피소드 시작
(기립 후 정책 인계)마다 reset() 을 호출해야 합니다.

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
        self._lstm = None
        if path.endswith(".npz"):
            self._load_numpy(path)
        elif path.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self._in_names = [i.name for i in self._sess.get_inputs()]
            self._out_names = [o.name for o in self._sess.get_outputs()]
            self._input_name = self._in_names[0]
            self._in_dim = int(self._sess.get_inputs()[0].shape[-1])
            # recurrent export 는 LSTM 상태가 외부 입출력으로 노출됩니다:
            # (obs, h_in, c_in) -> (actions, h_out, c_out)
            self._recurrent = len(self._in_names) >= 3
            if self._recurrent:
                self._hidden = int(self._sess.get_inputs()[1].shape[-1])
            self._backend = "onnx"
        else:
            import torch

            self._torch = torch
            self._module = torch.jit.load(path, map_location="cpu")
            self._module.eval()
            self._backend = "torchscript"
            self._recurrent = False
            self._in_dim = C.OBS_DIM
        self.reset()
        self._verify_reference_io()

    def _load_numpy(self, path):
        data = np.load(path)
        n = int(data["num_mlp_layers"]) if "num_mlp_layers" in data else 4
        self._layers = [
            (data["{}_weight".format(2 * i)].astype(np.float32),
             data["{}_bias".format(2 * i)].astype(np.float32))
            for i in range(n)
        ]
        if "lstm_weight_ih" in data:
            gate_order = data["gate_order"].tobytes().decode() \
                if "gate_order" in data else "ifgo"
            if gate_order != "ifgo":
                raise ValueError(
                    ".npz LSTM gate order {!r} != 'ifgo'; "
                    "scripts/export_policy_numpy.py 로 다시 변환하세요".format(gate_order))
            self._lstm = {
                "wih": data["lstm_weight_ih"].astype(np.float32),
                "whh": data["lstm_weight_hh"].astype(np.float32),
                "bih": data["lstm_bias_ih"].astype(np.float32),
                "bhh": data["lstm_bias_hh"].astype(np.float32),
            }
            self._hidden = int(self._lstm["whh"].shape[1])
            self._recurrent = True
            self._in_dim = int(self._lstm["wih"].shape[1])
        else:
            self._recurrent = False
            self._in_dim = int(self._layers[0][0].shape[1])
        bundle_obs = int(data["obs_dim"]) if "obs_dim" in data else None
        if bundle_obs is not None and bundle_obs != self._in_dim:
            raise ValueError(
                ".npz obs_dim {} 와 가중치 입력 {} 불일치 — 손상된 번들?".format(
                    bundle_obs, self._in_dim))
        if self._in_dim not in (C.OBS_DIM, C.POLICY_OBS_DIM):
            raise ValueError(
                ".npz 입력 {}차원: teacher {} 도 student {} 도 아닙니다; "
                "config.py 와 export 설정을 확인하세요".format(
                    self._in_dim, C.OBS_DIM, C.POLICY_OBS_DIM))
        self._backend = "numpy"

    @property
    def recurrent(self):
        return self._recurrent

    @property
    def hidden(self):
        """현재 LSTM hidden state (H,) — feed-forward 면 None.

        부상 파라미터 probe(injury_probe.npz)와 로깅에 사용합니다.
        """
        if not self._recurrent:
            return None
        h = self._h
        return h.reshape(-1)

    def reset(self):
        """LSTM hidden/cell 을 0 으로 — 에피소드 시작마다 필수 (feed-forward 는 no-op).

        학습에서 에피소드가 hidden=0 에서 시작하므로, 기립 후 정책 인계 시점과
        중단 후 재개 시점에 호출하지 않으면 이전 세션의 기억이 남습니다.
        """
        if not self._recurrent:
            return
        if self._backend == "numpy":
            self._h = np.zeros(self._hidden, dtype=np.float32)
            self._c = np.zeros(self._hidden, dtype=np.float32)
        elif self._backend == "onnx":
            self._h = np.zeros((1, 1, self._hidden), dtype=np.float32)
            self._c = np.zeros((1, 1, self._hidden), dtype=np.float32)

    def _verify_reference_io(self):
        """모델 옆의 reference_io.json (개발 PC onnxruntime 출력)과 대조.

        파일이 없으면 조용히 통과. 있는데 안 맞으면 모터를 건드리기 전에
        즉시 중단합니다 — 잘못된/오염된 export 를 하드웨어에 올리는 것을 방지.
        recurrent 모델의 pairs 는 hidden=0 에서 시작하는 **하나의 에피소드**라
        순서대로 실행하며, 검증 후 상태를 다시 리셋합니다.
        """
        ref_path = os.path.join(os.path.dirname(os.path.abspath(self._path)),
                                "reference_io.json")
        if not os.path.exists(ref_path):
            return
        with open(ref_path) as f:
            ref = json.load(f)
        self.reset()
        worst = 0.0
        for pair in ref["pairs"]:
            out = self(np.array(pair["obs"], dtype=np.float32))
            worst = max(worst, float(np.max(np.abs(out - np.array(pair["action"])))))
        self.reset()
        if worst > 1e-3:
            raise ValueError(
                "policy self-test MISMATCH: max|out-ref| = {:.3e} "
                "(reference: {})".format(worst, ref_path))
        print("policy self-test OK: {} pairs, max|out-ref| = {:.2e} "
              "[{}{}]".format(len(ref["pairs"]), worst, self._backend,
                              ", lstm" if self._recurrent else ""))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.reshape(-1).astype(np.float32)
        if x.shape[0] != C.OBS_DIM:
            raise ValueError(
                "obs dim {} != {}; "
                "export 시 GO1_ABS_JOINT_OBS=1 이었는지 확인하세요".format(
                    x.shape[0], C.OBS_DIM))
        if self._in_dim == C.POLICY_OBS_DIM:
            # student: policy 그룹(52)만 소비 — privileged tail(7)은 teacher 전용
            x = x[:C.POLICY_OBS_DIM]

        if self._backend == "numpy":
            h = x
            if self._lstm is not None:
                w = self._lstm
                gates = w["wih"] @ h + w["bih"] + w["whh"] @ self._h + w["bhh"]
                H = self._hidden
                i = 1.0 / (1.0 + np.exp(-gates[:H]))
                f = 1.0 / (1.0 + np.exp(-gates[H:2 * H]))
                g = np.tanh(gates[2 * H:3 * H])
                o = 1.0 / (1.0 + np.exp(-gates[3 * H:]))
                self._c = (f * self._c + i * g).astype(np.float32)
                self._h = (o * np.tanh(self._c)).astype(np.float32)
                h = self._h
            last = len(self._layers) - 1
            for k, (w, b) in enumerate(self._layers):
                h = h @ w.T + b
                if k < last:
                    h = np.where(h > 0.0, h, np.exp(h) - 1.0)  # ELU
            out = h
        elif self._backend == "onnx":
            x2 = x.reshape(1, -1)
            if self._recurrent:
                out, self._h, self._c = self._sess.run(
                    self._out_names,
                    {self._in_names[0]: x2,
                     self._in_names[1]: self._h,
                     self._in_names[2]: self._c})
            else:
                (out,) = self._sess.run(None, {self._input_name: x2})
        else:
            with self._torch.no_grad():
                out = self._module(
                    self._torch.from_numpy(x.reshape(1, -1))).numpy()
        action = np.asarray(out, dtype=np.float64).reshape(-1)
        if action.shape[0] != C.NUM_ACTIONS:
            raise ValueError("action dim {} != 12".format(action.shape[0]))
        return action
