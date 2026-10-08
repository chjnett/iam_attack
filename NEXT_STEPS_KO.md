# FlowGate 다음 단계 실행 가이드

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-07
- Verification Status: PROMPT DEVELOPMENT VERIFIED; BLIND RUN NOT STARTED
- Version Label: flowgate_pilot_execution_v1

이 문서는 `/Users/cheonhyeonjun/iam_attack`을 기준으로 한다. RTX 3090의 로컬
개발용 6건과 GPT-5.4의 원격 개발용 6건은 완료되었다. 최종 원격 프롬프트는
`prompts/remote_v2.txt`이며, 블라인드 18건과 프로토콜 동결은 아직 수행하지 않았다.

## 이번 주 목표

첫 목표는 논문 성능을 주장하는 것이 아니라 다음 한 문장을 검증하는 것이다.

> 동일한 권한 증거를 보더라도 로컬 모델이 틀리고 원격 모델이 맞는 사례가 존재하며,
> FlowGate가 그런 사례를 적은 원격 호출로 선택할 수 있는가?

실험은 반드시 아래 순서를 따른다.

1. 개발용 6건을 RTX 3090에서 실행한다.
2. 출력 계약과 판단 오류를 확인하고 프롬프트를 마지막으로 수정한다.
3. 코드·프롬프트·모델·가격·라우팅 규칙을 동결한다.
4. 블라인드 18건을 로컬/원격 모델에 각각 한 번만 실행한다.
5. 모든 출력을 봉인한 뒤 정답을 공개하고 평가한다.

블라인드 실행 전에 정답을 열거나, 결과를 본 뒤 프롬프트를 고치면 해당 실험은
무효다. 새 버전과 새 블라인드 세트가 필요하다.

## 0. 노트북 준비

```bash
cd /Users/cheonhyeonjun/iam_attack
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

성공 기준: `Ran 60 tests`와 `OK`가 출력된다.

개발 결과를 노트북에서 검증할 때 사용할 동일한 6건도 준비한다. 처음 한 번만
실행한다.

```bash
mkdir -p work/prompt_dev results/prompt_dev
flowgate-pilot export-split \
  --requests data/generated/gpu/requests.jsonl \
  --labels data/generated/private/prompt_dev_labels.jsonl \
  --split prompt_dev \
  --out work/prompt_dev/requests.jsonl \
  --manifest-out work/prompt_dev/requests.manifest.json
```

GPU 전송 파일의 무결성도 확인한다.

```bash
cd /Users/cheonhyeonjun/iam_attack/dist
shasum -a 256 -c flowgate-prompt-dev-gpu-v2.tar.gz.sha256
```

성공 기준: `flowgate-prompt-dev-gpu-v2.tar.gz: OK`. v2는 enum 배열을
JSON Schema로 강제해 Qwen의 `uncertainty_reasons` 형식 오류를 막는다.

## 1. 개발용 6건을 GPU PC로 전송

노트북에서 다음 명령을 실행한다. `<GPU_USER>`와 `<GPU_HOST>`만 바꾼다.

```bash
scp /Users/cheonhyeonjun/iam_attack/dist/flowgate-prompt-dev-gpu-v2.tar.gz \
  <GPU_USER>@<GPU_HOST>:~/
```

GPU가 Windows PC라면 WSL2 Ubuntu 터미널에서 이후 명령을 실행한다.

## 2. RTX 3090 환경 준비

GPU PC에서:

```bash
mkdir -p ~/flowgate-prompt-dev
tar -xzf ~/flowgate-prompt-dev-gpu-v2.tar.gz -C ~/flowgate-prompt-dev
cd ~/flowgate-prompt-dev

uv venv --python 3.12 --seed .venv-vllm
source .venv-vllm/bin/activate
uv pip install 'vllm==0.31.0' --torch-backend=auto
```

환경을 확인한다.

```bash
nvidia-smi
python3 scripts/capture_runtime.py
```

다음으로 로컬 모델 서버를 실행한다.

```bash
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key

vllm serve "$FLOWGATE_LOCAL_MODEL" \
  --revision "$FLOWGATE_LOCAL_REVISION" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --api-key "$FLOWGATE_LOCAL_KEY"
```

서버는 이 터미널에 계속 실행해 둔다. 인터넷에 노출하지 말고 `127.0.0.1`만
사용한다.

## 3. 개발용 6건 실행

GPU PC에서 새 터미널을 열고:

```bash
cd ~/flowgate-prompt-dev
source .venv-vllm/bin/activate
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
mkdir -p results

PYTHONPATH=src python3 -m flowgate.cli run-batch \
  --requests data/requests.jsonl \
  --prompt prompts/local_v1.txt \
  --out results/local_responses.jsonl \
  --records results/local_run_records.jsonl \
  --errors results/local_errors.jsonl \
  --run-id prompt-dev-local-v1 \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id "$FLOWGATE_LOCAL_MODEL" \
  --model-revision "$FLOWGATE_LOCAL_REVISION" \
  --quantization awq \
  --api-key-env FLOWGATE_LOCAL_KEY \
  --phase prompt_dev
```

다음 네 파일만 노트북으로 가져온다.

```text
results/local_responses.jsonl
results/local_run_records.jsonl
results/local_errors.jsonl
results/local_responses.jsonl.manifest.json
```

GPU PC에서 노트북으로 직접 보낼 수 있다면:

```bash
scp results/local_responses.jsonl \
    results/local_run_records.jsonl \
    results/local_errors.jsonl \
    results/local_responses.jsonl.manifest.json \
    <LAPTOP_USER>@<LAPTOP_HOST>:/Users/cheonhyeonjun/iam_attack/results/prompt_dev/
```

직접 접속이 어렵다면 USB나 암호화된 개인 저장소로 옮긴다. API 키, 모델 캐시,
원본 로그는 복사하지 않는다.

## 4. 개발 결과 확인

노트북에서:

```bash
cd /Users/cheonhyeonjun/iam_attack
source .venv/bin/activate

flowgate-pilot stats \
  --requests work/prompt_dev/requests.jsonl \
  --responses results/prompt_dev/local_responses.jsonl \
  --errors results/prompt_dev/local_errors.jsonl \
  --out results/prompt_dev/local_stats.json
```

다음 조건을 모두 확인한다.

- 여섯 요청이 모두 처리됨
- `local_errors.jsonl`이 비어 있음
- 여섯 응답이 JSON 계약을 통과함
- 입력에 존재하지 않는 event ID를 인용하지 않음
- family 이름만 보고 공격 여부를 결정하지 않음
- `abstain` 사례는 어떤 증거가 부족한지 설명함

이 단계에서는 `prompt_dev` 정답만 볼 수 있다. 블라인드 정답 파일
`data/generated/private/blind_labels.jsonl`은 열지 않는다.

## 5. 동결 전 결정

개발 6건 결과를 보고 다음 중 하나를 선택한다.

- **진행:** JSON 오류가 없고 판단 형식이 안정적이다.
- **프롬프트 1회 수정:** 형식 또는 판단 지시가 명백히 부족하다. 수정 후 새 run ID로
  개발 6건을 다시 실행한다. 기존 결과를 덮어쓰지 말고 `prompt_dev_v2`처럼 새 출력
  디렉터리와 새 GPU 번들을 사용한다.
- **중단:** 3090에서 모델이 안정적으로 실행되지 않거나 출력 계약을 지키지 못한다.

진행을 선택하면 그 이후에는 `src/`, `prompts/`, `schemas/`, 모델 ID, 모델 revision,
vLLM 버전, 라우팅 점수, 호출 예산을 바꾸지 않는다.

## 6. 프로토콜 동결 이후

동결부터 평가까지의 정확한 명령은 `README.md`의 다음 절을 순서대로 실행한다.

1. `Freeze the protocol`
2. `Stage 2 — export and run all 18 blind cases once`
3. 로컬 블라인드 18건 실행
4. 노트북에서 원격 모델 18건 counterfactual 실행
5. FlowGate가 정확히 4건 선택
6. `seal-outputs`
7. 블라인드 정답 공개
8. `evaluate` 및 `compare`

원격 모델은 GPU PC가 아니라 신뢰된 노트북에서만 호출한다. GPU PC에는 원격 API
키, 원격 프롬프트, 블라인드 정답을 보내지 않는다.

## 최종 산출물

| 산출물 | 위치 | 성공 기준 |
|---|---|---|
| 개발 로컬 출력 | `results/prompt_dev/` | 6건, 첫 시도 오류 0 |
| 프로토콜 동결 | `results/protocol-freeze.json` | 블라인드 실행 전 생성 |
| 블라인드 로컬/원격 출력 | `results/blind/` | 각각 18건, 재시도 없음 |
| 출력 봉인 | `results/blind/output-seal.json` | 정답 공개 전 생성 |
| 평가 | `results/blind/evaluation.json` | Go/Pivot/Kill/Inconclusive 포함 |
| Pareto 비교 | `results/blind/pareto.json`, `.csv` | 비용·MCC·FNR·coverage 포함 |

## 해석 제한

24건은 구현 가능성과 로컬/원격 상보성을 확인하는 dry run이다. 이 결과로 SOTA,
일반화 성능, 위험 보장, 실제 기업 환경 우월성을 주장하지 않는다. `CONTINUE`가
나오면 그때 계정과 공격 family를 분리한 더 큰 train/calibration/test 데이터셋을
설계한다.
