# FlowGate 파일럿: 지금 시작하는 순서

명령어를 포함한 단계별 실행 절차는 [`NEXT_STEPS_KO.md`](NEXT_STEPS_KO.md)를 먼저
따르십시오. 이 문서는 전체 흐름을 짧게 요약합니다.

지금 목표는 논문 성능을 증명하는 것이 아니라, **한 달짜리 연구를 계속할 가치가 있는지 24건으로 판정하는 것**입니다. 현재 노트북에 GPU가 없어도 준비·검증·봉인은 모두 할 수 있고, RTX 3090 PC는 로컬 모델 추론만 담당합니다.

## 오늘 할 일

1. 이 노트북에 이미 만들어진 `dist/flowgate-prompt-dev-gpu.tar.gz`만 RTX 3090 PC로 복사합니다.
2. GPU PC에서 압축을 풀고 내부 `GPU_README.md`의 명령을 실행합니다.
3. 다음 네 파일만 다시 이 노트북으로 가져옵니다.
   - `local_responses.jsonl`
   - `local_run_records.jsonl`
   - `local_errors.jsonl`
   - `local_responses.jsonl.manifest.json`
4. 이 여섯 개발 사례에서 JSON 오류와 명백한 판단 오류만 고칩니다. 블라인드 18건은 아직 실행하거나 정답을 열지 않습니다.

예시 전송 명령은 다음과 같습니다. 주소와 경로만 자신의 GPU PC에 맞게 바꾸면 됩니다.

```bash
scp dist/flowgate-prompt-dev-gpu.tar.gz <GPU_USER>@<GPU_HOST>:~/
```

GPU PC에서는:

```bash
mkdir -p ~/flowgate-prompt-dev
tar -xzf ~/flowgate-prompt-dev-gpu.tar.gz -C ~/flowgate-prompt-dev
cd ~/flowgate-prompt-dev
less GPU_README.md
```

Windows PC라면 WSL2 Ubuntu에서 같은 순서로 실행하는 편이 가장 단순합니다. vLLM 서버는 `127.0.0.1`에만 열리므로 외부 네트워크에 노출할 필요가 없습니다.

## 개발 6건이 정상일 때

다음 네 가지가 확인되면 프롬프트와 설정을 동결합니다.

- 여섯 요청이 모두 처리되고 `local_errors.jsonl`이 비어 있음
- 출력이 전부 엄격한 JSON 계약을 통과함
- 공격/정상 판정이 family 이름만 따라가지 않음
- 근거로 입력에 존재하는 event ID만 인용함

그다음 `README.md`의 순서대로 protocol freeze를 만들고, 블라인드 18건을 로컬 모델과 원격 모델에 각각 한 번만 실행합니다. 원격 모델은 이 노트북에서만 호출하며 GPU PC에는 API 키를 보내지 않습니다.

## 원격 모델 예산

권장 고정 스냅샷은 `gpt-5.4-mini-2026-03-17`입니다. 2026-10-07 공식 표시 가격은 입력 USD 0.75/M, 출력 USD 4.50/M입니다. 18건 전체를 700 출력 토큰 상한으로 호출할 때 출력비 상한은 USD 0.0567이고 입력비가 추가되므로, 만 원 예산보다 충분히 작습니다. 다만 실제 비용은 provider usage로 기록하며, 호출 직전에 가격과 계정의 데이터 보존 설정을 다시 확인해야 합니다.

## 결과를 해석하는 기준

- `CONTINUE`: 증거 재구성이 안정적이고, 원격 모델이 로컬 오답을 실제로 구조하며, 4회 호출 예산에서 MCC가 개선됨
- `PIVOT`: 권한 증거 그래프는 유용하지만 큰 모델이 로컬 모델의 오류를 충분히 구조하지 못함
- `KILL`: 개인정보/계약을 지킬 수 없거나, 정답을 아는 Oracle조차 라우팅 이득이 없음
- `INCONCLUSIVE`: 로컬 오답이 너무 적거나 첫 시도 출력이 부족해 비교할 수 없음

24건 결과로 SOTA나 일반화 성능을 주장하지 않습니다. 이 단계가 통과하면 계정·공격 family를 분리한 더 큰 train/calibration/test 실험으로 확장합니다.
