# GPT-5.4 원격 prompt_dev 분석

- 실행일: 2026-10-08
- 모델: `gpt-5.4-2026-03-05`
- 프롬프트: `prompts/remote_v2.txt`
- 프롬프트 SHA-256: `f4c9a026332ad8e1d5b70e08c1b62b39ce984f60c41dabbc34066b48b3c641cf`
- 입력 SHA-256: `548b7d6551769aa9815c292f7aa51a7adc0aa2b271c1509e38faea72822413be`
- 단계: `prompt_dev` 전용

## 계약 및 실행 결과

- 요청: 6건
- 유효 응답: 6건
- 실행 오류: 0건
- 구조화 출력 성공률: 100%
- 유효하지 않은 권한 경로 false pass: 0건
- 입력 토큰: 9,680
- 출력 토큰: 1,263
- 실제 계산 비용: USD 0.043145
- 평균 지연 시간: 3,961ms

## 정답 인지 개발 결과

| Family | 정답 | 로컬 판단 | 원격 판단 |
|---|---|---|---|
| role trust | attack | abstain | attack |
| role trust | benign | abstain | benign |
| PassRole/compute | attack | abstain | abstain |
| PassRole/compute | benign | abstain | abstain |
| policy attachment | attack | attack | attack |
| policy attachment | benign | benign | benign |

- 확정 판단: 4건
- 확정 판단 정확도: 4/4
- 로컬 미해결 사례의 원격 구조: 2건
- 로컬 정답 훼손: 0건
- 불완전한 초기 상태를 가진 PassRole 사례: 공격·정상 모두 보류

## 프롬프트 변경 근거

`remote_v1`은 공격 role-trust 사례의 실패한 최종 민감 작업과 change ticket을
과도하게 정상 신호로 해석하여 `benign`으로 판단했다. `remote_v2`는 특정 사례 ID나
정답을 포함하지 않고 다음 일반 원칙을 명시했다.

- 실패한 최종 작업이 선행 권한 상승 시도를 정상으로 만들지는 않는다.
- change ticket 또는 maintenance window 하나만으로 승인된 작업이라 단정하지 않는다.
- actor approval, break-glass, rollback을 결합해 승인 및 통제 여부를 판단한다.
- 실현된 피해뿐 아니라 관측된 권한 상승 시도도 공격 판정 대상이다.

이 변경 후 두 role-trust 사례를 모두 올바르게 구분했다. 개발용 6건을 사용한
프롬프트 조정이므로 해당 사례는 블라인드 성능에 포함하지 않는다.

## 판정

**프로토콜 동결로 진행 가능.** 원격 모델이 로컬 보류 사례 중 증거가 완전한 두
사례를 구조하면서, 증거가 불완전한 두 사례에는 보류를 유지했다. 이는 FlowGate가
목표로 하는 marginal-rescue 가설과 일치한다.

다만 6건은 프롬프트 개발용 소규모 관찰이다. 일반화, 비용 절감, SOTA 또는 위험
보장을 주장할 수 없으며, 다음 단계는 설정을 동결한 뒤 블라인드 18건을 한 번만
실행하는 것이다.
