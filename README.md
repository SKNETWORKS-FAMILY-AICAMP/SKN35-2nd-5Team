# SKN35-2nd-5Team

## Duolingo 데이터셋 준비

대용량 원본 로그를 사용자 단위로 축소한 최종 데이터셋은 다음 경로에 있습니다.

```text
data/duolingo_churn_dataset.csv
```

이 파일은 GitHub의 일반 파일 크기 제한을 초과하므로 **Git LFS**로 관리합니다.
CSV는 행 단위로 임의 추출하지 않았으며, 선정된 사용자의 전체 학습 기록을
`user_id`, `timestamp` 순으로 보존합니다.

### 최초 1회 설정

Git LFS를 설치한 뒤 저장소별로 다음 명령을 실행합니다.

```bash
git lfs install
```

Windows에서 Git LFS가 설치되어 있는지는 다음 명령으로 확인할 수 있습니다.

```bash
git lfs version
```

### 저장소 클론 및 데이터 다운로드

Git LFS가 설치된 상태에서 저장소를 평소처럼 클론하면 CSV도 함께 다운로드됩니다.

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN35-2nd-5Team.git
cd SKN35-2nd-5Team
git lfs pull
```

CSV 대신 몇 줄짜리 LFS 포인터 파일만 보이는 경우 다음 명령을 실행합니다.

```bash
git lfs install
git lfs pull
```

현재 LFS로 관리되는 파일은 다음 명령으로 확인할 수 있습니다.

```bash
git lfs ls-files
```

### 데이터셋 재생성

원본 `data/learning_traces.13m.csv`가 있을 때 다음 명령으로 동일한 방식의
축소 데이터셋을 재생성할 수 있습니다.

```bash
uv run python prepare_duolingo_dataset.py --output data/duolingo_churn_dataset.csv
```

원본 CSV는 용량이 매우 크므로 Git에 추가하지 않습니다. 재생성된 최종 CSV는
반드시 위 경로에 저장하여 기존 Git LFS 추적 규칙을 적용합니다.
