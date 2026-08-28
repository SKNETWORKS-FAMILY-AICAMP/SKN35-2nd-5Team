"""퇴사 예측 생성과 DB 적재를 시작하는 실행 파일.

실행 방법:
    uv run python insert_database.py
"""

from src.data.insert_dataset import main

if __name__ == "__main__":
    main()
