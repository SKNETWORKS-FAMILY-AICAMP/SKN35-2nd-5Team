from src.config import PROCESSED_DIR
from src.data.loader import load_raw_test, load_raw_train
from src.data.preprocess import preprocess_pipeline


def main():
    train_raw = load_raw_train()
    test_raw = load_raw_test()

    train_processed = preprocess_pipeline(train_raw)
    test_processed = preprocess_pipeline(test_raw, reference=train_raw)

    train_processed.to_csv(PROCESSED_DIR / "train_processed_v2.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test_processed_v2.csv", index=False)

    print("train_processed shape:", train_processed.shape)
    print("test_processed shape:", test_processed.shape)


if __name__ == "__main__":
    main()
