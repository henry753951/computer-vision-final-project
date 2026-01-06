import os
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # 忽略未來警告（可選）[web:113]
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 關 oneDNN（可選）[web:114]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 不顯示 INFO（可選）[web:115]


def enrty_point():
    from cli import app
    app()

if __name__ == "__main__":
    enrty_point()
