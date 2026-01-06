import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {"info": "cyan", "warning": "yellow", "error": "bold red", "success": "bold green"}
)

console = Console(theme=custom_theme)


def setup_logger(name: str = "AI_CLI", level: str = "INFO") -> logging.Logger:
    """
    設定並回傳一個整合 Rich 的 Logger
    """
    # 避免重複設定
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = setup_logger()
