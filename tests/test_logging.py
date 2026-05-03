import logging
from pathlib import Path

from src.utils.logging_utils import configure_logging


def test_logger_initialization():

    logger = logging.getLogger("truthlens")

    logger.info("test log")

    assert logger is not None


def test_configure_logging_allows_adding_file_handler_later(tmp_path: Path) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        root_logger.handlers.clear()

        configure_logging(level=logging.INFO)

        log_path = tmp_path / "app.log"
        configure_logging(level=logging.INFO, log_file=log_path)

        logging.getLogger("truthlens.logging").info("file logging check")
        for handler in root_logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        assert log_path.exists()
        assert "file logging check" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in list(root_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
