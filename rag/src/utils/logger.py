from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}")
logger.add("logs/rag.log", rotation="10 MB", retention="10 days", level="DEBUG")