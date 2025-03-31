import logging
import os
import sys
from datetime import datetime
import colorlog

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

class OrbitLogger:
    """Extension wrapper for Logger with orbit-specific formatting"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def header(self, text):
        """Display a clean header in logs without standard prefixes"""
        width = len(text)
        border = "=" * width
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream = handler.stream
                stream.write(f"\n{Colors.YELLOW}{border}{Colors.RESET}\n")
                stream.write(f"{Colors.YELLOW}  {Colors.BOLD}{text}{Colors.RESET}{Colors.YELLOW}  {Colors.RESET}\n")
                stream.write(f"{Colors.YELLOW}{border}{Colors.RESET}\n\n")
                stream.flush()
                
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.info("\n" + border)
                self.logger.info(f"{text}")
                self.logger.info(border + "\n")
        
    def section(self, text):
        """Display a clean section header without standard prefixes"""
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream = handler.stream
                stream.write(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.RESET}\n")
                stream.write(f"{Colors.CYAN}{'─' * (len(text) + 2)}{Colors.RESET}\n\n")
                stream.flush()
                
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.info(f"\n▶ {text}")
                self.logger.info("─" * (len(text) + 3) + "\n")  
    
    def result_box(self, title, data_dict):
        """Display a box containing results"""
        
        max_key_length = max([len(str(k)) for k in data_dict.keys()])
        max_val_length = max([len(str(v)) for v in data_dict.values()])
        width = max(len(title), max_key_length + max_val_length + 5)
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream = handler.stream
                stream.write(f"\n{Colors.GREEN}┌{'─' * (width + 2)}┐{Colors.RESET}\n")
                stream.write(f"{Colors.GREEN}│ {Colors.BOLD}{title}{Colors.RESET}{Colors.GREEN}{' ' * (width - len(title) + 1)}│{Colors.RESET}\n")
                stream.write(f"{Colors.GREEN}├{'─' * (width + 2)}┤{Colors.RESET}\n")
                
                for key, value in data_dict.items():
                    key_str = str(key)
                    val_str = str(value)
                    padding = width - len(key_str) - len(val_str) - 2
                    stream.write(f"{Colors.GREEN}│ {Colors.BOLD}{key_str}{Colors.RESET}: {Colors.CYAN}{val_str}{Colors.RESET}{' ' * padding} {Colors.GREEN}│{Colors.RESET}\n")
                
                stream.write(f"{Colors.GREEN}└{'─' * (width + 2)}┘{Colors.RESET}\n\n")
                stream.flush()
                
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.info(f"\n┌{'─' * (width + 2)}┐")
                self.logger.info(f"│ {title}{' ' * (width - len(title) + 1)}│")
                self.logger.info(f"├{'─' * (width + 2)}┤")
                
                for key, value in data_dict.items():
                    key_str = str(key)
                    val_str = str(value)
                    padding = width - len(key_str) - len(val_str) - 2
                    self.logger.info(f"│ {key_str}: {val_str}{' ' * padding} │")

                self.logger.info(f"└{'─' * (width + 2)}┘\n")
                    
    # standar logging methods
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


def setup_logger(name="orbit_determination", log_level=logging.INFO, console=True, log_file=False, log_dir="logs"):
    """
    Setup a colored logger with file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (default: INFO)
        console: Whether to log to console
        log_file: Whether to log to file
        log_dir: Directory to store log files

    Returns:
        OrbitLogger: Enhanced logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []
    
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s - [%(name)s] -  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return OrbitLogger(logger)