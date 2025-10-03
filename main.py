# main.py
from modules.logging_config import init_logger
from modules.automation import start_scheduler

init_logger()
try:
    start_scheduler()
except Exception:
    pass

if __name__ == "__main__":
    import sys
    mode = "gui"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        mode = "cli"

    if mode == "cli":
        from modules.cli import cli_main
        cli_main()
    else:
        from modules.gui import start_gui
        start_gui()
