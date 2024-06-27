"""Generate logger"""

from logging import StreamHandler, FileHandler, Formatter, Logger, INFO
from datetime import datetime
import pprint
import os


class LoggerEx(Logger):
    """
    Extend Logger for print multi-line info
    ------
    Instances of the Logger class represent a single logging channel. A
    "logging channel" indicates an area of an application. Exactly how an
    "area" is defined is up to the application developer. Since an
    application can have any number of areas, logging channels are identified
    by a unique string. Application areas can be nested (e.g. an area
    of "input processing" might include sub-areas "read CSV files", "read
    XLS files" and "read Gnumeric files"). To cater for this natural nesting,
    channel names are organized into a namespace hierarchy where levels are
    separated by periods, much like the Java or Python package namespace. So
    in the instance given above, channel names might be "input" for the upper
    level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
    There is no arbitrary limit to the depth of nesting.
    """

    def infoml(self: Logger, msg: object):
        """Expand info method for multi-line text
        ----
        Log 'msg % args' with severity 'INFO'.
        To pass exception information, use the keyword argument exc_info with a true value, e.g.
        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)

        Args:
            msg (str): message
        """

        if not isinstance(msg, str):
            msg = pprint.pformat(msg)
            msg: str

        sp_txt = msg.split("\n")
        for line in sp_txt:
            self.info(line)
        return


def set_logger(
    name: str, filename: str = "log/main.log", add_date: bool = True
) -> LoggerEx:
    """Generate logger

    Args:
        name (str): Displayed name in logs.
        filename (str, optional): Log file name. Defaults to "log/main.log".
        add_date (bool, optional): If `add_date=True`, date and time are added to the log file name.
        Defaults to True.

    Returns:
        LoggerEx: Configured logger instance.
    """

    if add_date:
        dt_now = datetime.now()
        dtime = dt_now.strftime("%Y%m%d_%H%M%S")
        dirname, filename = os.path.split(filename)
        filename = dtime + "_" + filename
        filename = os.path.join(dirname, filename)

    logger = LoggerEx(name)
    handler1 = StreamHandler()
    handler1.setFormatter(
        Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(
        Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    handler1.setLevel(INFO)
    handler2.setLevel(INFO)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(INFO)
    return logger
