"""
A helper class for progress logging
"""
from datetime import datetime
import logging


def logger(frequency=1000, maximum=0, initial=0, fixed_message=''):
    """
    Closure to provide a progress logger; the logger needs to be run every loop iteration
    Usage:
        some_own_logger = algo_logger.logger(frequency=100, maximum=some_maximum)
        each loop iteration: some_own_logger(one_time_message) (one_time_message is optional)
    :param frequency: message will be printed every 'frequency' step; default 1000
    :param maximum: value at which the iteration will stop; useful for ETA; default 0: no maximum set
    :param initial: initial counter value (iteration may not start from 0); default 0
    :param fixed_message: a message to display each time; default ''
    :return: The logging function
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s: %(message)s')
    start_time = datetime.now()
    counter = initial

    def _log_progress(additional_text=''):
        """
        This function should be called every main loop iteration
        It increments the internal counter, and displays a message each frequency step
        The message contains the estimated time of arrival (ETA) if maximum has been defined, time elapsed,
        the defined fixed message, and the additional text. Miliseconds are truncated
        :param additional_text: additional text to display in the log
        :return: None
        """
        nonlocal  counter
        counter += 1
        if counter % frequency == 0:
            max_text = f' of {maximum:,} ' if maximum else ' '
            processed = f'{counter:,}{max_text}processed; '
            eta_text = '; ETA: ' + (start_time + (datetime.now() - start_time)
                                    * (maximum / counter)).strftime("%Y-%m-%d %H:%M:%S") + '; ' if maximum else ''
            elapsed = 'time elapsed: ' + str(datetime.now() - start_time).split(".")[0] + eta_text
            additional_text = ' ' if additional_text else '' + additional_text

            logging.info('%s%s%s%s', processed, elapsed, fixed_message, additional_text)

    return _log_progress
