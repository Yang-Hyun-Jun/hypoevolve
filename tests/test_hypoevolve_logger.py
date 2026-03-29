import tempfile
import unittest
from pathlib import Path

from hypoevolve.logger import configure_logger, logger


class TestHypoEvolveLogger(unittest.TestCase):
    def test_configure_logger_writes_file_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'hypoevolve.log'
            configured = configure_logger('INFO', log_path)
            configured.info('hello logger')
            self.assertTrue(log_path.exists())
            self.assertIn('hello logger', log_path.read_text(encoding='utf-8'))
            self.assertIs(configured, logger)


if __name__ == '__main__':
    unittest.main()
