import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CatalogTest(unittest.TestCase):
    def test_conflicts_and_csv_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/'images'
            images.mkdir()
            for name, content in [('vino_0123456789.jpg', b'same'), ('other_0123456789.jpg', b'same'),
                                  ('small_vino_0123456789.jpg', b'small'),
                                  ('pair_0123456789.jpg', b'a'), ('pair_abcdef1234.jpg', b'b')]:
                (images/name).write_bytes(content)
            catalog = root/'catalog.csv'
            with catalog.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Slug', 'Название фото', 'Описание'])
                writer.writerows([['a', 'vino.jpg', 'Серия "Тест", белое'],
                                  ['a', 'vino.jpg', 'Серия "Тест", белое'], ['b', 'other.jpg', ''],
                                  ['c', 'pair.jpg', ''], ['d', 'missing.jpg', '']])
            subprocess.run([sys.executable, 'scripts/prepare_catalog.py', '--images', str(images),
                            '--catalog', str(catalog), '--out', str(root/'out')], check=True, capture_output=True)
            report = json.loads((root/'out/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['csv_rows'], 5)
            self.assertEqual(report['unique_rows'], 4)
            self.assertEqual(report['content_conflict_groups'], 1)
            self.assertEqual(report['link_statuses'], {'candidate': 2, 'ambiguous': 1, 'missing': 1})
            links = json.loads((root/'out/candidate_links.json').read_text(encoding='utf-8'))
            self.assertFalse(any(row['verified'] for row in links))
            self.assertEqual(links[0]['card']['Описание'], 'Серия "Тест", белое')


if __name__ == '__main__':
    unittest.main()
