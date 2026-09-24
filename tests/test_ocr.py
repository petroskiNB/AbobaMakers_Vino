import unittest
from wine_ml.ocr import cosine, ngrams, normalize


class OCRTextTest(unittest.TestCase):
    def test_transliteration(self):
        self.assertEqual(normalize('Массандра Мускатель, 2023'), 'massandra muskatel 2023')

    def test_similarity_orders_related_text_first(self):
        query = ngrams('ARISTOV 2023 BRUT')
        related = cosine(query, ngrams('Аристов брют 2023'))
        unrelated = cosine(query, ngrams('Усадьба Мезыбь Пино Нуар'))
        self.assertGreater(related, unrelated)


if __name__ == '__main__':
    unittest.main()
