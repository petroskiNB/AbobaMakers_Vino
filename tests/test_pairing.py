import unittest
from wine_ml.pairing import assess_pairing

class PairingTest(unittest.TestCase):
    white = {'Категория': 'Белое', 'Название вина': 'Белое сухое'}
    red = {'Категория': 'Красное', 'Название вина': 'Красное сухое'}

    def test_dish_changes_score_and_advice(self):
        fish = assess_pairing(self.white, 'рыба')
        meat = assess_pairing(self.white, 'мясо')
        self.assertGreater(fish['score'], meat['score'])
        self.assertNotEqual(fish['recommendations'], meat['recommendations'])
        self.assertGreater(assess_pairing(self.red, 'мясо')['score'], assess_pairing(self.red, 'рыба')['score'])

    def test_preparation_and_spice(self):
        plain = assess_pairing(self.red, 'курица')['score']
        self.assertLess(assess_pairing(self.red, 'острая курица с чили')['score'], plain)
        self.assertEqual(assess_pairing(self.red, 'курица не острая')['score'], plain)
        self.assertGreater(assess_pairing(self.white, 'рыба в сливочном соусе')['score'], assess_pairing(self.white, 'рыба')['score'])

    def test_unknown_has_no_score(self):
        for dish in ['абракадабра', '12345', 'самолет', 'острое']:
            self.assertIsNone(assess_pairing(self.white, dish)['score'])
        self.assertIsNone(assess_pairing({}, 'рыба')['score'])

    def test_sweetness_and_sparkling(self):
        sweet = {'Категория': 'Белое', 'Название вина': 'Белое сладкое'}
        self.assertGreater(assess_pairing(sweet, 'торт')['score'], assess_pairing(self.white, 'торт')['score'])
        self.assertIsNone(assess_pairing({'Категория': 'Белое'}, 'торт')['score'])
        self.assertEqual(assess_pairing({'Категория': 'Белое', 'Название вина': 'Спуманте брют'}, 'креветки')['wine_style'], 'игристое')

    def test_compounds_exclusions_and_case(self):
        self.assertEqual(assess_pairing(self.white, 'РЫБА')['score'], assess_pairing(self.white, 'рыба')['score'])
        result = assess_pairing(self.white, 'паста с креветками без мяса')
        self.assertNotIn('мясо', result['recognized_foods'])
        self.assertIn('морепродукты', result['recognized_foods'])
        self.assertTrue(0 <= result['score'] <= 100)

if __name__ == '__main__':
    unittest.main()
