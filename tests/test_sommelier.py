import json
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from wine_ml.sommelier import Advice, yandex_pairing, PairingUnavailable, recommend

class SommelierTest(unittest.TestCase):
    config = {'YANDEX_API_KEY': 'test-secret', 'YANDEX_FOLDER_ID': 'test-folder', 'YANDEX_MODEL': 'yandexgpt/rc'}

    def test_missing_key_makes_no_request(self):
        with patch('wine_ml.sommelier.request.urlopen') as call:
            with self.assertRaises(PairingUnavailable):
                yandex_pairing({}, 'рыба', {**self.config, 'YANDEX_API_KEY': ''})
            call.assert_not_called()

    def test_success_payload_and_no_secret_in_response(self):
        answer = dict(status='estimated', score=80, label='Хорошо', explanation='Подойдёт', reasons=['Причина'], recommendations=['Совет'], question='')
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}).encode()
        with patch('wine_ml.sommelier.request.urlopen', return_value=response) as call:
            result = yandex_pairing({'Название вина':'Тест','Описание':'Свежий вкус','private':'omit'}, 'лосось на гриле', self.config)
            self.assertEqual(result['method'], 'yandexgpt')
            payload = json.loads(call.call_args.args[0].data)
            self.assertNotIn('private', payload['messages'][1]['content'])
            self.assertNotIn('test-secret', json.dumps(result))

    def test_auth_error_is_sanitized(self):
        with patch('wine_ml.sommelier.request.urlopen', side_effect=HTTPError('url',401,'test-secret',{},None)):
            with self.assertRaises(PairingUnavailable) as ctx:
                yandex_pairing({},'рыба',self.config)
            self.assertNotIn('test-secret', str(ctx.exception))

    def test_invalid_and_truncated_response(self):
        for choice in [{'finish_reason':'length'}, {'finish_reason':'stop','message':{'content':'not json'}}]:
            response=MagicMock()
            response.__enter__.return_value.read.return_value=json.dumps({'choices':[choice]}).encode()
            with patch('wine_ml.sommelier.request.urlopen',return_value=response):
                with self.assertRaises(PairingUnavailable):
                    yandex_pairing({},'рыба',self.config)

    def test_clarification_has_no_score(self):
        with self.assertRaises(ValueError):
            Advice(status='needs_details',score=80,label='Уточните',explanation='Нужны данные',reasons=[],recommendations=[],question='Какой соус?')

    def test_explicit_rules_mode(self):
        with patch('wine_ml.sommelier.settings',return_value={'PAIRING_PROVIDER':'rules'}):
            self.assertEqual(recommend({'Категория':'Белое'},'рыба')['method'],'rules-v1')
