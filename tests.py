from main import *

class UnitTest(unittest.TestCase):
    def test_NASA_status(self):
        status_code = requests.get(NASA_VOYAGER_1_URL, headers).status_code
        self.assertEqual(status_code, 200)

    def test_RFC1149_history_status(self):
        status_code = requests.get(RFC1149_HISTORY_URL, headers).status_code
        self.assertEqual(status_code, 200)

    def test_unicode_status(self):
        status_code = requests.get(UNICODE_URL, headers).status_code
        self.assertEqual(status_code, 200)

    def test_genesis_block_status(self):
        status_code = requests.get(GENESIS_BLOCK_BITCOIN_URL, headers).status_code
        self.assertEqual(status_code, 200)

    def test_kr2_isbn_status(self):
        status_code = requests.get(KR2_ISBN10_URL, headers).status_code
        self.assertEqual(status_code, 200)

    def test_correct_flag(self):
        answer = main()
        self.assertEqual(sha256(answer.encode("utf-8")).hexdigest(),
                         "d311f26ea1a995af669a62758ad5e0ce2583331059fbfc5c04cc84b2d41f4aed")

if __name__ == '__main__':
    unittest.main()