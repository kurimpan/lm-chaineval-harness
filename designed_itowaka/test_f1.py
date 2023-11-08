import unittest
from memo2_f1 import load_evaluator

class TestF1Score(unittest.TestCase):
    def setUp(self):
        # テストデータと期待される結果をセットアップ
        self.test_data = [
            {"model_output": "1", "result": "1"},
            {"model_output": "0", "result": "0"},
            {"model_output": "1", "result": "0"},
            {"model_output": "0", "result": "1"}
        ]
        
        # キーマッピング
        self.keymap = {"result": "result"}
        
        # レコード（このテストでは使用しないが、フォーマットには含める）
        self.record = {}

        # Evaluator インスタンスの生成
        self.evaluator = load_evaluator("f1")
        
        # 期待されるF1スコア
        self.expected_f1_score = 0.5  # これはテストに適した値に置き換える

    def test_f1_score(self):
        # calculateメソッドを使用してF1スコアを計算
        calculated_f1_score = self.evaluator.calculate(self.test_data, self.keymap, self.record)
        
        # F1スコアが期待される値と一致するかアサートする
        self.assertAlmostEqual(calculated_f1_score, self.expected_f1_score, places=2,
                               msg=f"F1 score should be approximately {self.expected_f1_score}")

# テストスイートを実行する
if __name__ == '__main__':
    unittest.main()

