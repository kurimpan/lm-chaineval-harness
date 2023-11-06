import json

class Evaluator:
    def __init__(self, metrics_path):
        # ここでメトリックを読み込む
        with open(metrics_path, 'r') as file:
            self.metrics = json.load(file)
    
    def calculate(self, data, keymap, record):
        # 評価結果を記録する辞書
        evaluation_results = {}

        # すべてのデータポイントに対して評価を行う
        for item in data:
            # 各データポイントの評価結果を保持する辞書
            item_evaluation = {}
            # 定義されたメトリックごとに計算
            for metric in self.metrics:
                # メトリックに応じた評価関数を実行
                # ここでは例として、単純な一致率を計算している
                if metric == 'accuracy':
                    prediction = item[keymap.get('model_output')]
                    ground_truth = item[keymap.get('ground_truth')]
                    item_evaluation[metric] = self._calculate_accuracy(prediction, ground_truth)
                # 他のメトリクスに対する評価関数も同様に追加可能

            # このデータポイントの評価結果を追加
            evaluation_results[item[keymap.get('id', 'id')]] = item_evaluation
        
        # 結果を記録に追加
        record['evaluation_results'] = evaluation_results
        
        # 最後に全体の平均スコアなどを計算する場合も可能
        overall_score = self._calculate_overall_score(evaluation_results)
        record['overall_score'] = overall_score

        # 評価の完了後にrecordを返す
        return record
    
    def _calculate_accuracy(self, prediction, ground_truth):
        # 単純な正解率を計算する関数
        correct = 0
        total = len(ground_truth)
        for pred, truth in zip(prediction, ground_truth):
            if pred == truth:
                correct += 1
        return correct / total if total > 0 else 0
    
    def _calculate_overall_score(self, evaluation_results):
        # 全体の平均スコアを計算する関数
        total_score = 0
        count = 0
        for _, item_evaluation in evaluation_results.items():
            for score in item_evaluation.values():
                total_score += score
                count += 1
        return total_score / count if count > 0 else 0

# 以下のコードでEvaluatorクラスを使用する
# evaluator = Evaluator(metrics_path)
# results = evaluator.calculate(data, keymap, record)
