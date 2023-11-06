class Evaluator:
    """
    モデルを使用して、生成されたプロンプトに対する回答を取得。
    指定されたmetricsオプションに基づいて評価を行い、スコアを算出。
    """
    
    def __init__(self, metrics):
        # メトリクスを初期化する。メトリクスはスコア計算に使用される評価関数を含む
        self.metrics = metrics

    def evaluate(self, model_output, ground_truth):
        # モデルの出力と正解データを比較して評価し、スコアを算出する
        scores = {}
        for metric_name, metric_function in self.metrics.items():
            score = metric_function(model_output, ground_truth)
            scores[metric_name] = score
        return scores


class SampleEvaluator(Evaluator):
    def __init__(self, model, dataset, metrics):
        super().__init__(metrics)
        self.model = model
        self.dataset = dataset
    
    def evaluate(self):
        # データセット上でモデルを評価し、結果のリストを返す
        results = []
        for data in self.dataset:
            prompt = data['model_input']
            model_output = self.model.generate(prompt)
            ground_truth = data['ground_truth']  # 仮定：データセットには 'ground_truth' キーが含まれる
            scores = super().evaluate(model_output, ground_truth)
            results.append({
                "prompt": prompt,
                "model_output": model_output,
                "scores": scores
            })
        return results


#SampleEvaluator のインスタンス作成と評価を行うためには、以下のようなコードを使用します：
# モデル、データセット、メトリクスが定義されていると仮定
#model = load_model(model_path, model_args)
#dataset = load_testdata(source)
#metrics = {
#    'accuracy': accuracy_function,
#    'precision': precision_function,
    # その他のメトリクス関数...
#}

#evaluator = SampleEvaluator(model, dataset, metrics)
#evaluation_results = evaluator.evaluate()
