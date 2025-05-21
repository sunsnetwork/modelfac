import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score

class ModelFac:
    def __init__(self):
        self.model = None
        self.features = None
        self.target = None

    def train(self, df: pd.DataFrame, target: str) -> float:
        self.target = target
        self.features = [col for col in df.columns if col != target]
        X = df[self.features]
        y = df[target]

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'verbosity': 0,
            'seed': 42
        }

        self.model = xgb.train(params, dtrain, num_boost_round=100)

        preds = self.model.predict(dtrain)
        auc = roc_auc_score(y, preds)
        return auc

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        dtest = xgb.DMatrix(X)
        df['score'] = self.model.predict(dtest)
        return df

    def explain(self, df: pd.DataFrame, output_dir: str) -> str:
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(df[self.features])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, df[self.features], show=False)
        shap_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        return shap_path

    def lift_chart(self, df: pd.DataFrame, output_dir: str) -> str:
        df['decile'] = pd.qcut(df['score'], 10, labels=False)
        lift_df = df.groupby('decile').agg(
            total=('score', 'count'),
            responders=('target', 'sum')
        ).reset_index()
        lift_df['response_rate'] = lift_df['responders'] / lift_df['total']
        baseline = df['target'].mean()
        lift_df['lift'] = lift_df['response_rate'] / baseline

        plt.figure(figsize=(8, 5))
        plt.bar(lift_df['decile'].astype(str), lift_df['lift'])
        plt.xlabel('Decile')
        plt.ylabel('Lift')
        plt.title('Lift Chart by Decile')
        plt.tight_layout()
        lift_path = os.path.join(output_dir, 'lift_chart.png')
        plt.savefig(lift_path)
        plt.close()
        return lift_path