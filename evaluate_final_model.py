"""
Avaliação final do melhor modelo usando validação cruzada estratificada
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf

import config
import data_utils
import model_utils
import database_utils

class FinalModelEvaluator:
    """
    Classe para avaliação final do melhor modelo
    """
    
    def __init__(self, X_data, y_data):
        """
        Inicializa o avaliador
        
        Args:
            X_data: Features do dataset
            y_data: Labels do dataset
        """
        self.X_data = X_data
        self.y_data = y_data
        self.input_dim = X_data.shape[1]
        
        # Obter melhor partícula
        self.best_particle = database_utils.get_best_particle_overall()
        if self.best_particle is None:
            raise ValueError("Nenhum resultado PSO encontrado no banco de dados")
        
        # Decodificar parâmetros do melhor modelo
        self.best_params = model_utils.decode_particle_position(
            self.best_particle['position']
        )
        
        print(f"🏆 Melhor modelo encontrado:")
        print(f"  - Experimento: {self.best_particle['experiment']}")
        print(f"  - Iteração: {self.best_particle['iteration']}")
        print(f"  - F1-score: {self.best_particle['f1_score']:.4f}")
        print(f"  - Arquitetura: {self.best_params['neurons'][:self.best_params['n_layers']]}")
        print(f"  - Learning Rate: {self.best_params['learning_rate']:.6f}")
    
    def cross_validate_model(self, n_splits=None, random_state=None):
        """
        Avalia o modelo usando validação cruzada estratificada
        
        Args:
            n_splits: Número de folds
            random_state: Seed para reprodutibilidade
            
        Returns:
            dict: Resultados da validação cruzada
        """
        if n_splits is None:
            n_splits = config.EVAL_CONFIG['cv_folds']
        if random_state is None:
            random_state = config.EVAL_CONFIG['random_state']
        
        print(f"\n🔄 Executando validação cruzada estratificada ({n_splits} folds)...")
        
        # Configurar validação cruzada
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Listas para armazenar resultados
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        # Barra de progresso
        pbar = tqdm(
            enumerate(skf.split(self.X_data, self.y_data)), 
            total=n_splits,
            desc="Validação Cruzada"
        )
        
        for fold, (train_idx, val_idx) in pbar:
            # Dividir dados
            X_train_fold = self.X_data[train_idx]
            X_val_fold = self.X_data[val_idx]
            y_train_fold = self.y_data[train_idx]
            y_val_fold = self.y_data[val_idx]
            
            # Criar e treinar modelo
            model = model_utils.create_neural_network(self.input_dim, self.best_params)
            
            # Treinar com early stopping
            history = model_utils.train_neural_network(
                model, X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                epochs=config.NN_CONFIG['epochs'],
                verbose=0
            )
            
            # Avaliar modelo
            metrics = model_utils.evaluate_model(model, X_val_fold, y_val_fold)
            
            # Armazenar resultados do fold
            fold_result = {
                'fold': fold + 1,
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc'],
                'n_samples': len(y_val_fold)
            }
            fold_results.append(fold_result)
            
            # Armazenar predições para análise global
            all_y_true.extend(y_val_fold)
            all_y_pred.extend(metrics['predictions'])
            all_y_proba.extend(metrics['probabilities'])
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'F1': f'{metrics["f1_score"]:.4f}',
                'Acc': f'{metrics["accuracy"]:.4f}',
                'AUC': f'{metrics["auc"]:.4f}'
            })
            
            # Limpar memória
            del model
            tf.keras.backend.clear_session()
        
        pbar.close()
        
        # Calcular estatísticas finais
        f1_scores = [r['f1_score'] for r in fold_results]
        accuracies = [r['accuracy'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        
        # Converter para arrays numpy
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        
        # Calcular métricas globais
        global_f1 = f1_score(all_y_true, all_y_pred)
        global_accuracy = accuracy_score(all_y_true, all_y_pred)
        global_auc = roc_auc_score(all_y_true, all_y_proba)
        global_cm = confusion_matrix(all_y_true, all_y_pred)
        
        results = {
            'fold_results': fold_results,
            'cv_metrics': {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs)
            },
            'global_metrics': {
                'f1_score': global_f1,
                'accuracy': global_accuracy,
                'auc': global_auc,
                'confusion_matrix': global_cm
            },
            'predictions': {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_proba': all_y_proba
            },
            'best_params': self.best_params,
            'best_particle_info': self.best_particle
        }
        
        return results
    
    def generate_evaluation_report(self, cv_results, save_path=None):
        """
        Gera relatório detalhado da avaliação
        
        Args:
            cv_results: Resultados da validação cruzada
            save_path: Caminho para salvar o relatório
            
        Returns:
            str: Relatório formatado
        """
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE AVALIAÇÃO FINAL - CLASSIFICAÇÃO DE PARKINSON")
        report.append("=" * 80)
        
        # Informações do melhor modelo
        report.append("\n🏆 MELHOR MODELO ENCONTRADO:")
        report.append(f"  • Experimento: {self.best_particle['experiment']}")
        report.append(f"  • Iteração: {self.best_particle['iteration']}")
        report.append(f"  • Partícula: {self.best_particle['particle']}")
        
        # Arquitetura da rede
        report.append(f"\n🏗️  ARQUITETURA DA REDE NEURAL:")
        report.append(f"  • Número de camadas ocultas: {self.best_params['n_layers']}")
        neurons_str = ", ".join([str(n) for n in self.best_params['neurons'][:self.best_params['n_layers']]])
        report.append(f"  • Neurônios por camada: [{neurons_str}]")
        report.append(f"  • Taxa de aprendizado: {self.best_params['learning_rate']:.6f}")
        report.append(f"  • Função de ativação: ReLU (ocultas), Sigmoid (saída)")
        report.append(f"  • Regularização: BatchNormalization + Dropout(0.3)")
        
        # Resultados da validação cruzada
        cv_metrics = cv_results['cv_metrics']
        report.append(f"\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA ({config.EVAL_CONFIG['cv_folds']} folds):")
        report.append(f"  • F1-Score:  {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
        report.append(f"  • Accuracy:  {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
        report.append(f"  • AUC-ROC:   {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
        
        # Métricas globais
        global_metrics = cv_results['global_metrics']
        report.append(f"\n🎯 MÉTRICAS GLOBAIS (todas as predições):")
        report.append(f"  • F1-Score:  {global_metrics['f1_score']:.4f}")
        report.append(f"  • Accuracy:  {global_metrics['accuracy']:.4f}")
        report.append(f"  • AUC-ROC:   {global_metrics['auc']:.4f}")
        
        # Matriz de confusão
        cm = global_metrics['confusion_matrix']
        report.append(f"\n🔢 MATRIZ DE CONFUSÃO:")
        report.append(f"                 Predito")
        report.append(f"              0      1")
        report.append(f"  Real    0  {cm[0,0]:4d}   {cm[0,1]:4d}")
        report.append(f"          1  {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Calcular métricas adicionais
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        report.append(f"\n📈 MÉTRICAS DETALHADAS:")
        report.append(f"  • Precisão:      {precision:.4f}")
        report.append(f"  • Recall:        {recall:.4f}")
        report.append(f"  • Especificidade: {specificity:.4f}")
        report.append(f"  • Verdadeiros Positivos:  {tp}")
        report.append(f"  • Verdadeiros Negativos:  {tn}")
        report.append(f"  • Falsos Positivos:       {fp}")
        report.append(f"  • Falsos Negativos:       {fn}")
        
        # Resultados por fold
        report.append(f"\n📋 RESULTADOS POR FOLD:")
        for fold_result in cv_results['fold_results']:
            report.append(f"  Fold {fold_result['fold']:2d}: F1={fold_result['f1_score']:.4f}, "
                         f"Acc={fold_result['accuracy']:.4f}, AUC={fold_result['auc']:.4f}")
        
        # Informações do dataset
        report.append(f"\n📊 INFORMAÇÕES DO DATASET:")
        report.append(f"  • Total de amostras: {len(self.y_data)}")
        report.append(f"  • Número de features: {self.input_dim}")
        class_counts = np.bincount(self.y_data.astype(int))
        report.append(f"  • Classe 0 (Saudável): {class_counts[0]} ({class_counts[0]/len(self.y_data)*100:.1f}%)")
        report.append(f"  • Classe 1 (Parkinson): {class_counts[1]} ({class_counts[1]/len(self.y_data)*100:.1f}%)")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Salvar relatório se caminho fornecido
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Relatório salvo em: {save_path}")
        
        return report_text
    
    def plot_confusion_matrix(self, cv_results, save_path=None):
        """
        Plota matriz de confusão
        
        Args:
            cv_results: Resultados da validação cruzada
            save_path: Caminho para salvar o gráfico
        """
        cm = cv_results['global_metrics']['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Saudável', 'Parkinson'],
            yticklabels=['Saudável', 'Parkinson']
        )
        plt.title('Matriz de Confusão - Classificação de Parkinson')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de confusão salva em: {save_path}")
        
        plt.show()

def evaluate_best_model():
    """
    Função principal para avaliar o melhor modelo
    
    Returns:
        dict: Resultados da avaliação
    """
    print("🔍 Iniciando avaliação final do melhor modelo...")
    
    # Carregar dados
    df = data_utils.load_parkinson_data()
    X_data, y_data, scaler = data_utils.prepare_data_for_cv(df)
    
    # Criar avaliador
    evaluator = FinalModelEvaluator(X_data, y_data)
    
    # Executar validação cruzada
    cv_results = evaluator.cross_validate_model()
    
    # Gerar relatório
    report = evaluator.generate_evaluation_report(
        cv_results, 
        save_path="relatorio_avaliacao_final.txt"
    )
    
    # Plotar matriz de confusão
    evaluator.plot_confusion_matrix(
        cv_results,
        save_path="matriz_confusao.png"
    )
    
    # Imprimir resumo
    print("\n" + "="*60)
    print("📋 RESUMO DA AVALIAÇÃO FINAL")
    print("="*60)
    cv_metrics = cv_results['cv_metrics']
    print(f"F1-Score: {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    print(f"Accuracy: {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
    print(f"AUC-ROC:  {cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
    print("="*60)
    
    return cv_results

if __name__ == "__main__":
    results = evaluate_best_model()

