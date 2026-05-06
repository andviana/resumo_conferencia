#--------------IMPORTACOES E CONFIGURACAO INICIAL-------------------
# Importações principais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings

# Configurar warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Importar módulos do projeto
import config
import data_utils
import model_utils
import database_utils
import pso_optimizer
import evaluate_final_model

print("✅ Importações realizadas com sucesso!")
print(f"📊 Configuração PSO: {config.PSO_CONFIG['n_particles']} partículas, {config.PSO_CONFIG['iters']} iterações, {config.PSO_CONFIG['experimentos']} experimentos")


#--------------EXPLORACAO DOS DADOS-------------------
# Carregar e explorar dados
print("📊 Carregando dataset de Parkinson...")
df = data_utils.load_parkinson_data()

# Validar dados
data_utils.validate_data(df)

# Informações do dataset
info = data_utils.get_data_info(df)
print(f"\n📈 Informações do Dataset:")
print(f"  • Amostras: {info['n_samples']}")
print(f"  • Features: {info['n_features']}")
print(f"  • Distribuição de classes: {info['target_distribution']}")
print(f"  • Valores ausentes: {info['missing_values']}")

# Mostrar primeiras linhas
#display(df.head())


# Visualizar distribuição das classes
#plt.figure(figsize=(10, 6))

# Gráfico de barras
#plt.subplot(1, 2, 1)
class_counts = df['status'].value_counts()
#plt.bar(['Saudável (0)', 'Parkinson (1)'], class_counts.values, color=['lightblue', 'lightcoral'])
#plt.title('Distribuição das Classes')
#plt.ylabel('Número de Amostras')

# Gráfico de pizza
#plt.subplot(1, 2, 2)
#plt.pie(class_counts.values, labels=['Saudável', 'Parkinson'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
#plt.title('Proporção das Classes')

#plt.tight_layout()
#plt.show()

print(f"Dataset balanceado: {'Sim' if abs(class_counts[0] - class_counts[1]) / len(df) < 0.1 else 'Não'}")


#--------------PREPARACAO DOS DADOS-------------------
# Preparar dados para validação cruzada
print("🔄 Preparando dados para validação cruzada...")
X_data, y_data, scaler = data_utils.prepare_data_for_cv(df)

print(f"✅ Dados preparados:")
print(f"  • Shape X: {X_data.shape}")
print(f"  • Shape y: {y_data.shape}")
print(f"  • Tipo de normalização: MinMaxScaler")
print(f"  • Range dos dados: [{X_data.min():.3f}, {X_data.max():.3f}]")


#--------------TESTE RAPIDO DA REDE NEURAL-------------------
# Testar criação e treino de uma rede neural simples
print("🧪 Testando criação e treino da rede neural...")

# Parâmetros de teste
test_params = {
    'n_layers': 2,
    'neurons': [64, 32],
    'learning_rate': 0.001
}

# Dividir dados para teste
from sklearn.model_selection import train_test_split
X_train_test, X_val_test, y_train_test, y_val_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Criar e treinar modelo
test_model = model_utils.create_neural_network(X_data.shape[1], test_params)
print(f"✅ Modelo criado com arquitetura: {test_params['neurons']}")

# Treinar por algumas épocas
history = model_utils.train_neural_network(
    test_model, X_train_test, y_train_test, 
    X_val_test, y_val_test, epochs=10, verbose=1
)

# Avaliar modelo
metrics = model_utils.evaluate_model(test_model, X_val_test, y_val_test)
print(f"\n📊 Métricas do teste:")
print(f"  • F1-Score: {metrics['f1_score']:.4f}")
print(f"  • Accuracy: {metrics['accuracy']:.4f}")
print(f"  • AUC: {metrics['auc']:.4f}")

# Limpar memória
del test_model
import tensorflow as tf
tf.keras.backend.clear_session()

print("✅ Teste da rede neural concluído com sucesso!")

#--------------EXECUCAO OTIMIZACAO PSO-------------------
# Verificar configurações antes de executar
print("⚙️ Configurações atuais do PSO:")
print(f"  • Experimentos: {config.PSO_CONFIG['experimentos']}")
print(f"  • Partículas por experimento: {config.PSO_CONFIG['n_particles']}")
print(f"  • Iterações por experimento: {config.PSO_CONFIG['iters']}")
print(f"  • Total de treinamentos: {config.PSO_CONFIG['experimentos'] * config.PSO_CONFIG['n_particles'] * config.PSO_CONFIG['iters']:,}")

# Estimar tempo
estimated_time_hours = (config.PSO_CONFIG['experimentos'] * config.PSO_CONFIG['n_particles'] * config.PSO_CONFIG['iters'] * 2) / 3600
print(f"  • Tempo estimado: ~{estimated_time_hours:.1f} horas")

print("\n💡 Para teste rápido, edite config.py e reduza os valores.")

# Executar otimização PSO
print("🚀 Iniciando otimização PSO...")
print("📊 Progresso será mostrado com barras de progresso interativas.")
print("💾 Todos os dados serão salvos automaticamente no banco SQLite.")

# Executar otimização
pso_results = pso_optimizer.run_pso_optimization()

print("\n🎉 Otimização PSO concluída com sucesso!")
print(f"🏆 Melhor F1-Score encontrado: {pso_results['best_experiment']['best_f1_score']:.4f}")

#--------------ANALISE RSULTADOS PSO-------------------
# Obter estatísticas dos experimentos
stats = database_utils.get_experiment_statistics()

print("📊 Estatísticas dos Experimentos PSO:")
print(f"  • Total de experimentos: {int(stats.get('total_experiments', 0))}")
print(f"  • Total de partículas avaliadas: {int(stats.get('total_particles', 0)):,}")
print(f"  • F1-Score médio: {stats.get('avg_f1_score', 0):.4f}")
print(f"  • Melhor F1-Score: {stats.get('best_f1_score', 0):.4f}")
print(f"  • Pior F1-Score: {stats.get('worst_f1_score', 0):.4f}")
print(f"  • Desvio padrão F1: {stats.get('std_f1_score', 0):.4f}")
print(f"  • Tempo médio por experimento: {stats.get('avg_total_time', 0):.1f}s")
print(f"  • Uso médio de CPU: {stats.get('avg_cpu_usage', 0):.1f}%")
print(f"  • Uso máximo de memória: {stats.get('avg_max_memory', 0):.1f} MB")