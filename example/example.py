import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import time
import psutil
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Cargar el dataset
data = pd.read_csv('example/datos_con_fases.csv')
all_roc_data = []  # Lista global

# Codificar la columna 'fase' a números
label_encoder = LabelEncoder()
data['fase'] = label_encoder.fit_transform(data['fase'])
class_names = label_encoder.classes_

# Vector de características: RGBI + desviaciones + temp + ph = 10 características
features = [
    'value_R', 'value_G', 'value_B', 'value_I',
    'value_R_desv', 'value_G_desv', 'value_B_desv', 'value_I_desv',
    'temperatura', 'ph'
]
X = data[features].values
y = data['fase'].values

# Normalización Z-score (antes de KFold para evitar data leakage en escala)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(sampling_strategy='auto', random_state=42)

def create_model(input_dim, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(np.random.uniform(0.2, 0.3)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(np.random.uniform(0.2, 0.3)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(np.random.uniform(0.3, 0.4)),
        layers.Dense(len(class_names), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Almacenar resultados
experiments_results = []
num_experiments = 3  # Cambiar a 40 si quieres

per_class_metrics_accum = {
    "precision": {cls: [] for cls in class_names},
    "recall": {cls: [] for cls in class_names},
    "f1_score": {cls: [] for cls in class_names},
    "roc_auc": {cls: [] for cls in class_names},
}
macro_metrics_accum = {
    "precision": [],
    "recall": [],
    "f1_score": [],
    "roc_auc": []
}

all_train_loss_histories = []
all_val_loss_histories = []
all_train_acc_histories = []
all_val_acc_histories = []

conf_matrix_total = np.zeros((len(class_names), len(class_names)), dtype=np.float64)

# Para guardar métricas de uso de recursos y tiempos
resource_stats = []

for exp_id in range(1, num_experiments + 1):
    print(f"Ejecutando experimento {exp_id}/{num_experiments}")

    kf = KFold(n_splits=5, shuffle=True, random_state=exp_id*123)
    results, val_losses, val_accuracies, histories = [], [], [], []

    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []

    # Acumular métricas de recursos por fold
    cpu_usages = []
    ram_usages = []
    train_times = []
    inference_times = []
    early_stopping_epochs = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = create_model(input_dim=X_train_res.shape[1], seed=exp_id * 100 + fold)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Medición antes entrenamiento
        process = psutil.Process()
        cpu_percent_start = psutil.cpu_percent(interval=None)
        mem_info_start = process.memory_info().rss / (1024 ** 2)  # MB

        start_train = time.time()
        history = model.fit(
            X_train_res, y_train_res,
            epochs=50,
            batch_size=np.random.choice([16, 32, 64]),
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[early_stopping]
        )
        end_train = time.time()

        # Medición después entrenamiento
        cpu_percent_end = psutil.cpu_percent(interval=None)
        mem_info_end = process.memory_info().rss / (1024 ** 2)  # MB

        train_time = end_train - start_train
        cpu_usage = (cpu_percent_start + cpu_percent_end) / 2
        ram_usage = (mem_info_start + mem_info_end) / 2

        # Inferencia y tiempo inferencia
        start_inf = time.time()
        y_prob = model.predict(X_val)
        all_roc_data.append({
    "y_true": y_val,
    "y_prob": y_prob
})

        end_inf = time.time()
        inf_time = end_inf - start_inf

        histories.append(history.history)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)

        val_loss_eval, val_accuracy_eval = model.evaluate(X_val, y_val, verbose=0)
        val_losses.append(val_loss_eval)
        val_accuracies.append(val_accuracy_eval)

        y_pred = np.argmax(y_prob, axis=1)

        cm = confusion_matrix(y_val, y_pred)
        conf_matrix_total += cm

        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')

        precision_per_class = precision_score(y_val, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
        recall_per_class = recall_score(y_val, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
        f1_per_class = f1_score(y_val, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
        roc_auc_per_class = roc_auc_score(y_val, y_prob, multi_class='ovr', average=None)

        for i, cls in enumerate(class_names):
            per_class_metrics_accum["precision"][cls].append(precision_per_class[i])
            per_class_metrics_accum["recall"][cls].append(recall_per_class[i])
            per_class_metrics_accum["f1_score"][cls].append(f1_per_class[i])
            per_class_metrics_accum["roc_auc"][cls].append(roc_auc_per_class[i])

        macro_metrics_accum["precision"].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        macro_metrics_accum["recall"].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        macro_metrics_accum["f1_score"].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        macro_metrics_accum["roc_auc"].append(roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro'))

        results.append({
            "kfold_id": fold + 1,
            "precision": precision,
            "accuracy": val_accuracy_eval,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist()
        })

        # Guardar época donde convergió early stopping
        early_stopping_epochs.append(len(history.history['loss']))

        # Guardar recursos y tiempos
        cpu_usages.append(cpu_usage)
        ram_usages.append(ram_usage)
        train_times.append(train_time)
        inference_times.append(inf_time)

    max_epochs = max(len(l) for l in fold_train_losses)
    def pad_and_avg(list_of_lists):
        padded = [l + [l[-1]] * (max_epochs - len(l)) for l in list_of_lists]
        return np.mean(padded, axis=0)

    all_train_loss_histories.append(pad_and_avg(fold_train_losses))
    all_val_loss_histories.append(pad_and_avg(fold_val_losses))
    all_train_acc_histories.append(pad_and_avg(fold_train_accs))
    all_val_acc_histories.append(pad_and_avg(fold_val_accs))

    # Guardar las métricas y estadísticas de recursos en el JSON
    experiment_resource_stats = {
        "id_exp": exp_id,
        "cpu_usage_avg_percent": float(np.mean(cpu_usages)),
        "ram_usage_avg_mb": float(np.mean(ram_usages)),
        "train_time_avg_sec": float(np.mean(train_times)),
        "inference_time_avg_sec": float(np.mean(inference_times)),
        "early_stopping_epoch_avg": int(np.round(np.mean(early_stopping_epochs))),
        "metrics_avg": {
            "precision": float(np.mean([r["precision"] for r in results])),
            "recall": float(np.mean([r["recall"] for r in results])),
            "f1_score": float(np.mean([r["f1_score"] for r in results])),
            "roc_auc": float(np.mean([r["roc_auc"] for r in results]))
        }
    }
    resource_stats.append(experiment_resource_stats)

    experiment = {
        "id_exp": exp_id,
        "data": results,
        "stats": {
            "precision": np.mean([r["precision"] for r in results]),
            "accuracy_mean": np.mean([r["accuracy"] for r in results]),
            "recall_mean": np.mean([r["recall"] for r in results]),
            "f1_score_mean": np.mean([r["f1_score"] for r in results]),
            "roc_auc_mean": np.mean([r["roc_auc"] for r in results]),
            "loss_mean": np.mean(val_losses)
        }
    }

    experiments_results.append(experiment)

# Guardar resultados por experimento
with open('output/experiments_results.json', 'w') as f:
    json.dump(experiments_results, f, indent=4)

# Guardar recursos y tiempos por experimento en otro JSON
with open('output/resource_stats.json', 'w') as f:
    json.dump(resource_stats, f, indent=4)

# Estadísticas globales promedio
estadisticas_globales = {
    "per_class": {},
    "macro_average": {}
}

for cls in class_names:
    estadisticas_globales["per_class"][cls] = {
        "precision_mean": float(np.mean(per_class_metrics_accum["precision"][cls])),
        "recall_mean": float(np.mean(per_class_metrics_accum["recall"][cls])),
        "f1_score_mean": float(np.mean(per_class_metrics_accum["f1_score"][cls])),
        "roc_auc_mean": float(np.mean(per_class_metrics_accum["roc_auc"][cls])),
    }

estadisticas_globales["macro_average"] = {
    "precision_mean": float(np.mean(macro_metrics_accum["precision"])),
    "recall_mean": float(np.mean(macro_metrics_accum["recall"])),
    "f1_score_mean": float(np.mean(macro_metrics_accum["f1_score"])),
    "roc_auc_mean": float(np.mean(macro_metrics_accum["roc_auc"])),
}

with open('output/estadisticas_globales.json', 'w') as f:
    json.dump(estadisticas_globales, f, indent=4)

max_len_exp = max(len(hist) for hist in all_train_loss_histories)

def pad_to_max_len(list_of_hist):
    return [np.pad(hist, (0, max_len_exp - len(hist)), mode='edge') for hist in list_of_hist]

all_train_loss_histories = pad_to_max_len(all_train_loss_histories)
all_val_loss_histories = pad_to_max_len(all_val_loss_histories)
all_train_acc_histories = pad_to_max_len(all_train_acc_histories)
all_val_acc_histories = pad_to_max_len(all_val_acc_histories)

mean_train_loss = np.mean(all_train_loss_histories, axis=0)
mean_val_loss = np.mean(all_val_loss_histories, axis=0)
mean_train_acc = np.mean(all_train_acc_histories, axis=0)
mean_val_acc = np.mean(all_val_acc_histories, axis=0)

# === Visualization of Loss and Accuracy in English ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mean_train_loss, label='Average Train Loss')
plt.plot(mean_val_loss, label='Average Validation Loss')
plt.title(f'Average Learning Curve - Loss ({num_experiments} Experiments)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mean_train_acc, label='Average Train Accuracy')
plt.plot(mean_val_acc, label='Average Validation Accuracy')
plt.title(f'Average Learning Curve - Accuracy ({num_experiments} Experiments)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# === Normalized Confusion Matrix Visualization in English ===
conf_matrix_norm = conf_matrix_total / conf_matrix_total.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Average Normalized Confusion Matrix ({num_experiments} Experiments)')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = conf_matrix_norm.max() / 2.
for i, j in np.ndindex(conf_matrix_norm.shape):
    plt.text(j, i, f"{conf_matrix_norm[i, j]:.8f}",
             horizontalalignment="center",
             color="white" if conf_matrix_norm[i, j] > thresh else "black")

plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()

# === ROC Curves in English ===
from sklearn.preprocessing import label_binarize

fpr_grid = np.linspace(0, 1, 100)
tpr_interp_per_class = {cls: [] for cls in class_names}
auc_per_class = {cls: [] for cls in class_names}

for entry in all_roc_data:
    y_true = label_binarize(entry["y_true"], classes=range(len(class_names)))
    y_prob = entry["y_prob"]

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp_per_class[cls].append(tpr_interp)
        auc_score = auc(fpr, tpr)
        auc_per_class[cls].append(auc_score)

mean_tpr_per_class = {cls: np.mean(tprs, axis=0) for cls, tprs in tpr_interp_per_class.items()}
mean_auc_per_class = {cls: np.mean(aucs) for cls, aucs in auc_per_class.items()}

# Graficar y guardar curvas ROC
import os
os.makedirs("figures", exist_ok=True)

for cls in class_names:
    plt.figure(figsize=(6, 5), dpi=120)
    plt.plot(fpr_grid, mean_tpr_per_class[cls], label=f'ROC {cls} (AUC = {mean_auc_per_class[cls]:.5f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Average ROC Curve - Class: {cls}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


print("Average AUC per class:")
for cls in class_names:
    print(f"{cls}: {mean_auc_per_class[cls]:.8f}")

print("Proceso finalizado. Resultados guardados y gráficas generadas.")
