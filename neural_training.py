import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
from sklearn.model_selection import train_test_split

# =======================================================
# CONFIGURAÇÕES DE DESEMPENHO
# =======================================================
# Descomente as linhas abaixo se tiver uma GPU NVIDIA compatível
# tf.config.optimizer.set_jit(True)
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

print("="*50)
print(f"TensorFlow version: {tf.__version__}")
print("GPUs Disponíveis:", tf.config.list_physical_devices('GPU'))
print("GPU em uso:", tf.test.gpu_device_name() or "NENHUMA")
print("="*50)

# =======================================================
# FUNÇÕES DE PARSING DE DADOS (Inalteradas)
# =======================================================


def expand_numbers_safe(text):
    valores = []
    for token in text.split():
        if token == "/" or token.strip() == "":
            continue
        if "*" in token:
            try:
                n, val = token.split("*")
                valores.extend([float(val)] * int(n))
            except ValueError:
                continue
        else:
            try:
                valores.append(float(token))
            except ValueError:
                continue
    return valores


def extrair_keyword(linhas, keyword):
    valores = []
    capturando = False
    for linha in linhas:
        if keyword in linha.upper():
            capturando = True
            continue
        if capturando:
            if re.match(r"^[A-Z]", linha.strip(), re.IGNORECASE):
                break
            valores.extend(expand_numbers_safe(linha))
    return valores


def extrair_coord_numpy(arquivo):
    coords = []
    in_coord = False
    with open(arquivo, 'r') as f:
        for line in f:
            if "COORD" in line:
                in_coord = True
                continue
            if in_coord:
                if "/" in line:
                    break
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line.split('--')[0])
                coords.extend(map(float, numbers))
    return np.array(coords).reshape(-1, 3)


def extrair_zcorn(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        dentro_zcorn = False
        dados_zcorn = []
        for linha in arquivo:
            if 'ZCORN' in linha:
                dentro_zcorn = True
                continue
            if dentro_zcorn:
                if '/' in linha:
                    linha = linha.replace('/', '')
                    dados_zcorn.extend(map(float, linha.split()))
                    break
                else:
                    dados_zcorn.extend(map(float, linha.split()))
    return dados_zcorn


def ler_actnum_arquivo(caminho_arquivo):
    valores_str = []
    coletando = False
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if not coletando:
                if linha.upper() == "ACTNUM":
                    coletando = True
                continue
            valores_str.append(linha)
            if linha.endswith('/'):
                break
    texto_valores = " ".join(valores_str).replace('/', ' ')
    tokens = texto_valores.split()
    valores = [int(t) for t in tokens if t in ('0', '1')]
    return np.array(valores)

# =======================================================
# FUNÇÕES AUXILIARES DE IMAGEM
# =======================================================


def downscale(image, factor):
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def upscale(image, factor, original_shape=None):
    if original_shape is not None:
        target_size = (original_shape[1], original_shape[0])
    else:
        target_size = (image.shape[1] * factor, image.shape[0] * factor)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def prepare_input(image):
    # Garante que a imagem tenha o formato (batch, height, width, channels)
    img_float = image.astype(np.float32)
    if img_float.ndim == 2:
        return img_float[np.newaxis, ..., np.newaxis]
    elif img_float.ndim == 3:
        return img_float[np.newaxis, ...]
    return img_float


# =======================================================
# PARÂMETROS DO MODELO E TREINAMENTO
# =======================================================
NX, NY, NZ = 81, 58, 20
scale_factor = 2
num_samples = 50000  # Mais amostras (patches) para um treino robusto
patch_size = 48    # Tamanho do recorte (patch)
batch_size = 64  # back size do treino
epochs = 300  # Reduzido para um exemplo rápido, 100-200 pode ser melhor

# =======================================================
# LEITURA DOS DADOS
# =======================================================
# Certifique-se que estes arquivos estão no mesmo diretório ou forneça o caminho completo
try:
    file_path = "PETRO_0.INC"
    arquivo_data = "datas_.DATA"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        linhas = f.readlines()
    permx = extrair_keyword(linhas, "PERMX")
    coords_array = extrair_coord_numpy(arquivo_data)
    zcorn_dados = extrair_zcorn(arquivo_data)
    array_actnum = ler_actnum_arquivo(arquivo_data)
except FileNotFoundError:
    raise FileNotFoundError("Arquivos de dados não encontrados.")


assert len(permx) == NX*NY*NZ, "PERMX length mismatch"
assert len(array_actnum) == NX*NY*NZ, "ACTNUM length mismatch"

# =======================================================
# REMODELAR E PREPARAR O MAPA DE PERMEABILIDADE
# =======================================================
actnum_3d = array_actnum.reshape((NX, NY, NZ), order='F')
permX_3d = np.array(permx).reshape((NX, NY, NZ), order='F')

perm_media_bloco = np.zeros((NX, NY))
for i in range(NX):
    for j in range(NY):
        active_cells = actnum_3d[i, j, :] == 1
        if np.any(active_cells):
            perm_media_bloco[i, j] = np.mean(permX_3d[i, j, active_cells])
        else:
            perm_media_bloco[i, j] = 1.0  # Valor padrão para células inativas

perm_media_bloco = np.nan_to_num(perm_media_bloco, nan=1.0)
min_val, max_val = np.min(perm_media_bloco), np.max(perm_media_bloco)
if max_val > min_val:
    perm_media_bloco = (perm_media_bloco - min_val) / (max_val - min_val)
else:
    perm_media_bloco = np.ones_like(perm_media_bloco)

# =======================================================
# NOVA FUNÇÃO PARA GERAR DATASET COM PATCHES
# =======================================================


def generate_sr_dataset_from_patches(high_res_image, num_samples, patch_size, scale_factor):
    """Gera um dataset para Super-Resolução extraindo patches aleatórios de uma única imagem."""
    X_lr, y_hr = [], []
    h, w = high_res_image.shape

    for _ in range(num_samples):
        rand_h = np.random.randint(0, h - patch_size)
        rand_w = np.random.randint(0, w - patch_size)

        hr_patch = high_res_image[rand_h:rand_h +
                                  patch_size, rand_w:rand_w + patch_size]

        # ↓ Flip aleatório removido

        lr_patch = downscale(hr_patch, scale_factor)

        y_hr.append(hr_patch)
        X_lr.append(lr_patch)

    y_hr = np.array(y_hr, dtype=np.float32)[..., np.newaxis]
    X_bicubic_patches = np.array([upscale(
        img, scale_factor, (patch_size, patch_size)) for img in X_lr], dtype=np.float32)
    X_train_patches = X_bicubic_patches[..., np.newaxis]

    return X_train_patches, y_hr


print("Gerando dataset a partir de patches...")
X_train, y_train = generate_sr_dataset_from_patches(
    perm_media_bloco, num_samples, patch_size, scale_factor)

print(f"Formato do dataset de entrada (X_train): {X_train.shape}")
print(f"Formato do dataset de saída (y_train): {y_train.shape}")

# Divisão explícita dos dados de treino e validação
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# =======================================================
# MODELO SRCNN
# =======================================================
model = models.Sequential([
    layers.Input(shape=(None, None, 1)),  # Permite entrada de qualquer tamanho
    layers.Conv2D(256, (9, 9), padding='same', activation='relu'),
    layers.Conv2D(128, (5, 5), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(1, (5, 5), padding='same', activation='linear')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.MeanSquaredError(),  # Usando o objeto
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', save_best_only=True, monitor='val_loss')
]

# =======================================================
# TREINAMENTO
# =======================================================
history = model.fit(
    X_train_final,
    y_train_final,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# =======================================================
# AVALIAÇÃO E VISUALIZAÇÃO LADO A LADO (VERSÃO MELHORADA)
# =======================================================
print("\nIniciando avaliação e visualização final...")

# Carregar o melhor modelo salvo durante o treinamento
best_model = tf.keras.models.load_model(
    'best_model.h5',
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

# -------------------------------------------------------------------
# ETAPA 1: PREDIÇÃO E CÁLCULO DE MÉTRICAS NA RESOLUÇÃO ORIGINAL
# -------------------------------------------------------------------
# Preparar as imagens para comparação na resolução nativa (81x58)
original_hr = perm_media_bloco
original_lr = downscale(original_hr, scale_factor)
input_bicubic = upscale(original_lr, scale_factor, original_hr.shape)

# Fazer a predição com o modelo na imagem completa
input_tensor = prepare_input(input_bicubic)
predicted_hr = best_model.predict(input_tensor)[0, ..., 0]
# Garante que os valores fiquem no intervalo [0, 1]
predicted_hr = np.clip(predicted_hr, 0, 1)

# Calcular métricas de qualidade (PSNR) antes do redimensionamento para visualização
psnr_bicubic = tf.image.psnr(prepare_input(
    original_hr), prepare_input(input_bicubic), max_val=1.0).numpy()
psnr_model = tf.image.psnr(prepare_input(
    original_hr), prepare_input(predicted_hr), max_val=1.0).numpy()

print(f"PSNR (Bicúbico vs Orignal): {psnr_bicubic[0]:.2f} dB")
print(f"PSNR (Modelo SRCNN vs Original): {psnr_model[0]:.2f} dB")

# -------------------------------------------------------------------
# ETAPA 2: PREPARAÇÃO PARA VISUALIZAÇÃO (UPSCALE PARA 200x200)
# -------------------------------------------------------------------
target_shape_viz = (200, 200)  # (altura, largura) para a visualização final

# Criar o mapa de erro absoluto
error_map = np.abs(original_hr - predicted_hr)

# Redimensionar todas as imagens para o tamanho de visualização desejado
# Nota: cv2.resize espera (largura, altura)
target_size_cv2 = (target_shape_viz[1], target_shape_viz[0])

viz_bicubic = cv2.resize(input_bicubic, target_size_cv2,
                         interpolation=cv2.INTER_CUBIC)
viz_predicted = cv2.resize(
    predicted_hr, target_size_cv2, interpolation=cv2.INTER_CUBIC)
viz_original = cv2.resize(original_hr, target_size_cv2,
                          interpolation=cv2.INTER_CUBIC)
viz_error = cv2.resize(error_map, target_size_cv2,
                       interpolation=cv2.INTER_NEAREST)


# -------------------------------------------------------------------
# ETAPA 3: PLOTAGEM COM VISUALIZAÇÃO APRIMORADA
# -------------------------------------------------------------------
# Criar a visualização comparativa com 4 painéis
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
cmap_custom = LinearSegmentedColormap.from_list(
    "CustomMap", ["#ff00b3", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"])

# 1. Imagem Bicúbica
axes[0].imshow(viz_bicubic.T, cmap=cmap_custom, origin='lower',
               vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title(
    f'Entrada Bicúbica\nPSNR: {psnr_bicubic[0]:.2f} dB', fontsize=14)
axes[0].axis('off')

# 2. Imagem Prevista pelo Modelo
im = axes[1].imshow(viz_predicted.T, cmap=cmap_custom,
                    origin='lower', vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title(
    f'Saída do Modelo SRCNN\nPSNR: {psnr_model[0]:.2f} dB', fontsize=14)
axes[1].axis('off')

# 3. Imagem Original (Ground Truth)
axes[2].imshow(viz_original.T, cmap=cmap_custom, origin='lower',
               vmin=0, vmax=1, interpolation='nearest')
axes[2].set_title('Original (Ground Truth)', fontsize=14)
axes[2].axis('off')

# 4. Mapa de Erro Absoluto 💡
im_error = axes[3].imshow(viz_error.T, cmap='coolwarm',
                          origin='lower', interpolation='nearest')
axes[3].set_title('Mapa de Erro Absoluto\n(SRCNN vs Original)', fontsize=14)
axes[3].axis('off')


fig.suptitle(
    'Comparação de Super-Resolução (Visualização 200x200)', fontsize=20)
# Adiciona uma colorbar para os mapas de permeabilidade
cbar = fig.colorbar(im, ax=axes[:3], shrink=0.7,
                    label='Permeabilidade Normalizada')
# Adiciona uma colorbar separada para o mapa de erro
cbar_error = fig.colorbar(
    im_error, ax=axes[3], shrink=0.7, label='Diferença Absoluta')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()
