import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import glob
import numpy as np
import tensorflow as tf
import argparse

# --- 1. CONFIGURATION ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", "--data-dir", dest="data_folder", type=str, required=True, help="Path to data folder injected by Azure")
parser.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
parser.add_argument('--epochs', type=int, default=50, help='Number of Phase 3 epochs')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
parser.add_argument('--learning-rate', type=float, default=5e-5, help='Lower learning rate for fine-tuning')
# Set default to None so it trains from scratch if omitted
parser.add_argument('--phase2-weights', type=str, default=None, help='Path to previous model to fine-tune (leave empty to train from scratch)')
args, unknown = parser.parse_known_args()

DATA_DIR = args.data_folder
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ALL_EPOCHS_DIR = os.path.join(CHECKPOINT_DIR, "all_epochs")
os.makedirs(ALL_EPOCHS_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "attention_unet_model_phase3_final.keras")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "phase3_best_model.keras")

# --- 2. METRICS & LOSS FUNCTIONS ---
@tf.keras.utils.register_keras_serializable()
def r2_score_masked(y_true, y_pred):
    """Calculates R2 strictly over non-NaN pixels, avoiding zero-variance explosions."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_safe = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    mask_f = tf.cast(mask, tf.float32)

    valid_pixels = tf.reduce_sum(mask_f)
    
    # Safe division to find the mean
    mean_y = tf.math.divide_no_nan(tf.reduce_sum(y_true_safe), valid_pixels)
    
    ss_res = tf.reduce_sum(tf.square((y_true_safe - y_pred_safe) * mask_f))
    ss_tot = tf.reduce_sum(tf.square((y_true_safe - mean_y) * mask_f))
    return tf.where(
        ss_tot < 1e-4, 
        tf.where(ss_res < 1e-4, 1.0, 0.0), 
        1.0 - (ss_res / ss_tot)
    )

@tf.keras.utils.register_keras_serializable()
def dynamic_huber_ssim_loss(y_true, y_pred):
    """Uses dynamic max_val to prevent NaN explosions on high-value heads."""
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_safe = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    
    huber = tf.keras.losses.Huber(delta=1.0)
    base_loss = huber(y_true_safe, y_pred_safe, sample_weight=tf.cast(mask, tf.float32))
    
    # Dynamically scale SSIM max_val based on batch properties
    max_val = tf.reduce_max(y_true_safe) + 1e-3
    ssim = tf.image.ssim(y_true_safe, y_pred_safe, max_val=max_val)
    
    return base_loss + (0.5 * (1.0 - tf.reduce_mean(ssim)))

# --- 3. PHYSICS-INFORMED ARCHITECTURE ---
def attention_block(g, x, num_filters):
    Wg = tf.keras.layers.Conv2D(num_filters, 1, padding='same')(g)
    Wx = tf.keras.layers.Conv2D(num_filters, 1, padding='same')(x)
    out = tf.keras.layers.Activation('relu')(Wg + Wx)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(out)
    return tf.keras.layers.Multiply()([x, out])

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def safe_bilinear_upsample(x):
    """Bypasses the Keras 3 UpSampling2D dynamic shape bug."""
    shape = tf.shape(x)
    return tf.image.resize(x, [shape[1] * 2, shape[2] * 2], method='bilinear')

def build_phase2_unet(input_shape=(None, None, 5)):
    """Recreated so Keras 3 can natively load the .keras file into a matching topology."""
    inputs = tf.keras.Input(shape=input_shape)
    e1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D(2)(e3)
    b = conv_block(p3, 256)
    
    u3 = tf.keras.layers.Lambda(safe_bilinear_upsample)(b)
    a3 = attention_block(u3, e3, 128)
    u3 = tf.keras.layers.Concatenate()([u3, a3])
    d3 = conv_block(u3, 128)
    
    u2 = tf.keras.layers.Lambda(safe_bilinear_upsample)(d3)
    a2 = attention_block(u2, e2, 64)
    u2 = tf.keras.layers.Concatenate()([u2, a2])
    d2 = conv_block(u2, 64)
    
    u1 = tf.keras.layers.Lambda(safe_bilinear_upsample)(d2)
    a1 = attention_block(u1, e1, 32)
    u1 = tf.keras.layers.Concatenate()([u1, a1])
    d1 = conv_block(u1, 32)
    
    cot_out = tf.keras.layers.Conv2D(1, 1, activation='relu', name='cot_head')(d1)
    solar_out = tf.keras.layers.Conv2D(1, 1, activation='relu', name='solar_head')(d1)
    
    return tf.keras.Model(inputs=inputs, outputs=[cot_out, solar_out])

def build_phase3_unet(input_shape=(None, None, 5)):
    """Phase 3 architecture with bounded heads and Beer-Lambert physics modulation."""
    inputs = tf.keras.Input(shape=input_shape)
    
    e1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D(2)(e3)
    b = conv_block(p3, 256)
    
    u3 = tf.keras.layers.Lambda(safe_bilinear_upsample)(b)
    a3 = attention_block(u3, e3, 128)
    u3 = tf.keras.layers.Concatenate()([u3, a3])
    d3 = conv_block(u3, 128)
    
    u2 = tf.keras.layers.Lambda(safe_bilinear_upsample)(d3)
    a2 = attention_block(u2, e2, 64)
    u2 = tf.keras.layers.Concatenate()([u2, a2])
    d2 = conv_block(u2, 64)
    
    u1 = tf.keras.layers.Lambda(safe_bilinear_upsample)(d2)
    a1 = attention_block(u1, e1, 32)
    u1 = tf.keras.layers.Concatenate()([u1, a1])
    d1 = conv_block(u1, 32)
    
    # 1. BOUNDED COT HEAD
    cot_d = tf.keras.layers.Conv2D(1, 1, name='cot_pre_act')(d1)
    cot_out = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x, max_value=200.0), name='cot_head')(cot_d)
    
    # 2. BEER-LAMBERT PHYSICS MODULATION
    attenuation = tf.keras.layers.Lambda(lambda cot: tf.math.exp(-cot / 20.0), name='physics_attenuation')(cot_out)
    solar_base = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='solar_base')(d1)
    solar_modulated = tf.keras.layers.Multiply(name='physical_solar_multiplier')([solar_base, attenuation])
    
    solar_d = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(solar_modulated)
    solar_out_pre = tf.keras.layers.Conv2D(1, 1, name='solar_pre_act')(solar_d)
    
    # 3. BOUNDED SOLAR HEAD
    solar_out = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x, max_value=5.0), name='solar_head')(solar_out_pre)
    
    return tf.keras.Model(inputs=inputs, outputs=[cot_out, solar_out])

# --- 4. DATA PIPELINE ---
def pad_arrays(X, y_cot, y_solar):
    h, w = X.shape[:2]
    pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        X = np.pad(X, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        y_cot = np.pad(y_cot, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
        y_solar = np.pad(y_solar, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
    return X, np.expand_dims(y_cot, -1), np.expand_dims(y_solar, -1)

def data_generator(file_list):
    def generator():
        for f in file_list:
            try:
                # TF Data passes strings as bytes; decode if necessary
                if isinstance(f, bytes): f = f.decode('utf-8')
                data = np.load(f)
                X, y_cot, y_solar = pad_arrays(data['X'], data['y_cot'], data['y_solar'])
                yield X, {'cot_head': y_cot, 'solar_head': y_solar}
            except Exception:
                continue
    return generator

# --- 5. EXECUTION PIPELINE ---
def main():
    print("🚀 Initializing Phase 3 Physics-Informed Training...")
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not all_files:
        raise ValueError(f"❌ No .npz files found in {DATA_DIR}")
        
    np.random.shuffle(all_files)
    split_idx = int((1.0 - args.val_split) * len(all_files))
    
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    print(f"📁 Found {len(all_files)} total files.")
    print(f"   ┣━ Training on: {len(train_files)} files")
    print(f"   ┗━ Validating on: {len(val_files)} files")
    
    output_signature = (
        tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
        {
            'cot_head': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            'solar_head': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
        }
    )
    
    train_ds = tf.data.Dataset.from_generator(
        data_generator(train_files), output_signature=output_signature
    ).shuffle(50).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        data_generator(val_files), output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    print("🏗️ Building Models...")
    phase3_model = build_phase3_unet()
    
    # Conditional loading logic based on argument presence
    if args.phase2_weights:
        if os.path.exists(args.phase2_weights):
            print(f"📥 Loading Previous Weights from {args.phase2_weights}...")
            try:
                # Build exact Phase 2 topology to satisfy Keras 3 file format requirements
                temp_p2 = build_phase2_unet()
                temp_p2.load_weights(args.phase2_weights)
                
                # Isolate layers that actually contain weights
                p2_weight_layers = [l for l in temp_p2.layers if len(l.get_weights()) > 0]
                p3_weight_layers = [l for l in phase3_model.layers if len(l.get_weights()) > 0]
                
                transferred = 0
                # Iterate through both models concurrently based on sequence order
                for l2, l3 in zip(p2_weight_layers, p3_weight_layers):
                    w2 = l2.get_weights()
                    w3 = l3.get_weights()
                    
                    # If the layer shapes match, copy them. If they diverge (i.e. the heads), stop.
                    if len(w2) == len(w3) and all(w2[i].shape == w3[i].shape for i in range(len(w2))):
                        l3.set_weights(w2)
                        transferred += 1
                    else:
                        break
                        
                print(f"✅ Successfully transferred weights for {transferred} base layers based on shape matching.")
            except Exception as e:
                print(f"⚠️ Failed to load previous weights: {e}. Starting from scratch.")
        else:
            print(f"⚠️ Provided model path '{args.phase2_weights}' does not exist. Starting from scratch.")
    else:
        print("🌱 No previous weights provided. Training from scratch.")

    # Apply GLOBAL NORM Gradient Clipping to act as a massive circuit breaker
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    phase3_model.compile(
        optimizer=optimizer,
        loss={
            'cot_head': dynamic_huber_ssim_loss,
            'solar_head': dynamic_huber_ssim_loss
        },
        metrics={
            'cot_head': [r2_score_masked],
            'solar_head': [r2_score_masked]
        }
    )

    every_epoch_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ALL_EPOCHS_DIR, "model_epoch_{epoch:02d}_val_{val_loss:.4f}.keras"),
        save_best_only=False,
        save_weights_only=False,
        verbose=1
    )

    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks = [
        every_epoch_checkpoint,
        best_model_checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "tensorboard_logs_phase3"), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(OUTPUT_DIR, "training_log_phase3.csv"), separator=',', append=True)
    ]

    print("🔥 Starting Training Loop...")
    phase3_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    phase3_model.save(FINAL_MODEL_PATH)
    print(f"🎉 Training Complete. Final model saved to: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
