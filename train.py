import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from glob import glob
import os
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
# Make sure this points to your actual data folder
TRAIN_DIR = r"G:\Other computers\Latitude\INFRYNE\resarrch\Model\training_data_master"
MODEL_PATH = "grand_unified_model.keras"
TEST_FILES_LIST = "test_set_files.txt" 
BATCH_SIZE = 8
EPOCHS = 30

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# --- 0. CUSTOM MASKED LOSS FUNCTIONS ---
def masked_mse(y_true, y_pred):
    """Calculates Mean Squared Error, ignoring NaN values in y_true."""
    # Force y_true to perfectly match the (Batch, H, W, 1) shape of y_pred
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask_f32 = tf.cast(mask, tf.float32)
    
    y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_clean = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    
    squared_error = tf.square(y_true_clean - y_pred_clean)
    return tf.reduce_sum(squared_error) / tf.maximum(tf.reduce_sum(mask_f32), 1.0)

def masked_mae(y_true, y_pred):
    """Calculates Mean Absolute Error, ignoring NaN values in y_true."""
    # Force y_true to perfectly match the (Batch, H, W, 1) shape of y_pred
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask_f32 = tf.cast(mask, tf.float32)
    
    y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred_clean = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    
    abs_error = tf.abs(y_true_clean - y_pred_clean)
    return tf.reduce_sum(abs_error) / tf.maximum(tf.reduce_sum(mask_f32), 1.0)


# --- 1. THE UNIFIED DATA GENERATOR (SELF-HEALING) ---
class UnifiedDataGenerator(tf.keras.utils.Sequence):
    """
    Feeds the model with the 5-Channel Master Tensor.
    Input: [Vis, IR, ERA5_Cloud, ERA5_Rad, ERA5_Temp]
    Outputs: { 'cot_head': NASA_Official_Label, 'solar_head': CERES_Truth }
    """
    def __init__(self, file_list, batch_size, shuffle=True, augment=False):
        self.file_list = file_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment 
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        list_files = [self.file_list[k] for k in indexes]
        
        X_batch = []
        y_cot = []
        y_solar = []
        for f in list_files:
            while True:
                try:
                    data = np.load(f)
                    X = data['X']
                    if X.size > 0 and X.ndim == 3 and X.shape[-1] == 5:
                        X_batch.append(X)
                        y_cot.append(data['y_cot'])
                        y_solar.append(data['y_solar']) 
                        break # Success! Break out of the while loop
                    else:
                        raise ValueError("Corrupted shape or empty array.")
                except Exception:
                    # If the file is broken, pick a random new one and try again immediately
                    f = np.random.choice(self.file_list)
            
        X_batch = np.array(X_batch)
        
        # Explicitly add the missing Channel dimension so shape is (Batch, H, W, 1)
        y_cot = np.expand_dims(np.array(y_cot), axis=-1)
        
        # PREVENT EXPLODING GRADIENTS: Scale down the ERA5 Solar Joules
        y_solar_raw = np.expand_dims(np.array(y_solar), axis=-1)
        y_solar = y_solar_raw / 1000000.0

        if self.augment:
            # np.flip correctly handles and moves NaN values without breaking them
            if np.random.rand() > 0.5:
                X_batch = np.flip(X_batch, axis=2)
                y_cot = np.flip(y_cot, axis=2)
                y_solar = np.flip(y_solar, axis=2)
                
            if np.random.rand() > 0.5:
                X_batch = np.flip(X_batch, axis=1)
                y_cot = np.flip(y_cot, axis=1)
                y_solar = np.flip(y_solar, axis=1)

        return X_batch, {
            'cot_head': y_cot, 
            'solar_head': y_solar
        }

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)


# --- 2. THE ADAPTIVE FUSION U-NET (WITH BATCH NORM) ---
def build_unified_model(input_shape):
    inputs = layers.Input(input_shape)
    
    # 🌟 BATCH NORMALIZATION: Fixes the R^2 = 0 issue by balancing the 5 channels
    x = layers.BatchNormalization()(inputs)
    
    # --- ENCODER ---
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # BOTTLENECK
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # --- DECODER ---
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Resizing(c3.shape[1], c3.shape[2])(u1)
    u1 = layers.Concatenate()([u1, c3]) 
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Resizing(c2.shape[1], c2.shape[2])(u2)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Resizing(c1.shape[1], c1.shape[2])(u3)
    u3 = layers.Concatenate()([u3, c1])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u3)

    # --- HEADS ---
    cot_out = layers.Conv2D(1, 1, activation='linear', name='cot_head')(c6)
    rad_features = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)
    solar_out = layers.Conv2D(1, 1, activation='linear', name='solar_head')(rad_features)

    model = models.Model(inputs=inputs, outputs=[cot_out, solar_out])
    return model


def main():
    print("🚀 Starting Ultra-Robust Training Pipeline...")
    
    all_files = glob(os.path.join(TRAIN_DIR, "*.npz"))
    total_files = len(all_files)
    
    if total_files == 0:
        return print(f"❌ No files found in {TRAIN_DIR}. Check your path!")
    
    print(f"   📂 Found {total_files} total samples.")
    
    random.seed(42) 
    random.shuffle(all_files)
    
    train_end = int(total_files * TRAIN_RATIO)
    val_end   = int(total_files * (TRAIN_RATIO + VAL_RATIO))
    
    train_files = all_files[:train_end]
    val_files   = all_files[train_end:val_end]
    test_files  = all_files[val_end:]
    
    print(f"   📊 Training Set:   {len(train_files)} samples")
    print(f"   🧐 Validation Set: {len(val_files)} samples")
    print(f"   🧪 Test Set:       {len(test_files)} samples")
    
    with open(TEST_FILES_LIST, "w") as f:
        for item in test_files:
            f.write(f"{item}\n")
    print(f"   💾 Test file list saved to {TEST_FILES_LIST}")

    train_gen = UnifiedDataGenerator(train_files, BATCH_SIZE, augment=True)
    val_gen   = UnifiedDataGenerator(val_files, BATCH_SIZE, shuffle=False, augment=False)
    test_gen  = UnifiedDataGenerator(test_files, BATCH_SIZE, shuffle=False, augment=False)
    
    # Use the generator to get a safe sample (in case file 0 is corrupted)
    sample_X, _ = train_gen[0]
    sample_shape = sample_X[0].shape
    print(f"   📐 Input Shape detected: {sample_shape} (Should be H, W, 5)")
    
    # --- AUTO-RESUME LOGIC ---
    if os.path.exists(MODEL_PATH):
        print(f"   🔄 Found existing checkpoint: {MODEL_PATH}")
        print("   Resuming training from last saved weights...")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'masked_mse': masked_mse, 'masked_mae': masked_mae}
        )
    else:
        print("   ✨ No previous checkpoint found. Starting fresh model...")
        model = build_unified_model(sample_shape)
        model.compile(
            optimizer='adam',
            loss={'cot_head': masked_mse, 'solar_head': 'mse'},
            loss_weights={'cot_head': 0.2, 'solar_head': 1.0}, 
            metrics={'cot_head': masked_mae, 'solar_head': 'mae'}
        )
    
    print("   🏃 Starting Training Loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=2, # FIX: Prevents multi-line spam in Windows console!
        callbacks=[
            callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_solar_head_loss', save_best_only=True, verbose=1, mode='min'),
            callbacks.EarlyStopping(monitor='val_solar_head_loss', patience=5, restore_best_weights=True, mode='min')
        ]
    )
    
    print("\n" + "="*40)
    print("   🧪 FINAL EVALUATION ON TEST SET")
    print("="*40)
    results = model.evaluate(test_gen, verbose=2)
    
    print(f"   Test Total Loss:    {results[0]:.4f}")
    print(f"   Test Solar MAE:     {results[4]:.4f} (Objective 4 - Scaled)")
    print(f"   Test Cloud MAE:     {results[3]:.4f} (Objective 2 - Ignored NaNs)")
    
    # Note: Only plot if we actually trained (history is populated)
    if hasattr(history, 'history') and 'solar_head_mae' in history.history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['solar_head_mae'], label='Train Solar')
        plt.plot(history.history['val_solar_head_mae'], label='Val Solar')
        plt.title("Solar Forecasting Error (MAE)")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history.get('cot_head_masked_mae', []), label='Train Cloud')
        plt.plot(history.history.get('val_cot_head_masked_mae', []), label='Val Cloud')
        plt.title("Cloud Thickness Error (Masked MAE)")
        plt.legend()
        
        plt.savefig("training_history_split.png")
        print("🎉 Done! History saved to training_history_split.png")

if __name__ == "__main__":
    main()