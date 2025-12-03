# acdc_deep_unet_meta_dice_fixed_single.py
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
import cv2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import h5py


sys.path.append('C:\\Alex The Great\\Project\\medai-env\\Scikit-learn\\session10')

# ---------------------- ACDC Data Pipeline Fixed ----------------------
class ACDCDataPipeline:
    def __init__(self, data_loader, target_size=(192, 192)):
        self.data_loader = data_loader
        self.target_size = target_size
    
    def prepare_data(self, patient_ids, data_type='training_volumes', batch_size=32, augment=True):
       
        frames = []
        masks = []
        
        print(f"Preparing ACDC data for {len(patient_ids)} patients ({data_type})...")
        
        for patient_id in patient_ids:
            try:
                patient_data = self.data_loader.load_patient_data(patient_id)
                
                if not patient_data:
                    print(f"No data for patient {patient_id}")
                    continue
                
                for key, data_dict in patient_data.items():
                    if data_dict['type'] == data_type:
                        data = data_dict['data']
                        
                        if data is None:
                            continue
                            
                       
                        image_key = None
                        mask_key = None
                        
                        
                        for k in data.keys():
                            if 'image' in k.lower() or 'img' in k.lower():
                                image_key = k
                            elif 'mask' in k.lower() or 'label' in k.lower() or 'seg' in k.lower():
                                mask_key = k
                        
                        if image_key is None or mask_key is None:
                          
                            keys = list(data.keys())
                            if len(keys) >= 2:
                                image_key = keys[0]
                                mask_key = keys[1]
                            else:
                                continue
                        
                        image_data = data[image_key][:]
                        mask_data = data[mask_key][:]
                        
                      
                        if len(image_data.shape) == 3:  # حجم (depth, height, width)
                           
                            for slice_idx in range(image_data.shape[0]):
                                image_slice = image_data[slice_idx]
                                mask_slice = mask_data[slice_idx]
                                
                             
                                image_resized = cv2.resize(image_slice, self.target_size, interpolation=cv2.INTER_AREA)
                                mask_resized = cv2.resize(mask_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
                                
                              
                                if image_resized.max() > image_resized.min():
                                    image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
                                else:
                                    image_normalized = image_resized
                                
                                frames.append(image_normalized)
                                masks.append(mask_resized)
                        
                        elif len(image_data.shape) == 2:  
                            
                            image_resized = cv2.resize(image_data, self.target_size, interpolation=cv2.INTER_AREA)
                            mask_resized = cv2.resize(mask_data, self.target_size, interpolation=cv2.INTER_NEAREST)
                            
                          
                            if image_resized.max() > image_resized.min():
                                image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
                            else:
                                image_normalized = image_resized
                            
                            frames.append(image_normalized)
                            masks.append(mask_resized)
                            
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue
        
        if len(frames) == 0:
          
            print("Trying alternative data loading method...")
            frames, masks = self._load_data_directly(patient_ids, data_type)
        
        if len(frames) == 0:
            raise ValueError("No valid samples found after all attempts!")
        
        frames = np.array(frames)
        if len(frames.shape) == 3: 
            frames = frames[..., np.newaxis]
        
        masks = np.array(masks)
        
       
        all_mask_values = masks.flatten()
        class_counts = Counter(all_mask_values)
        print(f"Class distribution: {dict(class_counts)}")
        print(f"ACDC data prepared: {frames.shape} frames, {masks.shape} masks")
        
      
        dataset = tf.data.Dataset.from_tensor_slices((frames, masks))
        dataset = dataset.shuffle(1000)
        
        if augment:
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_data_directly(self, patient_ids, data_type):
      
        frames = []
        masks = []
        
        if data_type == 'training_volumes':
            file_list = self.data_loader.training_volumes
            base_path = self.data_loader.training_volumes_path
        elif data_type == 'training_slices':
            file_list = self.data_loader.training_slices
            base_path = self.data_loader.training_slices_path
        else:
            file_list = self.data_loader.testing_volumes
            base_path = self.data_loader.testing_volumes_path
        
        for patient_id in patient_ids:
          
            patient_files = [f for f in file_list if patient_id in f]
            
            for filename in patient_files:
                try:
                    file_path = os.path.join(base_path, filename)
                    
                    with h5py.File(file_path, 'r') as f:
                      
                        keys = list(f.keys())
                        print(f"File {filename} keys: {keys}")
                        
                        if len(keys) >= 2:
                        
                            image_data = f[keys[0]][:]
                            mask_data = f[keys[1]][:]
                            
                          
                            if len(image_data.shape) == 3:
                                for slice_idx in range(image_data.shape[0]):
                                    image_slice = image_data[slice_idx]
                                    mask_slice = mask_data[slice_idx]
                                    
                                    image_resized = cv2.resize(image_slice, self.target_size, interpolation=cv2.INTER_AREA)
                                    mask_resized = cv2.resize(mask_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
                                    
                                    if image_resized.max() > image_resized.min():
                                        image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
                                    else:
                                        image_normalized = image_resized
                                    
                                    frames.append(image_normalized)
                                    masks.append(mask_resized)
                            
                            elif len(image_data.shape) == 2:
                                image_resized = cv2.resize(image_data, self.target_size, interpolation=cv2.INTER_AREA)
                                mask_resized = cv2.resize(mask_data, self.target_size, interpolation=cv2.INTER_NEAREST)
                                
                                if image_resized.max() > image_resized.min():
                                    image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
                                else:
                                    image_normalized = image_resized
                                
                                frames.append(image_normalized)
                                masks.append(mask_resized)
                                
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
                    continue
        
        return frames, masks
    
    def _augment_data(self, frame, mask):
       
        mask_expanded = tf.expand_dims(mask, axis=-1)
        
       
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_left_right(frame)
            mask_expanded = tf.image.flip_left_right(mask_expanded)
        
       
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_up_down(frame)
            mask_expanded = tf.image.flip_up_down(mask_expanded)
        
     
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        frame = tf.image.rot90(frame, k)
        mask_expanded = tf.image.rot90(mask_expanded, k)
        
      
        frame = tf.image.random_brightness(frame, max_delta=0.1)
        frame = tf.clip_by_value(frame, 0.0, 1.0)
        
        mask_aug = tf.squeeze(mask_expanded, axis=-1)
        return frame, mask_aug

# ---------------------- Deep U-Net Model for ACDC ----------------------
class ACDCDeepUNet:
    def __init__(self, input_shape=(192, 192, 1), num_classes=4, name="acdc_deep_unet"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.model = self.build_deep_unet()
        self.checkpoint_dir = f"acdc_checkpoints_{name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def build_deep_unet(self):
       
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bridge
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
        
        # Decoder
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
        
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
        
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(c9)
        
        model = Model(inputs, outputs, name=self.name)
        return model
    
    def compile(self, learning_rate=1e-3):
      
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Compiled {self.name}")
    
    def train(self, train_dataset, val_dataset, epochs=100, initial_epoch=0):
      
        best_model_path = os.path.join(self.checkpoint_dir, f'best_{self.name}.h5')
        resume_checkpoint = os.path.join(self.checkpoint_dir, f'resume_{self.name}.h5')
        
      
        if initial_epoch > 0 and os.path.exists(resume_checkpoint):
            print(f"Resuming training from epoch {initial_epoch}")
            self.model.load_weights(resume_checkpoint)
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                resume_checkpoint,
                monitor='val_accuracy',
                save_best_only=False,
                verbose=0
            )
        ]
        
        print(f"Training {self.name} for {epochs} epochs...")
        
        try:
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=val_dataset,
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.model.save(best_model_path)
            print(f"FINISHED training {self.name}")
            
            if 'val_accuracy' in history.history:
                best_val_acc = max(history.history['val_accuracy'])
                print(f"Best validation accuracy: {best_val_acc:.4f}")
            
            return history
            
        except KeyboardInterrupt:
            print(f"Training interrupted. Model saved for resuming.")
            self.model.save(resume_checkpoint)
            return None

# ---------------------- Evaluator ----------------------
class Evaluator:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.results_dir = f"acdc_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self, model, X_test, y_test):
      
        print("="*60)
        print("SINGLE MODEL EVALUATION (U-Net V1)")
        print("="*60)
        
        results = {}
        
        print(f"Evaluating U-Net V1...")
        
       
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
     
        print("Predicting...")
        y_pred_proba = model.predict(X_test, verbose=1, batch_size=4)
        y_pred = np.argmax(y_pred_proba, axis=-1)
        
      
        print("Calculating metrics...")
        metrics = self._calculate_all_metrics(y_test, y_pred)
        results['acdc_unet_v1'] = metrics
        
        print(f"RESULTS for U-Net V1:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean Dice: {metrics['mean_dice']:.4f}")
        
        class_names = ['Background', 'RV', 'Myocardium', 'LV']
        print("Dice Scores:")
        for i, class_name in enumerate(class_names):
            dice = metrics['dice_scores'][str(i)]
            print(f"{class_name}: {dice:.4f}")
        
      
        self._save_results(results, X_test, y_test)
        
        return results
    
    def _calculate_all_metrics(self, y_true, y_pred):
      
        from sklearn.metrics import accuracy_score
        
        # Accuracy
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        
        # Dice scores
        dice_scores = {}
        for class_id in range(self.num_classes):
            true_binary = (y_true == class_id).astype(np.float32)
            pred_binary = (y_pred == class_id).astype(np.float32)
            
            intersection = np.sum(true_binary * pred_binary)
            union = np.sum(true_binary) + np.sum(pred_binary)
            
            if union == 0:
                dice = 1.0 if np.sum(true_binary) == 0 else 0.0
            else:
                dice = (2. * intersection) / (union + 1e-8)
            
            dice_scores[class_id] = dice
        
        mean_dice = np.mean(list(dice_scores.values()))
        
        return {
            'accuracy': float(accuracy),
            'mean_dice': float(mean_dice),
            'dice_scores': {str(k): float(v) for k, v in dice_scores.items()}
        }
    
    def _save_results(self, results, X_test, y_test):
      
        print("Saving results...")
        
     
        results_file = os.path.join(self.results_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'test_data_info': {
                    'samples': X_test.shape[0],
                    'shape': X_test.shape[1:],
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=4)
        
        print(f"Results saved: {results_file}")
        
        
        self._create_report(results)
        
       
        self._create_plots(results, X_test, y_test)
    
    def _create_report(self, results):
        
        report_file = os.path.join(self.results_dir, 'report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ACDC CARDIAC SEGMENTATION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("SINGLE MODEL EVALUATION (U-Net V1)\n")
            f.write("-"*40 + "\n")
            
            for model_name, model_results in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"Accuracy: {model_results['accuracy']:.4f}\n")
                f.write(f"Mean Dice: {model_results['mean_dice']:.4f}\n")
                
                f.write("\nDice Scores:\n")
                class_names = ['Background', 'RV', 'Myocardium', 'LV']
                for i, class_name in enumerate(class_names):
                    dice = model_results['dice_scores'][str(i)]
                    f.write(f"{class_name}: {dice:.4f}\n")
                
                f.write(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: U-Net V1 (acdc_unet_v1)\n")
                f.write(f"Architecture: Original U-Net\n")
        
        print(f"Report saved: {report_file}")
    
    def _create_plots(self, results, X_test, y_test):
      
        print("Creating plots for U-Net V1...")
        
        plt.figure(figsize=(15, 10))
        
       
        plt.subplot(2, 3, 1)
        
        if 'acdc_unet_v1' in results:
            model_results = results['acdc_unet_v1']
            
            class_names = ['Background', 'RV', 'Myocardium', 'LV']
            dice_scores = [model_results['dice_scores'][str(i)] for i in range(4)]
            
            colors_class = ['gray', 'blue', 'green', 'red']
            bars = plt.bar(class_names, dice_scores, color=colors_class, alpha=0.7)
            
            plt.title(f'Dice Scores - U-Net V1\nMean Dice: {model_results["mean_dice"]:.4f}')
            plt.ylabel('Dice Score')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            for bar, score in zip(bars, dice_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
      
        plt.subplot(2, 3, 2)
        if len(X_test) > 0:
            sample_idx = 0
            plt.imshow(X_test[sample_idx].squeeze(), cmap='gray')
            plt.title(f'Sample Image {sample_idx+1}')
            plt.axis('off')
        
        
        plt.subplot(2, 3, 3)
        if len(y_test) > 0:
            sample_idx = 0
            plt.imshow(y_test[sample_idx], cmap='jet')
            plt.title(f'Ground Truth {sample_idx+1}')
            plt.axis('off')
        
       
        plt.subplot(2, 3, 4)
        if len(X_test) > 0 and 'acdc_unet_v1' in results:
            sample_idx = 0
            
           
            if len(X_test) > sample_idx:
                plt.imshow(X_test[sample_idx].squeeze(), cmap='gray')
                plt.title(f'Input for Prediction')
                plt.axis('off')
        
    
        plt.subplot(2, 3, 5)
        if 'acdc_unet_v1' in results:
            model_results = results['acdc_unet_v1']
            
            class_names = ['Background', 'RV', 'Myocardium', 'LV']
            dice_scores = [model_results['dice_scores'][str(i)] for i in range(4)]
            
           
            plt.plot(class_names, dice_scores, 'o-', linewidth=2, markersize=8, color='blue')
            plt.fill_between(class_names, dice_scores, alpha=0.2, color='blue')
            
            plt.title('Dice Score Progression')
            plt.ylabel('Dice Score')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
          
            for i, score in enumerate(dice_scores):
                plt.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
      
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        if 'acdc_unet_v1' in results:
            model_results = results['acdc_unet_v1']
            
            summary_text = f"""
            U-Net V1 EVALUATION
            ====================
            
            Model: acdc_unet_v1
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            RESULTS:
            --------
            Accuracy: {model_results['accuracy']:.4f}
            Mean Dice: {model_results['mean_dice']:.4f}
            
            DICE SCORES:
            ------------
            Background: {model_results['dice_scores']['0']:.4f}
            RV: {model_results['dice_scores']['1']:.4f}
            Myocardium: {model_results['dice_scores']['2']:.4f}
            LV: {model_results['dice_scores']['3']:.4f}
            
            Test Samples: {X_test.shape[0]}
            Image Size: {X_test.shape[1:3]}
            """
        else:
            summary_text = "No results available for U-Net V1"
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'u-net_v1_evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved: {plot_path}")
        plt.show()
        
    
        self._create_detailed_plot(results, X_test, y_test)
    
    def _create_detailed_plot(self, results, X_test, y_test):
     
        if 'acdc_unet_v1' not in results:
            return
            
        plt.figure(figsize=(12, 8))
        
        model_results = results['acdc_unet_v1']
        
       
        plt.subplot(2, 2, 1)
        class_names = ['Background', 'RV', 'Myocardium', 'LV']
        dice_scores = [model_results['dice_scores'][str(i)] for i in range(4)]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = plt.bar(range(len(class_names)), dice_scores, color=colors, alpha=0.8)
        
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.ylabel('Dice Score')
        plt.title('U-Net V1: Dice Scores by Class', fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, dice_scores)):
            plt.text(i, bar.get_height() + 0.02, f'{score:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
     
        plt.subplot(2, 2, 2)
        explode = (0.05, 0.05, 0.05, 0.05)
        plt.pie(dice_scores, labels=class_names, autopct='%1.1f%%',
                startangle=90, colors=colors, explode=explode,
                shadow=True)
        plt.title('Dice Score Distribution', fontweight='bold')
        
       
        plt.subplot(2, 2, 3)
        if len(X_test) > 0:
            sample_idx = 10 
            plt.imshow(X_test[sample_idx].squeeze(), cmap='gray')
            plt.title(f'Input Image (Sample {sample_idx+1})')
            plt.axis('off')
        
       
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        metrics_text = f"""
        U-Net V1 Performance Metrics
        
        Overall Metrics:
        ----------------
        Accuracy:  {model_results['accuracy']:.4f}
        Mean Dice: {model_results['mean_dice']:.4f}
        
        Class-wise Dice:
        ----------------
        Background:   {model_results['dice_scores']['0']:.4f}
        RV:           {model_results['dice_scores']['1']:.4f}
        Myocardium:   {model_results['dice_scores']['2']:.4f}
        LV:           {model_results['dice_scores']['3']:.4f}
        
        Dataset Info:
        -------------
        Test Samples: {X_test.shape[0]}
        Image Size:   {X_test.shape[1]}x{X_test.shape[2]}
        """
        
        plt.text(0.1, 0.5, metrics_text, fontsize=9, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        detailed_plot_path = os.path.join(self.results_dir, 'u-net_v1_detailed_analysis.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plots saved: {detailed_plot_path}")
        plt.show()

# ---------------------- Main Execution ----------------------
def main():
    print("="*70)
    print("ACDC DEEP U-NET WITH SINGLE MODEL")
    print("="*70)
    
    try:
        
        from data_loaders.acdc_loader import ACDCLoader
        data_loader = ACDCLoader()
        print("ACDC Data loader loaded")
        
       
        all_patient_ids = []
        for filename in data_loader.training_volumes:
            info = data_loader.parse_filename(filename)
            if 'patient_id' in info:
                all_patient_ids.append(info['patient_id'])
        
        all_patient_ids = sorted(list(set(all_patient_ids)))
        print(f"Total patients: {len(all_patient_ids)}")
        
       
        train_patients, temp_patients = train_test_split(all_patient_ids, test_size=0.3, random_state=42)
        val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)
        
        print(f"Split: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")
        
       
        data_pipeline = ACDCDataPipeline(data_loader)
        
        print("Preparing training data...")
        train_dataset = data_pipeline.prepare_data(train_patients, 'training_volumes', batch_size=4, augment=True)
        
        print("Preparing validation data...")
        val_dataset = data_pipeline.prepare_data(val_patients, 'training_volumes', batch_size=4, augment=False)
        
        print("Preparing test data...")
        test_dataset = data_pipeline.prepare_data(test_patients, 'training_volumes', batch_size=4, augment=False)
        
       
        print("Building U-Net model...")
        model = ACDCDeepUNet(input_shape=(192, 192, 1), num_classes=4, name="acdc_unet_v1")
        model.compile(learning_rate=1e-3)
        
        print("="*50)
        print("Training Model...")
        print("="*50)
        
        model.train(train_dataset, val_dataset, epochs=100)
        
       
        print("="*50)
        print("Evaluating Model...")
        print("="*50)
        
       
        frames_list = []
        masks_list = []
        for batch in test_dataset:
            f, m = batch
            frames_list.append(f.numpy())
            masks_list.append(m.numpy())
        
        X_test = np.concatenate(frames_list, axis=0)
        y_test = np.concatenate(masks_list, axis=0)
        
        evaluator = Evaluator(num_classes=4)
        results = evaluator.evaluate(model.model, X_test, y_test)
        
        print("="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        
        if 'acdc_unet_v1' in results:
            model_results = results['acdc_unet_v1']
            
            print(f"FINAL RESULTS (U-Net V1):")
            print(f"Accuracy: {model_results['accuracy']:.4f}")
            print(f"Mean Dice: {model_results['mean_dice']:.4f}")
            
            print(f"Dice Scores:")
            class_names = ['Background', 'RV', 'Myocardium', 'LV']
            for i, class_name in enumerate(class_names):
                dice = model_results['dice_scores'][str(i)]
                print(f"{class_name}: {dice:.4f}")
        
        print(f"\nAll results saved in: {evaluator.results_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
