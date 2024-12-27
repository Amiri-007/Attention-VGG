import numpy as np
import pickle
import tensorflow as tf
import os

# Enable TF2 compatibility mode for session-based code:
tf.compat.v1.disable_eager_execution()

class AttentionMechanism:
    def __init__(self, TCpath, bd=1, attype=1):
        """
        Initialize attention mechanism
        bd: bidirectional or positive only
        attype: 1 = multiplicative, 2 = additive
        """
        self.TCpath = TCpath
        self.bd = bd
        self.attype = attype
        # If additive attention, we have baseline multipliers:
        self.lyrBL = [20, 100, 150, 150, 240, 240, 150, 150, 80, 20, 20, 10, 1] if attype == 2 else None
        
        # 5 groups of layers in VGG:
        self.layer_dims = [
            (224, 224, 64),
            (112, 112, 128),
            (56, 56, 256),
            (28, 28, 512),
            (14, 14, 512)
        ]
    
    def make_tuning_attention(self, object_idx, strength_vec):
        """
        Create attention matrices based on tuning curves (featvecs... .txt file).
        object_idx: index of the object in the tuning curve
        strength_vec: vector of attention strengths for each of the 13 conv layers
        """
        try:
            attnmats = []
            tc_file = os.path.join(self.TCpath, 'featvecs20_train35_c.txt')
            if not os.path.exists(tc_file):
                print(f"Warning: Tuning curves file not found: {tc_file}")
                return None
            
            # If Python 2 → 3 pickling issues, do: pickle.load(fp, encoding='latin1')
            with open(tc_file, "rb") as fp:
                tuning_curves = pickle.load(fp)

            # layer_groups correspond to 13 layers partitioned by (start, end)
            layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
            for group_idx, (start, end) in enumerate(layer_groups):
                h, w, c = self.layer_dims[group_idx]
                for li in range(start, end):
                    tc = tuning_curves[li]
                    fmvals = np.squeeze(tc[object_idx, :])
                    
                    if np.all(fmvals == fmvals[0]):
                        print(f"Warning: Constant tuning values for layer {li}")
                        fmvals = np.random.normal(1.0, 0.1, size=fmvals.shape)
                    
                    # Normalize 0 -> 2
                    fmvals = fmvals - np.min(fmvals)
                    if np.max(fmvals) > 0:
                        fmvals = fmvals / np.max(fmvals) * 2
                    
                    # Expand to shape [1,1,fmaps], then tile to [H,W,fmaps]
                    aval = np.reshape(fmvals, (1, 1, -1))
                    amat = np.tile(aval, [h, w, 1]) * strength_vec[li]

                    # Add small noise
                    noise = np.random.normal(0, 0.01, size=amat.shape)
                    amat += noise

                    # If multiplicative, ensure non-negative
                    if self.attype == 1:
                        amat = np.maximum(amat, 0)
                    
                    attnmats.append(amat)
            return attnmats
        except Exception as e:
            print(f"Error in make_tuning_attention: {e}")
            return None

    def make_gradient_attention(self, object_idx, strength_vec, imtype=1):
        """
        Create attention matrices based on gradient files (CATgrads... .txt).
        object_idx: index of the object in the gradient matrix
        strength_vec: vector of attention strengths
        imtype: image type (1=merge, 2=array) to pick the correct gradient file
        """
        try:
            attnmats = []
            grad_file = os.path.join(self.TCpath, f"CATgradsDetectTrainTCs_im{imtype}.txt")
            if not os.path.exists(grad_file):
                print(f"Warning: Gradient file not found: {grad_file}")
                return None
            
            with open(grad_file, "rb") as fp:
                grads = pickle.load(fp)
            
            print("\nDebug - Gradients:")
            print(f"Number of gradient matrices: {len(grads)}")
            print(f"Object index: {object_idx}")
            
            layer_groups = [(0,2), (2,4), (4,7), (7,10), (10,13)]
            for group_idx, (start, end) in enumerate(layer_groups):
                h, w, c = self.layer_dims[group_idx]
                for li in range(start, end):
                    fv = grads[li]
                    fmvals = np.squeeze(fv[object_idx, :])
                    
                    max_abs = np.amax(np.abs(fv), axis=0)
                    max_abs[max_abs == 0] = 1
                    fmvals = fmvals / max_abs
                    
                    if np.all(fmvals == fmvals[0]):
                        print(f"Warning: Constant gradient values for layer {li}")
                        fmvals = np.random.normal(0.0, 0.1, size=fmvals.shape)
                    
                    aval = np.reshape(fmvals, (1, 1, -1))

                    # If multiplicative:
                    if self.attype == 1:
                        amat = np.ones((h, w, c)) + np.tile(aval, [h, w, 1]) * strength_vec[li]
                        amat = np.maximum(amat, 0)
                    else:
                        # additive
                        amat = np.tile(aval, [h, w, 1]) * strength_vec[li] * self.lyrBL[li]
                    
                    noise = np.random.normal(0, 0.01, size=amat.shape)
                    amat += noise
                    print((
                        f"Layer {li} attention stats - "
                        f"Min: {amat.min():.3f}, Max: {amat.max():.3f}, "
                        f"Mean: {amat.mean():.3f}, Std: {amat.std():.3f}"
                    ))
                    
                    attnmats.append(amat)
            return attnmats
        except Exception as e:
            print(f"Error in make_gradient_attention: {e}")
            return None


class LayerAttention:
    """
    Manage attention strength for each of the 13 conv layers.
    """
    def __init__(self, num_layers=13):
        self.num_layers = num_layers
        
    def get_layer_mask(self, target_layer):
        if target_layer > self.num_layers:
            return np.ones(self.num_layers)
        else:
            mask = np.zeros(self.num_layers)
            mask[target_layer] = 1
            return mask
            
    def scale_attention_strength(self, strength, target_layer):
        mask = self.get_layer_mask(target_layer)
        if target_layer > self.num_layers:
            return strength * mask * 0.1
        return strength * mask
