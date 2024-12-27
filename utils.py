import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, pearsonr
from skimage.metrics import structural_similarity as ssim
import cv2
import pickle
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class AttentionAnalyzer:
    """Analyzer for attention effects in VGG16 network."""
    def __init__(self, vgg_model, session):
        self.vgg = vgg_model
        self.sess = session
    
    def analyze_attention_effects(self, tp_batch, tn_batch, attnmats, astrgs):
        """
        Analyze attention effects across layers with proper batch handling
        (computes responses with and without attention, etc.)
        """
        results = {
            'tp_responses': [],
            'tn_responses': [],
            'tp_scores': [],
            'tn_scores': [],
            'strength': astrgs
        }
        
        print("Computing baseline responses (no attention)...")
        baseline_dict = self._create_base_feed_dict(tp_batch)
        _ = self.sess.run(self.vgg.get_all_layers(), feed_dict=baseline_dict)

        # If needed, store them or compare them in detail:
        # tp_baseline = ...
        
        processed_attnmats = []
        for amat in attnmats:
            if len(amat.shape) == 4:
                processed_attnmats.append(amat[0])
            else:
                processed_attnmats.append(amat)
        
        print("Computing responses with attention...")
        try:
            feed_dict = self._create_feed_dict(tp_batch, processed_attnmats)
            tp_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
            tp_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
            
            feed_dict = self._create_feed_dict(tn_batch, processed_attnmats)
            tn_responses = self.sess.run(self.vgg.get_all_layers(), feed_dict=feed_dict)
            tn_score = self.sess.run(self.vgg.guess, feed_dict=feed_dict)
            
            results['tp_responses'].append(tp_responses)
            results['tn_responses'].append(tn_responses)
            results['tp_scores'].append(tp_score)
            results['tn_scores'].append(tn_score)
            
        except Exception as e:
            print(f"Error in feed_dict: {e}")
            placeholders = self.vgg.get_attention_placeholders()
            for i, p in enumerate(placeholders):
                print(f"Placeholder {i}: {p.get_shape()}")
            for i, a in enumerate(processed_attnmats):
                print(f"Attention {i}: {a.shape}")
            raise
        
        return results
    
    def _create_base_feed_dict(self, batch):
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        for placeholder in placeholders:
            shape = placeholder.get_shape().as_list()
            # Replace any None with 1
            if None in shape:
                shape = [s if s is not None else 1 for s in shape]
            feed_dict[placeholder] = np.ones(shape, dtype=np.float32)
        return feed_dict
    
    def _create_feed_dict(self, batch, attnmats):
        placeholders = self.vgg.get_attention_placeholders()
        feed_dict = {self.vgg.imgs: batch}
        
        print("\nCreating feed dict with attention maps:")
        for idx, (placeholder, attnmat) in enumerate(zip(placeholders, attnmats)):
            expected_shape = tuple(d if d is not None else 1 for d in placeholder.get_shape().as_list())
            current_shape = attnmat.shape
            print(f"Layer {idx} - Expected shape: {expected_shape}, Current shape: {current_shape}")
            
            if current_shape != expected_shape:
                if len(current_shape) == 4 and current_shape[0] == 1:
                    attnmat = np.squeeze(attnmat, axis=0)
                    print(f"Squeezed batch dimension for layer {idx}")
                else:
                    raise ValueError(f"Shape mismatch for layer {idx}: expected {expected_shape}, got {current_shape}")
            
            feed_dict[placeholder] = attnmat
        return feed_dict


def pad_batch(batch, target_size):
    """Pad the batch to target_size by concatenating zeros if needed."""
    current_size = batch.shape[0]
    if current_size < target_size:
        pad_size = target_size - current_size
        zeros = np.zeros((pad_size,) + batch.shape[1:], dtype=batch.dtype)
        return np.concatenate([batch, zeros], axis=0)
    return batch


def compute_saliency_map(sess, model, images, labels=None, attention_maps=None):
    """
    Compute gradient-based saliency maps under TF2.x (v1 compat).
    """
    if not hasattr(model, 'saliency_op'):
        target_logits = model.fc3l[:, 0]
        smoothed_logits = target_logits + 1e-8 * tf.reduce_mean(tf.square(model.imgs))
        model.saliency_op = tf.compat.v1.gradients(smoothed_logits, model.imgs)[0]

    feed_dict = {model.imgs: images}
    if attention_maps is not None:
        placeholders = model.get_attention_placeholders()
        if len(attention_maps) == len(placeholders):
            for p, amap in zip(placeholders, attention_maps):
                feed_dict[p] = amap

    try:
        sal = sess.run(model.saliency_op, feed_dict=feed_dict)
        sal[np.isnan(sal)] = 0.0
        sal = np.abs(sal)
        sal = np.sum(sal, axis=-1, keepdims=False)
        
        if np.all(np.abs(sal) < 1e-10):
            print("Warning: near-zero saliency detected, adding noise.")
            sal += np.random.normal(0, 0.01, sal.shape)
        
        # Normalize each map robustly
        out_maps = []
        for i in range(sal.shape[0]):
            smap = sal[i]
            smap += np.random.normal(0, 1e-6, smap.shape)
            v1, v99 = np.percentile(smap, [1, 99])
            if v99 > v1:
                smap = np.clip(smap, v1, v99)
                smap = (smap - v1) / (v99 - v1)
            else:
                # fallback min->max
                smap_min, smap_max = np.min(smap), np.max(smap)
                if smap_max > smap_min:
                    smap = (smap - smap_min) / (smap_max - smap_min)
                else:
                    smap = np.zeros_like(smap)
            out_maps.append(smap)
        sal = np.stack(out_maps)
        
        print("\nSaliency stats:")
        print(f"  shape = {sal.shape}")
        print(f"  min={np.min(sal):.5f}, max={np.max(sal):.5f}, mean={np.mean(sal):.5f}, std={np.std(sal):.5f}")
        
        return sal
    except Exception as e:
        print(f"Error computing saliency map: {e}")
        return None


def debug_saliency(saliency_maps):
    if saliency_maps is not None:
        print(f"Saliency shape: {saliency_maps.shape}")
        print(f"Non-zero elements: {np.count_nonzero(saliency_maps)}")
        print(f"Range: {np.min(saliency_maps)} to {np.max(saliency_maps)}")

def debug_print_shapes(saliency_map, attention_maps, msg=""):
    print(f"\n=== Debug Shapes {msg} ===")
    print(f"Saliency map: {saliency_map.shape}")
    for i, amap in enumerate(attention_maps):
        print(f"Attention map {i} shape: {amap.shape}")


def print_debug_info(saliency_maps, attention_maps, layer):
    print("Debug - shapes before comparison:")
    print(f"  saliency: {saliency_maps.shape}")
    print(f"  attention: {[amap.shape for amap in attention_maps]}")
    print(f"  comparing layer {layer}")


def compare_saliency_attention(saliency_map, attention_maps, layer_idx):
    att_map = attention_maps[layer_idx]
    
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]
    elif len(saliency_map.shape) == 4:
        saliency_map = np.mean(saliency_map[0], axis=-1)
    
    if len(att_map.shape) == 4:
        att_map = np.mean(att_map[0], axis=-1)
    elif len(att_map.shape) == 3:
        att_map = np.mean(att_map, axis=-1)
    
    saliency_map[np.isnan(saliency_map)] = 0.0
    att_map[np.isnan(att_map)] = 0.0
    
    min_size = 7
    th = max(min_size, min(saliency_map.shape[0], att_map.shape[0]))
    tw = max(min_size, min(saliency_map.shape[1], att_map.shape[1]))
    
    saliency_map = cv2.resize(saliency_map.astype(np.float32), (tw, th))
    att_map = cv2.resize(att_map.astype(np.float32), (tw, th))
    
    # Normalize
    def norm_map(x):
        x += np.random.normal(0, 1e-6, x.shape)
        v1, v99 = np.percentile(x, [1, 99])
        if v99 > v1:
            x = np.clip(x, v1, v99)
            x = (x - v1) / (v99 - v1)
        else:
            xmin, xmax = x.min(), x.max()
            if xmax > xmin:
                x = (x - xmin) / (xmax - xmin)
            else:
                x = np.zeros_like(x)
        return x
    
    s_norm = norm_map(saliency_map)
    a_norm = norm_map(att_map)
    
    metrics = {}
    s_flat = s_norm.flatten()
    a_flat = a_norm.flatten()
    
    if np.std(s_flat) > 1e-6 and np.std(a_flat) > 1e-6:
        corr, _ = pearsonr(s_flat, a_flat)
        metrics['pearson_correlation'] = float(corr)
    else:
        metrics['pearson_correlation'] = 0.0
    
    def calc_iou(sm, am, percentile=75):
        s_thr = np.percentile(sm, percentile)
        a_thr = np.percentile(am, percentile)
        sb = sm > s_thr
        ab = am > a_thr
        inter = np.logical_and(sb, ab).sum()
        union = np.logical_or(sb, ab).sum()
        return float(inter)/(union + 1e-8) if union>0 else 0.0
    
    metrics['iou'] = calc_iou(s_norm, a_norm)
    
    try:
        wsize = min(7, min(th, tw)-1)
        if wsize % 2 == 0:
            wsize -= 1
        metrics['ssim'] = float(ssim(s_norm, a_norm, win_size=wsize, gaussian_weights=True))
    except Exception as e:
        print(f"Warning: SSIM failed: {e}")
        metrics['ssim'] = 0.0
    
    eps = 1e-8
    p = s_norm + eps
    q = a_norm + eps
    p /= p.sum()
    q /= q.sum()
    metrics['kl_divergence'] = float(np.sum(p * np.log(p/q)))
    
    return metrics


def visualize_comparison(image, saliency_map, attention_maps, metrics, save_path, batch_idx=0):
    if len(image.shape) == 4:
        image = image[batch_idx]
    if len(saliency_map.shape) == 4:
        saliency_map = saliency_map[batch_idx]
    
    att_maps = [amap[batch_idx] if len(amap.shape)==4 else amap for amap in attention_maps]
    print((
        f"Saliency final stats: "
        f"min={saliency_map.min():.5f}, max={saliency_map.max():.5f}, "
        f"mean={saliency_map.mean():.5f}, std={saliency_map.std():.5f}"
    ))
    
    fig = plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1,3,2)
    if saliency_map.ndim > 2:
        saliency_map = np.mean(saliency_map, axis=-1)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Saliency")
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1,3,3)
    processed_maps = []
    target_h, target_w = image.shape[:2]
    for am in att_maps:
        if am.ndim > 2:
            am = np.mean(am, axis=-1)
        if am.shape != (target_h, target_w):
            am = cv2.resize(am, (target_w, target_h))
        processed_maps.append(am)
    avg_att = np.mean(processed_maps, axis=0)
    plt.imshow(avg_att, cmap='viridis')
    plt.title("Attention")
    plt.colorbar()
    plt.axis('off')
    
    text_str = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    hist_path = save_path.replace(".png","_hist.png")
    plt.figure()
    plt.hist(saliency_map.flatten(), bins=50, color='red', alpha=0.7)
    plt.title("Saliency Value Dist.")
    plt.xlabel("Saliency val.")
    plt.ylabel("Frequency")
    plt.savefig(hist_path)
    plt.close()


class DataLoader:
    """
    Class to load image data (like arr5_cX.npz) and tuning curves for the VGG16 network.
    Adjust to your local directory structure as needed.
    """
    def __init__(self, image_path, tc_path):
        self.image_path = image_path
        self.tc_path = tc_path
        
    def load_category_images(self, category, image_type=1, max_images=75):
        """
        Example usage for array images with category=5:
          descatpics = np.load(impath + '/arr5_c5.npz')['arr_0']
        """
        if image_type == 1:
            # merged images
            file_path = os.path.join(self.image_path, f"merg5_c{category}.npz")
            data = np.load(file_path)
            raw_data = data['arr_0']
            reshaped = raw_data.reshape(-1, 224, 224, 3)
            return reshaped[:max_images]
        elif image_type == 2:
            # array images
            file_path = os.path.join(self.image_path, f"arr5_c{category}.npz")
            data = np.load(file_path)
            raw_data = data['arr_0']
            reshaped = raw_data.reshape(-1, 224, 224, 3)
            return reshaped[:max_images]
        elif image_type == 3:
            # test images or 20 category
            file_path = os.path.join(self.image_path, "cats20_test15_c.npy")
            data = np.load(file_path)
            # data might be 20 x 15 x 224 x 224 x 3
            if len(data.shape) > 4:
                reshaped = data[category].reshape(-1, 224, 224, 3)
                return reshaped[:max_images]
            return data[category]
        else:
            print(f"Unsupported image_type: {image_type}")
            return np.zeros((0,224,224,3), dtype=np.float32)
            
    def load_tuning_curves(self):
        # e.g. "featvecs20_train35_c.txt"
        # if needed
        pass
            
    def prepare_batch(self, positive_images, negative_images, batch_size):
        """ Optionally used to create TP/TN batches in older code. """
        tp_batch = np.zeros((batch_size, 224,224,3), dtype=positive_images.dtype)
        tn_batch = np.zeros((batch_size, 224,224,3), dtype=negative_images.dtype)
        
        for i in range(min(batch_size, len(positive_images))):
            tp_batch[i] = positive_images[i]
        for i in range(min(batch_size, len(negative_images))):
            tn_batch[i] = negative_images[i]
        return tp_batch, tn_batch

class Visualizer:
    """Potential class to plot more advanced metrics if needed."""
    def __init__(self):
        mpl.rcParams['font.size'] = 22
        
    def plot_attention_effects(self, metrics, attention_strengths):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
        ax1.plot(attention_strengths, metrics['performance'], 'k-', linewidth=2)
        ax1.set_ylabel('Performance')
        ax1.set_xlabel('Attention Strength')
        
        ax2.plot(attention_strengths, metrics['criteria'], 'b-', linewidth=2)
        ax2.set_ylabel('Criteria')
        ax2.set_xlabel('Attention Strength')
        
        ax3.plot(attention_strengths, metrics['sensitivity'], 'r-', linewidth=2)
        ax3.set_ylabel('Sensitivity (d\')')
        ax3.set_xlabel('Attention Strength')
        
        plt.tight_layout()
        return fig
        
    def plot_layer_modulation(self, layer_effects, layer_names):
        fig, ax = plt.subplots(figsize=(10,6))
        means = [effects['mean_modulation'] for effects in layer_effects.values()]
        stds = [effects['std_modulation'] for effects in layer_effects.values()]
        ax.errorbar(range(len(means)), means, yerr=stds, fmt='o-', capsize=5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_ylabel('Modulation Factor')
        ax.set_xlabel('Layer')
        plt.tight_layout()
        return fig
