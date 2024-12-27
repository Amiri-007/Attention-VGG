import argparse
import numpy as np
import os
import tensorflow as tf

# For TF2 in v1-compat mode:
tf.compat.v1.disable_eager_execution()

from vgg_16 import VGG16Base
from attention import AttentionMechanism, LayerAttention
from utils import (
    DataLoader,
    AttentionAnalyzer,
    pad_batch,
    make_attention_maps_with_batch,
    compute_saliency_map,
    debug_saliency,
    debug_print_shapes,
    compare_saliency_attention,
    visualize_comparison
)

def parse_args():
    parser = argparse.ArgumentParser(description='VGG16 Attention Analysis')
    parser.add_argument('--imtype', type=int, default=1, choices=[1, 2, 3],
                        help='Image type: 1=merge, 2=array, 3=test')
    parser.add_argument('--category', type=int, default=19,
                        help='Object category to attend to (0-19)')
    parser.add_argument('--layer', type=int, default=11,
                        help='Layer to apply attention (0-12, >12 for all layers)')
    parser.add_argument('--attention_type', type=str, default='TCs',
                        choices=['TCs', 'GRADs'], help='Type of attention to apply')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--max_images', type=int, default=2,
                        help='Maximum number of images to load')
    return parser.parse_args()

def setup_paths():
    """
    Setup local paths for your data. 
    Adjust these so they point to the correct directories on your Windows machine.
    """
    base_path = r'C:\Users\mreza\OneDrive\Documents\GitHub\Attention-VGG\Data'
    return {
        # Path containing gradient/tuning-curve .txt files
        'tc_path': os.path.join(base_path, 'object_GradsTCs'),

        # Path containing "vgg16_weights.npz" and "catbins" subfolder
        'weight_path': base_path,

        # Path containing images or .npz files like 'arr5_c5.npz'
        'image_path': os.path.join(base_path, 'images'),

        # Where to save result npy/npz files
        'save_path': base_path
    }

def main():
    args = parse_args()
    paths = setup_paths()
    
    print("\nVerifying paths:")
    for path_name, path_val in paths.items():
        exists = os.path.exists(path_val)
        print(f"{path_name}: {path_val} -> {'exists' if exists else 'MISSING'}")
        if not exists:
            print(f"Warning: {path_name} path does not exist!")
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    print("\nInitializing model placeholders...")
    imgs = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 224, 224, 3], name="imgs")
    labs = tf.compat.v1.placeholder(tf.int32, [args.batch_size, 1], name="labs")
    
    # Check for pretrained weights
    weights_path = os.path.join(paths['weight_path'], 'vgg16_weights.npz')
    if not os.path.exists(weights_path):
        print(f"Error: VGG16 weights file not found at {weights_path}")
        return
    
    # Create model
    vgg = VGG16Base(imgs=imgs, labs=labs, weights=weights_path, sess=sess)
    
    # Attempt to load checkpoint from catbins
    try:
        saver = tf.compat.v1.train.Saver({"fc3": vgg.fc3w, "fcb3": vgg.fc3b})
        ckpt_dir = os.path.join(paths['weight_path'], 'catbins')
        ckpt_prefix = f"catbin_{args.category}.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_prefix)

        data_file = ckpt_path + ".data-00000-of-00001"
        index_file = ckpt_path + ".index"
        
        print("Checking checkpoint files for binary classifier:")
        print(f"Data file: {data_file} -> {'Found' if os.path.exists(data_file) else 'Missing'}")
        print(f"Index file: {index_file} -> {'Found' if os.path.exists(index_file) else 'Missing'}")
        
        if not os.path.exists(data_file) or not os.path.exists(index_file):
            print("Warning: No checkpoint found for catbins; skipping load.")
        else:
            print(f"Loading checkpoint from: {ckpt_path}")
            saver.restore(sess, ckpt_path)
            print("Checkpoint loaded successfully")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    print("\nSetting up data loader and attention objects...")
    data_loader = DataLoader(paths['image_path'], paths['tc_path'])
    attention = AttentionMechanism(paths['tc_path'])
    layer_attention = LayerAttention()
    analyzer = AttentionAnalyzer(vgg, sess)
    visualizer = Visualizer()
    
    print(f"\nLoading category {args.category} data for imtype={args.imtype} ...")
    try:
        pos_images = data_loader.load_category_images(args.category, args.imtype, args.max_images)
        print(f"Loaded positive images shape: {pos_images.shape}")
        if pos_images.shape[0] == 0:
            print("Error: No positive images loaded!")
            return
    except Exception as e:
        print(f"Error loading positive images: {e}")
        return
    
    # Prepare save path for results
    save_dir = os.path.join(paths['save_path'], f"attention_results_cat{args.category}_layer{args.layer}")
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except Exception as e:
            print(f"Error creating save directory: {e}")
            return
    
    print("\nProcessing data in batches...")
    n_batches = (len(pos_images) + args.batch_size - 1) // args.batch_size
    
    # Example attention strengths
    attention_strengths = np.array([0.2, 0.7])
    
    for strength in attention_strengths:
        strength_vec = np.zeros(13, dtype=np.float32)
        if args.layer > 12:
            strength_vec[:] = strength * 0.1
        else:
            strength_vec[args.layer] = strength
        
        print(f"\nAttention strength = {strength}")
        print(f"strength_vec = {strength_vec}")
        
        for batch_idx in range(0, len(pos_images), args.batch_size):
            batch_end = min(batch_idx + args.batch_size, len(pos_images))
            tp_batch = pos_images[batch_idx:batch_end]
            tp_batch = pad_batch(tp_batch, args.batch_size)
            
            # Create labels for the batch
            tplabs = np.full((args.batch_size, 1), args.category, dtype=np.int32)
            
            # Attempt to load negative images from another category
            other_categories = list(range(20))
            if args.category in other_categories:
                other_categories.remove(args.category)
            neg_category = np.random.choice(other_categories)
            
            try:
                neg_images = data_loader.load_category_images(neg_category, args.imtype)
                if len(neg_images) == 0:
                    print(f"Warning: No negative images loaded for category {neg_category}")
                    continue
                tn_batch = neg_images[:args.batch_size]
                if len(tn_batch) < args.batch_size:
                    tn_batch = pad_batch(tn_batch, args.batch_size)
            except Exception as e:
                print(f"Error loading negative images: {e}")
                continue
            
            print(f"\nGenerating attention maps for batch {batch_idx // args.batch_size + 1}/{n_batches} ...")
            try:
                if args.attention_type == 'TCs':
                    attention.attype = 1  # multiplicative
                    attention_maps = make_attention_maps_with_batch(
                        attention, args.category, strength_vec, args.batch_size
                    )
                else:
                    attention.attype = 1
                    attention_maps = make_attention_maps_with_batch(
                        attention, args.category, strength_vec, args.batch_size
                    )
                
                if attention_maps is None or len(attention_maps) == 0:
                    print("Warning: No attention maps generated")
                    continue
                print(f"Created {len(attention_maps)} attention maps")
            except Exception as e:
                print(f"Error generating attention maps: {e}")
                continue
            
            # 1. Saliency maps
            print("Computing saliency maps...")
            saliency_maps = compute_saliency_map(sess, vgg, tp_batch, tplabs, attention_maps)
            if saliency_maps is None:
                print("Error: saliency maps could not be generated")
                continue
            debug_saliency(saliency_maps)
            
            # 2. Compare saliency vs. attention
            print("Comparing saliency and attention maps...")
            all_layer_metrics = {}
            try:
                for layer_idx in range(len(attention_maps)):
                    metrics = compare_saliency_attention(saliency_maps, attention_maps, layer_idx)
                    all_layer_metrics[f"layer_{layer_idx}"] = metrics
                    print(f"Layer {layer_idx} metrics: {metrics}")
            except Exception as e:
                print(f"Error in compare_saliency_attention: {e}")
                continue
            
            # 3. Analyze attention effects
            print("Analyzing attention effects (TP vs. TN responses)...")
            try:
                attention_results = analyzer.analyze_attention_effects(tp_batch, tn_batch, attention_maps, [strength])
            except Exception as e:
                print(f"Error analyzing attention effects: {e}")
                continue
            
            # 4. Visualize
            print("Visualizing comparisons for each image in the batch...")
            for img_idx in range(min(len(tp_batch), args.batch_size)):
                try:
                    compare_key = f"layer_{args.layer}"
                    if compare_key not in all_layer_metrics:
                        print(f"Cannot find metrics for {compare_key}, skipping visualization.")
                        continue
                    layer_metrics = all_layer_metrics[compare_key]
                    save_path = os.path.join(save_dir, f"comparison_batch{batch_idx}_img{img_idx}_strength{strength}.png")
                    visualize_comparison(tp_batch, saliency_maps, attention_maps, layer_metrics, save_path, batch_idx=img_idx)
                    print(f"Saved visualization at {save_path}")
                except Exception as e:
                    print(f"Error visualizing image {img_idx}: {e}")
            
            # 5. Save results
            try:
                results_dict = {
                    'strength': strength,
                    'saliency_maps': saliency_maps,
                    'attention_maps': attention_maps,
                    'comparison_metrics': all_layer_metrics,
                    'attention_results': attention_results
                }
                np_file_path = os.path.join(save_dir, f"batch_{batch_idx}_strength_{strength}_results.npy")
                np.save(np_file_path, results_dict)
                print(f"Saved batch results to {np_file_path}")
            except Exception as e:
                print(f"Error saving batch results: {e}")
                continue
    
    print(f"\nDone! Results saved under {save_dir}")

if __name__ == '__main__':
    main()
