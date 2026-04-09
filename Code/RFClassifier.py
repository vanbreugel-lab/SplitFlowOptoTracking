
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV  # Added for calibration
import matplotlib.patches as mpatches
import matplotlib as mpl
import splitflow.TripletLoss as CCL
from splitflow.PlotUtilities import *

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def extract_latent_representations(model, dataloader, representation_type='mu'):
    """
    Extract latent features with different representation options
    
    Args:
        representation_type: 'mu', 'z', or 'both'
    """
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label, obj in dataloader:
            if representation_type == 'mu':
                mu, logvar = model.encode(data)
                latents.append(mu.numpy())
            elif representation_type == 'z':
                _, _, _, z = model(data)
                latents.append(z.numpy())
            elif representation_type == 'both':
                mu, logvar = model.encode(data)
                _, _, _, z = model(data)
                # Concatenate mu and z for richer representation
                combined = torch.cat([mu, z], dim=1)
                latents.append(combined.numpy())
                
            labels.append(label.numpy())
    
    return np.vstack(latents), np.hstack(labels)

def create_classification_matrix_with_ood(predictions, class_label_dict=None, include_ood=True):
    """
    Create classification matrix including OOD samples
    """
    
    # Get all classes including OOD
    all_classes = set()
    for dataset_name, pred_data in predictions.items():
        all_classes.update(pred_data['numeric_labels'])
    
    all_classes = sorted(list(all_classes))
    
    # Add OOD class if present
    if -1 in all_classes and include_ood:
        all_classes.remove(-1)  # Remove -1 to place OOD at the end
        all_classes.append(-1)
    
    # Create class names
    class_names = []
    for cls in all_classes:
        if cls == -1:
            class_names.append('OOD/Rejected')
        elif class_label_dict is not None:
            class_names.append(class_label_dict.get(cls, f'Class {cls}'))
        else:
            class_names.append(f'Class {cls}')
    
    # Initialize matrices
    datasets = list(predictions.keys())
    count_matrix = pd.DataFrame(0, index=class_names, columns=datasets)
    percentage_matrix = pd.DataFrame(0.0, index=class_names, columns=datasets)
    
    # Fill matrices
    for dataset_name, pred_data in predictions.items():
        numeric_labels = pred_data['numeric_labels']
        total_samples = len(numeric_labels)
        
        # Count occurrences
        unique, counts = np.unique(numeric_labels, return_counts=True)
        class_count_dict = dict(zip(unique, counts))
        
        # Update matrices
        for cls in all_classes:
            count = class_count_dict.get(cls, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            
            if cls == -1:
                class_name = 'OOD/Rejected'
            elif class_label_dict is not None:
                class_name = class_label_dict.get(cls, f'Class {cls}')
            else:
                class_name = f'Class {cls}'
            
            count_matrix.loc[class_name, dataset_name] = count
            percentage_matrix.loc[class_name, dataset_name] = percentage
    
    return percentage_matrix, count_matrix

def plot_classification_matrix_with_ood(percentage_matrix, count_matrix, figsize=(14, 8), path=None):
    """Plot classification matrix with OOD samples highlighted"""
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create custom colormap that highlights OOD
    import matplotlib.colors as mcolors
    cmap = plt.cm.bone_r.copy()
    if 'OOD/Rejected' in percentage_matrix.index:
        # Make OOD stand out with a different color
        #ood_index = list(percentage_matrix.index).index('OOD/Rejected')
        colors = list(plt.cm.bone_r(np.linspace(0, 1, len(percentage_matrix.index))))
        #colors[ood_index] = (0.7, 0.3, 0.3, 1.0)  # Red color for OOD
        cmap = mcolors.ListedColormap(colors)
    
    # Plot 1: Heatmap
    annotations = []
    for i in range(len(percentage_matrix)):
        row_annotations = []
        for j in range(len(percentage_matrix.columns)):
            pct = percentage_matrix.iloc[i, j]
            count = count_matrix.iloc[i, j]
            row_annotations.append(f'{pct:.1f}%\n({count})')
        annotations.append(row_annotations)
    annotations = np.array(annotations)
    
    sns.heatmap(percentage_matrix, annot=annotations, fmt='', cmap=cmap,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax1)
    ax1.set_title('Classification Distribution with OOD Rejection')
    ax1.set_xlabel('Datasets')
    ax1.set_ylabel('Classes (OOD highlighted)')
    
    plt.tight_layout()
    plt.show()
    if path is not None:
        fig.savefig(path, transparent=True)

def predict_with_ood_rejection(model, train_loader, unlabeled_datasets, dataset_names=None, 
                             class_label_dict=None, confidence_threshold=0.7, 
                             distance_threshold=None):
    """
    Predict labels with OOD rejection option
    
    Args:
        model: Trained TripletLossVAE model
        train_loader: DataLoader for training data
        unlabeled_datasets: List of DataLoaders for unlabeled datasets
        dataset_names: List of names for each dataset
        class_label_dict: Dictionary mapping numeric labels to class names
        confidence_threshold: Minimum confidence for accepting prediction (0-1)
        distance_threshold: Maximum distance from training distribution (if using distance-based)
        method: 'confidence', 'distance', or 'both'
    """
    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Number of unlabeled datasets: {len(unlabeled_datasets)}")
    for i, loader in enumerate(unlabeled_datasets):
        print(f"  Dataset {i}: {len(loader.dataset)} samples")
    # Extract training representations
    train_latents, train_labels = extract_latent_representations(model, train_loader)
    test_latents, test_labels = extract_latent_representations(model, test_loader)
    # Train classifier with calibration using Platt's method (sigmoid)
    base_rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf = CalibratedClassifierCV(base_rf, method='temperature', cv=5)
    #rf.fit(test_latents, test_labels)
    rf.fit(train_latents, train_labels)
    
    print("Random Forest classifier calibrated using Platt's method (sigmoid)")

    # Generate dataset names if not provided
    if dataset_names is None:
        dataset_names = [f'dataset_{i}' for i in range(len(unlabeled_datasets))]
    
    predictions = {}
    ood_analysis = {}
    
    for i, unlabeled_loader in enumerate(unlabeled_datasets):
        dataset_name = dataset_names[i]
        print(f"Processing {dataset_name} with OOD rejection...")
        
        # Extract latent representations
        unlabeled_latents, _ = extract_latent_representations(model, unlabeled_loader)
        #unlabeled_latents, _ = sextract_latent_representations(model, unlabeled_loader)
                    
        # Predict probabilities (now calibrated!)
        pred_proba = rf.predict_proba(unlabeled_latents)
        confidence_scores = np.max(pred_proba, axis=1)
        pred_labels = rf.predict(unlabeled_latents)
        
        # Apply OOD rejection
        ood_mask = np.zeros(len(pred_labels), dtype=bool)
        

        ood_mask = confidence_scores < confidence_threshold
        
        # Apply rejection
        final_labels = pred_labels.copy()
        final_labels[ood_mask] = -1  # Use -1 for OOD/rejected samples
        
        final_class_names = []
        for label, is_ood in zip(pred_labels, ood_mask):
            if is_ood:
                final_class_names.append('OOD/Rejected')
            elif class_label_dict is not None:
                final_class_names.append(class_label_dict[label])
            else:
                final_class_names.append(f'Class {label}')
        
        predictions[dataset_name] = {
            'numeric_labels': final_labels,
            'class_names': final_class_names,
            'original_numeric_labels': pred_labels,
            'original_class_names': [class_label_dict.get(l, f'Class {l}') for l in pred_labels] if class_label_dict else [f'Class {l}' for l in pred_labels],
           'probabilities': pred_proba,
            'confidence': confidence_scores,
            'ood_mask': ood_mask,
            'latent_representations': unlabeled_latents
        }
        
        # OOD analysis
        ood_count = np.sum(ood_mask)
        total_count = len(ood_mask)
        ood_analysis[dataset_name] = {
            'total_samples': total_count,
            'ood_count': ood_count,
            'ood_percentage': (ood_count / total_count) * 100,
            'accepted_count': total_count - ood_count,
            'accepted_percentage': ((total_count - ood_count) / total_count) * 100,
            'mean_confidence_accepted': np.mean(confidence_scores[~ood_mask]) if np.sum(~ood_mask) > 0 else 0,
            'mean_confidence_ood': np.mean(confidence_scores[ood_mask]) if ood_count > 0 else 0
        }
    
    return predictions, ood_analysis


def plot_classification_matrix_with_colorbars(
    percentage_matrix, 
    count_matrix, ax1, ax2, ax3, ax4=None,
    y_colors=None,
    x_colors=None,
    show_y_labels=True,
    show_x_labels=True,
    figsize=(14, 8), 
    path=None, 
    colorbar_width=0.02,
    colorbar_spacing=0.005,
    colorbar_border=False,
    show_percentage_cbar=True,
    heatmap_cmap='bone_r',
    vmin=0,
    vmax=100,
    row_order=None,   # ← add this
    ):
    
    # Reorder rows if specified
    if row_order is not None:
        percentage_matrix = percentage_matrix.iloc[row_order]
        count_matrix = count_matrix.iloc[row_order]
        if y_colors is not None:
            y_colors = [y_colors[i] for i in row_order]
    """
    Plot classification matrix with continuous heatmap and discrete colorbars for axes.
    """
    ax_heatmap = ax1
    ax_ycolorbar = ax2
    ax_xcolorbar = ax3
    
    # Determine what to do with the continuous percentage colorbar
    if show_percentage_cbar:
        if ax4 is not None:
            # Use the provided axes for the continuous colorbar
            cbar_ax = ax4
            cbar_kws = {'label': 'Percentage (%)'}
        else:
            # Create a continuous colorbar axes next to the heatmap
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_heatmap)
            cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
            cbar_kws = {'label': 'Percentage (%)'}
    else:
        # Don't show continuous percentage colorbar
        cbar_ax = None
        cbar_kws = {}
    
    # Prepare annotations
    annotations = []
    for i in range(len(percentage_matrix)):
        row_annotations = []
        for j in range(len(percentage_matrix.columns)):
            pct = percentage_matrix.iloc[i, j]
            row_annotations.append(f'{round(pct)}')
        annotations.append(row_annotations)
    annotations = np.array(annotations)

    # Plot heatmap with CONTINUOUS colormap
    if show_percentage_cbar and cbar_ax is not None:
        # Plot with continuous colorbar
        im = sns.heatmap(percentage_matrix, 
                        annot=annotations, 
                        fmt='', 
                        cmap=heatmap_cmap,  # Continuous colormap
                        cbar_kws=cbar_kws,
                        ax=ax_heatmap, 
                        vmin=vmin, 
                        vmax=vmax,
                        cbar=True, 
                        cbar_ax=cbar_ax)
    else:
        # Plot without continuous colorbar
        im = sns.heatmap(percentage_matrix, 
                        annot=annotations, 
                        fmt='', 
                        cmap=heatmap_cmap,  # Continuous colormap
                        ax=ax_heatmap, 
                        vmin=vmin, 
                        vmax=vmax,
                        cbar=False)
    
    # Remove axis labels from heatmap
    ax_heatmap.set_xlabel('')
    ax_heatmap.set_ylabel('')
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    
    # Remove borders/spines from DISCRETE colorbar axes
    for spine in ax_ycolorbar.spines.values():
        spine.set_visible(colorbar_border)
    for spine in ax_xcolorbar.spines.values():
        spine.set_visible(colorbar_border)
    
    # Create DISCRETE y-axis colorbar (classes)
    if y_colors is None:
        # Generate discrete colors if not provided
        y_colors = plt.cm.tab10(np.linspace(0, 1, len(percentage_matrix.index)))
    y_colors = list(reversed(y_colors))
    
    # Create DISCRETE vertical colorbar for y-axis
    y_norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(len(y_colors) + 1) - 0.5,
        ncolors=len(y_colors)
    )
    
    # Create DISCRETE colorbar for y-axis
    y_cbar = mpl.colorbar.ColorbarBase(
        ax_ycolorbar,
        cmap=mpl.colors.ListedColormap(y_colors),
        norm=y_norm,
        orientation='vertical',
        ticks=np.arange(len(percentage_matrix.index)),
        drawedges=False  # Remove edges between colors for discrete bars
    )
    
    # Remove tick lines for discrete colorbar
    y_cbar.ax.tick_params(size=0, width=0, pad=0)

    if show_y_labels:
        # Set labels on the LEFT side of the discrete colorbar
        y_cbar.ax.set_yticklabels(list(reversed(percentage_matrix.index.tolist())), 
                                 ha='right',
                                 va='center',
                                 position=(-0.01, 0))
        # Move y-axis to the left
        ax_ycolorbar.yaxis.set_label_position('left')
        ax_ycolorbar.yaxis.tick_left()
    else:
        # Remove labels if not showing
        y_cbar.ax.set_yticklabels([])
        y_cbar.ax.axis('off')
    
    # Create DISCRETE x-axis colorbar (datasets)
    if x_colors is None:
        # Generate discrete colors if not provided
        x_colors = plt.cm.Set3(np.linspace(0, 1, len(percentage_matrix.columns)))
    
    # Create DISCRETE horizontal colorbar for x-axis
    x_norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(len(x_colors) + 1) - 0.5,
        ncolors=len(x_colors)
    )
    
    x_cbar = mpl.colorbar.ColorbarBase(
        ax_xcolorbar,
        cmap=mpl.colors.ListedColormap(x_colors),
        norm=x_norm,
        orientation='horizontal',
        ticks=np.arange(len(percentage_matrix.columns)),
        drawedges=False  # Remove edges between colors for discrete bars
    )
    
    # Remove tick lines for discrete colorbar
    x_cbar.ax.tick_params(size=0, width=0, pad=0)
    
    if show_x_labels:
        # Get column names
        if hasattr(percentage_matrix.columns, 'tolist'):
            x_labels = percentage_matrix.columns.tolist()
        else:
            x_labels = [f'Dataset {i+1}' for i in range(len(percentage_matrix.columns))]
        
        # Set labels on top for discrete colorbar
        x_cbar.ax.set_xticklabels(x_labels, 
                                 rotation=45, ha="right", rotation_mode="anchor",
                                 position=(-1, -1))
    else:
        # Remove labels if not showing
        x_cbar.ax.set_xticklabels([])
        x_cbar.ax.axis('off')
        
    # Remove tick marks from DISCRETE colorbars
    ax_ycolorbar.tick_params(axis='both', length=0)
    ax_xcolorbar.tick_params(axis='both', length=0)
    
    # Return values based on whether we have a continuous percentage colorbar
    if ax4 is not None and show_percentage_cbar:
        return ax_heatmap, ax_ycolorbar, ax_xcolorbar, cbar_ax
    else:
        return ax_heatmap, ax_ycolorbar, ax_xcolorbar
