import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)



############# test on simple trajectories ##################
############################################################################################################################
############################################################################################################################


def generate_sine_wave(num_trajectories, seq_length=200, frequency_range=(0.1, 3), amplitude_range=(-10, 10)):
    trajectories = []
    for _ in range(num_trajectories):
        t = np.linspace(0, 4*np.pi, seq_length)
        freq = np.random.uniform(frequency_range[0], frequency_range[1])
        amp = np.random.uniform(amplitude_range[0], amplitude_range[1])
        phase = np.random.uniform(0, 2*np.pi)

        x = t
        y = amp * np.sin(freq * t + phase)
        x_shift = np.roll(x, 1)
        x_shift[0] = 0
        xvel = (x - x_shift)/0.01
        y_shift = np.roll(y, 1)
        y_shift[0] = 0
        yvel = (y - y_shift)/0.01
        trajectory = np.column_stack([x, y,xvel,yvel])
        trajectories.append(trajectory)

    return np.array(trajectories)

def generate_circle(num_trajectories, seq_length=200, radius_range=(0.1, 2)):
    trajectories = []
    for _ in range(num_trajectories):
        t = np.linspace(0, 2*np.pi, seq_length)
        radius = np.random.uniform(radius_range[0], radius_range[1])
        phase = np.random.uniform(0, 2*np.pi)

        x = radius * np.cos(t + phase)
        y = radius * np.sin(t + phase)
        x_shift = np.roll(x, 1)
        x_shift[0] = 0
        xvel = (x - x_shift)/0.01
        y_shift = np.roll(y, 1)
        y_shift[0] = 0
        yvel = (y - y_shift)/0.01
        
        trajectory = np.column_stack([x, y,xvel,yvel])
        trajectories.append(trajectory)

    return np.array(trajectories)

def generate_straight_line(num_trajectories, seq_length=200, slope_range=(-10, 10), intercept_range=(-5, 5)):
    trajectories = []
    for _ in range(num_trajectories):
        t = np.linspace(-2, 2, seq_length)
        slope = np.random.uniform(slope_range[0], slope_range[1])
        intercept = np.random.uniform(intercept_range[0], intercept_range[1])

        x = t
        y = slope * t + intercept
        x_shift = np.roll(x, 1)
        x_shift[0] = 0
        xvel = (x - x_shift)/0.01
        y_shift = np.roll(y, 1)
        y_shift[0] = 0
        yvel = (y - y_shift)/0.01
        
        trajectory = np.column_stack([x, y,xvel,yvel])
        trajectories.append(trajectory)

    return np.array(trajectories)

def generate_dataset(samples_per_class=1000, seq_length=200):
    sine_trajectories = generate_sine_wave(samples_per_class, seq_length)
    circle_trajectories = generate_circle(samples_per_class, seq_length)
    line_trajectories = generate_straight_line(samples_per_class, seq_length)

    trajectories = np.concatenate([sine_trajectories, circle_trajectories,line_trajectories, ], axis=0)

    # Create labels: 0=sine, 1=circle, 2=straight line
    labels = np.concatenate([
        np.zeros(samples_per_class),  # sine
        np.ones(samples_per_class),   # circle
        np.full(samples_per_class, 2) # straight line
    ])

    return trajectories, labels

def plot_example_simple_trajectories(trajectories, labels, num_examples=5):
    """Plot example trajectories for each type"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_names = ['Sine Wave', 'Circle', 'Straight Line']
    
    for label in range(3):
        # Find indices of this label
        indices = np.where(labels == label)[0]
        examples = trajectories[indices[:num_examples]]
        
        ax = axes[label]
        for i in range(min(num_examples, len(examples))):
            traj = examples[i]  # Already in shape (seq_length, 2)
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2)
            ax.scatter(traj[0, 0], traj[0, 1], color='red', s=50, zorder=5)  # Start point
            ax.scatter(traj[-1, 0], traj[-1, 1], color='green', s=50, zorder=5)  # End point
        
        ax.set_title(f'{label_names[label]} Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def classify_latent_space(latent_vectors, true_labels, class_labels=None):
    """Classify trajectories using latent space features"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        latent_vectors, true_labels, test_size=0.3, random_state=42, stratify=true_labels
    )
    
    # Train classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    if class_labels is None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, )
    #  display_labels=['Sine', 'Circle', 'Line'])

    else: 
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Latent Space Classification')
    plt.show()
    
    return accuracy




#### unsupervised pipeline ##################
############################################################################################################################
############################################################################################################################
'''
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels):
        # Flatten and normalize trajectories
        trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
        
        # Normalize each trajectory independently
        self.trajectories = self.normalize_trajectories(trajectories_flat)
        self.labels = torch.LongTensor(labels)
    
    def normalize_trajectories(self, trajectories):
        """Normalize each trajectory to have zero mean and unit variance"""
        trajectories_norm = trajectories.copy()
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            mean = np.mean(traj)
            std = np.std(traj)
            if std > 0:
                trajectories_norm[i] = (traj - mean) / std
        return torch.FloatTensor(trajectories_norm)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]
'''
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels, obj_ids=None, normalization_type='zero_mean'):
        # Flatten and normalize trajectories
        trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
        
        # Normalize each trajectory independently
        if normalization_type == 'zero_mean':
            self.trajectories = self.normalize_zero_mean(trajectories_flat)
        elif normalization_type == 'minmax':
            self.trajectories = self.normalize_minmax(trajectories_flat)
        else:
            raise ValueError("normalization_type must be 'zero_mean' or 'minmax'")
            
        self.labels = torch.LongTensor(labels)
        self.obj_ids = obj_ids  # Store obj_id_unique
        
    def normalize_zero_mean(self, trajectories):
        """Normalize each trajectory to have zero mean and unit variance"""
        trajectories_norm = trajectories.copy()
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            mean = np.mean(traj)
            std = np.std(traj)
            if std > 0:
                trajectories_norm[i] = (traj - mean) / std
        return torch.FloatTensor(trajectories_norm)
    
    def normalize_minmax(self, trajectories, feature_range=(-1, 1)):
        """Normalize each trajectory to the given feature range (default: [0, 1])"""
        trajectories_norm = trajectories.copy()
        min_val, max_val = feature_range
        
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            traj_min = np.min(traj)
            traj_max = np.max(traj)
            
            # Avoid division by zero
            if traj_max - traj_min > 0:
                # Scale to [0, 1] first, then to target range
                trajectories_norm[i] = (traj - traj_min) / (traj_max - traj_min)
                trajectories_norm[i] = trajectories_norm[i] * (max_val - min_val) + min_val
            else:
                # If all values are the same, set to the middle of the range
                trajectories_norm[i] = np.full_like(traj, (min_val + max_val) / 2)
                
        return torch.FloatTensor(trajectories_norm)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        if self.obj_ids is not None:
            return self.trajectories[idx], self.labels[idx], self.obj_ids[idx]
        else:
            return self.trajectories[idx], self.labels[idx]
            
class ImprovedVAE(nn.Module):
    def __init__(self, input_dim=800, hidden_dim=[256,128,64], latent_dim=20):  # Updated input_dim for 4 features
        super(ImprovedVAE, self).__init__()
        
        # Larger encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.1),
            
            #nn.Linear(hidden_dim[1], hidden_dim[2]),
            #nn.BatchNorm1d(hidden_dim[2]),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.1),
            
            nn.Linear(hidden_dim[1], hidden_dim[-1]),
            nn.BatchNorm1d(hidden_dim[-1]),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim[-1], latent_dim)
        
        # Larger decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[-1]),
            nn.BatchNorm1d(hidden_dim[-1]),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.2),
            
            #nn.Linear(hidden_dim[-1], hidden_dim[2]),
            #nn.BatchNorm1d(hidden_dim[2]),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.1),
            
            nn.Linear(hidden_dim[-1], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.1),
            
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            
            nn.Linear(hidden_dim[0], input_dim),
            # No activation here - we'll handle normalization separately
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def plot_example_trajectories_simple(trajectories, labels, num_examples=5):
    """Plot example trajectories for each type"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_names = ['Sine Wave', 'Circle', 'Straight Line']
    
    for label in range(3):
        # Find indices of this label
        indices = np.where(labels == label)[0]
        examples = trajectories[indices[:num_examples]]
        
        ax = axes[label]
        for i in range(min(num_examples, len(examples))):
            traj = examples[i]  # Already in shape (seq_length, 2)
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2)
            ax.scatter(traj[0, 0], traj[0, 1], color='red', s=50, zorder=5)  # Start point
            ax.scatter(traj[-1, 0], traj[-1, 1], color='green', s=50, zorder=5)  # End point
        
        ax.set_title(f'{label_names[label]} Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_example_trajectories(trajectories, labels, num_examples=5):
    """Plot example trajectories for each type"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_names = np.unique(labels)#['Sine Wave', 'Circle', 'Straight Line']
    
    for label in np.unique(labels):
        # Find indices of this label
        indices = np.where(labels == label)[0]
        examples = trajectories[indices[:num_examples]]
        
        ax = axes[label]
        for i in range(min(num_examples, len(examples))):
            # Plot positions (first two columns)
            traj = examples[i, :, 0:2]  # Only use x, y positions for plotting
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2)
            ax.scatter(traj[0, 0], traj[0, 1], color='red', s=50, zorder=5)  # Start point
            ax.scatter(traj[-1, 0], traj[-1, 1], color='green', s=50, zorder=5)  # End point
        
        ax.set_title(f'{label_names[label]} Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_velocity_fields(trajectories, labels, num_examples=3):
    """Plot velocity fields for example trajectories"""
    fig, axes = plt.subplots(3, num_examples, figsize=(15, 10))
    label_names = np.unique(labels)
    
    for label in np.unique(labels):
        # Find indices of this label
        indices = np.where(labels == label)[0]
        examples = trajectories[indices[:num_examples]]
        
        for example_idx in range(num_examples):
            ax = axes[label, example_idx]
            traj = examples[example_idx]
            
            # Plot trajectory
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=2)
            
            # Plot velocity vectors (plot every 10th point for clarity)
            skip = max(1, traj.shape[0] // 20)
            for i in range(0, traj.shape[0], skip):
                ax.arrow(traj[i, 0], traj[i, 1], 
                        traj[i, 2] * 0.1, traj[i, 3] * 0.1,  # Scale velocities for visualization
                        head_width=0.1, head_length=0.1, fc='r', ec='r', alpha=0.6)
            
            ax.scatter(traj[0, 0], traj[0, 1], color='red', s=50, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color='green', s=50, zorder=5)
            
            if example_idx == 0:
                ax.set_ylabel(f'{label_names[label]}')
            if label == 2:
                ax.set_xlabel('X')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Example {example_idx + 1}')
    
    plt.tight_layout()
    plt.show()


def plot_specific_reconstruction(model, test_loader, target_obj_id, color, n_input_feats=4, device='cpu', fifi=False, ax1=None, ax2=None ):
    """
    Find and plot original vs reconstructed trajectory for a specific obj_id.
    
    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        target_obj_id: The specific object ID to plot
        n_input_feats: Number of features per timestep (default=4)
        device: Device to run model on ('cpu' or 'cuda')
    """
    model.eval()
    model.to(device)
    
    found = False
    target_data = None
    target_obj_id_actual = None
    
    # Search through the test loader for the specific obj_id
    with torch.no_grad():
        for batch in test_loader:
            try:
                # Try to get obj_ids if available
                data, labels, obj_ids = batch
            except ValueError:
                # Fallback if obj_ids not available
                data, labels = batch
                obj_ids = None
            
            # If obj_ids are available, search for the target
            if obj_ids is not None:
                # Convert to list for easier searching
                obj_ids_list = list(obj_ids.numpy() if hasattr(obj_ids, 'numpy') else obj_ids)
                
                # Check if target_obj_id is in this batch
                if target_obj_id in obj_ids_list:
                    idx = obj_ids_list.index(target_obj_id)
                    target_data = data[idx:idx+1].to(device)  # Keep batch dimension
                    target_obj_id_actual = obj_ids[idx]
                    found = True
                    break
            else:
                print("Warning: obj_ids not available in the data")
                break
    
    if not found:
        print(f"Object ID {target_obj_id} not found in test data.")
        return
    
    # Get reconstruction for the found data
    with torch.no_grad():
        try:
            recon_data, _, _ = model(target_data)
        except:
            recon_data, _, _, _ = model(target_data)
    
    # Convert to numpy
    target_data_np = target_data.cpu().numpy()
    recon_data_np = recon_data.cpu().numpy()

    # Original trajectory
    orig_traj_full = target_data_np[0].reshape(-1, n_input_feats)
    orig_traj = orig_traj_full[:, 0:2]  # Only x, y positions
    

        # Reconstructed trajectory
    recon_traj_full = recon_data_np[0].reshape(-1, n_input_feats)
    recon_traj = recon_traj_full[:, 0:2]  # Only x, y positions
    
    if fifi==False:
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(5,5))
    
        axes[0].plot(orig_traj[:, 0], orig_traj[:, 1], linewidth=1, color=color, label='Original', alpha=1)
        axes[0].axis('off')
        axes[0].set_aspect('equal',)
        
        axes[1].plot(recon_traj[:, 0], recon_traj[:, 1], '--', linewidth=1, color=color, label='Reconstructed', alpha=1)
        axes[1].axis('off')
        axes[1].set_aspect('equal','box')
        plt.tight_layout()
        plt.show()
    
    # Save the figure
        fig.savefig(f'reconstruction_obj_{target_obj_id}.svg', transparent=True, bbox_inches='tight')
        print(f"Plot saved as 'reconstruction_obj_{target_obj_id}.svg'")
    else:
       
        ax1.plot(orig_traj[:, 0], orig_traj[:, 1], linewidth=1, color=color, label='Original', alpha=1)
        ax1.axis('off')
        ax1.set_aspect('equal',)
        
        
        ax2.plot(recon_traj[:, 0], recon_traj[:, 1], '--', linewidth=1, color=color, label='Reconstructed', alpha=1)
        ax2.axis('off')
        ax2.set_aspect('equal','box')
        #plt.tight_layout()
        #plt.show()
        return ax1,ax2


def circular_distance(pred_vel, true_vel):
    """
    Compute circular distance between predicted and true velocity directions
    """
    # Compute angles from velocities
    true_angle = torch.atan2(true_vel[:, 1::2], true_vel[:, 0::2])  # [yvel, xvel] pairs
    pred_angle = torch.atan2(pred_vel[:, 1::2], pred_vel[:, 0::2])
    
    # Circular distance - handle periodicity
    diff = torch.atan2(torch.sin(true_angle - pred_angle), 
                       torch.cos(true_angle - pred_angle))
    

    return torch.mean(torch.abs(diff))

def improved_vae_loss_with_course(recon_x, x, mu, logvar, beta=0.001, course_weight=0.01):
    """VAE loss with circular course direction penalty"""
    # Original reconstruction loss
    recon_loss = nn.SmoothL1Loss()(recon_x, x)
    
    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Course direction loss
    course_loss = circular_distance(recon_x, x)
    
    return recon_loss + beta*kld_loss + course_weight*course_loss

def improved_vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """Improved VAE loss with adjustable beta and better reconstruction loss"""
    # Use smooth L1 loss for better gradient behavior
    recon_loss = nn.SmoothL1Loss()(recon_x, x)
    
    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #print('KL:', kld_loss)
    return recon_loss + beta * kld_loss, kld_loss

def train_vae(model, train_loader, test_loader, epochs=200):
    """Train the VAE model with validation"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    model.train()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            #loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=0.01)
            loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=0.001)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                recon_batch, mu, logvar = model(data)
                loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=0.001)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses



def create_umap_distillation_model(input_dim, hidden_dims=[128, 64, 32], output_dim=2):
    """Train encoder to reproduce UMAP embeddings"""
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dims[0], activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_dims[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(hidden_dims[2], activation='relu'),
        tf.keras.layers.Dense(output_dim, name='embedding')
    ])
    return encoder

def distill_umap_embedding(latent_vectors, umap_embedding, epochs=200):
    """Train encoder to mimic UMAP projection"""
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latent_vectors)
    
    # Create and train model
    encoder = create_umap_distillation_model(X_scaled.shape[1])
    
    encoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['CosineSimilarity']
    )
    
    # Train to reproduce UMAP coordinates
    history = encoder.fit(
        X_scaled, umap_embedding,
        batch_size=32,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
    )
    
    return encoder, scaler, history

# Usage:
# First get good UMAP embedding
    #original_umap = umap.UMAP(n_components=2, random_state=42)
    #umap_embedding = original_umap.fit_transform(latent_vectors)

    #tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    #latent_tsne = tsne.fit_transform(latent_vectors)


    #with tf.device('/cpu:2'):
    # Distill into parametric model
     #   encoder, scaler, history = distill_umap_embedding(latent_vectors,umap_embedding)

    # Project new data
     #   new_data_projected = encoder.predict(latent_vectors)

    #with tf.device('/cpu:2'):
        #real_data_projected = encoder.predict(real_latent_vectors)
        
        
    '''    
    # Plot first two dimensions of latent space
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First two dimensions
    scatter1 = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                          c=true_labels, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('Original Latent Dimensions')
    plt.colorbar(scatter1, ax=ax1, label='Class')


    scatter2 = ax2.scatter(new_data_projected[:, 0], new_data_projected[:, 1], 
                          c=true_labels, cmap='viridis', alpha=0.7)


    scatter2 = ax2.scatter(real_data_projected[:, 0], real_data_projected[:, 1], 
                          c=real_true_labels, cmap='bwr', alpha=0.7)


    ax2.set_xlabel('New Latent Dim 1')
    ax2.set_ylabel('New Latent Dim 2')
    ax2.set_title('2D Latent Space from Vanilla AE using UMAP projection in training')
    plt.colorbar(scatter2, ax=ax2, label='Class')


    plt.tight_layout()
    plt.show()
    '''
    
'''  
def plot_reconstructions(model, test_loader, n_input_feats = 4,  num_samples=20):
    """Plot original vs reconstructed trajectories (positions only)"""
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data, labels = next(iter(test_loader))
        try:
            recon_data, _, _ = model(data)
        except:
            recon_data, _, _,_ = model(data)
        fig, axes = plt.subplots(2, num_samples, figsize=(45, 6))

        
        for i in range(num_samples):
            # Original - reshape back to (seq_length, 4) and take only positions

            orig_traj_full = data[i].numpy().reshape(-1, n_input_feats)
            
            orig_traj = orig_traj_full[:, 0:2]  # Only x, y positions
            
            axes[0, i].plot(orig_traj[:, 0], orig_traj[:, 1], 'b-', linewidth=2, label='Original')
            axes[0, i].scatter(orig_traj[0, 0], orig_traj[0, 1], color='red', s=50)
            axes[0, i].scatter(orig_traj[-1, 0], orig_traj[-1, 1], color='green', s=50)
            #axes[0, i].set_title(f'Original ({label_names[labels[i].item()]})')
            axes[0, i].grid(True, alpha=0.3)
            
            # Reconstructed - reshape back to (seq_length, 4) and take only positions

            recon_traj_full = recon_data[i].numpy().reshape(-1, n_input_feats)

            recon_traj = recon_traj_full[:, 0:2]  # Only x, y positions
            
            axes[1, i].plot(recon_traj[:, 0], recon_traj[:, 1], 'r--', linewidth=2, label='Reconstructed')
            axes[1, i].scatter(recon_traj[0, 0], recon_traj[0, 1], color='red', s=50)
            axes[1, i].scatter(recon_traj[-1, 0], recon_traj[-1, 1], color='green', s=50)
            #axes[1, i].set_title(f'Reconstructed ({label_names[labels[i].item()]})')
            axes[1, i].grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.show()
'''

def plot_reconstructions(model, test_loader, n_input_feats=4, num_samples=20):
    """Plot original vs reconstructed trajectories (positions only)"""
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        data_iter = iter(test_loader)
        try:
            # Try to get obj_ids if available
            data, labels, obj_ids = next(data_iter)
        except ValueError:
            # Fallback if obj_ids not available
            data, labels = next(data_iter)
            obj_ids = None
        
        try:
            recon_data, _, _ = model(data)
        except:
            recon_data, _, _,_ = model(data)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(45, 6))

        for i in range(num_samples):
            # Create title with obj_id if available
            if obj_ids is not None:
                #title_suffix = f" ({obj_ids[i]})"
                title_suffix = ""
                print(obj_ids[i])
            else:
                title_suffix = ""
            
            # Original - reshape back to (seq_length, 4) and take only positions
            orig_traj_full = data[i].numpy().reshape(-1, n_input_feats)
            orig_traj = orig_traj_full[:, 0:2]  # Only x, y positions
            
            axes[0, i].plot(orig_traj[:, 0], orig_traj[:, 1], 'b-', linewidth=2, label='Original')
            axes[0, i].scatter(orig_traj[0, 0], orig_traj[0, 1], color='red', s=50)
            axes[0, i].scatter(orig_traj[-1, 0], orig_traj[-1, 1], color='green', s=50)
            axes[0, i].set_title(f'Original{title_suffix}')
            #axes[0, i].grid(False, alpha=0.3)
            axes[0, i].axis('off')
            # Reconstructed - reshape back to (seq_length, 4) and take only positions
            recon_traj_full = recon_data[i].numpy().reshape(-1, n_input_feats)
            recon_traj = recon_traj_full[:, 0:2]  # Only x, y positions
            
            axes[1, i].plot(recon_traj[:, 0], recon_traj[:, 1], 'r--', linewidth=2, label='Reconstructed')
            axes[1, i].scatter(recon_traj[0, 0], recon_traj[0, 1], color='red', s=50)
            axes[1, i].scatter(recon_traj[-1, 0], recon_traj[-1, 1], color='green', s=50)
            axes[1, i].set_title(f'Reconstructed{title_suffix}')
            axes[1, i].axis('off')
            #axes[1, i].grid(False, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        fig.savefig('reconstructions.svg', transparent=True)

def plot_latent_space(model, test_loader, labels, real_data_loader=None, real_data_labels=None):
    """Plot latent space and t-SNE visualization"""
    model.eval()
    latent_vectors = []
    true_labels = []
    obj_ids = []    
    with torch.no_grad():
        try:
            for data, label, obj in test_loader:
                mu, logvar = model.encode(data)
                latent_vectors.append(mu.numpy())
                true_labels.extend(label.numpy())
        except:    
            for data, label in test_loader:
                mu, logvar = model.encode(data)
                latent_vectors.append(mu.numpy())
                true_labels.extend(label.numpy())
        
    latent_vectors = np.vstack(latent_vectors)
    true_labels = np.array(true_labels)
    
    if real_data_loader is not None:
        real_latent_vectors = []
        real_true_labels = []
    
        with torch.no_grad():
            for data, label in real_data_loader:
                mu, logvar = model.encode(data)
                real_latent_vectors.append(mu.numpy())
                real_true_labels.extend(label.numpy())

        real_latent_vectors = np.vstack(real_latent_vectors)
        real_true_labels = np.array(real_true_labels)
    
    # Plot first two dimensions of latent space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First two dimensions
    scatter1 = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                          c=true_labels, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('First Two Latent Dimensions')
    plt.colorbar(scatter1, ax=ax1, label='Class')
    
    # t-SNE visualization
    #tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    #from umap.parametric_umap import ParametricUMAP
    #embedder = ParametricUMAP()
    #embedding = embedder.fit_transform(my_data)
    tsne = umap.UMAP(n_components=2,random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    scatter2 = ax2.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                          c=true_labels, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.set_title('UMAP Visualization of Latent Space')
    plt.colorbar(scatter2, ax=ax2, label='Class')
    
    
    if real_data_loader is not None:
        real_latent_tsne = tsne.transform(real_latent_vectors)
        
        scatter1 = ax1.scatter(real_latent_vectors[:, 0], real_latent_vectors[:, 1], 
                          c=real_true_labels, cmap='bwr', alpha=0.7)
        scatter2 = ax2.scatter(real_latent_tsne[:, 0], real_latent_tsne[:, 1], 
                              c=real_true_labels, cmap='bwr', alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    if real_data_loader is not None:
        return real_latent_vectors, real_true_labels
    
    else:
        return latent_vectors, true_labels

    
############################################################################################################################
############################################################################################################################
'''
def plot_latent_space_with_labels(model, test_loader, labels, reduce='PCA', KDE=True, n_feats= 48, plotting_labels=None, real_data_loader=None, real_data_labels=None, train_colors=None, real_colors=None):
    """Plot latent space and t-SNE visualization"""
    # Define class label mapping
    if plotting_labels == None: 
        class_labels = {
            0: 'circle sims',
            1: 'casting sims', 
            2: 'brownian',
            3: 'gaussian rw',
            4: 'laplace',
            5: '300ms_turns_same_direction',
            6: '300ms_random_turns',
            7: 'random_turns_150_to_600ms'
        }
    else: 
        class_labels= plotting_labels
    model.eval()
    latent_vectors = []
    true_labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            mu, logvar = model.encode(data)
            latent_vectors.append(mu.numpy())
            true_labels.extend(label.numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    true_labels = np.array(true_labels)
    
    if real_data_loader is not None:
        real_latent_vectors = []
        real_true_labels = []
    
        with torch.no_grad():
            for data, label in real_data_loader:
                mu, logvar = model.encode(data)
                real_latent_vectors.append(mu.numpy())
                real_true_labels.extend(label.numpy())

        real_latent_vectors = np.vstack(real_latent_vectors)
        real_true_labels = np.array(real_true_labels)
    
    # Create discrete colormap
    unique_labels = np.unique(true_labels)
    num_classes = len(unique_labels)+1 
    
    if train_colors is None:
    # Create a discrete version of viridis colormap
        cmap_train = plt.cm.get_cmap('viridis', num_classes)
    else:
        cmap_train = colors
        
    # Plot first two dimensions of latent space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First two dimensions - use discrete colormap
    scatter1 = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                          c=true_labels, cmap=cmap_train, alpha=0.7, 
                          vmin=min(unique_labels), vmax=max(unique_labels))
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('First Two Latent Dimensions')
    
    # Create discrete colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=unique_labels)
    cbar1.set_label('Class')
    cbar1.set_ticklabels([class_labels.get(label, f'Class {label}') for label in unique_labels])
    
    if reduce=='UMAP':
        # t-SNE visualization
        import umap
        tsne = umap.UMAP(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)
    
    if reduce=='PCA':
        from sklearn.decomposition import PCA
        tsne = PCA(n_components=2)
        latent_tsne = tsne.fit_transform(latent_vectors[:,:n_feats])
        print('Using PCA, explained variance:', tsne.explained_variance_ratio_)
        
        
    # Second plot with discrete colormap
    if KDE == True:
        sns.kdeplot(x=latent_tsne[:, 0],
            y=latent_tsne[:, 1],
            hue=true_labels,
            levels=10, thresh=0.05, fill=True, alpha=.2,
            palette=cmap_train,
            legend=False, ax=ax2)
    else:    
        scatter2 = ax2.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                          c=true_labels, cmap=cmap_train, alpha=0.7,
                          vmin=min(unique_labels), vmax=max(unique_labels))
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.set_title('UMAP Visualization of Latent Space')
    

    if real_data_loader is not None:
        
        real_latent_tsne = tsne.transform(real_latent_vectors[:,:n_feats])
        
        if real_colors is None:
        # Create discrete colormap for real data (using different colormap)
            cmap_real = plt.cm.get_cmap('bwr', len(np.unique(real_true_labels)))
            # Plot real data with discrete colormap
            ax1.scatter(real_latent_vectors[:, 0], real_latent_vectors[:, 1], 
                       c=real_true_labels, cmap=cmap_real, alpha=0.7,
                       vmin=min(np.unique(real_true_labels)), vmax=max(np.unique(real_true_labels)))
            ax2.scatter(real_latent_tsne[:, 0], real_latent_tsne[:, 1], 
                       c=real_true_labels, cmap=cmap_real, alpha=0.7, 
                       vmin=min(np.unique(real_true_labels)), vmax=max(np.unique(real_true_labels)))
        else:
            cmap_real = real_colors
            from matplotlib import colors
            divnorm=colors.TwoSlopeNorm(vmin=0, vcenter=0.25, vmax=.5)
            
            ax1.scatter(real_latent_vectors[:, 0], real_latent_vectors[:, 1], 
                       c=cmap_real,cmap="coolwarm", norm=divnorm, alpha=0.7,)
            im = ax2.scatter(real_latent_tsne[:, 0], real_latent_tsne[:, 1], 
                       c=cmap_real,cmap="coolwarm", norm=divnorm, alpha=0.7, )
            #fig.colorbar(im, ax=ax2, fraction=.03, label='median altitude between 1.1-3s')   

                # Create discrete colorbar for the second plot
            cbar2 = fig.colorbar(im, ax=ax2, fraction=.03, label='median altitude between 1.1-3s')   
            #cbar2.set_label('Class')
            #cbar2.set_ticklabels([class_labels.get(label, f'Class {label}') for label in unique_labels])
            
    plt.tight_layout()
    plt.show()
    
    if real_data_loader is not None:
        return latent_vectors, true_labels, latent_tsne, real_latent_vectors, real_true_labels, real_latent_tsne
    else:
        return latent_vectors, true_labels, latent_tsne
    
'''

def plot_latent_space_with_labels(model, test_loader, labels, reduce='PCA', KDE=True, n_feats=48, path = 'output.svg', plotting_labels=None, real_data_loader=None, real_data_labels=None, train_colors=None, real_colors=None):
    """Plot latent space and t-SNE visualization"""
    # Define class label mapping
    if plotting_labels is None: 
        class_labels = {
            0: 'circle sims',
            1: 'casting sims', 
            2: 'brownian',
            3: 'gaussian rw',
            4: 'laplace',
            5: '300ms_turns_same_direction',
            6: '300ms_random_turns',
            7: 'random_turns_150_to_600ms'
        }
    else: 
        class_labels = plotting_labels
    
    model.eval()
    latent_vectors = []
    true_labels = []
    obj_ids = []  # Store obj_ids
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different return formats from DataLoader
            if len(batch) == 3:  # (data, labels, obj_ids)
                data, label, obj_id_batch = batch
                obj_ids.extend(obj_id_batch)
            elif len(batch) == 2:  # (data, labels) - fallback
                data, label = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(mu.numpy())
            true_labels.extend(label.numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    true_labels = np.array(true_labels)
    
    # Initialize real data variables
    real_latent_vectors = None
    real_true_labels = None
    real_obj_ids = None

    if real_data_loader is not None:
        real_latent_vectors = []
        real_true_labels = []
        real_obj_ids = []  # Store real data obj_ids
        
        with torch.no_grad():
            for batch in real_data_loader:
                # Handle different return formats from DataLoader
                if len(batch) == 3:  # (data, labels, obj_ids)
                    data, label, obj_id_batch = batch
                    real_obj_ids.extend(obj_id_batch)
                elif len(batch) == 2:  # (data, labels) - fallback
                    data, label = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
                
                mu, logvar = model.encode(data)
                z = model.reparameterize(mu, logvar)                
                real_latent_vectors.append(mu.numpy())
                real_true_labels.extend(label.numpy())
        
        real_latent_vectors = np.vstack(real_latent_vectors)
        real_true_labels = np.array(real_true_labels)
    
    # Create discrete colormap
    unique_labels = np.unique(true_labels)
    num_classes = len(unique_labels) + 1 
    
    if train_colors is None:
        # Create a discrete version of viridis colormap
        cmap_train = plt.cm.get_cmap('viridis', num_classes)
    else:
        # Create the colormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_gradient",train_colors)
        cmap_train = custom_cmap 
        


    # Plot first two dimensions of latent space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if KDE:
        scatter1= sns.kdeplot(x=latent_vectors[:, 0],
            y=latent_vectors[:, 1],
            hue=true_labels,
            levels=10, thresh=0.02, fill=True, alpha=.6,
            palette=cmap_train,
            legend=False, ax=ax1)
    # First two dimensions - use discrete colormap

    scatter1 = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                          c=true_labels, cmap=cmap_train, alpha=0.7, 
                          vmin=min(unique_labels), vmax=max(unique_labels), rasterized=True)
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('First Two Latent Dimensions')
    
    # Create discrete colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=unique_labels)
    cbar1.set_label('Class')
    cbar1.set_ticklabels([class_labels.get(label, f'Class {label}') for label in unique_labels])
    
    if reduce == 'UMAP':
        # UMAP visualization

        reducer = umap.UMAP(n_components=2, random_state=42)
        latent_reduced = reducer.fit_transform(latent_vectors)
    
    if reduce == 'PCA':
        reducer = PCA(n_components=2)
        latent_reduced = reducer.fit_transform(latent_vectors[:, :n_feats])
        print('Using PCA, explained variance:', reducer.explained_variance_ratio_)
        
    # Second plot with discrete colormap
    if KDE:
        sns.kdeplot(x=latent_reduced[:, 0],
            y=latent_reduced[:, 1],
            hue=true_labels,
            levels=10, thresh=0.05, fill=True, alpha=.2,
            palette=cmap_train,
            legend=False, ax=ax2)
    #else:    
    #scatter2 = ax2.scatter(latent_reduced[:, 0], latent_reduced[:, 1], 
    #                      c=true_labels, cmap=cmap_train, alpha=0.7,
    #                      vmin=min(unique_labels), vmax=max(unique_labels), rasterized=True)
    
    ax2.set_xlabel(f'{reduce} Component 1')
    ax2.set_ylabel(f'{reduce} Component 2')
    ax2.set_title(f'{reduce} Visualization of Latent Space')

    if real_data_loader is not None:
        # Transform real data using the same reducer
        real_latent_reduced = reducer.transform(real_latent_vectors[:, :n_feats])
        
        if real_colors is None:
            # Create discrete colormap for real data (using different colormap)
            cmap_real = plt.cm.get_cmap('managua', len(np.unique(real_true_labels)))
            # Plot real data with discrete colormap
            ax1.scatter(real_latent_vectors[:, 0], real_latent_vectors[:, 1], 
                       c=real_true_labels, cmap=cmap_real, alpha=0.7,
                       vmin=min(np.unique(real_true_labels)), vmax=max(np.unique(real_true_labels)))
            scatter2 = ax2.scatter(real_latent_reduced[:, 0], real_latent_reduced[:, 1], 
                       c=real_true_labels, cmap=real_true_labels, alpha=0.7, 
                       vmin=min(np.unique(real_true_labels)), vmax=max(np.unique(real_true_labels)))
            cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=np.unique(real_true_labels))
        else:
            cmap_real = real_colors
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vmin=0.1, vcenter=0.25, vmax=0.4)
            
            im = ax1.scatter(real_latent_vectors[:, 0], real_latent_vectors[:, 1], 
                       c=cmap_real, cmap="coolwarm", norm=divnorm, alpha=0.7)
            im = ax2.scatter(real_latent_reduced[:, 0], real_latent_reduced[:, 1], 
                       c=cmap_real, cmap="coolwarm", norm=divnorm, alpha=0.7)
            
            # Create discrete colorbar for the second plot
            cbar2 = fig.colorbar(im, ax=ax2, fraction=0.03, label='mean altitude between 1.1-3s')   
            
    plt.tight_layout()
    plt.show()
    #print(obj_ids==[])
    
    fig.savefig(path,)# transparent=True)
    # Return values including obj_ids
    if real_data_loader is not None:
        if obj_ids == []:
            return latent_vectors, true_labels, latent_reduced, real_latent_vectors, real_true_labels, real_latent_reduced, real_obj_ids
        else:
            return latent_vectors, true_labels, latent_reduced,obj_ids,real_latent_vectors, real_true_labels, real_latent_reduced, real_obj_ids
    else:
        if obj_ids == []:
            return latent_vectors, true_labels, latent_reduced
        else:
            return latent_vectors, true_labels, latent_reduced, obj_ids




def plot_training_progress(train_losses, val_losses, cluster_losses, recon_losses):
    """Plot training progress with all loss components"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Total Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(cluster_losses, label='Cluster Loss', color='red')
    plt.plot(recon_losses, label='Recon Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_losses[50:], label='Total Train Loss')
    plt.plot(val_losses[50:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Last Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
#################################################################################


class TripletLossVAE(nn.Module):
    def __init__(self, input_dim=800, hidden_dim=512, latent_dim=20, margin=1.0, dropout=0.2):
        super(TripletLossVAE, self).__init__()
        
        # Larger encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim//4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//4, latent_dim)
        
        # Larger decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Triplet loss
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def compute_triplet_loss(self, z, labels):
        """Compute triplet loss for the latent representations"""
        batch_size = z.size(0)
        
        # We need at least 2 samples per class for triplet loss
        anchors = []
        positives = []
        negatives = []
        
        # Create triplets
        for i in range(batch_size):
            anchor = z[i]
            anchor_label = labels[i]
            
            # Find positive samples (same class)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            pos_indices = torch.where(pos_mask)[0]
            
            # Find negative samples (different class)
            neg_mask = (labels != anchor_label)
            neg_indices = torch.where(neg_mask)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Randomly select one positive and one negative
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                
                anchors.append(anchor.unsqueeze(0))
                positives.append(z[pos_idx])
                negatives.append(z[neg_idx])
        
        if len(anchors) > 0:
            anchors = torch.cat(anchors)
            positives = torch.cat(positives)
            negatives = torch.cat(negatives)
            
            return self.triplet_loss(anchors, positives, negatives)
        else:
            # Return zero loss if we can't form triplets
            return torch.tensor(0.0).to(z.device)

def circular_distance(pred_vel, true_vel):
    """
    Compute circular distance between predicted and true velocity directions
    """
    # Compute angles from velocities
    true_angle = torch.atan2(true_vel[:, 1::2], true_vel[:, 0::2])  # [yvel, xvel] pairs
    pred_angle = torch.atan2(pred_vel[:, 1::2], pred_vel[:, 0::2])
    
    # Circular distance - handle periodicity
    diff = torch.atan2(torch.sin(true_angle - pred_angle), 
                       torch.cos(true_angle - pred_angle))
    
    return torch.mean(torch.abs(diff))

def improved_vae_loss_with_course(recon_x, x, mu, logvar, beta=0.01, course_weight=0.01):
    """VAE loss with circular course direction penalty"""
    # Original reconstruction loss
    recon_loss = nn.SmoothL1Loss()(recon_x, x)
    
    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Course direction loss
    course_loss = circular_distance(recon_x, x)
    
    return recon_loss + beta * kld_loss + course_weight * course_loss

def triplet_loss_with_mu_sigma(mu_anchor, mu_positive, mu_negative, 
                              sigma_anchor=None, sigma_positive=None, sigma_negative=None,
                              method='mu_sigma_l2', lambda_sigma=0.1):
    
    if method == 'mu_only':
        # Most common approach - stable and effective
        pos_distance = torch.sum((mu_anchor - mu_positive)**2, dim=1)
        neg_distance = torch.sum((mu_anchor - mu_negative)**2, dim=1)
        
    elif method == 'mu_sigma_l2':
        # Combine μ and σ distances
        pos_distance = (torch.sum((mu_anchor - mu_positive)**2, dim=1) + 
                       lambda_sigma * torch.sum((sigma_anchor - sigma_positive)**2, dim=1))
        neg_distance = (torch.sum((mu_anchor - mu_negative)**2, dim=1) + 
                       lambda_sigma * torch.sum((sigma_anchor - sigma_negative)**2, dim=1))
        
    elif method == 'kl_based':
        # Use KL divergence between distributions
        pos_distance = kl_divergence(mu_anchor, sigma_anchor, mu_positive, sigma_positive)
        neg_distance = kl_divergence(mu_anchor, sigma_anchor, mu_negative, sigma_negative)
    
    loss = F.relu(pos_distance - neg_distance + margin)
    return loss
    
def train_with_kl_annealing(model, train_loader, test_loader, epochs=250, 
                            final_beta=1.0, warmup_epochs=100):
    """Gradually increase β from 0 to final_beta"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        # Linear warmup
        if epoch < warmup_epochs:
            current_beta = final_beta * (epoch / warmup_epochs)
        else:
            current_beta = final_beta
        
        model.train()
        for data, labels, obj in train_loader:
            optimizer.zero_grad()
            
            recon_batch, mu, logvar, z = model(data)
            
            recon_loss = nn.MSELoss()(recon_batch, data)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Weighted by current β
            loss = recon_loss + current_beta * kld_loss
            
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: β={current_beta:.3f}")
            
def train_vae_with_triplet(model, train_loader, test_loader, epochs=200, triplet_weight=0.1, beta=0.0013):
    """Train the VAE model with triplet loss - PROPER VALIDATION"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    triplet_losses = []
    val_triplet_losses = []  # Track validation triplet loss too
    val_total_losses = []    # Track combined validation loss
    kl_losses=[]
    val_kl_losses=[]
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_triplet_loss = 0
        total_vae_loss = 0
        total_kl_loss = 0
        for batch_idx, (data, labels,obj) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar, z = model(data)
            
            # Compute VAE loss
            vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
            
            # Compute triplet loss
            triplet_loss = model.compute_triplet_loss(z, labels)
            
            # Combined loss
            total_loss = vae_loss + triplet_weight * triplet_loss
            
            total_loss.backward()
            
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item() * data.size(0)
            total_vae_loss += vae_loss.item() * data.size(0)
            total_triplet_loss += triplet_loss.item() * data.size(0)
            total_kl_loss += kl_loss.item() * data.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_triplet_loss = total_triplet_loss / len(train_loader.dataset)
        avg_vae_loss = total_vae_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        train_losses.append(avg_train_loss)
        triplet_losses.append(avg_triplet_loss)
        kl_losses.append(avg_kl_loss)
        
        # VALIDATION - Include triplet loss!
        model.eval()
        total_val_loss = 0
        total_val_triplet_loss = 0
        total_val_vae_loss = 0
        total_val_kl_loss = 0
        with torch.no_grad():
            for data, labels, obj in test_loader:
                recon_batch, mu, logvar, z = model(data)
                
                # Compute both losses for validation
                vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
                triplet_loss = model.compute_triplet_loss(z, labels)
                
                # Combined validation loss (same weighting as training)
                combined_val_loss = vae_loss + triplet_weight * triplet_loss
                
                total_val_loss += combined_val_loss.item() * data.size(0)
                total_val_triplet_loss += triplet_loss.item() * data.size(0)
                total_val_vae_loss += vae_loss.item() * data.size(0)
                total_val_kl_loss += kl_loss.item() * data.size(0)
                
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        avg_val_triplet_loss = total_val_triplet_loss / len(test_loader.dataset)
        avg_val_vae_loss = total_val_vae_loss / len(test_loader.dataset)
        avg_val_kl_loss = total_val_kl_loss / len(test_loader.dataset)
        
        val_losses.append(avg_val_loss)
        val_triplet_losses.append(avg_val_triplet_loss)
        val_total_losses.append(avg_val_loss)
        val_kl_losses.append(avg_val_kl_loss)
        
        # Use COMBINED loss for scheduler
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train - Total: {avg_train_loss:.6f}, VAE: {avg_vae_loss:.6f}, Triplet: {avg_triplet_loss:.6f}, KL: {avg_kl_loss:.6f}')
            print(f'  Val   - Total: {avg_val_loss:.6f}, VAE: {avg_val_vae_loss:.6f}, Triplet: {avg_val_triplet_loss:.6f}, KL: {avg_val_kl_loss:.6f} ')
            print('')
    
    return train_losses, val_losses, triplet_losses, val_triplet_losses, kl_losses, val_kl_losses






def sample_from_aggregated_posterior(model, dataloader, device, n_samples_per_class=100, 
                                     n_gmm_components=None, plot_results=True, 
                                     generate_examples=True, n_generated_examples=5):
    """
    Sample from the aggregated posterior using Gaussian Mixture Model per class.
    
    Args:
        model: Your trained Beta-VAE model
        dataloader: DataLoader with your dataset (should have labels accessible)
        device: 'cuda' or 'cpu'
        n_samples_per_class: Number of samples to generate per class
        n_gmm_components: Number of GMM components per class (None = auto-determine)
        plot_results: Whether to create visualizations
        generate_examples: Whether to generate and plot example reconstructions
        n_generated_examples: Number of example generations to plot per class
        
    Returns:
        Dictionary with GMM models, sampled latents, and generated samples
    """
    
    model.eval()
    
    # Step 1: Collect latent representations for each class
    print("Collecting latent representations...")
    
    latents_by_class = {}
    reconstructions_by_class = {}
    original_data_by_class = {}
    
    with torch.no_grad():
        for batch_idx, (data, labels, obj_ids) in enumerate(dataloader):
            data = data.to(device)
            
            # Forward pass through encoder
            # Assuming your model returns (mu, logvar) or similar
            if hasattr(model, 'encode'):
                mu, logvar = model.encode(data)
            else:
                # Adjust based on your model's encoder output
                encoded = model.encoder(data)
                mu = encoded  # or split if needed
                
            # Get latent sample (using reparameterization trick for consistency)
            if hasattr(mu, 'shape') and len(mu.shape) > 1 and mu.shape[1] >= 2:
                # Assuming mu is the full latent space
                z = mu.cpu().numpy()
            else:
                # If your model uses stochastic sampling during training,
                # but for analysis we can use just the mean
                std = torch.exp(0.5 * logvar) if logvar is not None else 1.0
                eps = torch.randn_like(std)
                z = (mu + eps * std).cpu().numpy()
            
            labels_np = labels.cpu().numpy()
            
            # Organize latents by class
            for i, label in enumerate(labels_np):
                label_str = str(label)
                if label_str not in latents_by_class:
                    latents_by_class[label_str] = []
                    reconstructions_by_class[label_str] = []
                    original_data_by_class[label_str] = []
                
                latents_by_class[label_str].append(z[i])
                
                # Store reconstructions and originals for visualization
                if generate_examples and len(reconstructions_by_class[label_str]) < n_generated_examples:
                    if hasattr(model, 'decode'):
                        recon = model.decode(torch.from_numpy(z[i:i+1]).to(device)).cpu()
                    else:
                        recon = model.decoder(torch.from_numpy(z[i:i+1]).to(device)).cpu()
                    reconstructions_by_class[label_str].append(recon.numpy())
                    original_data_by_class[label_str].append(data[i].cpu().numpy())
    
    # Convert lists to arrays
    for label in latents_by_class:
        latents_by_class[label] = np.array(latents_by_class[label])
        if reconstructions_by_class.get(label):
            reconstructions_by_class[label] = np.array(reconstructions_by_class[label])
            original_data_by_class[label] = np.array(original_data_by_class[label])
    
    # Step 2: Fit GMM for each class
    print("Fitting Gaussian Mixture Models...")
    
    gmm_models = {}
    sampled_latents = {}
    
    for label, latents in latents_by_class.items():
        n_data = latents.shape[0]
        
        # Auto-determine number of components if not specified
        if n_gmm_components is None:
            # Simple heuristic: 1 component per 50 samples, min 1, max 10
            n_components = max(1, min(n_data // 50, 10))
        else:
            n_components = n_gmm_components
        
        # Ensure we don't have more components than data points
        n_components = min(n_components, n_data)
        
        print(f"  Class {label}: {n_data} samples, using {n_components} GMM components")
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        gmm.fit(latents)
        gmm_models[label] = gmm
        
        # Sample from the GMM
        if n_data > 0:
            sampled_latents[label] = gmm.sample(n_samples_per_class)[0]
        else:
            sampled_latents[label] = np.array([])
    
    # Step 3: Generate samples from the decoder
    print("Generating samples from decoder...")
    
    generated_samples = {}
    if generate_examples:
        with torch.no_grad():
            for label, latents in sampled_latents.items():
                if len(latents) == 0:
                    continue
                
                # Take first few samples for visualization
                n_to_generate = min(n_generated_examples, len(latents))
                sample_latents = latents[:n_to_generate]
                
                generated = []
                for z_sample in sample_latents:
                    z_tensor = torch.FloatTensor(z_sample).unsqueeze(0).to(device)
                    if hasattr(model, 'decode'):
                        output = model.decode(z_tensor)
                    else:
                        output = model.decoder(z_tensor)
                    generated.append(output.cpu().numpy())
                
                generated_samples[label] = np.array(generated)
    
    # Step 4: Visualization
    if plot_results:
        plot_gmm_results(latents_by_class, sampled_latents, gmm_models, 
                        original_data_by_class, reconstructions_by_class, 
                        generated_samples, n_generated_examples)
    
    return {
        'gmm_models': gmm_models,
        'sampled_latents': sampled_latents,
        'latents_by_class': latents_by_class,
        'generated_samples': generated_samples
    }


def plot_gmm_results(latents_by_class, sampled_latents, gmm_models,
                    original_data_by_class, reconstructions_by_class,
                    generated_samples, n_examples):
    """Create visualization plots."""
    
    n_classes = len(latents_by_class)
    
    # If we have 2D latent space, we can plot it directly
    # Otherwise, we need to reduce dimensionality
    
    # Get all latents for dimensionality check
    all_latents = np.vstack([latents_by_class[label] for label in latents_by_class])
    latent_dim = all_latents.shape[1]
    
    # Create figure with subplots
    if latent_dim == 2:
        fig = plt.figure(figsize=(16, 4 * n_classes))
        
        for idx, label in enumerate(sorted(latents_by_class.keys())):
            # Plot 1: Original latents vs GMM samples
            ax1 = plt.subplot(n_classes, 4, idx*4 + 1)
            
            original_latents = latents_by_class[label]
            gmm_samples = sampled_latents[label]
            
            if len(original_latents) > 0:
                plt.scatter(original_latents[:, 0], original_latents[:, 1], 
                           alpha=0.5, label='Original', s=20)
            
            if len(gmm_samples) > 0:
                plt.scatter(gmm_samples[:, 0], gmm_samples[:, 1], 
                           alpha=0.5, label='GMM Samples', s=20, marker='x')
            
            # Plot GMM contours
            if label in gmm_models:
                gmm = gmm_models[label]
                x_min, x_max = ax1.get_xlim()
                y_min, y_max = ax1.get_ylim()
                
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                    np.linspace(y_min, y_max, 100))
                Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.contour(xx, yy, Z, levels=10, alpha=0.5, linewidths=1)
            
            plt.title(f'Class {label}: Latent Space')
            plt.xlabel('Latent dim 1')
            plt.ylabel('Latent dim 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Marginal distributions for first latent dimension
            ax2 = plt.subplot(n_classes, 4, idx*4 + 2)
            
            if len(original_latents) > 0:
                sns.kdeplot(original_latents[:, 0], label='Original', fill=True, alpha=0.3)
            
            if len(gmm_samples) > 0:
                sns.kdeplot(gmm_samples[:, 0], label='GMM Samples', fill=True, alpha=0.3)
            
            # Plot standard normal for reference
            x = np.linspace(-3, 3, 100)
            plt.plot(x, norm.pdf(x, 0, 1), 'k--', alpha=0.5, label='N(0,1)')
            
            plt.title(f'Class {label}: Latent dim 1 Distribution')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Example reconstructions (if available)
            ax3 = plt.subplot(n_classes, 4, idx*4 + 3)
            if (label in original_data_by_class and 
                label in reconstructions_by_class and
                len(original_data_by_class[label]) > 0):
                
                # Take first example
                original_img = original_data_by_class[label][0]
                recon_img = reconstructions_by_class[label][0]
                
                # For image data
                if len(original_img.shape) >= 2:
                    # Assuming image data - adjust based on your data
                    if len(original_img.shape) == 3:  # CHW format
                        original_img = original_img.transpose(1, 2, 0)
                        recon_img = recon_img.transpose(1, 2, 0)
                    
                    # Show original and reconstruction side by side
                    combined = np.hstack([original_img, recon_img])
                    plt.imshow(combined, cmap='gray')
                    plt.title(f'Class {label}: Original (L) vs Recon (R)')
                    plt.axis('off')
                else:
                    # For 1D data (e.g., trajectories)
                    plt.plot(original_img, label='Original', alpha=0.7)
                    plt.plot(recon_img, label='Reconstruction', alpha=0.7)
                    plt.title(f'Class {label}: Example Reconstruction')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            # Plot 4: Generated samples (if available)
            ax4 = plt.subplot(n_classes, 4, idx*4 + 4)
            if label in generated_samples and len(generated_samples[label]) > 0:
                
                gen_img = generated_samples[label][0]
                
                if len(gen_img.shape) >= 2:
                    # For image data
                    if len(gen_img.shape) == 3:  # CHW format
                        gen_img = gen_img.transpose(1, 2, 0)
                    
                    plt.imshow(gen_img, cmap='gray')
                    plt.title(f'Class {label}: Generated from GMM')
                    plt.axis('off')
                else:
                    # For 1D data (trajectories)
                    for i in range(min(3, len(generated_samples[label]))):
                        plt.plot(generated_samples[label][i], alpha=0.7, 
                                label=f'Sample {i+1}')
                    plt.title(f'Class {label}: Generated Trajectories')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # For higher-dimensional latent spaces, use dimensionality reduction
        from sklearn.decomposition import PCA
        
        print(f"Latent dimension is {latent_dim} > 2, using PCA for visualization")
        
        # Apply PCA to all latents
        pca = PCA(n_components=2)
        all_latents_pca = pca.fit_transform(all_latents)
        
        # Split back by class
        latents_pca_by_class = {}
        sampled_pca_by_class = {}
        
        start_idx = 0
        for label in sorted(latents_by_class.keys()):
            n_samples = len(latents_by_class[label])
            latents_pca_by_class[label] = all_latents_pca[start_idx:start_idx + n_samples]
            
            if len(sampled_latents[label]) > 0:
                sampled_pca_by_class[label] = pca.transform(sampled_latents[label])
            
            start_idx += n_samples
        
        # Create simplified plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: PCA of original latents
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for idx, (label, color) in enumerate(zip(sorted(latents_pca_by_class.keys()), colors)):
            pca_latents = latents_pca_by_class[label]
            axes[0].scatter(pca_latents[:, 0], pca_latents[:, 1], 
                          alpha=0.6, label=f'Class {label}', s=30, color=color)
        
        axes[0].set_title('Original Latents (PCA)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: PCA of GMM samples
        for idx, (label, color) in enumerate(zip(sorted(sampled_pca_by_class.keys()), colors)):
            if label in sampled_pca_by_class:
                pca_samples = sampled_pca_by_class[label]
                axes[1].scatter(pca_samples[:, 0], pca_samples[:, 1], 
                              alpha=0.6, label=f'Class {label}', s=30, 
                              color=color, marker='x')
        
        axes[1].set_title('GMM Samples (PCA)')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Example generated trajectories (if available)
        ax3 = axes[2]
        if generated_samples:
            for idx, label in enumerate(sorted(generated_samples.keys())):
                if label in generated_samples and len(generated_samples[label]) > 0:
                    color = colors[idx % len(colors)]
                    for i in range(min(2, len(generated_samples[label]))):
                        traj = generated_samples[label][i].flatten()
                        ax3.plot(traj, alpha=0.7, color=color, 
                                label=f'Class {label}' if i == 0 else None)
            
            ax3.set_title('Generated Trajectories')
            ax3.set_xlabel('Time step')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Plot distribution statistics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot KL divergence between GMM and standard normal
    kl_divergences = []
    class_labels = []
    
    for label in sorted(gmm_models.keys()):
        gmm = gmm_models[label]
        # Approximate KL divergence (simplified)
        if hasattr(gmm, 'means_') and len(gmm.means_) > 0:
            # Simple approximation: average distance from origin
            mean_distance = np.mean(np.linalg.norm(gmm.means_, axis=1))
            kl_divergences.append(mean_distance)
            class_labels.append(label)
    
    axes[0].bar(class_labels, kl_divergences)
    axes[0].set_title('Average Distance from N(0,I) per Class')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Mean ||μ||')
    axes[0].grid(True, alpha=0.3)
    
    # Plot number of components per class
    n_components = [gmm_models[label].n_components for label in sorted(gmm_models.keys())]
    axes[1].bar(sorted(gmm_models.keys()), n_components)
    axes[1].set_title('GMM Components per Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Components')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Now you can generate new samples from the GMM:
def generate_from_gmm(results, model, class_label, n_samples=10):
    gmm = results['gmm_models'][str(class_label)]
    sampled_z = gmm.sample(n_samples)[0]
    
    # Decode to get generated data
    with torch.no_grad():
        z_tensor = torch.FloatTensor(sampled_z).to("cpu")
        generated = model.decode(z_tensor).cpu().numpy()
    
    return generated