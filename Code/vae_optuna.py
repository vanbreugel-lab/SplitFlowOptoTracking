import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset

# Device configuration
DEVICE = torch.device("cpu")#"mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels, obj_ids=None, normalization_type='zero_mean'):
        # Flatten and normalize trajectories
        trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)

        # Normalize each trajectory independently
        if normalization_type == 'zero_mean':
            self.trajectories = self.normalize_zero_mean(trajectories_flat).to(DEVICE)
        elif normalization_type == 'minmax':
            self.trajectories = self.normalize_minmax(trajectories_flat).to(DEVICE)
        else:
            raise ValueError("normalization_type must be 'zero_mean' or 'minmax'")

        self.labels = torch.LongTensor(labels).to(DEVICE)
        self.obj_ids = obj_ids

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


class VAEOptimizerWithVariance:
    """VAE optimization with variance explained and reconstruction error minimization"""

    def __init__(self, trajectories, labels, obj_ids,
                 val_split=0.3, n_trials=150, timeout=3600):
        """
        Args:
            trajectories: Input trajectories
            labels: Class labels
            obj_ids: Object IDs
            val_split: Validation split ratio
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.trajectories = trajectories
        self.labels = labels
        self.obj_ids = obj_ids
        self.val_split = val_split
        self.n_trials = n_trials
        self.timeout = timeout

        # Calculate input dimension
        self.input_dim = trajectories.shape[1] * trajectories.shape[2]
        print(f"Input dimension: {self.input_dim}")
        print(f"Number of samples: {trajectories.shape[0]}")
        print(f"Number of classes: {len(np.unique(labels))}")

        # Create integer labels if they're strings
        if isinstance(labels[0], str):
            unique_labels = np.unique(labels)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.labels_int = np.array([label_to_idx[label] for label in labels])
        else:
            self.labels_int = labels.astype(np.int64)

    def calculate_variance_explained(self, latent_vectors):
        """Calculate total variance explained by 2D PCA of latent vectors"""
        if len(latent_vectors) < 2:
            return 0.0

        # Convert to numpy if it's a tensor
        if isinstance(latent_vectors, torch.Tensor):
            try:
                latent_vectors = latent_vectors.mps().detach().numpy()
            except:
                latent_vectors = latent_vectors.cpu().detach().numpy()
        # If latent dimension is less than 2, return 0
        if latent_vectors.shape[1] < 2:
            return 0.0

        try:
            # Calculate PCA
            pca = PCA(n_components=2)
            pca.fit(latent_vectors)

            # Return total variance explained by first 2 components as Python float
            total_variance_explained = np.sum(pca.explained_variance_ratio_)
            return float(total_variance_explained)
        except Exception as e:
            print(f"Warning: PCA calculation failed: {e}")
            return 0.0

    def calculate_total_network_size(self, model):
        """Calculate total number of parameters in the network"""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    def calculate_triplet_accuracy(self, model, data_loader, margin=0.5):
        """Calculate triplet accuracy as described in the paper"""
        model.eval()
        correct_triplets = 0
        total_triplets = 0

        with torch.no_grad():
            for data, labels, _ in data_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # Get latent representations
                _, _, _, z = model(data)
                batch_size = z.size(0)

                # Generate triplets
                for i in range(batch_size):
                    anchor = z[i]
                    anchor_label = labels[i]

                    # Find positive (same class, different sample)
                    #pos_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
                    pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=z.device) != i)
                    pos_indices = torch.where(pos_mask)[0]

                    # Find negative (different class)
                    neg_mask = (labels != anchor_label)
                    neg_indices = torch.where(neg_mask)[0]

                    if len(pos_indices) > 0 and len(neg_indices) > 0:
                        # For each positive-negative pair
                        for pos_idx in pos_indices[:1]:  # Use first positive
                            for neg_idx in neg_indices[:1]:  # Use first negative
                                positive = z[pos_idx]
                                negative = z[neg_idx]

                                # Calculate distances
                                d_ap = torch.norm(anchor - positive, p=2)
                                d_an = torch.norm(anchor - negative, p=2)

                                # Check if triplet condition is satisfied
                                if d_ap < d_an - margin:
                                    correct_triplets += 1
                                total_triplets += 1

        # Return accuracy as percentage
        if total_triplets > 0:
            accuracy = correct_triplets / total_triplets * 100.0
        else:
            accuracy = 0.0

        return accuracy, total_triplets


    def create_vae_model(self, trial):
        """Create VAE model based on trial suggestions - MATCHING TRAINING SCRIPT ARCHITECTURE"""

        # Get hyperparameters from trial
        hidden_dim = trial.suggest_int("hidden_dim", 128, 1024, step=32)
        latent_dim = trial.suggest_int("latent_dim", 10, 150, step=5)
        n_layers = trial.suggest_int("n_layers", 2, 5)

        # Architecture choices
        activation_fn = trial.suggest_categorical(
            "activation_fn", ["leaky_relu", "relu", "elu"]
        )

        dropout_enabled = trial.suggest_categorical("dropout_enabled", [True, False])
        dropout_rate = 0.0
        if dropout_enabled:
            dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.1)

        margin = trial.suggest_float("margin", 0.25, 2.0, step=0.25)

        class SimpleVAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim, n_layers,
                         activation_fn, dropout_rate, margin):
                super().__init__()

                self.margin = margin
                self.input_dim = input_dim
                self.latent_dim = latent_dim

                # Track network architecture for size calculation
                self.layer_sizes = []

                # ENCODER - MATCHING TRAINING SCRIPT
                encoder_layers = []
                in_dim = input_dim

                for i in range(n_layers):
                    # Match training script: hidden_dim // (2 ** i)
                    out_dim = hidden_dim // (2 ** i)
                    self.layer_sizes.append((in_dim, out_dim))

                    encoder_layers.append(nn.Linear(in_dim, out_dim))
                    encoder_layers.append(nn.BatchNorm1d(out_dim))

                    # Activation function
                    if activation_fn == "leaky_relu":
                        encoder_layers.append(nn.LeakyReLU(0.2))
                    elif activation_fn == "relu":
                        encoder_layers.append(nn.ReLU())
                    else:  # elu
                        encoder_layers.append(nn.ELU(alpha=1.0))

                    # Dropout only on first layer if dropout > 0 (matching training script)
                    if (i == 0) and (dropout_rate > 0):
                        encoder_layers.append(nn.Dropout(dropout_rate))

                    in_dim = out_dim

                self.encoder = nn.Sequential(*encoder_layers)
                self.encoder_output_dim = in_dim

                # Latent space
                self.fc_mu = nn.Linear(in_dim, latent_dim)
                self.fc_logvar = nn.Linear(in_dim, latent_dim)
                self.layer_sizes.append((in_dim, latent_dim))
                self.layer_sizes.append((in_dim, latent_dim))

                # DECODER - MATCHING TRAINING SCRIPT (symmetric to encoder)
                decoder_layers = []
                in_dim = latent_dim
                decoder_n_layers = n_layers

                for i in range(n_layers):
                    # Match training script: hidden_dim // (2 ** (decoder_n_layers - 1))
                    out_dim = hidden_dim // (2 ** (decoder_n_layers - 1))
                    self.layer_sizes.append((in_dim, out_dim))

                    decoder_layers.append(nn.Linear(in_dim, out_dim))
                    decoder_layers.append(nn.BatchNorm1d(out_dim))

                    # Activation function
                    if activation_fn == "leaky_relu":
                        decoder_layers.append(nn.LeakyReLU(0.2))
                    elif activation_fn == "relu":
                        decoder_layers.append(nn.ReLU())
                    else:  # elu
                        decoder_layers.append(nn.ELU(alpha=1.0))

                    # Dropout only on last layer if dropout > 0 (matching training script)
                    if (i == (n_layers - 1)) and (dropout_rate > 0):
                        decoder_layers.append(nn.Dropout(dropout_rate))

                    in_dim = out_dim
                    decoder_n_layers -= 1

                # Final output layer
                decoder_layers.append(nn.Linear(in_dim, input_dim))
                self.layer_sizes.append((in_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

                # Triplet loss
                self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

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
            '''
            def compute_triplet_loss(self, z, labels):
                """Assumes batch always contains at least 2 classes"""
                batch_size = z.size(0)

                anchors = []
                positives = []
                negatives = []

                for i in range(batch_size):
                    anchor = z[i]
                    anchor_label = labels[i]

                    # Find positives and negatives
                    pos_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
                    pos_indices = torch.where(pos_mask)[0]

                    neg_mask = (labels != anchor_label)
                    neg_indices = torch.where(neg_mask)[0]

                    # We're guaranteed these exist by batch construction
                    pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,), device=z.device)]
                    neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,), device=z.device)]

                    anchors.append(anchor.unsqueeze(0))
                    positives.append(z[pos_idx])
                    negatives.append(z[neg_idx])

                anchors = torch.cat(anchors)
                positives = torch.cat(positives)
                negatives = torch.cat(negatives)

                return self.triplet_loss(anchors, positives, negatives)

            '''
            def compute_triplet_loss(self, z, labels):
                batch_size = z.size(0)

                if len(torch.unique(labels)) < 2:
                    return torch.tensor(0.0).to(z.device)

                anchors = []
                positives = []
                negatives = []

                for i in range(batch_size):
                    anchor = z[i]
                    anchor_label = labels[i]

                    # Find positive samples (same class, different sample)
                    #pos_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
                    pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=z.device) != i)
                    pos_indices = torch.where(pos_mask)[0]

                    # Find negative samples (different class)
                    neg_mask = (labels != anchor_label)
                    neg_indices = torch.where(neg_mask)[0]

                    if len(pos_indices) > 0 and len(neg_indices) > 0:
                        # Randomly select one positive and one negative
                        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,), device=z.device)]
                        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,), device=z.device)]

                        anchors.append(anchor.unsqueeze(0))
                        positives.append(z[pos_idx])
                        negatives.append(z[neg_idx])

                if len(anchors) > 0:
                    anchors = torch.cat(anchors)
                    positives = torch.cat(positives)
                    negatives = torch.cat(negatives)

                    return self.triplet_loss(anchors, positives, negatives)
                else:
                    print('no triplet loss computed')
                    return torch.tensor(0.0).to(z.device)

        return SimpleVAE(self.input_dim, hidden_dim, latent_dim, n_layers,
                         activation_fn, dropout_rate, margin)

    def improved_vae_loss(recon_x, x, mu, logvar, beta=0.001):
        """Improved VAE loss with adjustable beta and better reconstruction loss"""
        # Use smooth L1 loss for better gradient behavior
        recon_loss = nn.SmoothL1Loss()(recon_x, x)

        # KL divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # print('KL:', kld_loss)
        return recon_loss + beta * kld_loss, kld_loss

    def train_vae(self,model, train_loader, test_loader, optimizer, epochs=200, triplet_weight=0.1, alpha=1, beta=0.001):
        """Train the VAE model with triplet loss - PROPER VALIDATION"""

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses = []
        val_losses = []
        triplet_losses = []
        val_triplet_losses = []  # Track validation triplet loss too
        val_total_losses = []  # Track combined validation loss
        kl_losses = []
        val_kl_losses = []
        val_recon_losses = []
        recon_losses = []

        val_triplet_accuracies = []


        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            total_triplet_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            for batch_idx, (data, labels, obj) in enumerate(train_loader):
                if len(torch.unique(labels)) < 2:
                    continue

                optimizer.zero_grad()

                # Forward pass
                recon_batch, mu, logvar, z = model(data)

                # Compute VAE loss
                #vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
                recon_loss = nn.SmoothL1Loss()(recon_batch, data)
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                #kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
                # Compute triplet loss
                triplet_loss = model.compute_triplet_loss(z, labels)

                # Combined loss
                total_loss = alpha*recon_loss + beta*kld_loss + triplet_weight * triplet_loss

                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += total_loss.item() * data.size(0)
                total_recon_loss += recon_loss.item() *data.size(0)
                total_triplet_loss += triplet_loss.item() *data.size(0)
                total_kl_loss += kld_loss.item() *data.size(0)


            avg_train_loss = total_train_loss /len(train_loader.dataset)
            avg_triplet_loss = total_triplet_loss /len(train_loader.dataset)
            avg_recon_loss = total_recon_loss /len(train_loader.dataset)
            avg_kl_loss = total_kl_loss /len(train_loader.dataset)

            train_losses.append(avg_train_loss)
            triplet_losses.append(avg_triplet_loss)
            kl_losses.append(avg_kl_loss)
            recon_losses.append(avg_recon_loss)

            # VALIDATION - Include triplet loss!
            model.eval()
            total_val_loss = 0
            total_val_triplet_loss = 0
            total_val_recon_loss = 0
            total_val_kl_loss = 0
            n_batches=0

            # NEW: Calculate triplet accuracy
            val_triplet_accuracy, val_total_triplets = self.calculate_triplet_accuracy(
                model, test_loader, margin=model.margin)
            val_triplet_accuracies.append(val_triplet_accuracy)


            with torch.no_grad():
                for data, labels, obj in test_loader:
                    if len(torch.unique(labels)) < 2:
                        continue
                    recon_batch, mu, logvar, z = model(data)

                    # Compute both losses for validation
                    #vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
                    recon_loss = nn.SmoothL1Loss()(recon_batch, data)
                    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    #kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
                    # Compute triplet loss
                    triplet_loss = model.compute_triplet_loss(z, labels)

                    # Combined loss
                    total_loss = alpha*recon_loss + beta * kld_loss + triplet_weight * triplet_loss


                    # Combined validation loss (same weighting as training)
                    #combined_val_loss = vae_loss + triplet_weight * triplet_loss

                    total_val_loss += total_loss.item()  * data.size(0)
                    total_val_triplet_loss += triplet_loss.item()* data.size(0)
                    total_val_recon_loss += recon_loss.item()* data.size(0)
                    total_val_kl_loss += kld_loss.item()* data.size(0)


            avg_val_loss = total_val_loss /len(test_loader.dataset)
            avg_val_triplet_loss = total_val_triplet_loss /len(test_loader.dataset)
            avg_val_recon_loss = total_val_recon_loss /len(test_loader.dataset)
            avg_val_kl_loss = total_val_kl_loss /len(test_loader.dataset)

            val_losses.append(avg_val_loss)
            val_triplet_losses.append(avg_val_triplet_loss)
            #val_total_losses.append(avg_val_loss)
            val_recon_losses.append(avg_val_recon_loss)
            val_kl_losses.append(avg_val_kl_loss)

            # Use COMBINED loss for scheduler
            scheduler.step(avg_val_loss)

            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}:')
                print(
                    f'  Train - Total: {avg_train_loss:.6f}, Recon: {avg_recon_loss:.6f}, Triplet: {avg_triplet_loss:.6f}, KL: {avg_kl_loss:.6f}')
                print(
                    f'  Val   - Total: {avg_val_loss:.6f}, Recon: {avg_val_recon_loss:.6f}, Triplet: {avg_val_triplet_loss:.6f}, KL: {avg_val_kl_loss:.6f} ')
                print('')

        return train_losses, val_losses, triplet_losses, val_triplet_losses, kl_losses, val_kl_losses, recon_losses, val_recon_losses, val_triplet_accuracies

    def evaluate_pca(self, model, val_loader, alpha, beta, triplet_weight):
        """Evaluate VAE model and calculate variance explained - MATCHING TRAINING SCRIPT"""
        model.eval()


        all_latent_vectors = []
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                # Unpack based on what the dataset returns
                if len(batch) == 3:
                    data, labels, _ = batch  # Ignore obj_ids
                else:
                    data, labels = batch

                # Skip batches that are too small
                if data.size(0) < 2:
                    continue

                data, labels = data.to(DEVICE), labels.to(DEVICE)
                batch_size = data.size(0)
                total_samples += batch_size

                recon_data, mu, logvar, z = model(data)

                # Collect latent vectors for variance calculation
                try:
                    all_latent_vectors.append(z.mps())
                except:
                    all_latent_vectors.append(z.cpu())

        if total_samples == 0:
            return float('inf'), float('inf'), float('inf'), float('inf'), 0.0

        # Calculate variance explained
        if all_latent_vectors:
            latent_vectors = torch.cat(all_latent_vectors, dim=0)
            variance_explained = self.calculate_variance_explained(latent_vectors)
        else:
            variance_explained = 0.0

        return float(variance_explained)

    def objective(self, trial):
        """Objective function for Optuna"""
        # Get training hyperparameters
        alpha = trial.suggest_float("alpha", 0.1, 2, log=True)
        beta = trial.suggest_float("beta", 0.001, 5, log=True)
        triplet_weight = trial.suggest_float("triplet_weight", 0.1, 2.0, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
        epochs = 50  # Fixed number of epochs per trial

        # Create model
        model = self.create_vae_model(trial).to(DEVICE)

        # Calculate and store network size
        network_size = self.calculate_total_network_size(model)
        trial.set_user_attr("network_size", int(network_size))

        # Create optimizer - MATCHING TRAINING SCRIPT (Adam with weight decay)
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:  # RMSprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        # Create data loaders for this trial
        trajectories_flat = self.trajectories#.reshape(self.trajectories.shape[0], -1)

        # Split data
        X_train, X_val, y_train, y_val, obj_train, obj_val = train_test_split(
            trajectories_flat, self.labels, self.obj_ids,
            test_size=0.3, random_state=42, stratify=self.labels
        )

        # Create datasets
        train_dataset = TrajectoryDataset(
            X_train,
            y_train,
            obj_train,
            normalization_type='zero_mean'
        )
        val_dataset = TrajectoryDataset(
            X_val,
            y_val,
            obj_val,
            normalization_type='zero_mean'
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train model
        (train_losses, val_losses, triplet_losses, val_triplet_losses,
         kl_losses, val_kl_losses, recon_losses, val_recon_losses,
         val_triplet_accuracies) = self.train_vae(  # NEW: receive accuracies
            model=model, train_loader=train_loader, test_loader=val_loader,
            optimizer=optimizer, alpha=alpha, beta=beta, triplet_weight=triplet_weight, epochs=epochs
        )

        # Evaluate model
        #variance_explained = self.evaluate_pca(
        #    model, val_loader, alpha, beta, triplet_weight
        #)

        # Extract FINAL loss values (from the last epoch)
        #final_recon_loss = recon_losses[-1]  # Last element in the list
        final_val_recon_loss = val_recon_losses[-1]
        #final_kl_loss = kl_losses[-1]
        final_val_kl_loss = val_kl_losses[-1]
        #final_triplet_loss = triplet_losses[-1]
        final_val_triplet_loss = val_triplet_losses[-1]
        final_total_loss = val_losses[-1]  # Use validation loss for optimization
        final_val_triplet_accuracy = val_triplet_accuracies[-1] if val_triplet_accuracies else 0.0

        # Evaluate model for variance explained
        variance_explained = self.evaluate_pca(model, val_loader, alpha, beta, triplet_weight)

        print(f'  Final recon loss: {final_val_recon_loss:.6f}')
        print(f'  Final total loss: {final_total_loss:.6f}')
        print(f'  PCA explained variance: {variance_explained:.6f}')
        print(f'  Triplet Accuracy: {final_val_triplet_accuracy:.6f}')


        # Store all metrics as user attributes for tracking
        trial.set_user_attr("recon_loss", float(final_val_recon_loss))
        trial.set_user_attr("kl_loss", float(final_val_kl_loss))
        trial.set_user_attr("triplet_loss", float(final_val_triplet_loss))
        trial.set_user_attr("total_loss", float(final_total_loss))
        #trial.set_user_attr("train_recon_loss", float(final_recon_loss))
        #trial.set_user_attr("train_kl_loss", float(final_kl_loss))
        #trial.set_user_attr("train_triplet_loss", float(final_triplet_loss))
        trial.set_user_attr("variance_explained", float(variance_explained))
        trial.set_user_attr("val_triplet_accuracy", float(final_val_triplet_accuracy))

        # Primary objectives: maximize variance explained and minimize reconstruction error
        # Return negative variance for maximization
        return float(final_total_loss),  float(final_val_triplet_accuracy)  #float(final_triplet_loss) #float(variance_explained)

    def run_optimization(self, study_name="vae_variance_optimization"):
        """Run the optimization"""
        print("\n" + "=" * 80)
        print("VAE Optimization - Maximizing Variance Explained & Minimizing Reconstruction Error")
        print("=" * 80)

        # Create study with multi-objective optimization
        study = optuna.create_study(
            directions=["minimize","maximize"],  # minimize total loss, maximize variance
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )

        # Run optimization
        print(f"\nRunning {self.n_trials} trials (timeout: {self.timeout}s)...")
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)

        print(f"\nNumber of finished trials: {len(study.trials)}")

        # Analyze results
        #self.analyze_results(study)

        return study

    def analyze_results(self, study):
        """Analyze and display optimization results"""

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)

        # Get Pareto front
        pareto_trials = study.best_trials
        print(f"\nNumber of trials on Pareto front: {len(pareto_trials)}")

        # Show top Pareto solutions
        print("\nPareto front solutions (sorted by total loss):")
        sorted_trials = sorted(pareto_trials, key=lambda t: t.values[0])

        for i, trial in enumerate(sorted_trials[:10]):  # Show top 10
            print(f"\nSolution {i + 1}:")
            print(f"  Total Loss: {trial.values[0]:.6f}")
            print(f"  Triplet Accuracy: {trial.values[1]:.6f}")
            print(f"  Reconstruction Loss: {trial.user_attrs.get('recon_loss', 'N/A'):.6f}")
            print(f"  KL Loss: {trial.user_attrs.get('kl_loss', 'N/A'):.6f}")
            print(f"  Triplet Loss: {trial.user_attrs.get('triplet_loss', 'N/A'):.6f}")
            print(f"  PCA Explained Variance: {trial.user_attrs.get('variance_explained', 'N/A'):.6f}")
            print(f"  Network Size: {trial.user_attrs.get('network_size', 'N/A'):,}")

            # Show key parameters
            key_params = ['hidden_dim', 'latent_dim', 'n_layers',
                          'beta', 'triplet_weight', 'learning_rate', 'batch_size', 'optimizer']
            print("  Key parameters:")
            for param in key_params:
                if param in trial.params:
                    print(f"    {param}: {trial.params[param]}")

        # Find balanced solution (max variance with reasonable reconstruction)
        if pareto_trials:
            # Extract values
            total_loss_vals = [t.values[0] for t in pareto_trials]
            #variance_vals = [t.values[1] for t in pareto_trials]

            # Normalize for distance calculation
            loss_min, loss_max = min(total_loss_vals), max(total_loss_vals)
            variance_min, variance_max = min(variance_vals), max(variance_vals)

            # Avoid division by zero
            loss_range = loss_max - loss_min if loss_max != loss_min else 1
            variance_range = variance_max - variance_min if variance_max != variance_min else 1

            # Find solution that maximizes variance while keeping loss reasonable
            balanced_trial = max(pareto_trials, key=lambda t:
            0.25* ((t.values[1] - variance_min) / variance_range) -
             0.75 ((t.values[0] - loss_min) / loss_range))

            print("\n" + "=" * 80)
            print("RECOMMENDED HYPERPARAMETERS (Balanced Solution - High Variance)")
            print("=" * 80)
            print(f"\nPerformance Metrics:")
            print(f"  Recon Loss: {balanced_trial.values[0]:.6f}")
            print(f"  Variance Explained: {balanced_trial.values[1]:.6f}")
            print(f"  Reconstruction Loss: {balanced_trial.user_attrs.get('recon_loss', 'N/A'):.6f}")
            print(f"  KL Loss: {balanced_trial.user_attrs.get('kl_loss', 'N/A'):.6f}")
            print(f"  Triplet Loss: {balanced_trial.user_attrs.get('triplet_loss', 'N/A'):.6f}")
            print(f"  Network Size: {balanced_trial.user_attrs.get('network_size', 'N/A'):,}")

            print("\nHyperparameters:")
            for key, value in balanced_trial.params.items():
                print(f"  {key}: {value}")

        # Parameter importance analysis
        try:
            print("\n" + "=" * 80)
            print("PARAMETER IMPORTANCE ANALYSIS")
            print("=" * 80)

            # Importance for total loss
            loss_study = optuna.create_study(direction="minimize")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    loss_study.add_trial(optuna.trial.create_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.values[0]  # Total loss
                    ))

            print("\nImportance for Total Loss:")
            loss_importances = optuna.importance.get_param_importances(loss_study)
            for param_name, importance in loss_importances.items():
                print(f"  {param_name}: {importance:.4f}")

            # Importance for variance explained
            variance_study = optuna.create_study(direction="maximize")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    variance_study.add_trial(optuna.trial.create_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.values[1]  # Variance explained (convert back to positive)
                    ))

            print("\nImportance for Variance Explained:")
            variance_importances = optuna.importance.get_param_importances(variance_study)
            for param_name, importance in variance_importances.items():
                print(f"  {param_name}: {importance:.4f}")

        except Exception as e:
            print(f"\nCould not compute parameter importance: {e}")

        return pareto_trials

    def get_best_model_config(self, study):
        """Get the best model configuration from the study"""
        if study.best_trials:
            # Get the trial with maximum variance explained
            best_trial = max(study.best_trials, key=lambda t: t.values[1])  # Negative variance

            # Create model with best parameters
            model = self.create_vae_model(best_trial).to(DEVICE)

            # Get training parameters
            params = {
                'beta': best_trial.params.get('beta'),
                'triplet_weight': best_trial.params.get('triplet_weight'),
                'learning_rate': best_trial.params.get('learning_rate'),
                'batch_size': best_trial.params.get('batch_size'),
                'optimizer': best_trial.params.get('optimizer'),
                'hidden_dim': best_trial.params.get('hidden_dim'),
                'latent_dim': best_trial.params.get('latent_dim'),
                'n_layers': best_trial.params.get('n_layers'),
                'activation_fn': best_trial.params.get('activation_fn'),
                'dropout_rate': best_trial.params.get('dropout_rate', 0.0),
                'margin': best_trial.params.get('margin')
            }

            return model, params, best_trial
        return None, None, None


