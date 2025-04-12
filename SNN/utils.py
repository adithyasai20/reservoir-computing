import torch
def rate_encoding(images, device, time_steps=20):
    """converts image into spike trains using rate encoding"""
    # Remove the channel dimension (if present) and normalize to [0, 1]
    images = images.squeeze(1).float() / 255.0  # Adjust normalization if already normalized
    return torch.bernoulli(images.unsqueeze(-1).repeat(1, 1, 1, time_steps)).view(images.size(0), -1, time_steps).to(device)


def temporal_encode(images, time_steps=20):
    """
    Encodes images using temporal coding.
    :param images: Batch of images (batch_size, 1, height, width).
    :param time_steps: Total time steps for encoding.
    :return: Encoded spike trains (batch_size, height * width, time_steps).
    """
    batch_size, _= images.shape
    flattened_images = images.view(batch_size, -1)  # Flatten to (batch_size, 784)
    
    # Calculate spike times (brighter pixels spike earlier)
    spike_times = (1.0 - flattened_images) * (time_steps - 1)
    
    # Create spike trains initialized to zeros
    spike_trains = torch.zeros(batch_size, flattened_images.size(1), time_steps, device=images.device)
    
    # Convert spike times to indices
    spike_times_rounded = spike_times.long()  # Round spike times to integer indices
    
    # Use advanced indexing to set spikes at the corresponding time step
    spike_trains[torch.arange(batch_size).unsqueeze(1), torch.arange(flattened_images.size(1)).unsqueeze(0), spike_times_rounded] = 1.0
    
    return spike_trains  # Shape: (batch_size, 784, time_steps)

