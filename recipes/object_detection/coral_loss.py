import torch

def compute_covariance_matrix(features):
    # features: tensor of shape [batch_size, channels, height, width]
    batch_size, channels, height, width = features.shape
    features = features.view(batch_size, channels, -1)  # reshape to [batch_size, channels, height*width]
    features = features.permute(0, 2, 1)  # reshape to [batch_size, height*width, channels]
    batch_size, num_pixels, channels = features.shape
    
    mean_features = torch.mean(features, dim=1, keepdim=True)  # mean across spatial dimensions
    features_centered = features - mean_features
    covariance_matrix = torch.bmm(features_centered.permute(0, 2, 1), features_centered) / num_pixels  # batch matrix multiplication

    return covariance_matrix

def coral_loss(source, target):
    d = source.shape[1]  # number of channels
    source_cov = compute_covariance_matrix(source)
    target_cov = compute_covariance_matrix(target)
    
    coral_loss = torch.mean((source_cov - target_cov) ** 2) / (4 * d ** 2)
    return coral_loss

def total_coral_loss(x1, x2):
    # assert x1, x2 same data structure
    total_loss = 0.0
    for source, target in zip(x1, x2):
        total_loss += coral_loss(source, target)
    return total_loss
