import torch
import gudhi as gd
import gudhi.representations as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_ph_features(image):
    # Detach tensor before NumPy conversion
    image = image.detach().squeeze().cpu().numpy()
    cc = gd.CubicalComplex(top_dimensional_cells=image)
    cc.persistence()
    return cc.persistence_intervals_in_dimension(1)

def ph_distance(img1, img2):
    # PH is non-differentiable → no gradients needed
    with torch.no_grad():
        diag1 = extract_ph_features(img1)
        diag2 = extract_ph_features(img2)

        if len(diag1) == 0 or len(diag2) == 0:
            return torch.tensor(0.0, device=device)

        wasserstein = gr.WassersteinDistance(order=1)
        dist = wasserstein(diag1, diag2)

    return torch.tensor(dist, device=device)
