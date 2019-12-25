from torch.nn.functional import interpolate


def feature_transform(img, n_upsample):
    """
    transform img like feature to be visualized
    :param img: [B, C, H, W]
    :return: visualized img range from 0 to 1
    """
    img = img[0, 1:2].repeat(3, 1, 1)
    img = interpolate(((img - img.min()) / (img.max() - img.min())).unsqueeze(0), scale_factor=n_upsample,
                      align_corners=False, mode='bilinear')

    return img

def edge_transform(img):
    """
    transform img like feature to be visualized
    :param img: [B, C, H, W]
    :return: visualized img range from 0 to 1
    """
    img = img[0].repeat(3, 1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).unsqueeze(0)

    return img
