import torch
from torch import nn
from torchsummary import summary


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # Split qkv before rearranging
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Replace rearrange with native PyTorch operations for q, k, v
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Perform attention on v
        out = torch.matmul(attn, v)

        # Combine the heads and project to output dimension
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for layer in self.layers:
            attn = layer[0]
            ff = layer[1]
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CustomViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, num_image_channels=4, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = num_image_channels * patch_height * patch_width

        # Manual patch embedding to replace Rearrange
        self.patch_size = patch_size
        self.dim = dim
        self.patch_dim = patch_dim

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        b, c, h, w = img.shape

        # Create patches
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, -1, self.patch_dim)

        # Patch embedding
        x = self.patch_to_embedding(patches)
        x = self.norm(x)

        x += self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        x = self.transformer(x)

        return x
if __name__ == "__main__":
    input = torch.randn(1, 4, 192, 192)
    hidded_dim = 256
    depth = 8
    num_heads = 12
    patch_size = 16
    model = CustomViT(image_size=192,
                 patch_size=patch_size,
                 dim=hidded_dim,
                 depth=depth,
                 heads=num_heads,
                 mlp_dim=hidded_dim*4, dim_head=hidded_dim)
    x0 = torch.randn(2, 4, 192, 192)


    summary(model, (4, 192, 192))
    pass