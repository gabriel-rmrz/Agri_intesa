DEBUG = True
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from torch.utils.data import random_split
from torchvision import transforms

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Down part of UNet
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up part of UNet
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(feature*2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for decoder

        # Decoder
        for up, conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)

            # Pad if needed (in case of uneven sizes)
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])

            x = torch.cat((skip, x), dim=1)
            x = conv(x)

        return self.final_conv(x)

class TinySegNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 16,3, padding = 1), nn.ReLU(),
        nn.Conv2d(16, 32,3, padding=1), nn.ReLU()
        )
    self.decoder = nn.Sequential(
        nn.Conv2d(32, 16,3, padding = 1), nn.ReLU(),
        nn.Conv2d(16, 1,1) 
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


def custom_collate_fn(batch):
  images, masks = zip(*batch)
  
  max_height = max(img.shape[1] for img in images)
  max_width = max(img.shape[2] for img in images)
  #print(max_height)
  #print(max_width)

  padded_images = []
  padded_masks = []

  for im, mask in zip(images, masks):
    pad_height = max_height - im.shape[1]
    pad_width = max_width - im.shape[2]

    img_padded = pad(im, (0, pad_width, 0, pad_height), value=0)
    mask_padded = pad(mask, (0, pad_width, 0, pad_height), value=0)
    #print(img_padded.shape)
    #print(mask_padded.shape)

    padded_images.append(img_padded)
    padded_masks.append(mask_padded)

  return torch.stack(padded_images), torch.stack(padded_masks)


def main():
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with open('input_files.txt', 'r') as f:
    input_files = f.readlines()
  '''
  input_files = ['Santo_Stefano_del_Sole_1_1_0',
                      'Atripalda_4_89_0']
  '''

  dataset = []
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5,0.5,0.5],
                           std=[0.23, 0.23, 0.23])
      ])
  for in_file in input_files:
    in_file = in_file.rstrip()
    mask_name = f"samples/array_mask_label_region_prov_{in_file}.npz" 
    data_file_name = f"samples/label_region_prov_{in_file}.png"
    data = iio.imread(data_file_name)
    data = transform(data)
    print(torch.max(data))
    print(torch.min(data))
    #data = np.transpose(data, (2,0,1))
    mask = np.load(mask_name)['arr_0']
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    mask = torch.from_numpy(mask).float()
    dataset.append((data,mask))
  train_size = int(0.15*len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
  train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=custom_collate_fn, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=custom_collate_fn, shuffle=False)
  #model = TinySegNet().to(device)
  model = UNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.BCEWithLogitsLoss()

  for epoch in range(10):
    model.train()
    for images, masks in train_loader:
      images = images.to(device)
      masks = masks.to(device)
      #print(images)

      preds = model(images)
      loss = loss_fn(preds, masks)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

  if DEBUG:
    print(f"Using device: {device}")
  #padded = custom_collate_fn(batch)
  del dataset
  #del padded
  del model
  del train_loader

if __name__=='__main__':
  main()
