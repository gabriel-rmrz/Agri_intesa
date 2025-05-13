DEBUG = True
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from torch.utils.data import random_split
from torchvision import transforms
import subprocess

def get_gpu_memory():
  result = subprocess.run(
      ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
      stdout=subprocess.PIPE, text=True
  )
  return int(result.stdout.strip().split('\n')[0])  # in MB


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
  def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256]):
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


def get_dataset():
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
    mask = np.load(mask_name)['arr_0']
    mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
    mask = torch.from_numpy(mask).float()
    dataset.append((data,mask))
  return dataset

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
  model.train()
  running_loss = 0

  for images, masks in dataloader:
    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()
    with autocast("cuda", enabled=scaler is not None):
      preds = model(images)
      loss = loss_fn(preds, masks)
    if scaler:
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      optimizer.step()
    running_loss += loss.item()* images.size(0)
  if DEBUG:
    print(loss.item())
  return running_loss/len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn, device):
  model.eval()
  running_loss = 0.0

  with torch.no_grad():
    for images, masks in dataloader:
      images = images.to(device)
      masks = masks.to(device)

      preds = model(images)
      loss = loss_fn(preds, masks)
      running_loss += loss.item()*images.size(0)
  return running_loss/ len(dataloader.dataset)

def test(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
      for images, _ in dataloader:
        images = image.to(device)
        preds = torch.sigmoid(model(images))
        preds = preds > 0.5
        predictions.append(preds.cpu())
    return predictions


def main():
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if DEBUG:
    print('here3')
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
    print("GPU Memory Used (MB):", get_gpu_memory())
  print("- Reading images into dataset.")
  dataset = get_dataset()
  print("- Spliting dataset into training, validation and test samples.")

  train_size = int(0.6*len(dataset))
  val_size = int(0.2*len(dataset))
  test_size = len(dataset) - train_size - val_size
  batch_size= 1
  num_epochs = 10
  
  print(f"n_images: {train_size}")


  train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
  if DEBUG:
    print(f"test_dataset size: {len(test_dataset)}")
    print("here 1")
  print("- Defining loaders.")
  train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False)
  #model = TinySegNet().to(device)
  print("Instantiating model")
  model = UNet(in_channels= 3, out_channels=1).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.BCEWithLogitsLoss()
  scaler = GradScaler('cuda')

  if DEBUG:
    print('here: before training')
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
    print("GPU Memory Used (MB):", get_gpu_memory())
  print("Beginning training.")

  
  for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
  '''
  for epoch in range(num_epochs):
    model.train()
    if DEBUG:
      print(f"1. Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
    for j, (images, masks) in enumerate(train_loader):
      # Size per image (in bytes)
      if DEBUG:
        print(f"mask shape: {masks.shape}")
        print(f"image shape: {images.shape}")
      batch_size, height, width = mask.shape
      channels = 1
      bytes_per_element = mask.element_size()  # usually 4 bytes (float32)
      total_elements_per_image = channels * height * width
      image_size_MB = (total_elements_per_image * bytes_per_element) / (1024 ** 2)

      if DEBUG:
        print(f"Each image ≈ {image_size_MB:.2f} MB")
      images = images.to(device, non_blocking=True)
      if DEBUG:
        print(f"{j}_2. Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
        print("GPU Memory Used (MB):", get_gpu_memory())
      masks = masks.to(device, non_blocking=True)
      if DEBUG:
        print(f"{j}_3. Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
        print("GPU Memory Used (MB):", get_gpu_memory())
      #print(images)

      with autocast('cuda'):
        preds = model(images)
        loss = loss_fn(preds, masks)
      if DEBUG:
        print(f"{j}_loss. Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
        print("GPU Memory Used (MB):", get_gpu_memory())

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      #optimizer.zero_grad()
      #loss.backward()
      #optimizer.step()
      if DEBUG:
        print(f"{j}_8. Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
        print("GPU Memory Used (MB):", get_gpu_memory())
      del images, masks, preds
      torch.cuda.empty_cache()
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
    if DEBUG:
      print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 
      print("GPU Memory Used (MB):", get_gpu_memory())
   # print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
  '''

  if DEBUG:
    print(f"Using device: {device}")
  #padded = custom_collate_fn(batch)
  del dataset
  #del padded
  del model
  del train_loader

if __name__=='__main__':
  main()
