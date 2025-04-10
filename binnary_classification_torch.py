DEBUG = True
import imageio.v3 as iio
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad



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
  print(max_height)
  print(max_width)

  padded_images = []
  padded_masks = []

  for im, mask in zip(images, masks):
    pad_height = max_height - im.shape[1]
    pad_width = max_width - im.shape[2]

    img_padded = pad(im, (0, pad_width, 0, pad_height), value=0)
    mask_padded = pad(mask, (0, pad_width, 0, pad_height), value=0)
    print(img_padded.shape)
    print(mask_padded.shape)

    padded_images.append(img_padded)
    padded_masks.append(mask_padded)

  return torch.stack(padded_images), torch.stack(padded_masks)


def main():
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_files = ['Santo_Stefano_del_Sole_1_1_0',
                      'Atripalda_4_89_0']
  dataset = []
  for in_file in input_files:
      mask_name = f"samples/array_mask_label_region_prov_{in_file}.npz" 
      data_file_name = f"samples/label_region_prov_{in_file}.png"
      data = iio.imread(data_file_name)
      data = np.transpose(data, (2,0,1))
      data = torch.from_numpy(data).float()
      mask = np.load(mask_name)['arr_0']
      mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
      mask = torch.from_numpy(mask).float()
      dataset.append((data,mask))

  loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
  model = TinySegNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.BCEWithLogitsLoss()

  for epoch in range(1):
    model.train()
    for images, masks in loader:
      images = images.to(device)
      masks = masks.to(device)
      print(images)

      preds = model(images)
      loss = loss_fn(preds, masks)

      optimizer.step()
      print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

  if DEBUG:
    print(f"Using device: {device}")
  #padded = custom_collate_fn(batch)
  del dataset
  #del padded
  del model
  del loader

if __name__=='__main__':
  main()
